import argparse
import json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainSegmentationDataset as Dataset
from logger import Logger
from loss import DiceLoss
from transform import transforms
from unet import Unet
from utils import log_images, dsc


def main(args):
    device, loaders, unet, optimizer, dsc_loss, logger = setup(args)
    best_validation_dsc = 0.0
    step = 0

    for _ in tqdm(range(args.epochs), total=args.epochs):
        step = train(loaders["train"], unet, optimizer, dsc_loss, device, logger, step)
        best_validation_dsc = validate_and_save(loaders["valid"], unet, dsc_loss, device, logger, args, best_validation_dsc, step)

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))

def setup(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}
    unet = Unet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)
    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    dsc_loss = DiceLoss()
    logger = Logger(args.logs)
    return device, loaders, unet, optimizer, dsc_loss, logger

def train(loader, unet, optimizer, dsc_loss, device, logger, step):
    unet.train()
    loss_train = []
    for data in loader:
        step += 1
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)
        optimizer.zero_grad()
        y_pred = unet(x)
        loss = dsc_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
        if step % 10 == 0:
            log_loss_summary(logger, loss_train, step)
            loss_train = []
    return step

def validate_and_save(loader, unet, dsc_loss, device, logger, args, best_validation_dsc, step):
    unet.eval()
    validation_pred, validation_true, loss_valid = [], [], []
    with torch.no_grad():
        for data in loader:
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)
            y_pred = unet(x)
            loss = dsc_loss(y_pred, y_true)
            loss_valid.append(loss.item())
            validation_pred.extend(y_pred.detach().cpu().numpy())
            validation_true.extend(y_true.detach().cpu().numpy())
    log_loss_summary(logger, loss_valid, step, prefix="val_")
    mean_dsc = np.mean(dsc_per_volume(validation_pred, validation_true, loader.dataset.patient_slice_index))
    logger.scalar_summary("val_dsc", mean_dsc, step)
    if mean_dsc > best_validation_dsc:
        best_validation_dsc = mean_dsc
        torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
    return best_validation_dsc

def log_loss_summary(logger, losses, step, prefix=""):
    avg_loss = np.mean(losses)
    logger.scalar_summary(f"{prefix}loss", avg_loss, step)



def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    )
    valid = Dataset(
        images_dir=args.images,
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
    )
    return train, valid


def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    args = parser.parse_args()
    main(args)
