import os 
import random 
import numpy as np
import torch 
from skimage.io import imread
from torch.utils.data import Dataset

from utils import crop_sample, pad_sample, resize_sample, normalize_volume

class BrainSegmentationDataset(Dataset):
    in_channels = 3 
    out_channels = 1
    def __init__(
            self,
            images_dir,
            transform=None,
            image_size=256,
            subset = "train",
            random_sampling = True,
            validation_cases=10,
            seed=42
    ):
        assert subset in ["all", "train", "validation"]
        volumes = {}
        masks = {}
        print(f"reading {subset} images ...")
        for (dirpath, _, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                filter(lambda f: ".tif" in f, filenames),
                key=lambda x: int(x.split(".")[-2].split("_")[4]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    mask_slices.append(imread(filepath, as_gray=True))
                else:
                    image_slices.append(imread(filepath))
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])
        self.patients = sorted(volumes)

        if subset != "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=validation_cases)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )

        print(f"preprocessing {subset} volumes ...")
        self.volumes = [(volumes[i], masks[i]) for i in self.patients]

        self.volumes = [crop_sample(v) for v in self.volumes]
        self.volumes = [pad_sample(v) for v in self.volumes]
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

        print(f"preprocessing {subset} volumes done!")   

        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for _, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]
        self.volumes = [(v, m[..., None]) for v, m in self.volumes]
        num_slices = [v.shape[0] for v, _ in self.volumes]
        self.patent_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )
        self.random_sampling = random_sampling
        self.transform = transform
    def __len__(self):
        return len(self.patent_slice_index)
    
    def __getitem__(self, idx):
        patient = self.patent_slice_index[idx][0]
        slice_n = self.patent_slice_index[idx][1]
        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )
        volume, mask = self.volumes[patient]
        image = volume[slice_n]
        mask = mask[slice_n]
        if self.transform:
            image, mask = self.transform((image, mask))
        
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        return image, mask