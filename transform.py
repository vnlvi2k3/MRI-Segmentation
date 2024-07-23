import numpy as np 
from skimage.transform import rescale, rotate 
from torchvision.transforms import Compose 

class Scale:
    def __init__(self, scale):
        self.scale_ratio = scale 

    def __call__(self, sample):
        image, mask = sample

        original_size = image.shape[0]
        scale = np.random.uniform(low=1.0 - self.scale_ratio, high = 1.0 + self.scale_ratio)

        image = rescale(
            image,
            (scale, scale),
            channel_axis=-1,
            preserve_range=True,
            mode = 'constant',
            anti_aliasing = False
        )
        mask = rescale(
            mask,
            (scale, scale),
            order = 0, #nearest-neighbor interpolation
            channel_axis=-1, #preserve all channels
            preserve_range = True, #keep the original range of values
            mode = 'constant', #fill the extra pixels with zeros
            anti_aliasing = False #preserve the edges
        )

        if scale < 1.0:
            diff = (original_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - original_size) // 2
            x_max = x_min + original_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask

class Rotate:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample
        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(
            image,
            angle,
            resize=False,
            preserve_range=True,
            mode="constant",
        )
        mask = rotate(
            mask,
            angle,
            resize=False,
            order=0,
            preserve_range=True,
            mode="constant",
        )
        return image, mask
    
class HorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample 

        if np.random.rand() > self.flip_prob:
            return image, mask
        
        image = np.fliplr(image).copy() #flip left-right
        mask = np.fliplr(mask).copy()
        return image, mask
    
def transforms(scale=None, angle=None, flip_prob=None):
    transforms_list = []
    if scale is not None:
        transforms_list.append(Scale(scale))
    if angle is not None:
        transforms_list.append(Rotate(angle))
    if flip_prob is not None:
        transforms_list.append(HorizontalFlip(flip_prob))
    return Compose(transforms_list)
