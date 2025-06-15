from typing import Literal, Optional
import torch
import torchvision
import math

# for default usage we use the CIFAR10 dataset
from src.dataset.cifar10 import CIFAR10_MEAN, CIFAR10_STD
from src.dataset.mask import RandomMask, MultiBlock

# ===============================
# NORMALIZATION (TRAININGAND DENORMALIZATION (RECONSTRUCTION)
# ===============================
def denormalize_image_fn(mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """
    Denormalizes an image from a given dataset for reconstruction.
    """
    mean, std = torch.as_tensor(mean), torch.as_tensor(std)
    inverse_std = 1 / std + 1e-7
    inverse_mean = -mean / std
    return torchvision.transforms.Normalize(inverse_mean, inverse_std)

def denormalize_image(image: torch.Tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """
    Applies the denormalization on to the image.
    """
    return denormalize_image_fn(mean, std)(image)

def normalize_image_fn(mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """
    Normalizes an image from a given dataset for training.
    """
    mean, std = torch.as_tensor(mean), torch.as_tensor(std)
    return torchvision.transforms.Normalize(mean, std)

def normalize_image(image: torch.Tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """
    Applies the normalization on to the image.
    """
    return normalize_image_fn(mean, std)(image)

# ===============================
# DATASET AUGMENTATION
# ===============================
def get_transforms(
    normalization,
    unsample=None,
    crop_size=28,
    scale=(0.3, 1.0),
    pretrain=False
):
    """
    Returns a transformation pipeline for the given dataset. From PIL to tensor with [0, 1] range for each channel.
    """
    # resize the image to the given size
    resize = torchvision.transforms.Resize(unsample) if unsample is not None else torch.nn.Identity()
    # if pretrain, we randomly select a scaled portion of the image and then resize it to the given crop size
    random_resized_crop = torchvision.transforms.RandomResizedCrop(crop_size, scale=scale) if pretrain else torchvision.transforms.CenterCrop(crop_size)
    # convert the image to a tensor
    to_tensor = torchvision.transforms.ToTensor()
    # compose the transformations
    return torchvision.transforms.Compose([
        resize,
        random_resized_crop,
        to_tensor,
        normalization
    ])

# ===============================
# MASKING STRATEGY
# ===============================
def collate_supervised(batch: list[dict[str, torch.Tensor]]):
    """
    Collate a batch of data for supervised training.
    """
    # a full batch is a list of dictionaries with the following keys: 'image', 'label'
    images, labels = zip(*batch)
    # [1, C, H, W], [1, C, H, W] ... B -> [B, C, H, W]
    stacked_images = torch.stack(images, dim=0).squeeze(1)
    # [1], [1] ... B -> [B]
    stacked_labels = torch.stack(labels, dim=0)

    return stacked_images, stacked_labels

def collate_masked(
    mask_generator: RandomMask | MultiBlock = RandomMask,
    downsampling_variant: Literal['fine',
                                  'coarse',
                                  'full_downsampled',
                                  'full_resampled',
                                  'none',
                                  ] = 'full_downsampled',
    num_scales: int = 1, 
    num_patches: int = 16
):
    """
    Collate a batch of data for masked training.
    """
    # NOTE: assuming it is square images
    latent_height = latent_width = int(math.sqrt(num_patches))
    # contraints the number of scales if exceeded the nummber of square root downsampling
    # EXAMPLE: for a latent width of 25 x 40 and the parameter num_scales = 5, the number of scales will be 1 + min(5, math.floor(math.log2(min(25, 40)))) = 3
    # which means you can only have 4 scales.
    num_scales = 1 + min(num_scales, math.floor(math.log2(min(latent_height, latent_width))))

    mask_data 
    # create a mask generator
    mask_generator = mask_generator(
        latent_height,
        latent_width,
        num_scales,
        downsampling_variant
    )


    
    # a full batch is a list of dictionaries with the following keys: 'image', 'label'
    images, labels = zip(*batch)
    # [1, C, H, W], [1, C, H, W] ... B -> [B, C, H, W]


def get_dataset(dataset: Literal['cifar10', 'cifar100', 'imagenet'] = 'cifar10'):
    """

    """
    if dataset == 'cifar10':
        return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=normalize_image_fn())
    elif dataset == 'cifar100':
        return torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=normalize_image_fn())
    elif dataset == 'imagenet':
        return torchvision.datasets.ImageNet(root='./data', split='train', download=True, transform=normalize_image_fn())