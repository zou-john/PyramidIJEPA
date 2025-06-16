from typing import Literal, Optional, Tuple
import torch
import torchvision
from torch.utils.data import random_split

from src.data.blurred_imagenet import BLURRED_IMAGENET_MEAN, BLURRED_IMAGENET_STD, make_blurred_imagenet_dataset

# ===============================
# NORMALIZATION CONSTANTS
# ===============================
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
BLURRED_IMAGENET_MEAN, BLURRED_IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# ===============================
# NORMALIZATION (TRAINING) AND DENORMALIZATION (RECONSTRUCTION)
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
# DATASET AUGMENTATION (DEFAULT)
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
# GET DATASET
# ===============================
def get_dataset(
        dataset: Literal['cifar10', 'cifar100', 'blurred_imagenet'] = 'cifar10', # datasets
        pretrain: bool = True, # there is a different set for pretraining and finetuning
        validation: bool = False, # whether we use the validation set
        crop_size: int = 28, # the size of the crop
        download: bool = True, # whether to download the dataset
        unsample: Optional[int] = None, # the size of the image after resizing
        scale: Tuple[float, float] = (0.3, 1.0), # the scale of the crop    
        imagenet_num_classes: Literal[10, 100, 1000] = 1000, # the number of classes in the imagenet dataset
        **imagenet_download_kwargs
    ):
    """
    Returns a dataset object for the given dataset.
    If dataset is 'cifar10', 'cifar100', or 'imagenet', it will return a torchvision.datasets.Dataset object.
    If dataset is 'blurred_imagenet', it will return a HuggingFaceDataset object.
    """
    # get the mean and std of the dataset
    if dataset == 'cifar10':
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset == 'cifar100':
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    elif dataset == 'blurred_imagenet':
        mean, std = BLURRED_IMAGENET_MEAN, BLURRED_IMAGENET_STD
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    # get the transforms for the dataset
    transforms = get_transforms(
        normalization=normalize_image_fn(mean, std),
        unsample=unsample,
        crop_size=crop_size,
        scale=scale,
        pretrain=pretrain
    )

    # pretrain and validation sets are different for each dataset
    match (pretrain, validation):
        case (True, True):
            split = 'pretrain' 
        case (True, False):
            split = 'finetune_train' 
        case (False, True):
            split = 'finetune_validation'
    
    if dataset == 'cifar10':
        my_dataset = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=pretrain, download=download, transform=transforms)
    elif dataset == 'cifar100':
        my_dataset = torchvision.datasets.CIFAR100(root='/tmp/cifar100', train=pretrain, download=download, transform=transforms)
    elif dataset == 'blurred_imagenet':
        # lobotomized iterabledataset
        split = 'train' if split in ['pretrain', 'finetune_train'] else 'validation'
        my_dataset = make_blurred_imagenet_dataset(split=split, 
                                                   num_classes=imagenet_num_classes, 
                                                   transform=transforms, 
                                                   **imagenet_download_kwargs)
        return my_dataset
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    if split == 'pretrain': return my_dataset
    elif split == 'finetune_train': return random_split(my_dataset, [0.75, 0.25])[0]
    elif split == 'finetune_validation': return random_split(my_dataset, [0.25, 0.75])[1]
    else: raise ValueError(f"Invalid split: {split}")
