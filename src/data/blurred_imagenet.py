import os
from typing import Literal, Optional, Callable

import datasets as hf_datasets
from datasets import DownloadConfig, Dataset as HuggingFaceDataset
from aiohttp import ClientTimeout


# ===============================
# Constants & Environment Utils
# ===============================
_DEFAULT_TIMEOUT = 1e11  # intentionally very high to avoid timeout
_BASE_DATASET_PATH = 'randall-lab/face-obfuscated-imagenet'

def get_user() -> str:
    """
    Returns current username from environment.
    """
    return os.environ['USER']


def _get_blurred_imagenet_name(num_classes: Literal[10, 100, 1000]) -> str:
    """
    Maps number of classes to dataset variant name
    """
    if num_classes == 10:
        return 'noface-10'
    elif num_classes == 100:
        return 'noface-100'
    elif num_classes == 1000:
        return 'noface-1k'
    raise ValueError(f'Invalid number of classes: {num_classes}')


# ===============================
# Dataset Loading Utility
# ===============================
def make_blurred_imagenet_dataset(
    split: Literal['train', 'validation'],
    num_classes: Literal[10, 100, 1000] = 1000,
    transform: Optional[Callable] = None,
    **download_cfg_kwargs
) -> HuggingFaceDataset:
    """
    Loads a version of ImageNet with obfuscated human faces (via HuggingFace).
    """
    dataset_name = _get_blurred_imagenet_name(num_classes)

    # custom timeout to avoid aiohttp timeout errors
    storage_options = {
        "client_kwargs": {
            "timeout": ClientTimeout(total=_DEFAULT_TIMEOUT)
        }
    }

    # temporary cache directory per user and dataset
    cache_dir = f'/jobtmp/{get_user()}/{dataset_name}/'

    download_config = DownloadConfig(
        cache_dir=cache_dir,
        num_proc=1,
        storage_options=storage_options,
        **download_cfg_kwargs
    )

    dataset = hf_datasets.load_dataset(
        _BASE_DATASET_PATH,
        name=dataset_name,
        split=split,
        trust_remote_code=True,
        streaming=False,  # randall-lab does not support streaming yet
        download_config=download_config
    )

    if transform:
        dataset.set_transform(transform)

    dataset.set_format('torch')
    return dataset
