"""Data operations, will be used in eval.py"""
import math
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler

# values that should remain constant
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# data preprocess configs
SCALE = (0.08, 1.0)
RATIO = (3./4., 4./3.)

torch.manual_seed(1)
def create_dataset_train(batch_size=64, train_data_url='', workers=8, distributed=False,
                            input_size=224):
    """Create ImageNet training dataset"""
    if not os.path.exists(train_data_url):
        raise ValueError('Path not exists')

    scale_size = None
    if isinstance(input_size, tuple):
        assert len(input_size) == 2
        if input_size[-1] == input_size[-2]:
            scale_size = int(math.floor(input_size[0] / DEFAULT_CROP_PCT))
        else:
            scale_size = tuple([int(x / DEFAULT_CROP_PCT) for x in input_size])


    else:
        scale_size = int(math.floor(input_size / DEFAULT_CROP_PCT))

    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=SCALE, ratio=RATIO, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    dataset = ImageFolder(train_data_url, transform=transform)

    if distributed:
        sampler = DistributedSampler(dataset)

    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None),
                            num_workers=workers, pin_memory=True, sampler=sampler)

    return dataloader

def create_dataset_val(batch_size=64, val_data_url='', workers=8, distributed=False,
                       input_size=224):
    """Create ImageNet validation dataset"""
    if not os.path.exists(val_data_url):
        raise ValueError('Path not exists')

    scale_size = None
    if isinstance(input_size, tuple):
        assert len(input_size) == 2
        if input_size[-1] == input_size[-2]:
            scale_size = int(math.floor(input_size[0] / DEFAULT_CROP_PCT))
        else:
            scale_size = tuple([int(x / DEFAULT_CROP_PCT) for x in input_size])
    else:
        scale_size = int(math.floor(input_size / DEFAULT_CROP_PCT))

    transform = transforms.Compose([
        transforms.Resize(scale_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    # create dataset in val_data url, in folder there is images folder with images and val_annotations.txt with labels in second column
    dataset = ImageFolder(val_data_url, transform=transform)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=True
    )

    return dataloader

# Note: The split_imgs_and_labels function is not needed in PyTorch as DataLoader handles this automatically
