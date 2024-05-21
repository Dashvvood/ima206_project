import os
import motti
motti.append_parent_dir(__file__)
thisfile = os.path.basename(__file__).split(".")[0]
o_d = motti.o_d()

from functools import partial
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.transforms.functional as VisionF

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18
from torchvision.utils import make_grid
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import WandbLogger

import matplotlib.pyplot as plt
import numpy as np

# self define
from args import opts
os.makedirs(opts.log_dir, exist_ok=True)
os.makedirs(opts.ckpt_dir, exist_ok=True)

from augmentation import BarlowTwinsTransform, pathmnist_normalization
from dataset import PathMNIST

from model.barlow_twins import (
    BarlowTwins,
    BarlowTwinsLoss,
    get_modified_resnet18,
    OnlineFineTuner,
)

train_transform = BarlowTwinsTransform(
    train=True, 
    input_height=opts.img_size, 
    gaussian_blur=False, jitter_strength=0.5, 
    normalize=pathmnist_normalization()
)

val_transform = BarlowTwinsTransform(
    train=False, 
    input_height=opts.img_size, 
    gaussian_blur=False, jitter_strength=0.5, 
    normalize=pathmnist_normalization()
)

train_dataset = PathMNIST(
    split="train", download=False, 
    transform=train_transform,
    root="../../data/medmnist2d/"
)

val_dataset = PathMNIST(
    split="val", download=False, 
    transform=val_transform,
    root="../../data/medmnist2d/"
)


train_loader = DataLoader(
    train_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=opts.num_workers, 
    drop_last=True
)

val_loader = DataLoader(
    val_dataset, batch_size=opts.batch_size, 
    shuffle=False, num_workers=opts.num_workers, 
    drop_last=True
)

