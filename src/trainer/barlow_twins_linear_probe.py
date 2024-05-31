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
    BarlowTwinsForImageClassification
)

train_transform = transforms.Compose([
    transforms.RandomCrop(opts.img_size, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    pathmnist_normalization(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    pathmnist_normalization(),
])

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
    drop_last=True,
    collate_fn=PathMNIST.collate_fn,
)

val_loader = DataLoader(
    val_dataset, batch_size=opts.batch_size, 
    shuffle=False, num_workers=opts.num_workers, 
    drop_last=False,
    collate_fn=PathMNIST.collate_fn,

)

encoder = get_modified_resnet18()
encoder_out_dim = 512
z_dim = 128

if opts.ckpt != "" and os.path.exists(opts.ckpt):
    barlow_model = BarlowTwins.load_from_checkpoint(
        opts.ckpt,
        encoder=encoder,
        encoder_out_dim=encoder_out_dim,
        num_training_samples=len(train_dataset),
        batch_size=opts.batch_size,
        z_dim=z_dim,
    )
else:
    barlow_model = BarlowTwins(
        encoder=encoder,
        encoder_out_dim=encoder_out_dim,
        num_training_samples=len(train_dataset),
        batch_size=opts.batch_size,
        z_dim=z_dim,
    )

model = BarlowTwinsForImageClassification(
    backbone=barlow_model,
    embedding_dim=encoder_out_dim,
    num_classes=train_dataset.n_classes,
    criterion=nn.CrossEntropyLoss(),
    finetune=True,
    warmup_steps=opts.warmup_epochs * len(train_loader),
    train_steps=opts.max_epochs * len(train_loader),
)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1, save_last=True,
    dirpath=os.path.join(opts.ckpt_dir, o_d),
    monitor="val_acc", mode="max"
)

wandblogger = WandbLogger(
    name=f"{o_d}_{thisfile}_{opts.ps}", 
    save_dir=opts.log_dir, 
    project="barlow_twins",
)

trainer = L.Trainer(
    max_epochs=opts.max_epochs,
    accelerator="gpu",
    devices=opts.device_num,
    fast_dev_run=opts.fast,
    logger=wandblogger,
    accumulate_grad_batches=opts.accumulate_grad_batches,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback],
)

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)