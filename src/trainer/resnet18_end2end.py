import os
import motti
motti.append_parent_dir(__file__)
thisfile = os.path.basename(__file__).split(".")[0]
o_d = motti.o_d()

from args import opts
os.makedirs(opts.log_dir, exist_ok=True)
os.makedirs(opts.ckpt_dir, exist_ok=True)

from model.resnet18 import (
    ResNet18Classifier,
    get_modified_resnet18
)

from dataset import PathMNIST

# data meta
from constant import PathMNISTmeta
import torchvision.transforms as transforms
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import lightning as L


train_transform = transforms.Compose([
    transforms.RandomCrop(opts.img_size, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=PathMNISTmeta.MEAN, std=PathMNISTmeta.STD)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=PathMNISTmeta.MEAN, std=PathMNISTmeta.STD)
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
    collate_fn=PathMNIST.collate_fn
)

if opts.ckpt != "" and os.path.exists(opts.ckpt):
    model = ResNet18Classifier.load_from_checkpoint(
        checkpoint_path=opts.ckpt,
        lr=opts.lr, 
        num_classes=train_dataset.n_classes, 
        warmup_steps=opts.warmup_epochs * len(train_loader), 
        train_steps=opts.max_epochs * len(train_loader),
        criterion= nn.CrossEntropyLoss()
    )
else:
    model = ResNet18Classifier(
        lr=opts.lr, 
        num_classes=train_dataset.n_classes, 
        warmup_steps=opts.warmup_epochs * len(train_loader), 
        train_steps=opts.max_epochs * len(train_loader),
        criterion= nn.CrossEntropyLoss()
    )

checkpoint_callback = ModelCheckpoint(
    save_top_k=1, save_last=True,
    dirpath=os.path.join(opts.ckpt_dir, o_d),
    monitor="val_acc", mode="max"
)

wandblogger = WandbLogger(
    name=f"{o_d}_{thisfile}_{opts.ps}", 
    save_dir=opts.log_dir, 
    project=opts.project,
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