import os
import motti
motti.append_parent_dir(__file__)
thisfile = os.path.basename(__file__).split(".")[0]
o_d = motti.o_d()

# lightning
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

import numpy as np
import itertools

# self define
from args import opts
os.makedirs(opts.log_dir, exist_ok=True)
os.makedirs(opts.ckpt_dir, exist_ok=True)

# dataset
from medmnist import PathMNIST
from dataset import pathmnist_collate_fn
from augmentation import BarlowTwinPretainTransform
from torch.utils.data import SubsetRandomSampler
from utils.medmnist_subset import get_subset_indices

# model
from model.barlow_twins import BarlowTwinsPretain

# utils
from utils.cross_correlation import LogCrossCorrMatrix

train_transform = BarlowTwinPretainTransform(img_size=opts.img_size)
val_transform = BarlowTwinPretainTransform(img_size=opts.img_size)


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

np.random.seed(42)
subset_indices = get_subset_indices(dataset=train_dataset,  proportion=opts.proportion)
subset_indices = list(itertools.chain(*subset_indices.values())) # inplace


train_loader = DataLoader(
    train_dataset, batch_size=opts.batch_size, 
    num_workers=opts.num_workers, drop_last=True,
    collate_fn=pathmnist_collate_fn,
    sampler=SubsetRandomSampler(indices=subset_indices)
)

val_loader = DataLoader(
    val_dataset, batch_size=opts.batch_size, 
    shuffle=False, num_workers=opts.num_workers, 
    drop_last=False,
    collate_fn=pathmnist_collate_fn,
)


z_dim = 128

if opts.ckpt != "" and os.path.exists(opts.ckpt):
    barlow_model = BarlowTwinsPretain.load_from_checkpoint(
        opts.ckpt,
        lr = opts.lr,
        z_dim=z_dim,
        warmup_steps=opts.warmup_epochs * len(train_loader), 
        train_steps=opts.max_epochs * len(train_loader),
    )
else:
    barlow_model = BarlowTwinsPretain(
        lr = opts.lr,
        z_dim=z_dim,
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
    project=opts.project,
)

trainer = L.Trainer(
    max_epochs=opts.max_epochs,
    accelerator="gpu",
    devices=opts.device_num,
    fast_dev_run=opts.fast,
    logger=wandblogger,
    accumulate_grad_batches=opts.accumulate_grad_batches,
    log_every_n_steps=opts.log_step,
    callbacks=[checkpoint_callback, LogCrossCorrMatrix()],
)

trainer.fit(
    model=barlow_model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
