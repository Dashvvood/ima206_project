from typing import Any

import torch
import torch.nn as nn
import lightning as L
import torchvision
from transformers import (
    AutoImageProcessor, 
    ViTMAEForPreTraining,
    ViTImageProcessor,
    ViTMAEConfig
)

from constant import PathMNIST_MEAN, PathMNIST_STD
from utils.cos_warmup_scheduler import get_cosine_schedule_with_warmup


## "facebook/vit-mae-base"
class LitViTMAEForPreTraining(L.LightningModule):
    def __init__(
        self, 
        lr=1e-4,
        warmup_steps=1e3,
        train_steps=1e5,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters()
        self.processor = X = ViTImageProcessor(
            do_normalize=True,
            do_rescale=True,
            do_resize=True,
            image_mean=PathMNIST_MEAN,
            image_std=PathMNIST_STD,
            size=64,
        )
        self.config = ViTMAEConfig(image_size=64)
        self.model = ViTMAEForPreTraining(config=self.config)
        self.lr = lr
        
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.processor(x, return_tensors="pt").pixel_values
        out = self(x)
        loss = out.loss
        
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.processor(x, return_tensors="pt").pixel_values
        out = self(x)
        loss = out.loss
        self.log("val_loss", out.loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, betas=[0.9, 0.999], 
            weight_decay=1e-2,
        )
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.train_steps,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            }
        } 
        
    
        
        