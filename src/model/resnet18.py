from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import resnet18
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
# from torchmetrics.functional import accuracy
from torchmetrics import Accuracy

def get_modified_resnet18(num_classes):
    model = resnet18(zero_init_residual=True, weights=None)

    # for CIFAR10, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    model.fc = nn.Linear(512, num_classes, bias=True)
    
    return model


class ResNet18Classifier(L.LightningModule):
    def __init__(self, lr, num_classes, criterion, warmup_steps, train_steps) -> None:
        super().__init__()
        
        model = get_modified_resnet18(num_classes=num_classes)
        self.model = model
        
        self.num_classes = num_classes
        self.acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        
        self.lr = lr
        self.criterion = criterion
        
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        
    def forward(self, x):
        self.model(x)
        
    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self.model(X)
        loss = self.criterion(pred, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self.model(X)
        loss = self.criterion(pred, y)
        val_acc = self.acc(pred, y)
        self.log_dict({
            "val_loss": loss,
            "val_acc": val_acc
        }, on_step=False, on_epoch=True)
        
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
