from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import Accuracy

import lightning as L
from dataclasses import dataclass, asdict

def _get_modified_resnet18(num_classes):
    model = resnet18(zero_init_residual=True, weights=None)

    # for CIFAR10, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
    model.fc = nn.Linear(512, num_classes, bias=True)
    
    return model


@dataclass
class ResNet18ClassifierOutput:
    X: torch.Tensor
    y: torch.Tensor
    loss: torch.Tensor
    out: Any
    

class ResNet18Classifier(L.LightningModule):
    def __init__(self, lr, num_classes, criterion, warmup_steps, train_steps, save_training_output=False) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # model = get_modified_resnet18(num_classes=num_classes)
        model = self._resnet18_n_class(num_classes=num_classes)
        self.model = model
        
        self.num_classes = num_classes
        self.acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        
        self.lr = lr
        self.criterion = criterion
        
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps

        # Epoch-level Operations
        self.save_training_output = save_training_output
        self.training_step_output: ResNet18ClassifierOutput = None
        
    @staticmethod
    def _resnet18_n_class(num_classes):
        model = resnet18(zero_init_residual=True, weights=None)
        model.fc = nn.Linear(512, num_classes, bias=True)
        return model

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        X, y = batch
        out = self.model(X)
        loss = self.criterion(out, y)
        
        self.log_dict({
            "train_loss": loss,
        }, on_step=True, on_epoch=True)
        
        if self.save_training_output:
            self.training_step_output = ResNet18ClassifierOutput(X=X, y=y, loss=loss, out=out)
        
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        out = self.model(X)
        loss = self.criterion(out, y)
        val_acc = self.acc(out, y)
        
        self.log_dict({
            "val_loss": loss,
            "val_acc": val_acc
        }, on_step=False, on_epoch=True)
        
        return ResNet18ClassifierOutput(X=X, y=y, loss=loss, out=out)

    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch
        out = self(X)
        loss = self.criterion(out, y)
        test_acc = self.acc(out, y)
        
        self.log_dict({
            "test_loss": loss, 
            "test_acc": test_acc,
        }, on_epoch=True)
        
        return ResNet18ClassifierOutput(X=X, y=y, out=out, loss=loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
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
