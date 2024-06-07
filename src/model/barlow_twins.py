from typing import Any

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from torchmetrics import Accuracy

from deprecated import deprecated
from dataclasses import dataclass


@dataclass
class BarlowTwinsOutput:
    x1: torch.Tensor
    x2: torch.Tensor
    z1: torch.Tensor
    z2: torch.Tensor
    loss: torch.Tensor

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        N = z1.shape[0]
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / N

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag
    
@deprecated("Don't use this, keep the original structure")
def get_modified_resnet18():
    encoder = resnet18(zero_init_residual=True)

    # for CIFAR10, replace the first 7x7 conv with smaller 3x3 conv and remove the first maxpool
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    encoder.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

    # replace classification fc layer of Resnet to obtain representations from the backbone
    encoder.fc = nn.Identity()
    
    return encoder


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.input_dim = input_dim,
        self.hidden_dim = hidden_dim,
        self.output_dim = output_dim
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),  # original is False
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)
    

class BarlowTwinsPretain(L.LightningModule):
    def __init__(
        self,
        lambda_coeff=5e-3,
        z_dim=128,
        lr=1e-4,
        warmup_steps=1e3,
        train_steps=1e5,
        save_training_output=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = self._resnet18_backbone()
        self.backbone_out_dim = 512
        
        self.projector = ProjectionHead(input_dim=self.backbone_out_dim , hidden_dim=self.backbone_out_dim , output_dim=z_dim)
        self.critetion = BarlowTwinsLoss(lambda_coeff=lambda_coeff, z_dim=z_dim)
        self.lr = lr
        # self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        
        self.save_training_output = save_training_output
        self.training_step_output: BarlowTwinsOutput = None
        
    @staticmethod
    def _resnet18_backbone():
        model = resnet18(zero_init_residual=True, weights=None)
        model.fc = nn.Identity()
        return model
    
    def forward(self, x):
        return self.backbone(x)

    def shared_step(self, batch):
        (x1, x2), y = batch
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        loss = self.critetion(z1, z2)
        
        return BarlowTwinsOutput(x1=x1, x2=x2, z1=z1, z2=z2, loss=loss)

    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch)
        self.log("train_loss", out.loss, on_step=True, on_epoch=True)
        if self.save_training_output:
            self.training_step_output = out
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch)
        self.log("val_loss", out.loss, on_step=False, on_epoch=True)
        return out
    
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


@dataclass
class BarlowTwinsForImageClassificationOutput:
    X: torch.Tensor
    y: torch.Tensor
    out: Any
    loss: torch.Tensor

class BarlowTwinsForImageClassification(L.LightningModule):
    def __init__(
        self,
        pretrained_model: BarlowTwinsPretain,
        num_classes: int,
        criterion=nn.CrossEntropyLoss(),
        lr=1e-4,
        warmup_steps=1e3,
        train_steps=1e5,
        frozen=True,
        save_training_output=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pretrained_model = pretrained_model
        self.num_classes = num_classes
        
        self.classifier = nn.Linear(
            in_features=pretrained_model.projector.input_dim[0], # this <=> output_dim of resnet18 
            out_features=num_classes
        )
        
        self.frozen = frozen
        if frozen:
            self.pretrained_model.freeze()
        else:
            self.pretrained_model.unfreeze()

        self.criterion = criterion
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        self.acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)

        self.save_training_output = save_training_output
        self.training_step_output: BarlowTwinsOutput = None
        
    def forward(self, inputs):
        x = self.pretrained_model(inputs)
        return self.classifier(x)
    
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
        
    def predict_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        labels_hat = torch.argmax(out.logits, dim=1)
        return labels_hat
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        loss = self.criterion(out, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        
        if self.save_training_output:
            self.training_step_output = BarlowTwinsForImageClassificationOutput(X=X, y=y, out=out, loss=loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        loss = self.criterion(out, y)
        val_acc = self.acc(out, y)
        self.log_dict({
            "val_loss": loss,
            "val_acc": val_acc
        }, on_step=False, on_epoch=True)

        return BarlowTwinsForImageClassificationOutput(X=X, y=y, out=out, loss=loss)
    