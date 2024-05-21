from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from torchmetrics.functional import accuracy

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag
    
    
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

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),  # original is False
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)
    
    
class BarlowTwins(L.LightningModule):
    def __init__(
        self,
        encoder,
        encoder_out_dim,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=128,
        learning_rate=1e-4,
        warmup_steps=1e3,
        train_steps=1e5,
        max_epochs=200,
    ):
        super().__init__()

        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=encoder_out_dim, hidden_dim=encoder_out_dim, output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)

        self.learning_rate = learning_rate
        # self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        self.max_epochs = max_epochs

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        (x1, x2, _), _ = batch

        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))

        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)  
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, betas=[0.9, 0.999], 
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
        
        
class OnlineFineTuner(Callback):
    def __init__(
        self,
        encoder_output_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # add linear_eval layer and optimizer
        pl_module.online_finetuner = nn.Linear(self.encoder_output_dim, self.num_classes).to(pl_module.device)
        self.optimizer = torch.optim.AdamW(pl_module.online_finetuner.parameters(), lr=1e-4)

    def extract_online_finetuning_view(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (_, _, finetune_view), y = batch
        finetune_view = finetune_view.to(device)
        y = y.to(device)

        return finetune_view, y

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y.squeeze())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        acc = accuracy(F.softmax(preds, dim=1), y.squeeze(), task="multiclass", num_classes=self.num_classes)
        pl_module.log("online_train_acc", acc, on_step=True, on_epoch=False)
        pl_module.log("online_train_loss", loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y.squeeze())

        acc = accuracy(F.softmax(preds, dim=1), y.squeeze(), task="multiclass", num_classes=self.num_classes)
        pl_module.log("online_val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

class BarlowTwinsForImageClassification(L.LightningModule):
    def __init__(
        self, 
        backbone,
        embedding_dim: int,
        num_class: int,
        criterion=nn.CrossEntropyLoss(),
        learning_rate=1e-4,
        warmup_steps=1e3,
        train_steps=1e5,
        max_epochs=200,
        finetune=True,
        
    ):
        super().__init__()
        
        self.backbone = backbone
        
        self.classifier = nn.Linear(
            in_features=embedding_dim, 
            out_features=num_class
        )
        self.finetune = finetune
        if finetune:
            self.backbone.eval()
        else:
            self.backbone.train()
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        self.max_epochs = max_epochs
        
    def forward(self, inputs):
        if self.finetune:
            with torch.no_grad():
                x = self.backbone.encoder(inputs)
                x = self.backbone.projection_head(x)
                x = x.detach()
        else:
            x = self.backbone(inputs)
        return self.classifier(x)
    
    def configure_optimizers(self):
        if self.finetune:
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.learning_rate, betas=[0.9, 0.999], 
                weight_decay=1e-2,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.classifier.parameters(), 
                lr=self.learning_rate, betas=[0.9, 0.999], 
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
        x, y = batch
        out = self(x)
        labels_hat = torch.argmax(out.logits, dim=1)
        return labels_hat
    
    def training_step(self, batch, batch_idx):
        (x1, x2, x), y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        preds = self(x)
        
        loss = self.criterion(preds, y.squeeze())
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        (x1, x2, x), y = batch
        x = x.to(self.device)
        y = y.squeeze()
        y = y.to(self.device)
        out = self(x)
        
        loss = self.criterion(out, y)
        
        labels_hat = torch.argmax(out, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log_dict({'val_loss': loss, 'val_acc': val_acc}, on_step=False, on_epoch=True)
