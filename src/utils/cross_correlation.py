from typing import Any
import lightning as L
from lightning.pytorch.callbacks import Callback

from torch import Tensor
from typing import Mapping
import torch
from collections import defaultdict
from torchmetrics import ConfusionMatrix
import wandb
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.classification import MulticlassConfusionMatrix
import PIL

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from matplotlib.collections import QuadMesh
import matplotlib.colors as colors

from utils.common import figure2pil

def pp_cc_matrix(X, figsize=[32, 32]):
    fig = plt.figure(figsize=figsize)
    plt.imshow(X, cmap='Blues', interpolation='nearest')
    plt.axis("off")
    # for i in range(len(X)):
    #     for j in range(len(X[0])):
    #         plt.text(j, i, f"{X[i][j]:.1f}", ha='center', va='center', color='white')
    return fig
    

class LogCrossCorrMatrix(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.memory = {}
    
    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.memory["z1"] = []
        self.memory["z2"] = []
        
    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.memory["z1"].append(outputs.z1.cpu())
        self.memory["z2"].append(outputs.z2.cpu())
        
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        z1 = torch.cat(self.memory["z1"], dim=0)
        z2 = torch.cat(self.memory["z2"], dim=0)
        
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)
        cross_corr = torch.matmul(z1_norm.T, z2_norm) / z1.shape[0]
        fig = pp_cc_matrix(cross_corr, figsize=(32, 32))
        img = figure2pil(fig=fig)

        trainer.logger.log_image(key="val_epoch_cm", images=[img], step=pl_module.current_epoch)
