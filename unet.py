from datamodule import DataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
import munch
import yaml
from pathlib import Path
import os
import monai.networks
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import torch.nn.functional as F
import torchio as tio

# Baseline model
# Based on code from Aladdin Persson and the Pytorch Lightning tutorial posted in BB
class DoubleConv(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5),  # Add dropout layer with dropout probability of 0.5
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.5)  # Add dropout layer with dropout probability of 0.5
        )

    def forward(self, x):
        return self.double_conv(x)


class U_Net(pl.LightningModule):
    def __init__(
            self, config, features=[64, 128, 256, 512]
    ):
        super(U_Net, self).__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.num_classes

        self.loss_fn = DiceFocalLoss(sigmoid=True) 
        #self.loss_fn = HausdorffDistanceMetric()
        #self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.acc_fn = DiceMetric(include_background=True, reduction="mean", num_classes=self.out_channels, get_not_nans=False) # HD95 is an alternative

        # ModuleList -> To do model.eval(), model.train() etc
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 

        # Down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(self.in_channels, feature))
            self.in_channels = feature
        
        # Up part of U-Net
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            ) # Doubles the height and width of the input
            # Can be improved by implementing something else (ref video) 
            self.ups.append(DoubleConv(feature*2, feature))
        
        
        # Bottom of the U-Net
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], self.out_channels, kernel_size=1) # Changes number of channels to out_channels

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse the list

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] 

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.max_lr) #, weight_decay=self.config.weight_decay
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat > 0.5, y)
        loss = self.loss_fn(y_hat, y)

        if torch.isnan(acc).any() or torch.isnan(loss).any():
            return None

        self.log_dict({
            "train/loss": loss,
            "train/acc": acc.mean()
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat > 0.5, y)
        loss = self.loss_fn(y_hat, y)

        if torch.isnan(acc).any() or torch.isnan(loss).any():
            return None
            
        self.log_dict({
            "val/loss":loss,
            "val/acc": acc.mean()
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

    
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
            
        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat > 0.5, y)

        if torch.isnan(acc).any():
            return None

        self.log_dict({
            "test/acc": acc.mean(),
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)