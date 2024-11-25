import torch
import lightning.pytorch as pl
from monai.networks.nets import AttentionUnet
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from torch.optim import AdamW
import time
import os
import yaml
import munch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('medium')
here = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(here, 'config.yaml')
config = munch.munchify(yaml.load(open(config_file), Loader=yaml.FullLoader))

class AttentionUNetLightningModule(pl.LightningModule):
    def __init__(self, weight_decay=1e-5):
        super().__init__()
        self.model = AttentionUnet(
            spatial_dims=2,  # 2D attention UNet
            in_channels=1,   # Input channels (grayscale images)
            out_channels=config.num_classes,  # Output channels 
            channels=(32, 64, 128, 256, 512),  # Number of filters in each layer
            strides=(2, 2, 2, 2),  # Strides for downsampling
        ).to(self.device)
        
        self.loss_function = DiceCELoss(softmax=True, include_background=False, to_onehot_y=True) # Softmax for multi-class segmentation
        #self.loss_function = DiceFocalLoss(softmax=True, include_background=False, to_onehot_y=True) # Softmax for multi-class segmentation
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, num_classes=config.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        
        # Forward pass
        #y_hat = torch.softmax(self.forward(x), dim=1)
        y_hat = self.forward(x)

        loss = self.loss_function(y_hat, y.long())
        dice = self.dice_metric(y_hat, y.long())

        if torch.isnan(dice).any() or torch.isnan(loss).any():
            return None
                
        self.log_dict({"train/loss": loss, "train/dice": dice.mean()}, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        #y_hat = torch.softmax(self.forward(x), dim=1)
        y_hat = self.forward(x)

        loss = self.loss_function(y_hat, y.long())
        dice = self.dice_metric(y_hat, y.long())

        if torch.isnan(dice).any() or torch.isnan(loss).any():
            return None
        
        self.log_dict({"val/loss": loss, "val/dice": dice.mean()}, on_epoch=True, prog_bar=True)
        return dice
    
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        #y_hat = torch.softmax(self.forward(x), dim=1)
        y_hat = self.forward(x)

        dice = self.dice_metric(y_hat, y.long())

        if torch.isnan(dice).any():
            return None
        
        self.log_dict({"test/dice": dice.mean()}, on_epoch=True, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]
