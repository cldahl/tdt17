import torch
import lightning.pytorch as pl
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.optim import AdamW
from datamodule import DataModule
import os
import yaml
import munch
from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
import time


# ========== Step 1: Set Device ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('medium')
here = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(here, 'config.yaml')
config = munch.munchify(yaml.load(open(config_file), Loader=yaml.FullLoader))

# ========== Step 2: Define the Lightning Module ==========
class SegResNetLightningModule(pl.LightningModule):
    def __init__(self, weight_decay=1e-5):
        super().__init__()
        self.model = SegResNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            init_filters=16, # TODO: Possibly change this to 32
            dropout_prob=0.2
        ).to(device)
        self.out_channels = config.num_classes
        self.loss_function = DiceCELoss(include_background=False, sigmoid=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=self.out_channels, get_not_nans=False)
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        y_hat = self.forward(x)
        dice = self.dice_metric(y_hat > 0.5, y)
        loss = self.loss_function(y_hat, y)

        if torch.isnan(dice).any() or torch.isnan(loss).any():
            return None

        self.log_dict({
            "train/loss": loss,
            "train/acc": dice.mean()
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        y_hat = self.forward(x)
        dice = self.dice_metric(y_hat > 0.5, y)
        loss = self.loss_function(y_hat, y)

        #outputs = sliding_window_inference(inputs, (512, 512), 4, self.model)
        #val_loss = self.loss_function(outputs, labels)

        if torch.isnan(dice).any() or torch.isnan(loss).any():
            return None
        
        self.log_dict({
            "val/loss":loss,
            "val/acc": dice.mean()
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return dice
    
    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        #outputs = sliding_window_inference(inputs, (512, 512), 4, self.model)
        #test_loss = self.loss_function(outputs, labels)
        start_time = time.time()  # Record start time
        y_hat = self.forward(x)
        end_time = time.time()  # Record end time
        inference_time = end_time - start_time  # Calculate inference time
        test_loss = self.loss_function(y_hat, y)
        #self.log("test_loss", test_loss, prog_bar=True)

        # Calculate Dice score
        dice_score = self.dice_metric(y_hat > 0.5, y)
        if torch.isnan(dice_score).any():
            return None
        
        self.log_dict({
            "test/acc": dice_score.mean(),
            "test/inference_time": inference_time  # Log inference time
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return dice_score

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=config.max_lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

# ========== Step 3: Train the Model ==========
def main():
    # Initialize DataModule
    dm = DataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        data_root=config.data_root,
        train_split_ratio=config.train_split_ratio,
        val_split_ratio=config.val_split_ratio,
    )

    # Initialize the model
    model = SegResNetLightningModule()

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        devices=config.devices,
        log_every_n_steps=10,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_progress_bar=config.enable_progress_bar,
        precision="bf16-mixed",
        logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
        callbacks=[
            EarlyStopping(monitor="val/acc", patience=config.early_stopping_patience, mode="max", verbose=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
                            filename='best_model', # :epoch={epoch:02d}-val_acc={val/acc:.4f}
                            auto_insert_metric_name=False,
                            save_weights_only=True,
                            save_top_k=1),]
    )

    # Train the model
    if not config.test_model:
        trainer.fit(model, datamodule=dm)

    # Test the model
    trainer.test(model, datamodule=dm)

# ========== Step 4: Run the Script ==========
if __name__ == "__main__":
    main()
