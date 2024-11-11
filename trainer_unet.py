import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import munch
import yaml
import os
from pathlib import Path
import monai
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, EnsureType
from monai.data import CacheDataset, decollate_batch
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
import torchio as tio
from datamodule import DataModule
import wandb

# Set global precision setting for matrix multiplication
torch.set_float32_matmul_precision('medium')

# Load the configuration file
here = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(here, 'config.yaml')
config = munch.munchify(yaml.load(open(config_file), Loader=yaml.FullLoader))

# Fix random seed for reproducibility
torch.manual_seed(42)

# Initialize WandB logging
wandb.init(project=config.wandb_project, name=config.wandb_experiment_name, config=config)

# Data module initialization
dm = DataModule(
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    train_split_ratio=config.train_split_ratio,
    data_root=config.data_root
)

# Model definition using MONAI's UNet
model = UNet(
    spatial_dims=3,
    in_channels=config.in_channels,
    out_channels=config.num_classes,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2)
).to(config.device)

# Loss function and optimizer
criterion = DiceCELoss(softmax=True)
optimizer = optim.AdamW(model.parameters(), lr=config.max_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# Metrics
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Directory for saving checkpoints
checkpoint_dir = Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

def train_one_epoch(epoch, model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs, labels = batch["image"].to(config.device), batch["label"].to(config.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch} - Training Loss: {epoch_loss:.4f}")
    return epoch_loss

def validate_one_epoch(epoch, model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["image"].to(config.device), batch["label"].to(config.device)
            outputs = sliding_window_inference(inputs, roi_size=(128, 128, 128), sw_batch_size=4, predictor=model)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Compute Dice Metric
            outputs = torch.softmax(outputs, 1)
            dice_metric(y_pred=outputs, y=labels)
    
    epoch_loss = running_loss / len(dataloader)
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"Epoch {epoch} - Validation Loss: {epoch_loss:.4f}, Dice Score: {dice_score:.4f}")
    return epoch_loss, dice_score

def save_checkpoint(model, optimizer, epoch, val_loss):
    checkpoint_path = checkpoint_dir / f"model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

def main():
    train_loader = dm.train_dataloader()
    val_loader = dm.train_dataloader()

    best_val_loss = float("inf")

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion)
        val_loss, dice_score = validate_one_epoch(epoch, model, val_loader, criterion)

        # Log metrics to WandB
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "dice_score": dice_score, "epoch": epoch})

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss)

    print("Training Complete!")

if __name__ == "__main__":
    main()
