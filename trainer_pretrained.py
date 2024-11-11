import time

from matplotlib import pyplot as plt
from datamodule_asoca import ASOCADataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchmetrics import Accuracy
import munch
import yaml
from pathlib import Path
import os
import monai
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.auto3dseg import LabelStats
from monai.inferers import SliceInferer, SlidingWindowInferer
from torchvision.models.resnet import resnet50
import torch.nn.functional as F


torch.set_float32_matmul_precision('medium')
here = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(here, 'config.yaml')
config = munch.munchify(yaml.load(open(config_file), Loader=yaml.FullLoader))


class ConvRelu(pl.LightningModule):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_, out, kernel_size=3, padding=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class DecoderBlock(pl.LightningModule):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                padding=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.block(x)


class VGGUnet(pl.LightningModule):
    def __init__(self, config, num_filters=32, pretrained=True):
        super().__init__()
        self.config = config
        self.out_channels = config.num_classes

        self.loss_fn = DiceFocalLoss(sigmoid=True)
        self.acc_fn = DiceMetric(include_background=True, reduction="mean", num_classes=self.out_channels, get_not_nans=False) # HD95 is an alternative

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg16(pretrained=pretrained).features

        # Freezing encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   self.relu,
                                   #nn.Dropout2d(p=0.1),
                                   self.encoder[2],
                                   nn.BatchNorm2d(64),
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   nn.BatchNorm2d(128),
                                   self.relu,
                                   #nn.Dropout2d(p=0.1),
                                   self.encoder[7],
                                   nn.BatchNorm2d(128),
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   nn.BatchNorm2d(256),
                                   self.relu,
                                   #nn.Dropout2d(p=0.1),
                                   self.encoder[12],
                                   nn.BatchNorm2d(256),
                                   self.relu,
                                   #nn.Dropout2d(p=0.1),
                                   self.encoder[14],
                                   nn.BatchNorm2d(256),
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   nn.BatchNorm2d(512),
                                   self.relu,
                                   #nn.Dropout2d(p=0.1),
                                   self.encoder[19],
                                   nn.BatchNorm2d(512),
                                   self.relu,
                                   #nn.Dropout2d(p=0.1),
                                   self.encoder[21],
                                   nn.BatchNorm2d(512),
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   nn.BatchNorm2d(512),
                                   self.relu,
                                   #nn.Dropout2d(p=0.1),
                                   self.encoder[26],
                                   nn.BatchNorm2d(512),
                                   self.relu,
                                   #nn.Dropout2d(p=0.1),
                                   self.encoder[28],
                                   nn.BatchNorm2d(512),
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, self.out_channels, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        x_out = self.final(dec1)

        return x_out
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.max_lr) # For L2: weight_decay=self.config.weight_decay
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
        
        start_time = time.time()  # Record start time
        y_hat = self.forward(x)
        end_time = time.time()  # Record end time
        inference_time = end_time - start_time  # Calculate inference time
        acc = self.acc_fn(y_hat > 0.5, y)

        if torch.isnan(acc).any():
            return None

        self.log_dict({
            "test/acc": acc.mean(),
            "test/inference_time": inference_time  # Log inference time
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)


# Visualizes the segmentation of an image
def visualize_segmentation(image, true_label, predicted_label, acc, type):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(type)  # Set overall title

    axs[0].imshow(image.permute(1, 2, 0))  # Original image
    axs[0].set_title('Original Image')

    axs[1].imshow(true_label.squeeze(), cmap='gray')  # True label
    axs[1].set_title('True Label')

    axs[2].imshow(predicted_label.squeeze(), cmap='gray')  # Predicted label
    axs[2].set_title('Predicted Label')

    # Display accuracy
    axs[2].set_xlabel(f"Accuracy: {acc.mean():.4f}")

    plt.show()

if __name__ == "__main__":
    
    pl.seed_everything(42)

    dm = ASOCADataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split_ratio=config.train_split_ratio,
        data_root=config.data_root)
    
    print(dm)

    if config.checkpoint_path:
        model = VGGUnet.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        print("Loading weights from checkpoint...")
    else:
        model = VGGUnet(pretrained=True, config=config)

    trainer = pl.Trainer(
        devices=config.devices, 
        max_epochs=config.max_epochs, 
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_progress_bar=config.enable_progress_bar,
        precision="bf16-mixed",
        # deterministic=True,
        logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
        callbacks=[
            EarlyStopping(monitor="val/acc", patience=config.early_stopping_patience, mode="max", verbose=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
                            filename='best_model', # :epoch={epoch:02d}-val_acc={val/acc:.4f}
                            auto_insert_metric_name=False,
                            save_weights_only=True,
                            save_top_k=1),
        ])
    
    if not config.test_model:
        print("Fitting model...")
        trainer.fit(model, datamodule=dm)
    
    trainer.test(model, datamodule=dm)

    # Visualize examples of segmentation
    """ model.eval()
    data_iter = iter(dm.normal_test_dataloader())
    num_batches_to_skip = 8
    num_images_to_visualize = 5
    acc_fn = DiceMetric(include_background=True, reduction="mean", num_classes=1, get_not_nans=False)
    for _ in range(0, num_images_to_visualize):
        for _ in range(0, num_batches_to_skip):
            next(data_iter)
        batch = next(data_iter)
        images = batch["image"]
        labels = batch["label"]
        with torch.no_grad(): # Disable gradient calculation
            y_hat = model(images)
            acc = acc_fn(y_hat > 0.5, labels)
        visualize_segmentation(images[0], labels[0], y_hat[0], acc, "Normal")

    data_iter = iter(dm.diseased_test_dataloader())
    for _ in range(0, num_images_to_visualize):
        for _ in range(0, num_batches_to_skip):
            next(data_iter)
        batch = next(data_iter)
        images = batch["image"]
        labels = batch["label"]
        with torch.no_grad(): # Disable gradient calculation
            y_hat = model(images)
            acc = acc_fn(y_hat > 0.5, labels)
        visualize_segmentation(images[0], labels[0], y_hat[0], acc, "Diseased") """