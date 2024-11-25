from datamodule import DataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
import munch
import yaml
from pathlib import Path
import os
from unet import U_Net
from attentionunet import AttentionUNetLightningModule


torch.set_float32_matmul_precision('medium')
here = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(here, 'config.yaml')
config = munch.munchify(yaml.load(open(config_file), Loader=yaml.FullLoader))

if __name__ == "__main__":
    
    pl.seed_everything(42)

    dm = DataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split_ratio=config.train_split_ratio,
        val_split_ratio=config.val_split_ratio,
        data_root=config.data_root)
    
    #unet = monai.networks.nets.UNet(
            #in_channels=config.in_channels,
            #out_channels=config.num_classes,
            #channels=(16, 32, 64, 128, 256),
            #strides=(2, 2, 2, 2),
            #spatial_dims=3
        #)
    
    print(dm)

    if config.checkpoint_path:
        #model = U_Net.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        model = AttentionUNetLightningModule.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        #model = Model(unet.load_from_checkpoint(checkpoint_path=config.checkpoint_path, in_channels=config.in_channels, out_channels=config.num_classes, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), spatial_dims=3))
        print("Loading weights from checkpoint...")
    else:
        #model = U_Net(config)
        model = AttentionUNetLightningModule(config)
        #model = Model(
            #net=unet,
            #criterion=monai.losses.DiceCELoss(softmax=True),
            #learning_rate=1e-2,
            #optimizer_class=torch.optim.AdamW,
        #)

    trainer = pl.Trainer(
        devices=config.devices, 
        max_epochs=config.max_epochs, 
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_progress_bar=config.enable_progress_bar,
        precision="bf16-mixed",
        log_every_n_steps=5,
        # deterministic=True,
        logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
        callbacks=[
            EarlyStopping(monitor="val/dice", patience=config.early_stopping_patience, mode="max", verbose=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
                            filename='best_model:epoch={epoch:02d}-val_acc={val/acc:.4f}',
                            auto_insert_metric_name=False,
                            save_weights_only=True,
                            save_top_k=1),
        ])
    
    if not config.test_model:
        print("Fitting model...")
        trainer.fit(model, datamodule=dm)
    
    trainer.test(model, datamodule=dm)