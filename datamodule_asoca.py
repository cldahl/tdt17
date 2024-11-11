from glob import glob
import os
import lightning.pytorch as pl
import torch
import monai
from torch.utils.data import DataLoader
from monai.data import CacheDataset, load_decathlon_datalist, PatchIterd, GridPatchDataset
import monai.transforms as transforms


class ASOCADataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=0, data_root="./data", train_split_ratio=0.8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.train_split_ratio = train_split_ratio

    def setup(self, stage=None):
        # Import filepaths
        training_files = load_decathlon_datalist('./asoca_filepaths.json', is_segmentation=True, data_list_key="training")
        validation_files = load_decathlon_datalist('./asoca_filepaths.json', is_segmentation=True, data_list_key="validation")
        testing_files = load_decathlon_datalist('./asoca_filepaths.json', is_segmentation=True, data_list_key="testing")

        # Visualization
        normal_testing_files = load_decathlon_datalist('./asoca_filepaths.json', is_segmentation=True, data_list_key="normal_testing")
        diseased_testing_files = load_decathlon_datalist('./asoca_filepaths.json', is_segmentation=True, data_list_key="diseased_testing")

        patch_func = PatchIterd(
            keys=["image", "label"],
            patch_size=(None, None, 1),  # dynamic first two dimensions
            start_pos=(0, 0, 0)
        )

        volume_ds = CacheDataset(data=training_files, transform=self.get_transforms(split="train"))
        val_volume_ds = CacheDataset(data=validation_files, transform=self.get_transforms(split="val"))
        test_volume_ds = CacheDataset(data=testing_files, transform=self.get_transforms(split="test"))
        
        self.train_dataset = GridPatchDataset(data=volume_ds, patch_iter=patch_func, transform=self.get_transforms(split="patch"), with_coordinates=False)
        self.val_dataset = GridPatchDataset(data=val_volume_ds, patch_iter=patch_func, transform=self.get_transforms(split="patch"), with_coordinates=False)
        self.test_dataset = GridPatchDataset(data=test_volume_ds, patch_iter=patch_func, transform=self.get_transforms(split="patch"), with_coordinates=False)

        # Visualization
        normal_test_dataset = CacheDataset(data=normal_testing_files, transform=self.get_transforms(split="test"))
        self.normal_test_dataset = GridPatchDataset(data=normal_test_dataset, patch_iter=patch_func, transform=self.get_transforms(split="patch"), with_coordinates=False)
        diseased_test_dataset = CacheDataset(data=diseased_testing_files, transform=self.get_transforms(split="test"))
        self.diseased_test_dataset = GridPatchDataset(data=diseased_test_dataset, patch_iter=patch_func, transform=self.get_transforms(split="patch"), with_coordinates=False)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):        
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True, shuffle=False)
    
    # Visualization
    def normal_test_dataloader(self):
        return DataLoader(self.normal_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
    
    def diseased_test_dataloader(self):
        return DataLoader(self.diseased_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
    
    
    def get_transforms(self,split):    
        shared_transforms = [
            transforms.Compose([
                # Preprocessing
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Resized(keys=["image", "label"], spatial_size=(512, 512, 168)),
                transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=5.0, upper=95.0, b_min=0.0, b_max=1.0, clip=True),
                transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
            ])
        ]
        
        if split == "train":
            return transforms.Compose([
                *shared_transforms,
                # Data augmentation
                transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=[0, 2]), # Performs it on all 3 axes
                transforms.RandZoomd(keys=["image", "label"], prob=0.1, min_zoom=0.9, max_zoom=1.1, keep_size=True),
                transforms.Rand3DElasticd(keys = ["image", "label"], prob = 0.1, magnitude_range =(50, 50), sigma_range =(5, 5)), # Train is applied before patch, so it is 3D
                transforms.RandRotate90d(keys = ["image", "label"], prob = 0.1, max_k = 3),
            ])
            
        elif split == "val":
            return transforms.Compose([
                *shared_transforms,
            ])
        elif split == "test":
            return transforms.Compose([
                *shared_transforms,

            ])
        elif split == "patch":
            # Splits the 3D image into 2D patches
            return transforms.Compose([
                transforms.SqueezeDimd(keys=["image", "label"], dim=-1),
                transforms.Resized(keys=["image", "label"], spatial_size=(512, 512)),
            ])
    