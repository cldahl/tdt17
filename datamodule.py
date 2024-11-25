from glob import glob
import os
import lightning.pytorch as pl
from matplotlib import pyplot as plt
import torch
import monai
from torch.utils.data import DataLoader
from monai.data import CacheDataset, load_decathlon_datalist, PatchIterd, GridPatchDataset
import monai.transforms as transforms
from sklearn.model_selection import train_test_split

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=0, data_root="./data", train_split_ratio=0.8, val_split_ratio=0.15):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio

    def setup(self, stage=None):
        # Import filepaths
        train_files = get_file_pairs(self.data_root, phase='train')
        test_files = get_file_pairs(self.data_root, phase='test')

         # Split the training files into training and validation sets
        train_files, val_files = train_test_split(train_files, test_size=self.val_split_ratio, random_state=42)


        #training_files = load_decathlon_datalist('./asoca_filepaths.json', is_segmentation=True, data_list_key="training")
        #validation_files = load_decathlon_datalist('./asoca_filepaths.json', is_segmentation=True, data_list_key="validation")
        #testing_files = load_decathlon_datalist('./asoca_filepaths.json', is_segmentation=True, data_list_key="testing")

        # Visualization
        #normal_testing_files = load_decathlon_datalist('./asoca_filepaths.json', is_segmentation=True, data_list_key="normal_testing")
        #diseased_testing_files = load_decathlon_datalist('./asoca_filepaths.json', is_segmentation=True, data_list_key="diseased_testing")

        patch_func = PatchIterd(
            keys=["image", "label"],
            patch_size=(None, None, 1),  # dynamic first two dimensions
            start_pos=(0, 0, 0)
        )

        volume_ds = CacheDataset(data=train_files, transform=self.get_transforms(split="train"))
        val_volume_ds = CacheDataset(data=val_files, transform=self.get_transforms(split="val"))
        test_volume_ds = CacheDataset(data=test_files, transform=self.get_transforms(split="test"))
        
        self.train_dataset = GridPatchDataset(data=volume_ds, patch_iter=patch_func, transform=self.get_transforms(split="patch"), with_coordinates=False)
        self.val_dataset = GridPatchDataset(data=val_volume_ds, patch_iter=patch_func, transform=self.get_transforms(split="patch"), with_coordinates=False)
        self.test_dataset = GridPatchDataset(data=test_volume_ds, patch_iter=patch_func, transform=self.get_transforms(split="patch"), with_coordinates=False)

        # Visualization
        #normal_test_dataset = CacheDataset(data=normal_testing_files, transform=self.get_transforms(split="test"))
        #self.normal_test_dataset = GridPatchDataset(data=normal_test_dataset, patch_iter=patch_func, transform=self.get_transforms(split="patch"), with_coordinates=False)
        #diseased_test_dataset = CacheDataset(data=diseased_testing_files, transform=self.get_transforms(split="test"))
        #self.diseased_test_dataset = GridPatchDataset(data=diseased_test_dataset, patch_iter=patch_func, transform=self.get_transforms(split="patch"), with_coordinates=False)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):        
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True, shuffle=False)
    
    # Visualization
    #def normal_test_dataloader(self):
        return DataLoader(self.normal_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
    
    #def diseased_test_dataloader(self):
        return DataLoader(self.diseased_test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)
    
    def inspect_labels(data):
        # Print unique values of the label
        print("Unique values in the label:", torch.unique(data["label"]))
        return data
    
    def get_transforms(self, split):
        shared_transforms = [
            transforms.Compose([
                # Preprocessing
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Resized(keys=["image", "label"], spatial_size=(512, 512, 75)),
                transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=5.0, upper=95.0, b_min=0.0, b_max=1.0, clip=True),
                transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
            ])
        ]

        if split == "train":
            return transforms.Compose([
                *shared_transforms,
                # Data augmentation
                transforms.RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=[0, 2]),
                transforms.RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.9, max_zoom=1.1, keep_size=True),
                transforms.Rand3DElasticd(keys=["image", "label"], prob=0.1, magnitude_range=(50, 50), sigma_range=(5, 5)),
                #transforms.RandRotated(keys=["image", "label"], range_x=0.1, prob=0.1),
                transforms.RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
                transforms.RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
                # Convert label to one-hot encoding with 3 classes
                transforms.ToTensord(keys=["image", "label"]),

            ])
        elif split in ["val", "test"]:
            return transforms.Compose([
                *shared_transforms,
                transforms.ToTensord(keys=["image", "label"]),
            ])
        elif split == "patch":
            return transforms.Compose([
                transforms.SqueezeDimd(keys=["image", "label"], dim=-1),
                transforms.Resized(keys=["image", "label"], spatial_size=(512, 512)),
            ])


def get_file_pairs(root_dir, phase='train'):
    image_paths = []
    label_paths = []
    # Traverse through the numbered folders
    for folder in sorted(os.listdir(os.path.join(root_dir, phase))):
        preRT_dir = os.path.join(root_dir, phase, folder, 'midRT')
        if os.path.exists(preRT_dir):
            folder_number = folder.zfill(1)  # Ensure the folder number is correctly matched

            # Construct the expected filenames
            image_pattern = os.path.join(preRT_dir, f"{folder_number}_preRT_T2_registered.nii.gz")
            label_pattern = os.path.join(preRT_dir, f"{folder_number}_preRT_mask_registered.nii.gz")
            image_pattern_2 = os.path.join(preRT_dir, f"{folder_number}_midRT_T2.nii.gz")
            label_pattern_2 = os.path.join(preRT_dir, f"{folder_number}_midRT_mask.nii.gz")

            # Check if both image and label files exist
            if os.path.exists(image_pattern) and os.path.exists(label_pattern) and os.path.exists(image_pattern_2) and os.path.exists(label_pattern_2):
                image_paths.append(image_pattern)
                label_paths.append(label_pattern)
                image_paths.append(image_pattern_2)
                label_paths.append(label_pattern_2)
    
    return [{'image': img, 'label': lbl} for img, lbl in zip(image_paths, label_paths)]