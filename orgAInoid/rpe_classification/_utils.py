from torch.utils.data import Dataset
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

class ClassificationDataset(Dataset):
    def __init__(self, image_arr: np.ndarray, classes: np.ndarray, transforms):
        self.image_arr = image_arr
        self.classes = classes
        self.transforms = transforms
    
    def __len__(self):
        return self.image_arr.shape[0]
    
    def __getitem__(self, idx):

        image = self.image_arr[idx, :, :, :]
        # Duplicate the single channel to create a 3-channel image
        image_3ch = np.repeat(image, 3, axis=0)  # [1, 224, 224] -> [3, 224, 224]

        # Transpose image to [224, 224, 3] for Albumentations
        image_3ch = np.transpose(image_3ch, (1, 2, 0))

        corr_class = torch.tensor(self.classes[idx])

        if self.transforms is not None:
            augmented = self.transforms(image = image_3ch)
            image = augmented["image"]

        assert isinstance(image, torch.Tensor)
        assert not torch.isnan(image).any()
        assert not torch.isinf(image).any()

        return (image, corr_class)

class CustomIntensityAdjustment(A.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(CustomIntensityAdjustment, self).__init__(always_apply, p)
        self.adjustment = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),  # Random brightness/contrast
                A.RandomGamma(p=0.5),               # Random gamma adjustment
            ], p=1.0)
        ])

    def apply(self, img, **params):
        # Apply intensity changes only to non-zero pixels
        non_zero_mask = img > 0
        img_augmented = self.adjustment(image=img)["image"]
        
        # Only change the intensity of non-zero pixels
        img = np.where(non_zero_mask, img_augmented, img)
        
        # Rescale the entire image to the 0-1 range
        img_min = img.min()
        img_max = img.max()
        
        if img_max > img_min:  # To avoid division by zero
            img = (img - img_min) / (img_max - img_min)
        
        return img

def val_transformations() -> A.Compose:
    return A.Compose([

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # Convert to PyTorch tensor
        ToTensorV2()
    ])


def train_transformations(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),    # Random vertical flip
        A.RandomRotate90(p=0.5),  # Random 90-degree rotation
        A.Rotate(limit=120, p=0.5),  # Random rotation by any angle between -45 and 45 degrees
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),  # Shift and scale (rotation already handled)
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.2), p=0.5),  # Resized crop
        
        # Apply intensity modifications only to non-masked pixels
        CustomIntensityAdjustment(p=0.5),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        # Convert to PyTorch tensor
        ToTensorV2()
    ])

def create_dataset(img_array: np.ndarray,
                   class_array: np.ndarray,
                   transformations) -> ClassificationDataset:
    return ClassificationDataset(img_array, class_array, transformations)

def create_dataloader(img_array: np.ndarray,
                      class_array: np.ndarray,
                      batch_size: int,
                      shuffle: bool,
                      train: bool) -> DataLoader:
    transformations = train_transformations() if train else val_transformations
    dataset = create_dataset(img_array, class_array, transformations)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)
