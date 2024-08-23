import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from typing import Optional, Literal, Union

from .model import UNet, DEEPLABV3, HRNET
from .._utils import val_transformations, CustomIntensityAdjustment

def train_transformations(image_size):
    segmentation_augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),    # Random vertical flip
        A.RandomRotate90(p=0.5),  # Random 90-degree rotation
        A.Rotate(limit=45, p=0.5),  # Random rotation by any angle between -45 and 45 degrees
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),  # Shift and scale (rotation already handled)
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), p=0.5),  # Resized crop

        # Apply intensity modifications only to non-masked pixels
        CustomIntensityAdjustment(p=0.5),
        
        # Normalization and final transformations
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value = 1),  # Normalization
        ToTensorV2()  # Convert to PyTorch tensor
    ], additional_targets={'mask': 'mask'})

    return segmentation_augmentation


class SegmentationDataset(Dataset):
    def __init__(self,
                 images: np.ndarray,
                 masks: np.ndarray,
                 model_type: Literal["UNet", "DEEPLABV3", "HRNET"] = 'UNet',
                 transforms: Optional[A.Compose] = None):
        self.images = images
        self.masks = masks
        self.model_type = model_type
        assert self.model_type in ["UNet", "DEEPLABV3", "HRNET"]
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert grayscale image to RGB if needed
        if self.model_type != 'UNet':
            image = image.repeat(3, 1, 1)  # Repeat the single channel to create an RGB image

        mask = mask.unsqueeze(0)

        return image, mask


def _run_segmentation_train_loop(dataset_dir: str,
                                 image_size: int,
                                 batch_size: int,
                                 model: Union[UNet, DEEPLABV3, HRNET],
                                 n_epochs: int = 200,
                                 init_lr = 0.0003,
                                 score_output_dir: str = "./results",
                                 model_output_dir: str = "./segmentators"):


    if not os.path.exists(score_output_dir):
        os.mkdir(score_output_dir)

    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)

    train_loader, val_loader = create_dataloaders(dataset_dir = dataset_dir,
                                                  image_size = image_size,
                                                  batch_size = batch_size,
                                                  model_name = model.__class__.__name__)

    batch_size = batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor = 0.2,
        patience = 10,
    )

    best_val_loss = float('inf')

    score_file = os.path.join(
        score_output_dir,
        f'losses_{model.__class__.__name__}_{image_size}.txt'
    )
    if not os.path.isfile(score_file):
        with open(score_file, 'w') as f:
            f.write('model,epoch,train_loss,val_loss,batch_size\n')
    
    for epoch in range(n_epochs):

        start = time.time()
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            (images, masks) = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (images, masks) in val_loader:
                (images, masks) = images.to(device), masks.to(device)
                output = model(images)
                loss = criterion(output, masks)
                
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step(val_loss)
        
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            # load best performing model before continuing
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        model_output_dir,
                        f'{model.__class__.__name__}_{image_size}_bs{batch_size}.pth'
                    )
                )
            )
            print(f"[INFO] Learning rate reduced from {current_lr} to {new_lr}")

        
        stop = time.time()

        # Print metrics
        print(f"[INFO] Epoch: {epoch+1}/{n_epochs}, "
              f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
              f"Time: {stop-start}")

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), 
                os.path.join(
                    model_output_dir,
                    f'{model.__class__.__name__}_{image_size}_bs{batch_size}.pth'
                )
            )
            print(f'Saved best model with val loss: {best_val_loss:.4f}')
        
        with open(score_file, 'a') as f:
            f.write(f'{model.__class__.__name__},{epoch+1},{train_loss},{val_loss},{batch_size}\n')
        

def create_dataloaders(dataset_dir: str,
                       image_size: float,
                       batch_size: int,
                       model_name: Literal["UNet", "DEEPLABV3", "HRNET"]) -> tuple[DataLoader, ...]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = True if device == "cuda" else False

    imgs = np.load(os.path.join(dataset_dir, f"unet_segmentation_images_{image_size}.npy"))
    masks = np.load(os.path.join(dataset_dir, f"unet_segmentation_masks_{image_size}.npy"))

    train_img, train_mask, test_img, test_mask = _apply_train_test_split(imgs, masks)

    train_dataset = SegmentationDataset(
        images = train_img,
        masks = train_mask,
        model_type = model_name,
    	transforms = train_transformations(image_size),
    )
    
    test_dataset = SegmentationDataset(
        images = test_img,
        masks = test_mask,
        model_type = model_name,
    	transforms = val_transformations(),
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
    	batch_size = batch_size,
        pin_memory = pin_memory,
    	num_workers = 1,
        drop_last = True if model_name == "DEEPLABV3" else False
    )
    val_dataloader = DataLoader(
        test_dataset,
        shuffle = False,
    	batch_size = batch_size,
        pin_memory = pin_memory,
    	num_workers = 1
    )

    return train_dataloader, val_dataloader
 

def _apply_train_test_split(imgs: list[str], masks: list[str],
                            test_split_ratio = 0.1) -> tuple[np.ndarray,...]:
    split = train_test_split(
        imgs, masks,
    	test_size=test_split_ratio,
        random_state = 187
    )
    train_img, test_img = split[:2]
    train_mask, test_mask = split[2:]
    return train_img, train_mask, test_img, test_mask
