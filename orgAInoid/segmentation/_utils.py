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

def get_augmentation_pipeline(image_size):
    segmentation_augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),    # Random vertical flip
        A.RandomRotate90(p=0.5),  # Random 90-degree rotation
        A.Rotate(limit=45, p=0.5),  # Random rotation by any angle between -45 and 45 degrees
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),  # Shift and scale (rotation already handled)
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), p=0.5),  # Resized crop
        
        # Apply image-specific augmentations (only to images)
        A.OneOf([
            A.GaussNoise(p=0.2),  # Add Gaussian noise (mild)
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ], p=0.2),  # Apply one of the blurs
            A.OneOf([
                A.RandomBrightnessContrast(p=0.2),  # Random brightness/contrast
                A.RandomGamma(p=0.2),  # Random gamma
            ], p=0.3),
        ], p=1.0),  # Ensure these are always applied, but only to the image

        # Normalization and final transformations
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
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

        return image, mask


def _run_segmentation_train_loop(dataset_dir: str,
                                 image_size: int,
                                 batch_size: int,
                                 model: Union[UNet, DEEPLABV3, HRNET],
                                 sub_batch_size: int = 4,
                                 score_output_dir: str = "./results",
                                 model_output_dir: str = "./segmentators"):

    def process_sub_batches(images, masks, sub_batch_size=8, training=True):
        sub_batch_losses = []
        
        if training:
            optimizer.zero_grad()  # Clear gradients before processing the main batch
        
        for i in range(0, len(images), sub_batch_size):
            sub_images = images[i:i + sub_batch_size]
            sub_masks = masks[i:i + sub_batch_size]
            
            with autocast():
                outputs = model(sub_images)
                loss = criterion(outputs, sub_masks)
            
            if training:
                scaler.scale(loss).backward()  # Accumulate gradients
            sub_batch_losses.append(loss.item() * sub_images.size(0))
        
        if training:
            scaler.step(optimizer)  # Update model parameters once after processing all sub-batches
            scaler.update()
        
        return sum(sub_batch_losses) / len(images)

    if not os.path.exists(score_output_dir):
        os.mkdir(score_output_dir)

    if not os.path.exists(model_output_dir):
        os.mkdir(model_output_dir)

    train_loader, val_loader = create_dataloaders(dataset_dir = dataset_dir,
                                                  image_size = image_size,
                                                  batch_size = batch_size,
                                                  model_name = model.__class__.__name__)
    init_lr = 0.001
    n_epochs = 200
    batch_size = batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor = 0.2,
        patience = 8,
    )

    scaler = GradScaler()
    
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
            
            train_loss += process_sub_batches(images, masks, sub_batch_size=sub_batch_size)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (images, masks) in val_loader:
                (images, masks) = images.to(device), masks.to(device)
                val_loss += process_sub_batches(images, masks, sub_batch_size=sub_batch_size, training=False)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Get the current learning rate before scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step the learning rate scheduler based on the validation loss
        scheduler.step(val_loss)
        
        # Check if the learning rate has been reduced
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
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

    transforms = get_augmentation_pipeline(image_size)

    train_dataset = SegmentationDataset(
        images = train_img,
        masks = train_mask,
        model_type = model_name,
    	transforms = transforms,
    )
    
    test_dataset = SegmentationDataset(
        images = test_img,
        masks = test_mask,
        model_type = model_name,
    	transforms = transforms,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
    	batch_size = batch_size,
        pin_memory = pin_memory,
    	num_workers = 1
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


