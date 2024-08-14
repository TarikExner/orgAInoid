import os

import numpy as np
import pandas as pd
from pathlib import Path
import cv2

import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from albumentations import (
    Compose, RandomCrop, Resize, Normalize, HorizontalFlip, VerticalFlip,
    Rotate, RandomResizedCrop, ImageOnlyTransform
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from torch.cuda.amp import autocast, GradScaler
import time

from .model import UNet


class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transforms, dimensions):
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.dimensions = dimensions
        self.bin_size = int(2048 / dimensions)
    
    def __len__(self):
        return len(self.images)
    
    def _normalize_image(self,
                         img: np.ndarray,
                         bitdepth: int) -> np.ndarray:
        return img / float(2**bitdepth)

    def _min_max_scale(self,
                       image: np.ndarray) -> np.ndarray:
        min_val = image.min()
        max_val = image.max()
        return (image - min_val) / (max_val - min_val)
        
    def _bin_image(self,
                   image: np.ndarray,
                   bin_size: int,
                   mask: bool = False):
        """
        Bins the image on demand.
        """
        height, width = image.shape[:2]
    
        new_height = height // bin_size
        new_width = width // bin_size
        
        binned_image = np.zeros((new_height, new_width), dtype=np.float32)
    
        for i in range(new_height):
            for j in range(new_width):
                binned_image[i, j] = np.mean(image[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size])
        if mask:
            binned_image[binned_image < 0.5] = 0
        return binned_image
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = self._normalize_image(image, bitdepth = 16)
        image = self._min_max_scale(image)
        
        mask = self.masks[idx]
        mask = self._normalize_image(mask, bitdepth = 8)
        mask = self._min_max_scale(mask)

        if self.bin_size != 1:
            image = self._bin_image(image, self.bin_size)
            mask = self._bin_image(mask, self.bin_size, mask = True)

        if self.transforms is not None:
            augmented = self.transforms(image = image, mask = mask)
            image = augmented['image']
            mask = augmented['mask']
        mask = np.expand_dims(mask, axis = 0)  # Add channel dimension
        return (image, mask)

def _apply_train_test_split(imgs: list[str], masks: list[str],
                            test_split_ratio = 0.1):
    split = train_test_split(
        imgs, masks,
    	test_size=test_split_ratio,
        random_state = 187
    )
    train_img, test_img = split[:2]
    train_mask, test_mask = split[2:]
    return train_img, train_mask, test_img, test_mask

def instantiate_transforms(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH):
    return 

def run_unet_training(dimensions: int = 2048,
                      batch_size: int = 128,
                      dataset_dir: Path = "./") -> None:

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True if DEVICE == "cuda" else False

    INIT_LR = 0.001
    NUM_EPOCHS = 200
    BATCH_SIZE = batch_size
    
    INPUT_IMAGE_WIDTH = dimensions
    INPUT_IMAGE_HEIGHT = dimensions

    imgs = np.load(os.path.join(dataset_dir, "unet_segmentation_images.npy"))
    masks = np.load(os.path.join(dataset_dir, "unet_segmentation_masks.npy"))
    train_img, train_mask, test_img, test_mask = _apply_train_test_split(imgs, masks)

    transforms = A.Compose([
        A.RandomResizedCrop(
            height=int(INPUT_IMAGE_HEIGHT*0.7),
            width=int(INPUT_IMAGE_HEIGHT*0.7),
            p = 0.5
        ),
        A.Resize(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ElasticTransform(p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GaussNoise(p=0.5),
        A.GaussianBlur(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=(-0.3, 0.3),  # ScaleFloatType
            scale_limit=(-0.6, 0.6),  # ScaleFloatType
            rotate_limit=(-180, 180),  # ScaleFloatType
            p = 0.5
        ),
        ToTensorV2()
    ])

    train_dataset = SegmentationDataset(
        images = train_img,
        masks = train_mask,
    	transforms = transforms,
        dimensions = dimensions
    )
    
    test_dataset = SegmentationDataset(
        images = test_img,
        masks = test_mask,
    	transforms = transforms,
        dimensions = dimensions
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle = True,
    	batch_size = BATCH_SIZE,
        pin_memory = PIN_MEMORY,
    	num_workers = 1
    )
    val_dataloader = DataLoader(
        test_dataset,
        shuffle = False,
    	batch_size = BATCH_SIZE,
        pin_memory = PIN_MEMORY,
    	num_workers = 1
    )
    
    model = UNet(n_class = 1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = INIT_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min',
        factor = 0.2,
        patience = 8,
    )

    scaler = GradScaler()
    
    best_val_loss = float('inf')
    score_file = f'losses_{dimensions}.txt'
    if not os.path.isfile(score_file):
        print("initializing score file")
        with open(score_file, 'w') as f:
            f.write('epoch,train_loss,val_loss,batch_size\n')

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
    
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        
        model.train()
        train_loss = 0
        for images, masks in train_dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            train_loss += process_sub_batches(images, masks, sub_batch_size=4)
        
        train_loss /= len(train_dataloader.dataset)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_dataloader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                val_loss += process_sub_batches(images, masks, sub_batch_size=4, training=False)
        
        val_loss /= len(val_dataloader.dataset)
        
        stop = time.time()
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {stop-start}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'segmentator_{dimensions}_bs{BATCH_SIZE}.pth')
            print(f'Saved best model with val loss: {best_val_loss:.4f}')
        
        with open(score_file, 'a') as f:
            f.write(f'{epoch+1},{train_loss},{val_loss},{BATCH_SIZE}\n')
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f'Learning rate reduced to {new_lr}')

if __name__ == "__main__":
    input_dir = os.getcwd()
    run_unet_training(input_dir = input_dir)