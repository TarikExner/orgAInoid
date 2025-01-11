import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Optional


class SequenceTransform:
    def __init__(self, transforms):
        """
        Initializes the SequenceTransform with a list of Albumentations transforms.

        Args:
            transforms (list): List of Albumentations transforms to apply.
        """
        self.transform = A.ReplayCompose(transforms)
    
    def __call__(self, images):
        """
        Applies the same transformations to all images in the sequence.

        Args:
            images (list or np.ndarray): List or array of images with shape (H, W) or (H, W, C).

        Returns:
            transformed_images (list or np.ndarray): List or array of transformed images.
        """
        transformed_images = []
        # Apply transform to the first image and capture the replay data
        transformed = self.transform(image=images[0])
        transformed_images.append(transformed['image'])
        replay = transformed['replay']
        
        # Apply the same transform using replay data to the rest of the images
        for img in images[1:]:
            transformed = A.ReplayCompose.apply_replay(self.transform, image=img, replay=replay)
            transformed_images.append(transformed['image'])
        
        return transformed_images

_train_transformations_pipeline = [
    A.RandomRotate90(p=0.5),                 # Randomly rotate the image by 90 degrees
    A.Flip(p=0.5),                           # Randomly flip the image horizontally or vertically
    A.Transpose(p=0.5),                      # Randomly transpose the image
    A.ShiftScaleRotate(
        shift_limit=0.0625, 
        scale_limit=0.1, 
        rotate_limit=45, 
        p=0.75
    ),                                        # Randomly shift, scale, and rotate the image
    A.ElasticTransform(
        alpha=1,
        sigma=50,
        p=0.5),  # Elastic deformation
    A.RandomResizedCrop(
        height=224, 
        width=224, 
        scale=(0.8, 1.0), 
        ratio=(0.9, 1.1), 
        p=0.5
    ),                                        # Randomly crop and resize the image
    A.GaussianBlur(blur_limit=(3, 7), p=0.5), # Apply Gaussian Blur
    A.GridDistortion(p=0.5),                 # Apply grid distortion
    ToTensorV2()
]

_val_transformations_pipeline = [
    ToTensorV2()
]

def train_transformations():
    return SequenceTransform(_train_transformations_pipeline)

def val_transformations():
    return SequenceTransform(_val_transformations_pipeline)

class ImageSequenceDataset(Dataset):
    def __init__(self, image_array, df, num_input=5, num_output=5, gap=0, transform=None):
        """
        Initializes the dataset with image sequences.

        Args:
            image_array (np.ndarray or torch.Tensor): Array of images with shape (num_images, height, width).
            df (pd.DataFrame): DataFrame with columns ["experiment", "well", "loop", "IMAGE_ARRAY_INDEX"].
            num_input (int): Number of input images in a sequence.
            num_output (int): Number of target images to predict.
            gap (int): Number of images to skip between input and target sequences.
            transform (SequenceTransform, optional): Transformations to apply to the image sequences.
        """
        self.image_array = image_array
        self.df = df.copy()
        self.num_input = num_input
        self.num_output = num_output
        self.gap = gap
        self.transform = transform
        
        # Preprocess the DataFrame to prepare sequences
        self._prepare_sequences()
    
    def _prepare_sequences(self):
        """
        Prepares a list of valid sequences where each sequence consists of num_input
        input images, followed by a gap of 'gap' images, and then num_output target images
        from the same experiment and well.
        """
        # Extract numeric loop number for proper sorting
        self.df['loop_num'] = self.df['loop'].str.extract('LO(\d+)').astype(int)
        
        # Sort the DataFrame by ['experiment', 'well', 'loop_num']
        self.df.sort_values(by=['experiment', 'well', 'loop_num'], inplace=True)
        
        # Group by ['experiment', 'well'] to ensure sequences are within the same group
        grouped = self.df.groupby(['experiment', 'well'])
        
        # List to store tuples of (input_indices, target_indices)
        self.sequence_indices = []
        
        for name, group in grouped:
            group = group.reset_index(drop=True)
            total_timepoints = len(group)
            # Calculate the maximum starting index to ensure sequences don't exceed available images
            max_start = total_timepoints - self.num_input - self.gap - self.num_output + 1
            for start in range(max_start):
                input_indices = group.loc[start:start + self.num_input - 1, 'IMAGE_ARRAY_INDEX'].values
                target_start = start + self.num_input + self.gap
                target_end = target_start + self.num_output
                target_indices = group.loc[target_start:target_end - 1, 'IMAGE_ARRAY_INDEX'].values
                self.sequence_indices.append((input_indices, target_indices))
        
        # Convert list to NumPy array for faster indexing
        self.sequence_indices = np.array(self.sequence_indices, dtype=np.int64)
        print(f"Total sequences available with gap={self.gap}: {len(self.sequence_indices)}")
    
    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        """
        Retrieves the input and target sequences for a given index.

        Args:
            idx (int): Index of the sequence.

        Returns:
            inputs (torch.Tensor): Tensor of shape (num_input, channels, height, width).
            targets (torch.Tensor): Tensor of shape (num_output, channels, height, width).
        """
        input_indices, target_indices = self.sequence_indices[idx]
        print(input_indices, target_indices)
        
        # Retrieve input images
        inputs = self.image_array[input_indices]  # Shape: (num_input, 1, 224, 224)
        # Retrieve target images
        targets = self.image_array[target_indices]  # Shape: (num_output, 1, 224, 224)
        
        # Concatenate inputs and targets for joint transformation
        all_images = np.concatenate((inputs, targets), axis=0)  # Shape: (num_input + num_output, 1, 224, 224)
        
        # Apply transformations if any
        if self.transform:
            # Albumentations expects images in HWC format; ensure all_images are in HWC
            # If images are grayscale, expand dims to (num_images, H, W, 1)
            if all_images.ndim == 3:
                all_images = np.expand_dims(all_images, -1)  # Shape: (num_input + num_output, 224, 224, 1)
            elif all_images.ndim == 4:
                all_images = np.transpose(all_images, (0,2,3,1)) # Transpose to (224, 224, 1) to confer with albumentations
            else:
                raise ValueError("Unexpected image dimensions.")
            
            # Convert to list for SequenceTransform
            all_images = list(all_images)
            transformed_images = self.transform(all_images)  # List of transformed images
            
            # Convert back to NumPy array
            transformed_images = np.stack(transformed_images)  # Shape: (num_input + num_output, H, W, C)
            
            # Remove channel dimension if grayscale
            if transformed_images.shape[-1] == 1:
                transformed_images = transformed_images.squeeze(-1)  # Shape: (num_input + num_output, H, W)
        else:
            transformed_images = all_images
        
        # Split back into inputs and targets
        transformed_inputs = transformed_images[:self.num_input]
        transformed_targets = transformed_images[self.num_input:]
        
        # Convert to torch tensors
        if isinstance(transformed_inputs, np.ndarray):
            transformed_inputs = torch.from_numpy(transformed_inputs)
        if isinstance(transformed_targets, np.ndarray):
            transformed_targets = torch.from_numpy(transformed_targets)
        
        # Add channel dimension if images are grayscale
        if transformed_inputs.ndim == 3:
            transformed_inputs = transformed_inputs.unsqueeze(1)  # Shape: (num_input, 1, 224, 224)
        if transformed_targets.ndim == 3:
            transformed_targets = transformed_targets.unsqueeze(1)  # Shape: (num_output, 1, 224, 224)
        
        return transformed_inputs, transformed_targets

def create_dataset(img_array: np.ndarray,
                   metadata: pd.DataFrame,
                   num_input: int,
                   num_output: int,
                   gap: int,
                   transformations) -> ImageSequenceDataset:
    return ImageSequenceDataset(
        img_array,
        metadata,
        num_input,
        num_output,
        gap,
        transformations
    )


def create_dataloader(img_array: np.ndarray,
                      metadata: pd.DataFrame,
                      num_input: int,
                      num_output: int,
                      gap: int,
                      batch_size: int,
                      shuffle: bool,
                      train: bool,
                      transformations: Optional[SequenceTransform] = None,
                      **kwargs) -> DataLoader:
    if transformations is None:
        transformations = train_transformations() if train else val_transformations()

    dataset = create_dataset(img_array, metadata, num_input, num_output, gap, transformations)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, **kwargs)
      
