import numpy as np
from torch.utils.data import DataLoader, Dataset
import skimage
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2


from typing import Optional

from .._augmentation import val_transformations, NormalizeSegmented


class DiscoverDataset(Dataset):
    def __init__(self, image_arr: np.ndarray, transforms):
        self.image_arr = image_arr
        self.transforms = transforms
        self.image_shape = image_arr.shape[2]

    def __len__(self):
        return self.image_arr.shape[0]

    def __getitem__(self, idx):
        image = self.image_arr[idx, :, :, :]

        if image.shape[0] == 1:
            binary_image = np.where(image > 0, 1, image)
            kernel = np.ones((10, 10), np.uint8)
            label = skimage.measure.label(
                binary_image.reshape(self.image_shape, self.image_shape)
            )
            assert isinstance(label, np.ndarray)
            label = cv2.morphologyEx(label.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            assert isinstance(label, np.ndarray)

            zero_pixel_mask = np.expand_dims(label, axis=0)

            # Duplicate the single channel to create a 3-channel image
            image_3ch = np.repeat(image, 3, axis=0)
            zero_pixel_mask = np.repeat(zero_pixel_mask, 3, axis=0)
        else:
            image_3ch = image

        # Transpose image to [224, 224, 3] for Albumentations
        image_3ch = np.transpose(image_3ch, (1, 2, 0))
        zero_pixel_mask = np.transpose(zero_pixel_mask, (1, 2, 0))
        # zero_pixel_mask = (image_3ch == 0).astype(np.float32)
        assert image_3ch.shape == zero_pixel_mask.shape

        if self.transforms is not None:
            augmented = self.transforms(image=image_3ch, mask=zero_pixel_mask)
            image = augmented["image"]

        # assert isinstance(image, torch.Tensor)
        # assert not torch.isnan(image).any()
        # assert not torch.isinf(image).any()

        return (image, 0)


def create_dataset(img_array: np.ndarray, transformations) -> DiscoverDataset:
    return DiscoverDataset(img_array, transformations)


def create_dataloader_discover(
    img_array: np.ndarray,
    batch_size: int,
    shuffle: bool,
    train: bool,
    transformations: Optional[A.Compose] = None,
    **kwargs,
) -> DataLoader:
    if transformations is None:
        transformations = discover_transformations() if train else val_transformations()

    dataset = create_dataset(img_array, transformations)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def discover_transformations(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=360, p=0.5),
            # A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
            # Normalization
            NormalizeSegmented(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Convert to PyTorch tensor
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )
