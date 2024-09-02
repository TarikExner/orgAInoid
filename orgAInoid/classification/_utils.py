from dataclasses import dataclass, field
from enum import Enum

from torch.utils.data import Dataset
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

from typing import Optional

from .._augmentation import val_transformations, CustomIntensityAdjustment


class SegmentatorModel(Enum):
    DEEPLABV3 = "DEEPLABV3"
    UNET = "UNET"
    HRNET = "HRNET"


@dataclass
class ImageMetadata:
    dimension: int
    cropped_bbox: bool
    scaled_to_size: bool
    crop_size: Optional[int]
    segmentator_model: SegmentatorModel
    segmentator_input_size: int
    mask_threshold: float
    cleaned_mask: bool
    scale_masked_image: bool
    crop_bounding_box: bool
    rescale_cropped_image: bool
    crop_bounding_box_dimension: Optional[int]

    def __repr__(self):
        return (
            f"ImageMetadata("
            f"dimension={self.dimension}, "
            f"cropped_bbox={self.cropped_bbox}, "
            f"scaled_to_size={self.scaled_to_size}, "
            f"crop_size={self.crop_size}, "
            f"segmentator_model={self.segmentator_model!r}, "
            f"segmentator_input_size={self.segmentator_input_size}, "
            f"mask_threshold={self.mask_threshold}, "
            f"cleaned_mask={self.cleaned_mask}, "
            f"scale_masked_image={self.scale_masked_image}, "
            f"crop_bounding_box={self.crop_bounding_box}, "
            f"rescale_cropped_image={self.rescale_cropped_image}, "
            f"crop_bounding_box_dimension={self.crop_bounding_box_dimension})"
        )

    def __eq__(self, other):
        if not isinstance(other, ImageMetadata):
            return NotImplemented
        return (
            self.dimension == other.dimension and
            self.cropped_bbox == other.cropped_bbox and
            self.scaled_to_size == other.scaled_to_size and
            self.crop_size == other.crop_size and
            self.segmentator_model == other.segmentator_model and
            self.segmentator_input_size == other.segmentator_input_size and
            self.mask_threshold == other.mask_threshold and
            self.cleaned_mask == other.cleaned_mask and
            self.scale_masked_image == other.scale_masked_image and
            self.crop_bounding_box == other.crop_bounding_box and
            self.rescale_cropped_image == other.rescale_cropped_image and
            self.crop_bounding_box_dimension == other.crop_bounding_box_dimension
        )


@dataclass
class DatasetMetadata:
    dataset_id: str
    experiment_dir: str
    readouts: list[str]
    start_timepoint: int
    stop_timepoint: int
    slices: list[str]
    n_slices: int = field(init = False)
    class_balance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n_slices = len(self.slices)

    def __repr__(self):
        return (
            f"DatasetMetadata("
            f"dataset_id={self.dataset_id!r}, "
            f"experiment_dir={self.experiment_dir!r}, "
            f"readouts={self.readouts!r}, "
            f"start_timepoint={self.start_timepoint}, "
            f"stop_timepoint={self.stop_timepoint}, "
            f"slices={self.slices!r}, "
            f"n_slices={self.n_slices}, "
            f"class_balance={self.class_balance!r})"
        )

    def __eq__(self, other):
        if not isinstance(other, DatasetMetadata):
            return NotImplemented
        return (
            self.dataset_id == other.dataset_id and
            self.experiment_dir == other.experiment_dir and
            self.readouts == other.readouts and
            self.start_timepoint == other.start_timepoint and
            self.stop_timepoint == other.stop_timepoint and
            self.slices == other.slices and
            self.class_balance == other.class_balance
        )

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


def train_transformations(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),    # Random vertical flip
        A.RandomRotate90(p=0.5),  # Random 90-degree rotation
        A.Rotate(limit=120, p=0.5),  # Random rotation by any angle between -45 and 45 degrees
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),  # Shift and scale (rotation already handled)
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1), p=0.5),  # Resized crop
        # A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-20, 20), shear=(-15, 15), p=0.5),
        # A.ElasticTransform(alpha=1.0, sigma=50.0, alpha_affine=50.0, p=0.5),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        # A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.5),


        # Apply intensity modifications only to non-masked pixels
        CustomIntensityAdjustment(p=0.5),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value = 1),

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
    transformations = train_transformations() if train else val_transformations()
    dataset = create_dataset(img_array, class_array, transformations)
    return DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

def _get_model_from_enum(model_name: str):
    return [
        model for model
        in SegmentatorModel
        if model.value == model_name
    ][0]

