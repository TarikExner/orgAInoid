from dataclasses import dataclass, field
from enum import Enum

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Union

from torch_lr_finder import LRFinder

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

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
        self.timepoints = list(range(self.start_timepoint, self.stop_timepoint + 1))

    def calculate_start_and_stop_timepoint(self):
        self.start_timepoint = min(self.timepoints)
        self.stop_timepoint = max(self.timepoints)

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
        if image.shape[0] == 1:
            # Duplicate the single channel to create a 3-channel image
            image_3ch = np.repeat(image, 3, axis=0)  # [1, 224, 224] -> [3, 224, 224]
        else:
            image_3ch = image

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

def find_ideal_learning_rate(model: nn.Module,
                             criterion: nn.Module,
                             optimizer: Optimizer,
                             train_loader: DataLoader,
                             start_lr: Optional[float] = None,
                             end_lr: Optional[float] = None,
                             num_iter: Optional[int] = None,
                             n_tests: int = 5,
                             return_dataframe: bool = False) -> Union[float, pd.DataFrame]:
    start_lr = start_lr or 1e-7
    end_lr = end_lr or 5e-2
    num_iter = num_iter or 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_data = pd.DataFrame()
    for i in range(n_tests):
        lr_finder = LRFinder(model, optimizer, criterion, device = device)
        lr_finder.range_test(train_loader, start_lr = start_lr, end_lr = end_lr, num_iter = num_iter)
        data = pd.DataFrame(lr_finder.history)
        data = data.rename(columns = {"loss": f"run{i}"})
        if i == 0:
            full_data = data
        else:
            full_data = full_data.merge(data, on = "lr")

        lr_finder.reset()
    full_data["mean"] = full_data.groupby(["lr"]).mean().mean(axis = 1).tolist()
    full_data["mean"] = _smooth_curve(full_data["mean"])

    if return_dataframe:
        return full_data

    return _calculate_ideal_learning_rate(full_data)

    # evaluation window is set to 1e-5 to 1e-2 for now

def _calculate_ideal_learning_rate(df: pd.DataFrame):
    
    lr_at_min_loss = df.loc[df["mean"] == df["mean"].min(), "lr"].iloc[0]
    window = df[df["lr"] <= lr_at_min_loss]
    window_start = window.index[0]
    inf_points = _calculate_inflection_points(window["mean"])
    
    # we take the last one which should be closest to the minimum of the fitted function
    inf_point = inf_points[-1] + window_start

    ideal_learning_rate = df.iloc[inf_point]["lr"]

    return ideal_learning_rate

def _smooth_curve(arr, degree: int = 5):
    x = np.arange(arr.shape[0])
    y = arr

    coeff = np.polyfit(x, y, degree)

    polynomial = np.poly1d(coeff)
    return polynomial(x)

def _calculate_inflection_points(y) -> np.ndarray:
    x = np.arange(y.shape[0])
    dy_dx = np.gradient(y,x)
    d2y_dx2 = np.gradient(dy_dx, x)
    inflection_points = np.where(np.diff(np.sign(d2y_dx2)))[0]
    return inflection_points


def train_transformations(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),    # Random vertical flip
        A.Rotate(limit=360, p=0.5),  # Random rotation by any angle between -45 and 45 degrees
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.2,
            rotate_limit=0,  # Set rotate limit to 0 if using Rotate separately
            p=0.5
        ),  # Shift and scale
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1), p=0.5),  # Resized crop
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.Affine(
            scale=1,
            translate_percent=(-0.3, 0.3),
            rotate=0,
            shear=(-15, 15),
            p=0.5
        ),
        A.CoarseDropout(
            max_holes=20,
            min_holes=10,
            max_height=8,
            max_width=8,
            p=0.5
        ),

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

