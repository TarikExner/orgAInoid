from ..segmentation.model import UNet
import torch
from pathlib import Path
import skimage
import numpy as np
import pandas as pd
import cv2
import os
from scipy.ndimage import zoom
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

from .._utils import (_generate_file_table,
                      _read_image,
                      _create_mask_from_image,
                      _clean_and_label_mask,
                      _threshold_mask,
                      _load_unet_model)

from . import methods

EXTRA_PROPERTIES = [
    methods.blur,
    methods.roi_contrast,
    methods.image_contrast,
    methods.intensity_median,
    methods.modal_value,
    methods.integrated_density,
    methods.raw_integrated_density,
    methods.skewness,
    methods.kurtosis
]

PROPERTIES = [
    "label",
    "area",
    "area_bbox",
    "area_convex",
    "area_filled",
    "axis_major_length",
    "axis_minor_length",
    "bbox",
    "coords",
    "centroid",
    "centroid_local",
    "centroid_weighted",
    "centroid_weighted_local",
    "eccentricity",
    "equivalent_diameter_area",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "image_convex",
    "image_filled",
    "image_intensity",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "intensity_max",
    "intensity_mean",
    "intensity_min",
    "intensity_std",
    "moments", # array
    "moments_central", # array
    "moments_hu", # array
    "moments_normalized", # array
    "moments_weighted", # array
    "moments_weighted_central", # array
    "moments_weighted_hu", # tuple
    "moments_weighted_normalized", # array
    "num_pixels",
    "orientation",
    "perimeter",
    "perimeter_crofton",
    "solidity"
]

def run_morphometrics(experiment_id: str,
                      image_dir: Path,
                      metadata_dir: Path,
                      output_dir: Path,
                      unet_dir: Path = "../segmentation",
                      unet_input_size: int = 128):
    """\
    Convention:
        annotations by Cassian are termed "E001_annotations.csv"
        file overviews are termed "E001_file_overview.csv"

        Both will be saved in ../metadata
    

    """
    overview_file = os.path.join(metadata_dir, f"{experiment_id}_file_overview.csv")
    if os.path.isfile(overview_file):
        file_frame = pd.read_csv(overview_file, index_col = None)
    else:
        annotations_file = os.path.join(metadata_dir, f"{experiment_id}_annotations.csv")
        file_frame = _generate_file_table(experiment_id = experiment_id,
                                          image_dir = image_dir,
                                          annotations_file = annotations_file)
        file_frame.to_csv(overview_file, index = False)

    file_frame = file_frame[file_frame["slice"] == "SL003"]
    file_list = file_frame["file_name"].tolist()

    model = _load_unet_model(unet_dir, unet_input_size)

    for file_name in file_list:
        print(file_name)
        image_path = os.path.join(image_dir, file_name)
        original_image = _read_image(image_path)
        if original_image is None:
            print("WARNING Corrupted image: {file_name}")
            continue

        original_image, mask = _create_mask_from_image(image = original_image,
                                                       model = model,
                                                       unet_input_size = unet_input_size,
                                                       output_size = 2048)
        
        mask = _threshold_mask(mask, threshold = 0.5)

        labeled_mask = _clean_and_label_mask(mask, min_size_perc = 5)
        if isinstance(labeled_mask, int) and labeled_mask == -1:
            print(f"Removed only label in image {file_name}... skipping.")
            continue
        if isinstance(labeled_mask, int) and labeled_mask == -2:
            print(f"Found more than one region in cleaned mask in image {file_name}... skipping.")
            continue

        reg_table = skimage.measure.regionprops_table(labeled_mask,
                                                      intensity_image = original_image,
                                                      properties = PROPERTIES,
                                                      extra_properties = EXTRA_PROPERTIES)
        reg_table["aspect_ratio"] = methods.aspect_ratio(reg_table)
        reg_table["roundness"] = methods.roundness(reg_table)
        reg_table["compactness"] = methods.compactness(reg_table)
        reg_table["circularity"] = methods.circularity(reg_table)
        reg_table["form_factor"] = methods.form_factor(reg_table)
        reg_table["effective_diameter"] = methods.effective_diameter(reg_table)
        reg_table["convexity"] = methods.convexity(reg_table)
        reg_table_df = pd.DataFrame(reg_table)
        reg_table_df = reg_table_df.drop(["coords", "image_convex", "image_filled", "image_intensity"], axis = 1)
        assert reg_table_df.shape[0] == 1
        file_frame.loc[file_frame["file_name"] == file_name, reg_table_df.columns] = reg_table_df.values

    file_frame.to_csv(os.path.join(output_dir, f"{experiment_id}_morphometrics.csv"))

    return file_frame

if __name__ == "__main__":
    run_shape_analysis()