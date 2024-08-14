from .segmentation.model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import cv2
import torch
from scipy.ndimage import zoom
import skimage
import os

from typing import Optional, Union
from pathlib import Path

def _load_unet_model(unet_dir: Path,
                     unet_input_size: int) -> UNet:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict_path = os.path.join(unet_dir, f"segmentator_{unet_input_size}_bs64.pth")
    
    model = UNet(1)
    model.load_state_dict(
        torch.load(
            state_dict_path,
            map_location=torch.device(device)
        )
    )
    model.eval()
    model.to(device)
    return model

def _generate_file_table(experiment_id: str,
                         image_dir: Path,
                         annotations_file: Path):
    annotations = pd.read_csv(annotations_file)
    annotations["well"] = [
        entry.split(experiment_id)[1]
        if not entry == f"{experiment_id}{experiment_id}" else experiment_id
        for entry in annotations["ID"].tolist()
    ]
    
    file_names = []
    wells = []
    positions = []
    loops = []
    slices = []
    rpe_annotations = []
    experiments = []
    files = os.listdir(image_dir)
    files = [file for file in files if file.endswith(".tif")]
    for file_name in files:
        
        contents = file_name.split("-")
        contents = [entry for entry in contents if entry != ""]
        if len(contents) != 14:
            print(f"Invalid image: {file_name}")
            continue
        file_names.append(file_name)
        wells.append(contents[0])
        positions.append(contents[1])
        loops.append(contents[2])
        slices.append(contents[4])
        rpe_annotations.append(annotations.loc[annotations["well"] == contents[0], "RPE"].iloc[0])
        experiments.append(experiment_id)
    
    file_dict = {}
    file_dict["experiment"] = experiments
    file_dict["file_name"] = file_names
    file_dict["well"] = wells
    file_dict["position"] = positions
    file_dict["loop"] = loops
    file_dict["slice"] = slices
    file_dict["RPE"] = rpe_annotations
    
    return pd.DataFrame(file_dict)
    
def _polynomial_upsample(image, scale_factor, order = 3):
    return zoom(image, scale_factor, order=order)

def _create_mask(image: np.ndarray,
                 model: UNet) -> np.ndarray:
    image_dimension = image.shape[0]
    transforms = A.Compose([
        A.Resize(image_dimension, image_dimension),
        ToTensorV2()
    ])
    preprocessed = transforms(image = image)
    img_tensor = preprocessed["image"].unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.to("cuda")
    with torch.no_grad():
        pred_mask = model(img_tensor.reshape(1,1,image_dimension,image_dimension)).squeeze()
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.cpu().detach().numpy()
    return pred_mask

def _bin_image(img: np.ndarray,
               bin_size: int,
               mask: bool = False):
    """
    Bins the image on demand.
    """
    height, width = img.shape[:2]

    new_height = height // bin_size
    new_width = width // bin_size

    if new_height == height and new_width == width:
        return img
    
    binned_image = np.zeros((new_height, new_width), dtype=np.float32)

    for i in range(new_height):
        for j in range(new_width):
            binned_image[i, j] = np.mean(img[i*bin_size:(i+1)*bin_size, j*bin_size:(j+1)*bin_size])
    if mask:
        binned_image[binned_image < 0.5] = 0
    return binned_image

def _normalize_image(img: np.ndarray,
                     bitdepth: int) -> np.ndarray:
    return img / float(2**bitdepth)

def _min_max_scale(img: np.ndarray) -> np.ndarray:
    min_val = img.min()
    max_val = img.max()
    return (img - min_val) / (max_val - min_val)

def _preprocess_image(img: np.ndarray,
                      bitdepth: int) -> np.ndarray:
    img = _normalize_image(img, bitdepth)
    img = _min_max_scale(img)
    return img

def _read_image(path: Path):
    return cv2.imread(path, -1)

def _create_mask_from_image(image: np.ndarray,
                            model: UNet,
                            unet_input_size: int,
                            output_size: int) -> np.ndarray:
    """
    Function creates a mask from an image.
    First, the image is downsampled to unet_input_size
    in order to confer with the input restrictions.
    The corresponding mask is then upsampled to output_size
    and the original image is brought to the same size.
    """
    
    binning_factor_segmentation = int(image.shape[0] / unet_input_size)
    binning_factor_image = int(image.shape[0] / output_size)
    upsample_factor = int(output_size / unet_input_size)
    
    bitdepth = 8 * image.itemsize

    image = image.astype(np.float32)
    original_image = image.copy()
    
    preprocessed = _preprocess_image(image, bitdepth = bitdepth)
    
    binned_image = _bin_image(preprocessed, binning_factor_segmentation)

    assert binned_image.shape[0] == unet_input_size
    
    binned_mask = _create_mask(binned_image, model)
    
    mask = _polynomial_upsample(binned_mask, upsample_factor, order = 5)

    original_image_binned = _bin_image(original_image, binning_factor_image)
    assert mask.shape == original_image_binned.shape
    
    return original_image_binned, mask

def _threshold_mask(mask: np.ndarray,
                    threshold: float = 0.5) -> np.ndarray:
    mask[mask >= threshold] = 1
    mask[mask < threshold] = 0
    return mask

def _get_masked_image(image: np.ndarray,
                      model: UNet,
                      unet_input_size: int,
                      output_size: int,
                      return_clean_only: bool = False,
                      normalized: bool = False,
                      scaled: bool = False,
                      threshold: float = 0.5,
                      min_size_perc = 5) -> np.ndarray:
    """
    Will mask an image. We will assume that we want a cleaned mask.
    If the cleaning procedure is for some reason messed up,
    we let the user decide what to return.
    """
    not_clean = False
    
    image, mask = _create_mask_from_image(image,
                                          model = model,
                                          unet_input_size = unet_input_size,
                                          output_size = output_size)

    if normalized:
        bitdepth = 16 # we hard code this because the image is np.float32, that leads to super inconsistent results.
        image = _normalize_image(image, bitdepth)
    if scaled:
        image = _min_max_scale(image)
    
    mask = _threshold_mask(mask, threshold = threshold)
    
    cleaned_mask = _clean_and_label_mask(mask, threshold = threshold, min_size_perc = min_size_perc)
    
    if isinstance(cleaned_mask, int) and cleaned_mask == -1:
        print("removed only label in mask...")
        not_clean = True
    if isinstance(cleaned_mask, int) and cleaned_mask == -2:
        print("more than one label found...")
        not_clean = True

    if not_clean is False:
        cleaned_mask = cleaned_mask.astype(bool)
        return image * cleaned_mask
    if return_clean_only and not_clean is True:
        return None
    elif not return_clean_only and not_clean is True:
        mask = mask.astype(bool)
        return image * mask

def _clean_and_label_mask(mask: np.ndarray,
                          threshold: float = 0.5,
                          min_size_perc: float = 5) -> Union[int, np.ndarray]:
    """
    Removes other, smaller objects.

    Returns -1 if after removal nothing is there
    Returns -2 if there are two big labels
    Returns mask if cleanup worked or wasnt necessary

    """
    min_size = mask.shape[0]**2 * (min_size_perc / 100)
    label_objects, num_labels = skimage.measure.label(mask,
                                                      background = 0,
                                                      return_num = True)
    
    if num_labels > 1:
        mask = skimage.morphology.remove_small_objects(label_objects, min_size=min_size).astype(np.float64)
        if np.max(mask) == 0:
            return -1
        mask /= np.max(mask)
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
        mask = mask.astype(np.uint8)
        label_objects, num_labels = skimage.measure.label(mask, background=0, return_num=True)
        if num_labels > 1:
            return -2

    return label_objects
















