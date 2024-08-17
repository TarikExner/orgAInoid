import numpy as np
import os
from pathlib import Path
import cv2

from .._utils import ImageHandler


def _assemble_file_paths(input_dir: str) -> tuple[list, list]:
    files = [file for file in os.listdir(input_dir)]
    imgs = [
        file for file in files
        if not "_mask" in file
        and file.endswith(".tif")
        and not file.startswith(".")
    ]
    masks = [
        file for file in files
        if "_mask" in file 
        and file.endswith(".tif")
        and not file.startswith(".")
    ]
    
    def _get_corresponding_mask(file_name, mask_list):
        unique_id = file_name.split(".tif")[0]
        corresponding_mask = mask_list[mask_list.index(unique_id + "_mask.tif")]
        assert unique_id in corresponding_mask
        return corresponding_mask
        
    matched = {file: _get_corresponding_mask(file, masks) for file in imgs}
    
    matched_imgs = []
    matched_masks = []
    for img, mask in matched.items():
        matched_imgs.append(os.path.join(input_dir, img))
        matched_masks.append(os.path.join(input_dir, mask))

    return matched_imgs, matched_masks

def assemble_data(target_size: float,
                  input_dir: str,
                  output_dir: str= "./raw_data"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    imgs, masks = _assemble_file_paths(input_dir)
    img_array = []
    mask_array = []

    img_handler = ImageHandler(
        target_image_size = target_size,
        unet_input_dir = None,
        unet_input_size = None
    )

    for img_path in imgs:
        img = img_handler.read_image(img_path)
        img.preprocess_for_unet(target_size)
        img_array.append(img.unet_preprocessed)
    for mask_path in masks:
        mask = img_handler.read_image(mask_path)
        mask.preprocess_for_unet(target_size)
        mask_array.append(mask.unet_preprocessed)

    img_array = np.array(img_array)
    mask_array = np.array(mask_array)

    np.save(os.path.join(output_dir, f"unet_segmentation_images_{target_size}.npy"), img_array)
    np.save(os.path.join(output_dir, f"unet_segmentation_masks_{target_size}.npy"), mask_array)
    print(f"Dataset assembled successfully for target size {target_size}")


if __name__ == "__main__":
    input_dir = os.getcwd()
