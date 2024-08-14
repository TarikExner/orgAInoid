import numpy as np
import os
from pathlib import Path
import cv2


def _assemble_file_paths(input_dir: Path) -> tuple[list, list]:
    files = [file for file in os.listdir(input_dir)]
    imgs = [file for file in files if not "_mask" in file and file.endswith(".tif") and not file.startswith(".")]
    masks = [file for file in files if "_mask" in file and file.endswith(".tif")]
    
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

def assemble_data(input_dir: Path,
                  output_dir: Path = os.getcwd()):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    imgs, masks = _assemble_file_paths(input_dir)
    img_array = []
    mask_array = []
    for img in imgs:
        img_array.append(cv2.imread(img, -1).astype(np.float32))
    for mask in masks:
        mask_array.append(cv2.imread(mask, -1).astype(np.float32))

    img_array = np.array(img_array)
    mask_array = np.array(mask_array)

    np.save(os.path.join(output_dir, "unet_segmentation_images.npy"), img_array)
    np.save(os.path.join(output_dir, "unet_segmentation_masks.npy"), mask_array)
    print("Dataset assembled successfully")


if __name__ == "__main__":
    input_dir = os.getcwd()
    assemble_data(input_dir)