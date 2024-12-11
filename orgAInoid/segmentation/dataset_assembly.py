import numpy as np
import os

from ..image_handling import ImageProcessor, OrganoidImage, OrganoidMaskImage

def _assemble_file_paths(input_dir: str) -> tuple[list, list]:
    files = [file for file in os.listdir(input_dir)]
    imgs = [
        file for file in files
        if "_mask" not in file
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
        try:
            corresponding_mask = mask_list[mask_list.index(unique_id + "_mask.tif")]
        except ValueError:
            print(f"No mask found for {file_name}.")
            return None
        assert unique_id in corresponding_mask
        return corresponding_mask

    matched = {}
    for file in imgs:
        mask = _get_corresponding_mask(file, masks)
        if mask is not None:
            matched[file] = mask
    
    matched_imgs = []
    matched_masks = []
    for img, mask in matched.items():
        matched_imgs.append(os.path.join(input_dir, img))
        matched_masks.append(os.path.join(input_dir, mask))

    return matched_imgs, matched_masks

def assemble_data(target_size: int,
                  input_dir: str,
                  output_dir: str= "./raw_data"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    imgs, masks = _assemble_file_paths(input_dir)
    img_array = []
    mask_array = []

    img_processor = ImageProcessor()

    for img_path in imgs:
        organoid_image = OrganoidImage(img_path)
        img = organoid_image.image
        preprocessed = img_processor.preprocess_for_segmentation(img, target_size)

        img_array.append(preprocessed)

    for mask_path in masks:
        mask_image = OrganoidMaskImage(mask_path)
        mask = mask_image.image
        preprocessed = img_processor.preprocess_for_segmentation(mask, target_size)
        preprocessed = img_processor._threshold_mask(preprocessed, threshold = 0.5)

        mask_array.append(preprocessed)

    img_array = np.array(img_array)
    mask_array = np.array(mask_array)

    np.save(os.path.join(output_dir, f"unet_segmentation_images_{target_size}.npy"), img_array)
    np.save(os.path.join(output_dir, f"unet_segmentation_masks_{target_size}.npy"), mask_array)

    print(f"Dataset assembled successfully for target size {target_size}")
