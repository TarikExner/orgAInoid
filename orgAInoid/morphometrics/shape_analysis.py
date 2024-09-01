import os
import skimage
import pandas as pd
import numpy as np


from typing import Literal

from . import methods
from ._utils import PROPERTIES, EXTRA_PROPERTIES

from ..image_handling import ImageHandler, OrganoidImage, OrganoidMask
from .._utils import _generate_file_table



def run_morphometrics(experiment_id: str,
                      image_dir: str,
                      metadata_dir: str,
                      output_dir: str,
                      segmentator_input_dir: str = "../segmentation/segmentators",
                      segmentator_input_size: int = 512,
                      segmentation_model_name: Literal["HRNET", "UNET", "DEEPLABV3"] = "DEEPLABV3",
                      ):
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
        file_frame = _generate_file_table(
            experiment_id = experiment_id,
            image_dir = image_dir,
            annotations_file = annotations_file
        )
        file_frame.to_csv(overview_file, index = False)

    file_frame = file_frame[file_frame["slice"] == "SL003"]
    file_list = file_frame["file_name"].tolist()

    img_handler = ImageHandler(
        segmentator_input_dir = segmentator_input_dir,
        segmentator_input_size = segmentator_input_size,
        segmentation_model_name = segmentation_model_name
    )

    for file_name in file_list:
        image_path = os.path.join(image_dir, file_name)
        original_image = OrganoidImage(image_path)
        if original_image is None:
            print("WARNING Corrupted image: {file_name}")
            continue

        img, mask = img_handler.get_mask_and_image(
            img = original_image,
            image_target_dimension = 2048,
            mask_threshold = 0.3,
            clean_mask = True,
            min_size_percentage = 7.5,
            crop_bounding_box = False
        )

        if img is None or mask is None:
            print(f"Skipping image {file_name} due to a masking error")
            continue

        mask = OrganoidMask(mask)

        labeled_mask = mask.label_mask(mask.image)

        reg_table = skimage.measure.regionprops_table(
            labeled_mask,
            intensity_image = img,
            properties = PROPERTIES,
            extra_properties = EXTRA_PROPERTIES
        )
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
    pass
