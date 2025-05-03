import os
import skimage
import pandas as pd
import numpy as np

from typing import Literal, Optional
import time

from . import methods
from ._utils import PROPERTIES, EXTRA_PROPERTIES

from ..image_handling import ImageHandler, OrganoidImage, OrganoidMask
from .._utils import _generate_file_table

def _run_z_projection(imgs: list[OrganoidImage],
                      projection: Optional[Literal["sum", "max"]]) -> OrganoidImage:

    if not projection:
        raise ValueError("please supply a projection")

    if len(imgs) == 1:
        return imgs[0]
    
    if projection == "max":
        projected_array = np.max(
            [org_image.image for org_image in imgs],
            axis = 0
        )
    elif projection == "sum":
        projected_array = np.sum(
            [org_image.image for org_image in imgs],
            axis = 0
        )
    else:
        raise ValueError(f"Unknown projection method: {projection}")

    img = OrganoidImage(path = None)
    img.set_image(projected_array)
    return img

def _calculate_morphometrics_on_image(original_image: OrganoidImage,
                                      img_handler: ImageHandler,
                                      experiment: str,
                                      well: str,
                                      timepoint: str,
                                      file_name: str,
                                      results_frame: pd.DataFrame) -> pd.DataFrame:
    img, mask = img_handler.get_mask_and_image(
        img = original_image,
        image_target_dimension = 2048,
        mask_threshold = 0.3,
        clean_mask = True,
        min_size_percentage = 7.5,
        crop_bounding_box = False
    )

    if img is None or mask is None:
        print(
            f"No analysis of organoid [{experiment}, {well}, {timepoint}] due to a masking error"
        )
        return results_frame

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

    results_frame.loc[
        (results_frame["experiment"] == experiment) &
        (results_frame["well"] == well) &
        (results_frame["loop"] == timepoint) &
        (results_frame["file_name"] == file_name),
        reg_table_df.columns
    ] = reg_table_df.values

    return results_frame

def run_morphometrics(experiment_id: str,
                      image_dir: str,
                      metadata_dir: str,
                      output_dir: str,
                      segmentator_input_dir: str = "../segmentation/segmentators",
                      segmentator_input_size: int = 512,
                      segmentation_model_name: Literal["HRNET", "UNET", "DEEPLABV3"] = "DEEPLABV3",
                      slices: Optional[list[str]] = None,
                      z_projection: Optional[Literal["max", "sum"]] = None) -> pd.DataFrame:
    """\
    Convention:
        annotations by Cassian are termed "E001_annotations.csv"
        file overviews are termed "E001_file_overview.csv"

        Both will be saved in ../metadata
    

    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
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

    img_handler = ImageHandler(
        segmentator_input_dir = segmentator_input_dir,
        segmentator_input_size = segmentator_input_size,
        segmentation_model_name = segmentation_model_name
    )

    if slices is None:
        slices = ["SL003"]

    file_frame = file_frame[file_frame["slice"].isin(slices)]
    individual_organoids = file_frame[["experiment", "well", "timepoint"]].drop_duplicates()

    result_dataframe = file_frame.copy()

    placeholder_file_name = "PROJECTED"

    if z_projection:
        constant_columns = [
            col for col in result_dataframe
            if col not in ["file_name", "slice"]
        ]
        result_dataframe = result_dataframe[constant_columns].drop_duplicates()
        assert result_dataframe.shape[0] == individual_organoids.shape[0]
        result_dataframe["slice"] = f"zproj_{z_projection}"
        result_dataframe["file_name"] = placeholder_file_name

    assert isinstance(result_dataframe, pd.DataFrame)

    start = time.time()
    for i, row in enumerate(individual_organoids):
        if i%100 == 0 and i != 0:
            stop = time.time()
            print(f"Processed {i}/{result_dataframe.shape[0]} images in {round(stop-start,2)} seconds")
            start = time.time()
        experiment, well, timepoint = row[1]
        organoid_data = file_frame[
            (file_frame["experiment"] == experiment) &
            (file_frame["well"] == well) &
            (file_frame["loop"] == timepoint)
        ]
        file_names = organoid_data["file_name"].tolist()

        if z_projection is not None:
            image_paths = [
                os.path.join(image_dir, file_name)
                for file_name in organoid_data["file_name"].tolist()
            ]
            imgs = [
                OrganoidImage(image_path)
                for image_path in image_paths
            ]
            original_image = _run_z_projection(imgs, z_projection)
            result_dataframe = _calculate_morphometrics_on_image(
                original_image,
                img_handler,
                experiment,
                well,
                timepoint,
                placeholder_file_name,
                result_dataframe
            )
        else:
            for file_name in file_names:
                image_path = os.path.join(image_dir, file_name)
                original_image = OrganoidImage(image_path)
                result_dataframe = _calculate_morphometrics_on_image(
                    original_image,
                    img_handler,
                    experiment,
                    well,
                    timepoint,
                    placeholder_file_name,
                    result_dataframe
                )

    result_dataframe.to_csv(
        os.path.join(output_dir, f"{experiment_id}_morphometrics.csv"),
        index = False
    )

    return result_dataframe

def _SAFETY_run_morphometrics(experiment_id: str,
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
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
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
    start = time.time()
    for i, file_name in enumerate(file_list):
        if i%100 == 0 and i != 0:
            stop = time.time()
            print(f"Processed {i}/{len(file_list)} images in {round(stop-start,2)} seconds")
            start = time.time()
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

    file_frame.to_csv(os.path.join(output_dir, f"{experiment_id}_morphometrics.csv"), index = False)

    return file_frame


if __name__ == "__main__":
    pass
