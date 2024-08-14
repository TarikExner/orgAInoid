import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from ..segmentation.model import UNet

from .._utils import _get_masked_image, _load_unet_model, _read_image


def _print_error_log(image_path):

    return None

def _create_timeseries(image_paths: list[str],
                       unet_dir: Path,
                       unet_input_size: int,
                       output_size: int) -> np.ndarray:
    model = _load_unet_model(unet_dir, unet_input_size)
    images = []
    start = time.time()
    print(f"A total of {len(image_paths)} images")
    for i, image_path in enumerate(image_paths):
        original_image = _read_image(image_path)
        if original_image is None:
            print(f"WARNING Corrupted image: {file_name}")
            return None
        masked_image = _get_masked_image(original_image,
                                         model = model,
                                         unet_input_size = unet_input_size,
                                         output_size = output_size,
                                         return_clean_only = True,
                                         normalized = True,
                                         scaled = True,
                                         threshold = 0.5,
                                         min_size_perc = 5)
        if masked_image is None:
            print("Masking went wrong")
            return None
        images.append(masked_image)
        if i%100 == 0:
            if i!= 0:
                print(f"{i} images done in {time.time() - start} seconds")
            start = time.time()
    image_array = np.array(images)
    return image_array


def create_timeseries_dataset(image_file_frame: pd.DataFrame,
                              unet_dir: Path,
                              unet_input_size: int,
                              image_size: int,
                              output_file_name: Path) -> None:
    experiment = image_file_frame["experiment"].unique().tolist()[0]
    input_dir = f"../../{experiment}"
    image_file_frame = image_file_frame[image_file_frame["slice"] == "SL003"]
    image_dataset = []
    for well in image_file_frame["well"].unique().tolist():
        well_specific = image_file_frame[image_file_frame["well"] == well]
        well_specific = well_specific.sort_values("loop", ascending = True)
        file_names = well_specific["file_name"].tolist()
        file_names = [os.path.join(input_dir, file) for file in file_names]
        series = _create_timeseries(file_names,
                                    unet_dir = unet_dir,
                                    unet_input_size = unet_input_size,
                                    output_size = image_size)
        if series is not None:
            image_dataset.append(series)
        else:
            print(f"Skipping sequence for well {well}")
    image_dataset = np.array(image_dataset)
    np.save(output_file_name, image_dataset)
    return
        