
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import cv2
import torch
from scipy.ndimage import zoom
import skimage
import os
from os import PathLike

from typing import Optional, Union

from abd import abstractmethod

from .segmentation import UNet, DEEPLABV3, HRNET
from ._augmentation import to_normalized_tensor

def _generate_file_table(experiment_id: str,
                         image_dir: str,
                         annotations_file: str):
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
