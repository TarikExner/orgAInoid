import os
from os import PathLike
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import time
from .._utils import ImageHandler



class OrganoidClassificationDataset:
    """\
    Base class to handle datasets associated with classification.

    """
    dataset_id: str
    start_timepoint: int
    stop_timepoint: int
    slices: list[str]
    image_dimension: int

    cropped_bbox: bool
    rescale_cropped_image: bool
    crop_size: Optional[int]

    train_wells: Optional[np.ndarray] = None
    test_wells: Optional[np.ndarray] = None

    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    n_train_images: int
    n_test_images: int

    def __init__(self,
                 dataset_id: str,
                 file_frame: pd.DataFrame,
                 start_timepoint: int,
                 stop_timepoint: int,
                 slices: list[str],
                 image_size: int,
                 crop_bbox: bool,
                 rescale_cropped_image: bool,
                 crop_size: Optional[int],
                 unet_dir: str,
                 unet_input_size: int,
                 experiment_dir: PathLike):
        self.dataset_id = dataset_id
        self.slices = slices
        self.start_timepoint = start_timepoint
        self.stop_timepoint = stop_timepoint
        self.image_dimension = image_size
        self.img_handler = ImageHandler(
            target_image_size = self.image_dimension,
            unet_input_dir = unet_dir,
            unet_input_size = unet_input_size
        )
        self.cropped_bbox = crop_bbox
        self.rescale_cropped_image = rescale_cropped_image
        self.crop_size = crop_size

        self.train_df, self.test_df = self._train_test_split_dataframe(
            file_frame = file_frame,
            experiment_dir = experiment_dir
        )
        self.create_datasets()

    def create_datasets(self):
        self.X_train, self.y_train = self._prepare_classification_data(df = self.train_df)
        self.X_test, self.y_test = self._prepare_classification_data(df = self.test_df)
        self.n_train_images = self.X_train.shape[0]
        self.n_test_images = self.X_test.shape[0]

    def _prepare_classification_data(self,
                                     df: pd.DataFrame,
                                     slice_to_mask: str = "SL003",
                                     train_set: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """\

        slice_to_mask
            We need to make sure that we use the correct slice for masking.
            EDIT: For now, we will treat both images as valid input for the UNET.

        """
        n_failed_images = 0

        images = []
        labels = []

        unique_experiment_well_combo = self._get_unique_experiment_well_combo(df, "experiment", "well")

        assert self.train_wells is not None
        assert self.test_wells is not None

        n_wells = unique_experiment_well_combo.shape[0]

        start = time.time()

        for i, (experiment, well) in enumerate(unique_experiment_well_combo):

            stop = time.time()

            if i != 0:
                print(f"{i}/{n_wells} wells completed in {round(stop-start, 2)} seconds..")

            start = time.time()

            well_df = df[
                (df["experiment"] == experiment) &
                (df["well"] == well)
            ].copy()

            # we loop through the timepoints in order to capture all slices
            for loop in well_df["loop"].unique():

                loop_data = well_df[well_df["loop"] == loop].copy()

                loop_label = list(set(loop_data["RPE"].tolist()))

                assert len(loop_label) == 1

                image_paths = loop_data["image_path"].tolist()

                loop_images = []
                for path in image_paths:
                    image = self.img_handler.read_image(path)
                    masked_image = self.img_handler.get_masked_image(image,
                                                                     normalized = True,
                                                                     scaled = True,
                                                                     crop_bounding_box = self.cropped_bbox,
                                                                     rescale = self.rescale_cropped_image,
                                                                     crop_size = self.crop_size)
                    if masked_image is not None:
                        loop_images.append(masked_image.img)
                    else:
                        loop_images = None
                        break

                if loop_images is not None:
                    images.append(np.array(loop_images))
                    labels.append(loop_label[0])
                else:
                    n_failed_images += 1
                    print(f"Dataset creation: skipping images {experiment}: {well}")

        images = np.array(images)
        labels = np.array(labels)

        assert images.shape[0] == labels.shape[0]
        assert images.shape[1] == len(self.slices)

        labels = self._one_hot_encode_labels(labels)

        print(f"In total, {n_failed_images}/{images.shape[0]} images were skipped.")

        return images, labels

    def _one_hot_encode_labels(self,
                               labels_array: np.ndarray) -> np.ndarray:
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels_array)
        
        onehot_encoder = OneHotEncoder()
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        classification = onehot_encoder.fit_transform(integer_encoded).toarray()
        return classification
                    
    def _train_test_split_dataframe(self,
                                    file_frame: pd.DataFrame,
                                    experiment_dir: PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
        file_frame["image_path"] = [
            os.path.join(experiment_dir, experiment, file_name)
            for experiment, file_name in zip(file_frame["experiment"].tolist(), file_frame["file_name"].tolist())
        ]
        timepoints = [
            f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}"
            for i in range(self.start_timepoint, self.stop_timepoint)
        ]
        files = file_frame[file_frame["slice"].isin(self.slices)]
        files = files.dropna()
        files = files[files["loop"].isin(timepoints)]
        assert isinstance(files, pd.DataFrame)

        unique_wells = self._get_unique_experiment_well_combo(files, "experiment", "well")

        train_wells, test_wells = train_test_split(unique_wells, test_size = 0.1, random_state = 187)
        assert isinstance(train_wells, np.ndarray)
        assert isinstance(test_wells, np.ndarray)

        train_df = self._filter_wells(files, train_wells, ["experiment", "well"])
        test_df = self._filter_wells(files, test_wells, ["experiment", "well"])

        self.train_wells = train_df[["experiment", "well"]].to_numpy()
        self.test_wells = test_df[["experiment", "well"]].to_numpy()
        return train_df, test_df

    def _get_unique_experiment_well_combo(self,
                                          df: pd.DataFrame,
                                          col1: str,
                                          col2: str):
        return df[[col1, col2]].drop_duplicates().reset_index(drop=True).to_numpy()

    def _filter_wells(self,
                      df: pd.DataFrame,
                      combinations: np.ndarray,
                      columns: list[str]):
        combinations_df = pd.DataFrame(combinations, columns=columns)
        return df.merge(combinations_df, on=columns, how='inner')

    def _save_metadata(self, output_dir: PathLike):
        """Save the metadata to a central JSON file."""
        metadata = self._create_metadata()

        metadata_file = Path(output_dir) / "datasets_metadata.json"

        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
        else:
            all_metadata = {}

        # Check if the dataset_id already exists
        if metadata.get("dataset_id") in all_metadata:
            raise ValueError(f"Dataset with ID {self.dataset_id} already exists in metadata.")

        # Add new metadata entry under the dataset_id key
        all_metadata[self.dataset_id] = metadata

        # Write the updated metadata back to the JSON file
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=4)

    def _create_metadata(self) -> dict:
        """Create a dictionary containing the metadata of the dataset."""
        metadata = {
            "start_timepoint": self.start_timepoint,
            "stop_timepoint": self.stop_timepoint,
            "slices": self.slices,
            "image_dimension": self.image_dimension,
            "cropped_bbox": self.cropped_bbox,
            "rescale_cropped_image": self.rescale_cropped_image,
            "crop_size": self.crop_size,
            "n_train_images": self.n_train_images,
            "n_test_images": self.n_test_images,
        }
        return metadata

    def save(self, output_dir: PathLike):
        """Save the dataset and its metadata to disk."""
        # Save the dataset itself
        with open(os.path.join(output_dir, f"{self.dataset_id}.cds"), "wb") as file:
            pickle.dump(self, file)

        # Save the metadata
        self._save_metadata(output_dir)



def read_classification_dataset(file_name) -> OrganoidClassificationDataset:
    with open(file_name, "rb") as file:
        dataset = pickle.load(file)
    return dataset
