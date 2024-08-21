import os
from os import PathLike
import numpy as np
import pandas as pd
from typing import Optional
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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

    train_wells: Optional[np.ndarray] = None
    test_wells: Optional[np.ndarray] = None

    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None

    def __init__(self,
                 dataset_id: str,
                 file_frame: pd.DataFrame,
                 start_timepoint: int,
                 stop_timepoint: int,
                 slices: list[str],
                 image_size: int,
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

        self.train_df, self.test_df = self._train_test_split_dataframe(
            file_frame = file_frame,
            experiment_dir = experiment_dir
        )
        self.create_datasets()

    def create_datasets(self):
        self.X_train, self.y_train = self._prepare_classification_data(df = self.train_df)
        self.X_test, self.y_test = self._prepare_classification_data(df = self.test_df)

    def _prepare_classification_data(self,
                                     df: pd.DataFrame,
                                     slice_to_mask: str = "SL003") -> tuple[np.ndarray, np.ndarray]:
        """\

        slice_to_mask
            We need to make sure that we use the correct slice for masking.
            EDIT: For now, we will treat both images as valid input for the UNET.

        """

        images = []
        labels = []

        unique_experiment_well_combo = self._get_unique_experiment_well_combo(df, "experiment", "well")

        for experiment, well in unique_experiment_well_combo:
            well = df[
                (df["experiment"] == experiment) &
                (df["well"] == well)
            ].copy()

            # we loop through the timepoints in order to capture all slices
            for loop in well["loop"].unique():

                loop_data = well[well["loop"] == loop].copy()

                loop_label = list(set(loop_data["RPE"].tolist()))

                assert len(loop_label) == 1

                image_paths = loop_data["image_path"].tolist()

                loop_images = []
                for path in image_paths:
                    image = self.img_handler.read_image(path)
                    masked_image = self.img_handler.get_masked_image(image,
                                                                     normalized = True,
                                                                     scaled = True)
                    if masked_image is not None:
                        loop_images.append(masked_image.img)
                    else:
                        loop_images = None
                        break

                if loop_images is not None:
                    images.append(np.array(loop_images))
                    labels.append(loop_label[0])
                else:
                    print(f"Dataset creation: skipping images {image_paths}")

        images = np.array(images)
        labels = np.array(labels)

        assert images.shape[0] == labels.shape[0]
        assert images.shape[1] == len(self.slices)

        labels = self._one_hot_encode_labels(labels)

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

    def save(self,
             output_dir: PathLike):
        with open(os.path.join(output_dir, f"{self.dataset_id}.cds"), "wb") as file:
            pickle.dump(self, file)


def read_classification_dataset(file_name) -> OrganoidClassificationDataset:
    with open(file_name, "rb") as file:
        dataset = pickle.load(file)
    return dataset
