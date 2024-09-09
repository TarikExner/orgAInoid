import os
from os import PathLike
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import json
from typing import Optional, Literal, Tuple, Union

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import time

from ._utils import ImageMetadata, DatasetMetadata, _get_model_from_enum
from ..image_handling import ImageHandler, OrganoidImage


# Validation dataset: no split
# Normal dataset: train/test_split
# crossvalidation dataset: multiple train/test_splits

class OrganoidDataset:
    """\
    Base class to handle datasets associated with classification.
    This class is not really meant to be used by the user. If
    a full dataset is required, use the OrganoidValidationDataset.

    The images are stored in X as a np.ndarray. The image dimension
    is determined by the input parameters 'slices' and 'image_size'.

    The annotations are stored in y as a dictionary. For each annotation
    in the 'file_frame', a separate np.ndarray is stored where the
    labels are one-hot-encoded. 

    We will store a dictionary to keep track of the images and their
    metadata, as this is important for proper cross-validation.

    readouts_n_classes: Necessary for whenever there is only one label
    in the dataset. It will be assumed that this label would be the
    first class and gets encoded that way!


    """

    def __init__(self,
                 dataset_id: str,
                 readouts: list[str],
                 readouts_n_classes: list[int],
                 file_frame: pd.DataFrame,
                 start_timepoint: int,
                 stop_timepoint: int,
                 slices: list[str],
                 image_size: int,
                 crop_bbox: bool,
                 rescale_cropped_image: bool,
                 crop_size: Optional[int],
                 segmentator_input_dir: str,
                 segmentator_input_size: int,
                 segmentation_model_name: Literal["HRNET", "UNET", "DEEPLABV3"],
                 experiment_dir: str,
                 _skip_init: bool = False):

        if _skip_init is True:
            return

        self.img_handler = ImageHandler(
            segmentator_input_dir = segmentator_input_dir,
            segmentator_input_size = segmentator_input_size,
            segmentation_model_name = segmentation_model_name
        )

        self._image_metadata = ImageMetadata(
            dimension = image_size,
            cropped_bbox = crop_bbox,
            scaled_to_size = rescale_cropped_image,
            crop_size = crop_size,
            segmentator_model = _get_model_from_enum(segmentation_model_name),
            segmentator_input_size = segmentator_input_size,
            mask_threshold = 0.3,
            cleaned_mask = True,
            scale_masked_image = True,
            crop_bounding_box = True,
            rescale_cropped_image = rescale_cropped_image,
            crop_bounding_box_dimension = crop_size
        )

        if not isinstance(readouts, list):
            readouts = [readouts]
        if not isinstance(readouts_n_classes, list):
            readouts_n_classes = [readouts_n_classes]

        assert len(readouts) == len(readouts_n_classes), "Provide n_classes for every readout"

        self._dataset_metadata = DatasetMetadata(
            dataset_id = dataset_id,
            experiment_dir = experiment_dir,
            readouts = readouts,
            readouts_n_classes = readouts_n_classes,
            start_timepoint = start_timepoint,
            stop_timepoint = stop_timepoint,
            slices = slices
        )

        self._metadata = self._preprocess_file_frame(file_frame)         

        self.create_full_dataset(self._metadata)

        self._create_class_counts()

    def _create_class_counts(self):
        class_balances = {}
        for readout in self.dataset_metadata.readouts:
            n_uniques = self.metadata.groupby(readout).nunique()["well"]
            class_balances[readout] = {
                n_uniques.index[i]: round(n_uniques.iloc[i] / n_uniques.sum(), 2)
                for i in range(n_uniques.shape[0])
            }

        self.dataset_metadata.class_balance = class_balances
        return

    def _preprocess_file_frame(self,
                               file_frame: pd.DataFrame) -> pd.DataFrame:
        """\
        Selects the necessary parts of the files, e.g. slices, loops and so forth.
        Appends 'image_path' and a placeholder 'IMAGE_ARRAY_INDEX' and sorts the values.
        """
        file_frame.loc[:, "image_path"] = [
            os.path.join(self.dataset_metadata.experiment_dir, experiment, file_name)
            for experiment, file_name
            in zip(file_frame["experiment"].tolist(), file_frame["file_name"].tolist())
        ]
        timepoints = [
            f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}"
            for i in range(self.dataset_metadata.start_timepoint, self.dataset_metadata.stop_timepoint)
        ]
        assert isinstance(timepoints, list)
        assert len(timepoints) != 0
        assert isinstance(self.dataset_metadata.slices, list)
        preprocessed = file_frame.loc[
            (file_frame["slice"].isin(self.dataset_metadata.slices)) &
            (file_frame["loop"].isin(timepoints)),
            ["image_path", "experiment", "well", "loop", "slice"] + self.dataset_metadata.readouts
        ]

        # important for the extraction of slices later
        preprocessed = preprocessed.sort_values(
            ["experiment", "well", "loop", "slice"]
        )

        # We keep track of the individual images in order to split them later
        # by individual wells. That way, we dont have to care about image
        # order or individual wells at the moment. We instantiate as -1
        # and replace it with actual numbers later
        preprocessed["IMAGE_ARRAY_INDEX"] = -1

        assert isinstance(preprocessed, pd.DataFrame)
        return preprocessed


    def create_full_dataset(self,
                            file_frame: pd.DataFrame) -> None:
        self.X, self.y = self._prepare_classification_data(df = file_frame)
        self.n_images = self.X.shape

    def _prepare_classification_data(self,
                                     df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        """\

        assembles the data

        """

        df = self.metadata
        n_failed_images = 0

        # images are ultimately stored as a np.ndarray
        images = []

        # labels are stored in multiple np.ndarrays in a dictionary
        labels = {
            readout: []
            for readout in self.dataset_metadata.readouts
        }

        start = time.time()

        unique_experiment_well_combo = self._get_unique_experiment_well_combo(df, "experiment", "well")
        n_wells = unique_experiment_well_combo.shape[0]

        image_array_index = 0
        for i, (experiment, well) in enumerate(unique_experiment_well_combo):

            stop = time.time()

            if i != 0:
                print(f"{i}/{n_wells} wells completed in {round(stop-start, 2)} seconds..")

            start = time.time()

            well_df = df[
                (df["experiment"] == experiment) &
                (df["well"] == well)
            ].copy()

            well_labels = well_df[self.dataset_metadata.readouts]
            assert isinstance(well_labels, pd.DataFrame)
            well_labels = well_labels.drop_duplicates()
            assert well_labels.shape[0] == 1

            # we loop through the timepoints in order to capture all slices
            for loop in well_df["loop"].unique():
                loop_data = well_df[well_df["loop"] == loop].copy()

                image_paths = loop_data["image_path"].tolist()

                loop_images = []
                for path in image_paths:
                    image = OrganoidImage(path)
                    masked_image = self.img_handler.get_masked_image(
                        image,
                        image_target_dimension = self.image_metadata.dimension,
                        mask_threshold = self.image_metadata.mask_threshold,
                        clean_mask = self.image_metadata.cleaned_mask,
                        scale_masked_image = self.image_metadata.scale_masked_image,
                        crop_bounding_box = self.image_metadata.crop_bounding_box,
                        rescale_cropped_image = self.image_metadata.rescale_cropped_image,
                        crop_bounding_box_dimension = self.image_metadata.crop_bounding_box_dimension
                    )
                    if masked_image is not None:
                        loop_images.append(masked_image)
                    else:
                        loop_images = None
                        break
                
                # we check if all slices are within the array
                if loop_images is not None:
                    if self.dataset_metadata.n_slices != len(loop_images):
                        loop_images = None
                
                if loop_images is not None:
                    images.append(np.array(loop_images))
                    self.metadata.loc[
                        (self.metadata["experiment"] == experiment) &
                        (self.metadata["well"] == well) &
                        (self.metadata["loop"] == loop),
                        "IMAGE_ARRAY_INDEX"
                    ] = image_array_index
                    image_array_index += 1

                    for label in well_labels.columns:
                        labels[label].append(well_labels[label].iloc[0])
                else:
                    n_failed_images += 1
                    print(f"Dataset creation: skipping images {experiment}: {well} : {loop}")

        images = np.array(images)
        labels = {
            key: np.array(label_list)
            for key, label_list in labels.items()
        }

        for label in labels:
            assert images.shape[0] == labels[label].shape[0]

        assert images.shape[1] == self.dataset_metadata.n_slices

        labels = {
            key: self._one_hot_encode_labels(label_list, key)
            for key, label_list in labels.items()
        }

        print(f"In total, {n_failed_images}/{images.shape[0]} images were skipped.")

        return images, labels

    def merge(self,
              other: "OrganoidDataset",
              copy: bool = False) -> "OrganoidDataset":

        if copy is True:
            raise NotImplementedError("Currently not supported to copy the instance")

        assert isinstance(other, type(self)), "Currently only one dataset is supported"

        assert self.image_metadata == other.image_metadata, "Can only merge datasets with identical settings"
        assert all(key in other.y for key in self.y), "Can only merge datasets with identical readouts"

        if not hasattr(other.dataset_metadata, "timepoints"):
            other.dataset_metadata.timepoints = list(
                range(
                    other.dataset_metadata.start_timepoint,
                    other.dataset_metadata.stop_timepoint + 1
                )
            )
        if not hasattr(self.dataset_metadata, "timepoints"):
            self.dataset_metadata.timepoints = list(
                range(
                    self.dataset_metadata.start_timepoint,
                    self.dataset_metadata.stop_timepoint + 1
                )
            )

        combined_timepoints = self.dataset_metadata.timepoints + other.dataset_metadata.timepoints

        pre_merge_X_shape = self.X.shape[0]

        # correct indices
        other_md = other.metadata.copy()
        other_md["IMAGE_ARRAY_INDEX"] = [
            index + pre_merge_X_shape
            if index != -1 else index
            for index in other_md["IMAGE_ARRAY_INDEX"]
        ]

        self.X = np.vstack([self.X, other.X])
        for key in self.y:
            self.y[key] = np.vstack([self.y[key], other.y[key]])

        self._metadata = pd.concat([self.metadata, other_md], axis = 0)

        self._create_class_counts()

        self.dataset_metadata.timepoints = combined_timepoints
        self.dataset_metadata.calculate_start_and_stop_timepoint()


        return self

    def add_annotations(self,
                        annotations: Union[list[str],str],
                        df: pd.DataFrame) -> None:
        if not isinstance(annotations, list):
            annotations = [annotations]
        assert "experiment" in df.columns, "'experiment' has to be one of the columns"
        assert "well" in df.columns, "'well' has to be one of the columns"
        new_metadata = df[["experiment", "well"] + annotations]
        self._metadata = self.metadata.merge(
            new_metadata,
            left_on = ["experiment", "well"],
            right_on = ["experiment", "well"]
        )
        # Copy it to not mess up the actual metadata
        merged = self._metadata[self._metadata["IMAGE_ARRAY_INDEX"] != -1].copy()
        merged = self._metadata.sort_values("IMAGE_ARRAY_INDEX", ascending = True)
        for annotation in annotations:
            self.dataset_metadata.readouts.append(annotation)
            merged_no_dups = merged[["experiment", "well", "loop", annotation]].copy().drop_duplicates()
            encoded_labels = self._one_hot_encode_labels(
                merged_no_dups[annotation].to_numpy(),
                readout = annotation
            )
            assert encoded_labels.shape[0] == self.X.shape[0]
            self.y[annotation] = encoded_labels
        self._create_class_counts()
        return

    def _one_hot_encode_labels(self,
                               labels_array: np.ndarray,
                               readout: str) -> np.ndarray:
        n_classes = self.dataset_metadata.n_classes_dict[readout]
        n_appended = 0
        if np.unique(labels_array).shape[0] != n_classes:
            # we have not enough labels. That means we look up how many
            # classes there are and provide the according array.

            # first, we provide every item there is potentially as an array
            if "classes" in readout:
                full_class_spectrum = np.array(list(range(4)))
            else:
                full_class_spectrum = np.array(["no", "yes"])
            n_appended = full_class_spectrum.shape[0]

            labels_array = np.vstack([labels_array, full_class_spectrum])

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels_array)
    
        onehot_encoder = OneHotEncoder()
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        classification = onehot_encoder.fit_transform(integer_encoded).toarray()

        if n_appended != 0:
            classification = classification[:-n_appended]
        return classification

    def _get_unique_experiment_well_combo(self,
                                          df: pd.DataFrame,
                                          col1: str,
                                          col2: str):
        return df[[col1, col2]].drop_duplicates().reset_index(drop=True).to_numpy()

    def _save_metadata(self,
                       output_dir: PathLike,
                       overwrite: bool = False
                       ):
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
        if not overwrite:
            if metadata.get("dataset_id") in all_metadata:
                raise ValueError(
                    f"Dataset with ID {self.dataset_metadata.dataset_id} already exists in metadata.")

        # Add new metadata entry under the dataset_id key
        all_metadata[self.dataset_metadata.dataset_id] = metadata

        # Write the updated metadata back to the JSON file
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=4)

    def _create_metadata(self) -> dict:
        """Create a dictionary containing the metadata of the dataset."""
        metadata = {
            "start_timepoint": self.dataset_metadata.start_timepoint,
            "stop_timepoint": self.dataset_metadata.stop_timepoint,
            "slices": self.dataset_metadata.slices,
            "image_dimension": self.image_metadata.dimension,
            "readouts": self.dataset_metadata.readouts,
            "cropped_bbox": self.image_metadata.cropped_bbox,
            "rescale_cropped_image": self.image_metadata.rescale_cropped_image,
            "crop_size": self.image_metadata.crop_size,
        }
        return metadata

    def save(self, output_dir: PathLike, overwrite: bool = False):
        """Save the dataset and its metadata to disk."""
        
        # we delete the img_handler because we want to be able to read
        # everything independent of a GPU. 
        del self.img_handler

        file_name = os.path.join(output_dir, f"{self.dataset_metadata.dataset_id}.cds")
        if os.path.isfile(file_name) and not overwrite:
            raise ValueError("Dataset exists, set overwrite to True")
        with open(file_name, "wb") as file:
            pickle.dump(self, file)

        # Save the metadata
        self._save_metadata(output_dir, overwrite)
 
    @property
    def image_metadata(self):
        """Returns metadata associated with image processing"""
        return self._image_metadata

    @property
    def dataset_metadata(self):
        """Returns metadata associated with dataset"""
        return self._dataset_metadata

    @property
    def metadata(self):
        """Returns metadata associated with the data"""
        return self._metadata

    @classmethod
    def read_classification_dataset(cls,
                                    file_name) -> "OrganoidDataset":
        with open(file_name, "rb") as file:
            dataset = pickle.load(file)
        return dataset

    @classmethod
    def from_instance(cls, old_instance: "OrganoidDataset") -> "OrganoidDataset":
        init_kwargs = {
            "dataset_id": None,
            "readouts": None,
            "readouts_n_classes": None,
            "file_frame": None,
            "start_timepoint": None,
            "stop_timepoint": None,
            "slices": None,
            "image_size": None,
            "crop_bbox": None,
            "rescale_cropped_image": None,
            "crop_size": None,
            "segmentator_input_dir": None,
            "segmentator_input_size": None,
            "segmentation_model_name": None,
            "experiment_dir": None,
            "_skip_init": True
        }
        new = cls(**init_kwargs)
        for attr, value in vars(old_instance).items():
            setattr(new, attr, value)
        return new

class OrganoidValidationDataset(OrganoidDataset):
    """\
    Class to hold a validation dataset. This means, that
    X and y are not subset but rather returned as is.
    """

    def __init__(self,
                 base_dataset: Union[PathLike, OrganoidDataset],
                 readout: str):
        if isinstance(base_dataset, OrganoidDataset):
            dataset = base_dataset
        else:
            dataset = self.read_classification_dataset(base_dataset)

        self.X = dataset.X
        self.y = dataset.y[readout]

class OrganoidDatasetSplitter:

    def __init__(self):
        pass

    def _filter_wells(self,
                      df: pd.DataFrame,
                      combinations: np.ndarray,
                      columns: list[str]):
        combinations_df = pd.DataFrame(combinations, columns=columns)
        return df.merge(combinations_df, on=columns, how='inner')

    def _get_array_indices_from_frame(self,
                                      df: pd.DataFrame,
                                      train_wells: np.ndarray,
                                      test_wells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        train_df = self._filter_wells(df, train_wells, ["experiment", "well"])
        test_df = self._filter_wells(df, test_wells, ["experiment", "well"])

        train_idxs = train_df["IMAGE_ARRAY_INDEX"].unique()
        test_idxs = test_df["IMAGE_ARRAY_INDEX"].unique()

        assert isinstance(train_idxs, np.ndarray)
        assert isinstance(test_idxs, np.ndarray)

        return train_idxs, test_idxs

    def _annotate_wells(self,
                        metadata: pd.DataFrame,
                        wells_to_annotate: np.ndarray,
                        colname: str = "set",
                        set_value: Literal["train", "test"] = "train"):
        for experiment, well in wells_to_annotate:
            metadata.loc[
                (metadata["experiment"] == experiment) &
                (metadata["well"] == well),
                colname
            ] = set_value

        return metadata

    def _annotate_train_test_wells(self,
                                   metadata: pd.DataFrame,
                                   colname: str,
                                   train_wells: np.ndarray,
                                   test_wells: np.ndarray):
        kwargs = {
            "metadata": metadata,
            "colname": colname
        }
        metadata = self._annotate_wells(
            wells_to_annotate = train_wells,
            set_value = "train",
            **kwargs
        )
        metadata = self._annotate_wells(
            wells_to_annotate = test_wells,
            set_value = "test",
            **kwargs
        )

        return metadata


class OrganoidCrossValidationDataset(OrganoidDataset, OrganoidDatasetSplitter):
    """\
    Class to implement a cross-validation dataset with n_splits.
    Example usage:
        dataset = OrganoidCrossValidationDataset()
        for fold_number, (X_train, X_test, y_train, y_test) in enumerate(dataset):
            [...]
    """

    def __init__(self,
                 base_dataset: Union[PathLike, OrganoidDataset],
                 readout: str,
                 n_splits: int = 5):
        self.readout = readout
        if isinstance(base_dataset, OrganoidDataset):
            self.dataset = base_dataset
        else:
            self.dataset = self.read_classification_dataset(base_dataset)

        self._metadata = self.dataset.metadata
        self._calculate_k_folds(n_splits, self._metadata)

    def __iter__(self) -> 'OrganoidCrossValidationDataset':
        self.current_fold = 0  # Reset fold index for new iteration
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.current_fold >= len(self.fold_indices):
            raise StopIteration
        data = self.get_fold_data(self.current_fold)
        self.current_fold += 1
        return data

    def get_fold_data(self, fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_idxs, test_idxs = self.fold_indices[fold]
        X_train = self.dataset.X[train_idxs]
        X_test = self.dataset.X[test_idxs]
        y_train = self.dataset.y[self.readout][train_idxs]
        y_test = self.dataset.y[self.readout][test_idxs]
        return X_train, X_test, y_train, y_test

    def _calculate_k_folds(self,
                           n_splits: int,
                           metadata: pd.DataFrame):
        self.fold_indices = {
            i: (np.array([]), np.array([]))
            for i in range(n_splits)
        }
        skf = KFold(n_splits = n_splits, shuffle = True, random_state = 187)
        unique_well_per_experiment = self._get_unique_experiment_well_combo(metadata, "experiment", "well")
        for i, (train_indices, test_indices) in enumerate(skf.split(unique_well_per_experiment)):
            train_wells = unique_well_per_experiment[train_indices]
            test_wells = unique_well_per_experiment[test_indices]
            self._metadata = self._annotate_train_test_wells(
                self._metadata,
                colname = f"fold{i}",
                train_wells = train_wells,
                test_wells = test_wells,
            )
            self.fold_indices[i] = (self._get_array_indices_from_frame(metadata, train_wells, test_wells))

        return


class OrganoidTrainingDataset(OrganoidDataset, OrganoidDatasetSplitter):

    def __init__(self,
                 base_dataset: Union[PathLike, OrganoidDataset],
                 readout: str,
                 test_size: float = 0.1):
        if isinstance(base_dataset, OrganoidDataset):
            self.dataset = base_dataset
        else:
            self.dataset = self.read_classification_dataset(base_dataset)
        self.readout = readout
        self._metadata = self.dataset.metadata.copy()
        self.train_idxs, self.test_idxs = self._calculate_train_test_split(
            test_size,
            self._metadata
        )

    @property
    def X_train(self):
        return self.dataset.X[self.train_idxs]

    @property
    def X_test(self):
        return self.dataset.X[self.test_idxs]

    @property
    def y_train(self):
        return self.dataset.y[self.readout][self.train_idxs]

    @property
    def y_test(self):
        return self.dataset.y[self.readout][self.test_idxs]

    @property
    def arrays(self):
        return self.X_train, self.X_test, self.y_train, self.y_test


    def _calculate_train_test_split(self,
                                    test_size: float,
                                    metadata: pd.DataFrame):
        unique_well_per_experiment = self._get_unique_experiment_well_combo(metadata, "experiment", "well")
        train_wells, test_wells = train_test_split(
            unique_well_per_experiment,
            test_size = test_size,
            random_state = 187
        )
        assert isinstance(train_wells, np.ndarray)
        assert isinstance(test_wells, np.ndarray)

        self._metadata = self._annotate_train_test_wells(
            self._metadata,
            colname = "set",
            train_wells = train_wells,
            test_wells = test_wells,
        )

        return self._get_array_indices_from_frame(metadata, train_wells, test_wells)
