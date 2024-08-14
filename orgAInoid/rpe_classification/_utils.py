import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import os
import torch
from ..segmentation.model import UNet


class ClassificationDataset(Dataset):
    def __init__(self, image_arr: np.ndarray, classes: np.ndarray, transforms):
        self.image_arr = image_arr
        self.classes = classes
        self.transforms = transforms
    
    def __len__(self):
        return self.image_arr.shape[0]
    
    def __getitem__(self, idx):
        image = self.image_arr[idx, :, :]
        rpe_type = torch.tensor(self.classes[idx])
        if self.transforms is not None:
            image = self.transforms(image)
        assert not torch.isnan(image).any()
        assert not torch.isinf(image).any()
        return (image, rpe_type)

def _prepare_classification_dataset(classification_id: str,
                                    file_frame: pd.DataFrame,
                                    start_timepoint: int,
                                    stop_timepoint: int,
                                    image_size: int = 256,
                                    unet_dir: Path = "../segmentation",
                                    unet_input_size: int = 256,
                                    experiment_dir: Path = "../../"):
    # We test if we already have data for this specific experiment
    classification_config = f"{classification_id}.cfg"
    if os.path.isfile(classification_config):
        print("Loading config...")
        with open(classification_config, "rb") as file:
            config = pickle.load(file)

        if unet_input_size != config["unet_input_size"]:
            raise ValueError("Unet Input Size of previously made config does not match")
        if start_timepoint != config["start_timepoint"]:
            raise ValueError("Start timepoint of previously made config does not match")
        if stop_timepoint != config["stop_timepoint"]:
            raise ValueError("Stop timepoint of previously made config does not match")
        if image_size != config["image_size"]:
            raise ValueError("Image Size of previously made config does not match")
    
        X_train = np.load(config["X_train_data"])
        y_train = np.load(config["y_train_data"])
        X_test = np.load(config["X_test_data"])
        y_test = np.load(config["y_test_data"])

    else:
        config = {}
        config["start_timepoint"] = start_timepoint
        config["stop_timepoint"] = stop_timepoint
        config["unet_input_size"] = unet_input_size
        config["image_size"] = image_size
        config["experiments"] = file_frame["experiment"].unique().tolist()
        
        model = _load_unet_model(unet_dir, unet_input_size)
        
        train_df, test_df = _split_dataframe_to_train_and_test(
            file_frame = file_frame,
            start_timepoint = start_timepoint,
            stop_timepoint = stop_timepoint,
            experiment_dir = experiment_dir
        )
        config["train_wells"] = train_df[["experiment", "well"]].to_numpy()
        config["test_wells"] = test_df[["experiment", "well"]].to_numpy()
        
        X_train, y_train = _prepare_classification_data(train_df,
                                                        model = model,
                                                        unet_input_size = unet_input_size,
                                                        output_size = image_size)
        X_test, y_test = _prepare_classification_data(test_df,
                                                      model = model,
                                                      unet_input_size = unet_input_size,
                                                      output_size = image_size)
        config["X_train_data"] = f"raw_data/{classification_id}_X_train.npy"
        config["y_train_data"] = f"raw_data/{classification_id}_y_train.npy"
        config["X_test_data"] = f"raw_data/{classification_id}_X_test.npy"
        config["y_test_data"] = f"raw_data/{classification_id}_y_test.npy"

        np.save(config["X_train_data"], X_train)
        np.save(config["y_train_data"], y_train)
        np.save(config["X_test_data"], X_test)
        np.save(config["y_test_data"], y_test)

        with open(classification_config, "wb") as file:
            pickle.dump(config, file)

    return X_train, y_train, X_test, y_test


def _filter_wells(df, combinations, columns):
    combinations_df = pd.DataFrame(combinations, columns=columns)
    return df.merge(combinations_df, on=columns, how='inner')

def _get_unique_experiment_well_combo(df: pd.DataFrame,
                                      col1: str,
                                      col2: str):
    return df[[col1, col2]].drop_duplicates().reset_index(drop=True).to_numpy()

def _split_dataframe_to_train_and_test(file_frame: pd.DataFrame,
                                       start_timepoint: int,
                                       stop_timepoint: int,
                                       experiment_dir: Path):
    file_frame["image_path"] = [
        os.path.join(experiment_dir, experiment, file_name)
        for experiment, file_name in zip(file_frame["experiment"].tolist(), file_frame["file_name"].tolist())
    ]
    timepoints = [f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}" for i in range(start_timepoint, stop_timepoint)]
    files = file_frame[file_frame["slice"].isin(["SL003"])]
    files = files.dropna()
    files = files[files["loop"].isin(timepoints)]

    unique_wells = _get_unique_experiment_well_combo(files, "experiment", "well")

    train_wells, test_wells = train_test_split(unique_wells, test_size = 0.1, random_state = 187)

    train_df = _filter_wells(files, train_wells, ["experiment", "well"])
    test_df = _filter_wells(files, test_wells, ["experiment", "well"])
    return train_df, test_df

def _prepare_classification_data(df: pd.DataFrame,
                                 model: UNet,
                                 unet_input_size: int,
                                 output_size: int,
                                 mask_threshold: float = 0.5,
                                 min_size_perc: float = 5):
    image_paths = df["image_path"].tolist()
    labels = df["RPE"].tolist()
    
    print(f"A total of {len(image_paths)} images")

    corrupted_indices = []
    images = []
    classification_labels = []
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        original_image = _read_image(image_path)
        if original_image is None:
            print("WARNING Corrupted image: {file_name}")
            corrupted_indices.append(i)
            continue
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
            corrupted_indices.append(i)
            continue
        
        images.append(masked_image)
        classification_labels.append(label)
        
        if i%100 == 0:
            if i!=0:
                print(f"{i} images done in {time.time() - start} seconds")
            start = time.time()

    if corrupted_indices:
        print(f"found {len(corrupted_indices)} corrupted indices")
    
    assert len(images) == len(classification_labels)
    image_array = np.array(images)
    labels_array = np.array(classification_labels)
    assert image_array.shape[0] == labels_array.shape[0]
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels_array)
    
    onehot_encoder = OneHotEncoder()
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    classification = onehot_encoder.fit_transform(integer_encoded).toarray()

    return image_array, classification