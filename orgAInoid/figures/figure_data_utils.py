import os
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from typing import Optional, Literal, Union, NoReturn, Sequence, cast

from . import figure_config as cfg

from ..classification.models import DenseNet121, ResNet50, MobileNetV3_Large
from ..classification._dataset import OrganoidDataset, OrganoidTrainingDataset
from ..classification._utils import (create_dataloader,
                                     _get_data_array,
                                     _get_labels_array,
                                     create_data_dfs)
from ..classification._evaluation import ModelWithTemperature

Readouts = Literal[
    "RPE_Final",
    "Lens_Final",
    "RPE_classes",
    "Lens_classes",
    "morph_classes"
]
BaselineReadouts = Literal[
    "Baseline_RPE_Final",
    "Baseline_Lens_Final",
    "Baseline_RPE_classes",
    "Baseline_Lens_classes"
]

EvaluationSets = Literal["test", "val"]

ModelNames = Literal["DenseNet121", "ResNet50", "MobileNetV3_Large"]

HumanReadouts = Literal[
    "RPE_Final_Contains",
    "Lens_Final_Contains",
    "RPE_Final",
    "Lens_Final",
    "RPE_classes",
    "Lens_classes"
]

Projections = Literal["max", "sum", "all_slices", ""]
ProjectionIDs = Literal["ZMAX", "ZSUM", "ALL_SLICES", "SL3", "SLICE3"]

METADATA_COLUMNS = [
    'experiment', 'well', 'file_name', 'position', 'slice', 'loop',
    'Condition', 'RPE_Final', 'RPE_Norin', 'RPE_Cassian',
    'Confidence_score_RPE', 'Total_RPE_amount', 'RPE_classes', 'Lens_Final',
    'Lens_Norin', 'Lens_Cassian', 'Confidence_score_lens', 'Lens_area',
    'Lens_classes', 'label', 'morph_classes'
]

_DEFAULT_HUMAN_COLS = [
    "human_eval_RPE_Final_Contains",
    "human_eval_Lens_Final_Contains",
    "human_eval_RPE_Final",
    "human_eval_Lens_Final",
    "human_eval_RPE_classes",
    "human_eval_Lens_classes"
]

N_CLASSES_DICT: dict[str, int] = {
    "RPE_Final": 2,
    "Lens_Final": 2,
    "RPE_classes": 4,
    "Lens_classes": 4,
    "RPE_Final_Contains": 2,
    "Lens_Final_Contains": 2,
    "morph_classes": 4
}

MODEL_NAMES = ["DenseNet121", "ResNet50", "MobileNetV3_Large"]

READOUT_BASELINE_READOUT_MAP: dict[Readouts, BaselineReadouts] = {
    "RPE_Final": "Baseline_RPE_Final",
    "Lens_Final": "Baseline_Lens_Final",
    "RPE_classes": "Baseline_RPE_classes",
    "Lens_classes": "Baseline_Lens_classes"
}

BASELINE_READOUT_TO_READOUT_MAP: dict[BaselineReadouts, Readouts] = {
    "Baseline_RPE_Final": "RPE_Final",
    "Baseline_Lens_Final": "Lens_Final",
    "Baseline_RPE_classes": "RPE_classes",
    "Baseline_Lens_classes": "Lens_classes"
}

PROJECTION_SAVE_MAP: dict[Union[Projections, ProjectionIDs], Union[Projections, ProjectionIDs]] = {
    "SL3": "SL3",
    "ZMAX": "max",
    "ZSUM": "sum",
    "ALL_SLICES": "all_slices",
    "": "SL3",
    "max": "max",
    "sum": "sum",
    "all_slices": "all_slices"
}

PROJECTION_TO_PROJECTION_ID_MAP: dict[Projections, ProjectionIDs] = {
    "max": "ZMAX",
    "sum": "ZSUM",
    "all_slices": "ALL_SLICES",
    "": "SL3"
}

BEST_CLASSIFIERS = {
    "RPE_Final": RandomForestClassifier,
    "RPE_classes": HistGradientBoostingClassifier,
    "Lens_Final": QuadraticDiscriminantAnalysis,
    "Lens_classes": QuadraticDiscriminantAnalysis,
    "morph_classes": DecisionTreeClassifier
}

_CONTAINS_MAP = {
    "RPE_Final_Contains": "RPE_Final",
    "Lens_Final_Contains": "Lens_Final"
}
_STRING_LABEL_COLS = {"RPE_Final", "Lens_Final"}

_BINARY_MAP = {"yes": 1, "no": 0}

def get_morphometrics_frame(results_dir: str,
                            external_experiment_id: Optional[str] = None,
                            suffix: Optional[str] = None) -> pd.DataFrame:
    if not suffix:
        suffix = ""

    if not external_experiment_id:
        external_experiment_id = []
    else:
        external_experiment_id = [external_experiment_id]

    df = pd.DataFrame()
    for i, exp in enumerate(cfg.EXPERIMENTS+external_experiment_id):
        input_dir = os.path.join(results_dir, f"{exp}_morphometrics{suffix}.csv")
        data = pd.read_csv(input_dir)
        if i == 0:
            df = data
        else:
            df = pd.concat([df, data], axis = 0)

    df = df.drop(["label"], axis = 1)

    # we remove columns and rows that only contain NA
    df = df.dropna(how = "all", axis = 1)
    
    data_columns = [col for col in df.columns if col not in METADATA_COLUMNS]  
    df = df.dropna(how = "all", axis = 0, subset = data_columns)
    assert isinstance(df, pd.DataFrame)
    # we have to read in the morph_classes differently:
    morph_classes = pd.read_csv(
        os.path.join(results_dir, f"morph_classes{suffix}.csv"),
        index_col = False
    )
    df = df.merge(
        morph_classes[["experiment", "well", "morph_classes"]],
        on = ["experiment", "well"],
        how = "left"
    )
    return df

def _loop_to_timepoint(loops: list[str]) -> list[int]:
    return [int(lo.split("LO")[1]) for lo in loops]

def _build_timeframe_dict(n_timeframes: int, total: int = 144) -> dict[str, int]:
    chunk = total // n_timeframes
    return {
        f"LO{i:03d}": min((i - 1) // chunk + 1, n_timeframes)
        for i in range(1, total + 1)
    }

def check_for_file(output_file: str) -> Optional[pd.DataFrame]:
    if os.path.isfile(output_file):
        return pd.read_csv(output_file, index_col = False)
    return

def get_data_columns_morphometrics(df: pd.DataFrame):
    return [col for col in df.columns if col not in METADATA_COLUMNS]

def rename_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    eval_id = df["Evaluator_ID"].unique()[0]
    eval_id = "human_eval"
    df = df.rename(
        columns = {
            "FileName": "file_name",
            "ContainsRPE": f"{eval_id}_RPE_Final_Contains",
            "WillDevelopRPE": f"{eval_id}_RPE_Final",
            "RPESize": f"{eval_id}_RPE_classes",
            "ContainsLens": f"{eval_id}_Lens_Final_Contains",
            "WillDevelopLens": f"{eval_id}_Lens_Final",
            "LensSize": f"{eval_id}_Lens_classes"
        }
    )
    return df

def f1_scores(df: pd.DataFrame,
              group_keys: list[str],
              average: str = "weighted") -> pd.DataFrame:
    """
    Compute weighted F1 scores for each human prediction column against its ground truth counterpart,
    grouped by specified keys. Ensures each class appears at least once for stable scoring.

    Returns one F1 column per prediction.
    """
    n_classes_dict = N_CLASSES_DICT
    pred_cols = _DEFAULT_HUMAN_COLS

    records = []
    for keys, grp in df.groupby(group_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        for pred in pred_cols:
            # resolve target gt col
            short = pred.replace('human_eval_', '')
            gt_col = _CONTAINS_MAP.get(short, short)
            # extract true/pred labels
            if gt_col in _STRING_LABEL_COLS:
                y_true = grp[gt_col].astype(str).str.lower().map(_BINARY_MAP).astype(int).to_numpy()
                y_pred = grp[pred].astype(str).str.lower().map(_BINARY_MAP).astype(int).to_numpy()
            else:
                y_true = grp[gt_col].astype(int).to_numpy()
                y_pred = grp[pred].astype(int).to_numpy()
            # pad missing classes
            n = n_classes_dict[gt_col]
            present = set(np.unique(y_true)) | set(np.unique(y_pred))
            missing = set(range(n)) - present
            if missing:
                pad = np.array(list(missing), dtype=int)
                y_true = np.concatenate([y_true, pad])
                y_pred = np.concatenate([y_pred, pad])
            # compute f1
            row[f"F1_{short}"] = f1_score(y_true, y_pred, average=average)
        records.append(row)
    return pd.DataFrame.from_records(records)

def add_loop_from_timeframe(df: pd.DataFrame,
                            n_timeframes: int = 12,
                            total_loops: int = 144,
                            timeframe_col: str = "timeframe",
                            loop_col: str = "loop") -> pd.DataFrame:
    """Calculates backwards and adds the maximum loop number for every given timeframe"""
    chunk = total_loops // n_timeframes
    max_loops = {
        tf: min(tf * chunk, total_loops)
        for tf in range(1, n_timeframes + 1)
    }
    df = df.copy()
    df[loop_col] = df[timeframe_col].map(max_loops)
    return df

## neural net evaluation
def _preprocess_results_file(df: pd.DataFrame) -> pd.DataFrame:
    df = cast(pd.DataFrame, df[df["ExperimentID"] != "ExperimentID"].copy())
    assert isinstance(df, pd.DataFrame)
    df["ValF1"] = df["ValF1"].astype(float)
    df["TestF1"] = df["TestF1"].astype(float)
    return df

def _create_f1_weights(results: pd.DataFrame) -> pd.DataFrame:
    """calculates the max f1 score by model"""
    return cast(
        pd.DataFrame,
        results.groupby(["ValExpID", "Model"]).max(["TestF1", "ValF1"]).reset_index()
    )

def read_classification_results(results_dir,
                                readout: Readouts) -> pd.DataFrame:
    scores = pd.read_csv(os.path.join(results_dir, f"{readout}.txt"))
    scores = _preprocess_results_file(scores)
    return scores

def _get_model(model_name: str,
               num_classes: int) -> torch.nn.Module:
    if model_name == "DenseNet121":
        return DenseNet121(num_classes = num_classes, dropout = 0.2)
    elif model_name == "ResNet50":
        return ResNet50(num_classes = num_classes, dropout = 0.2)
    elif model_name == "MobileNetV3_Large":
        return MobileNetV3_Large(num_classes = num_classes, dropout = 0.5)
    else:
        raise ValueError(f"Unknown Model {model_name}")

def _instantiate_model(model_name: str,
                       eval_set: EvaluationSets,
                       val_experiment_id: str,
                       readout: Readouts,
                       classifier_dir: str):
    num_classes = N_CLASSES_DICT[readout]
    model = _get_model(model_name, num_classes)
    model_name = model.__class__.__name__

    model_file_name = f"{model_name}_{eval_set}_f1_{val_experiment_id}_{readout}"
    print(f"...loading model {model_file_name}")

    state_dict_path = os.path.join(
        classifier_dir, f"{model_file_name}_base_model.pth"
    )
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    model.cuda()
    return model

def model_setup_with_temperature(model_name,
                                 eval_set: EvaluationSets,
                                 val_experiment_id: str,
                                 readout: Readouts,
                                 val_loader: DataLoader,
                                 classifier_dir: str):
    model = _instantiate_model(model_name = model_name,
                               eval_set = eval_set,
                               val_experiment_id = val_experiment_id,
                               readout = readout,
                               classifier_dir = classifier_dir)
    model = ModelWithTemperature(model, eval_set, val_experiment_id)
    model.set_temperature(val_loader)
    return model

def generate_neural_net_ensemble(val_experiment_id: str,
                                 readout: Union[Readouts, BaselineReadouts],
                                 val_loader: DataLoader,
                                 eval_set: EvaluationSets,
                                 experiment_dir: str,
                                 output_dir: str = "./figure_data",
                                 output_file_name: str = "model_ensemble") -> list[torch.nn.Module]:
    save_dir = os.path.join(output_dir, f"classification_{readout}")
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f"{output_file_name}.models")

    if "Baseline_" in readout:
        readout = BASELINE_READOUT_TO_READOUT_MAP[readout]

    if os.path.isfile(output_file):
        with open(output_file, "rb") as file:
            models = pickle.load(file)
        return models

    classifier_dir = os.path.join(experiment_dir, "classifiers")

    models = []
    for model_name in MODEL_NAMES:
        _model = model_setup_with_temperature(
            model_name = model_name,
            eval_set = eval_set,
            val_experiment_id = val_experiment_id,
            readout = readout,
            val_loader = val_loader,
            classifier_dir = classifier_dir
        )
        models.append(_model)

    with open(output_file, "wb") as file:
        pickle.dump(models, file)

    return models

def ensemble_probability_averaging(models, dataloader, weights=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """\
    Weights is a dictionary where the keys are the model names
    and the values is the val_f1_score.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_labels = []
    all_preds = []
    single_predictions = {model.original_name: [] for model in models}
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            prob_sum = 0
            total_weight = 0
            
            for model in models:
                logits = model(inputs)
                preds_single = torch.argmax(logits, dim=1).detach().cpu().numpy()
                single_predictions[model.original_name].append(preds_single)
                probs = F.softmax(logits, dim=1)
                
                if weights is not None:
                    weight = weights[model.original_name]
                else:
                    # If no weights provided, use equal weighting
                    weight = 1.0
                
                # Accumulate the weighted probabilities
                prob_sum += weight * probs
                total_weight += weight
            
            # Normalize the accumulated probabilities
            probs_avg = prob_sum / total_weight
            # Determine the ensemble predictions
            preds = torch.argmax(probs_avg, dim=1)
            # Collect labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), single_predictions

def _read_val_dataset_full(raw_data_dir: str,
                           file_name: str) -> OrganoidDataset:
    validation_dataset_file = os.path.join(raw_data_dir, file_name)
    return OrganoidDataset.read_classification_dataset(validation_dataset_file)

def _read_val_dataset_split(raw_data_dir: str,
                            file_name: str,
                            readout: Readouts) -> OrganoidTrainingDataset:
    ds = _read_val_dataset_full(raw_data_dir, file_name)
    return OrganoidTrainingDataset(ds, readout = readout)

def _create_val_loader_full_dataset(val_dataset: OrganoidDataset,
                                    readout: Readouts) -> DataLoader:
    X, y = val_dataset.X, val_dataset.y[readout]
    return create_dataloader(X, y, batch_size = 128, shuffle = False, train = False)

def _create_val_loader_split(val_dataset: OrganoidTrainingDataset) -> DataLoader:
    X, y = val_dataset.X_test, val_dataset.y_test
    return create_dataloader(X, y, batch_size = 128, shuffle = False, train = False)

def create_val_loader(val_dataset: Union[OrganoidDataset, OrganoidTrainingDataset],
                      readout: Readouts,
                      eval_set: EvaluationSets) -> Union[DataLoader, NoReturn]:
    if eval_set == "test":
        return _create_val_loader_split(val_dataset)
    elif eval_set == "val":
        return _create_val_loader_full_dataset(val_dataset,
                                               readout = readout)
    else:
        raise ValueError(f"eval_set has to be one of 'test' and 'val'. Received {eval_set}")

def calculate_f1_scores(df) -> pd.DataFrame:
    loops = df.sort_values(["loop"], ascending = True)["loop"].unique().tolist()
    f1_scores = pd.DataFrame(
        df.groupby('loop').apply(
            lambda x: f1_score(x['truth'], x['pred'], average = "weighted")
        ),
        columns = ["F1"]
    ).reset_index()
    f1_scores["loop"] = _loop_to_timepoint(loops)
    return f1_scores

def _create_cnn_filename(val_experiment_id: str,
                         val_dataset_id: str,
                         eval_set: EvaluationSets,
                         proj: ProjectionIDs) -> str:
    return f"{val_dataset_id}_{val_experiment_id}_{eval_set}_{proj}"

def _save_neural_net_results(output_dir: str,
                             readout: Union[Readouts, BaselineReadouts],
                             val_experiment_id: str,
                             val_dataset_id: str,
                             eval_set: EvaluationSets,
                             proj: ProjectionIDs,
                             f1_scores: pd.DataFrame,
                             confusion_matrices: np.ndarray) -> None:
    proj = PROJECTION_SAVE_MAP[proj]
    save_dir = os.path.join(output_dir, f"classification_{readout}")
    os.makedirs(save_dir, exist_ok=True)
    file_name = _create_cnn_filename(val_experiment_id = val_experiment_id,
                                     val_dataset_id = val_dataset_id,
                                     eval_set = eval_set,
                                     proj = proj)

    f1_scores.to_csv(
        os.path.join(save_dir, f"{file_name}_CNNF1.csv"),
        index = False
    )

    np.save(
        os.path.join(save_dir, f"{file_name}_CNNCM.npy"),
        confusion_matrices
    )
    return

def _read_neural_net_results(output_dir: str,
                             readout: Union[Readouts, BaselineReadouts],
                             val_experiment_id: str,
                             val_dataset_id: str,
                             eval_set: EvaluationSets,
                             proj: ProjectionIDs) -> tuple:
    proj = PROJECTION_SAVE_MAP[proj]
    save_dir = os.path.join(output_dir, f"classification_{readout}")
    file_name = _create_cnn_filename(val_experiment_id = val_experiment_id,
                                     val_dataset_id = val_dataset_id,
                                     eval_set = eval_set,
                                     proj = proj)
    csv_file = os.path.join(save_dir, f"{file_name}_CNNF1.csv")
    if os.path.isfile(csv_file):
        f1_scores = pd.read_csv(csv_file, index_col = False)
    else:
        return None, None

    numpy_file = os.path.join(save_dir, f"{file_name}_CNNCM.npy")
    if os.path.isfile(numpy_file):
        confusion_matrices = np.load(numpy_file)
    else:
        return None, None

    if f1_scores is not None and confusion_matrices is not None:
        print(f"Reading NeuralNet results for {val_experiment_id}: {readout} ({eval_set})")

    return f1_scores, confusion_matrices

def _get_labels(readout: Union[Readouts, BaselineReadouts]) -> list[int]:
    if "Baseline_" in readout:
        readout = BASELINE_READOUT_TO_READOUT_MAP[readout]
    return [0, 1] if readout in ["RPE_Final", "Lens_Final"] else [0, 1, 2, 3]

def _get_best_params(hyperparameter_dir: str,
                     projection: str,
                     classifier_name: str,
                     readout: str) -> dict:
    if not projection:
        projection = "SLICE3"

    if "Baseline_" in readout:
        readout = readout.split("Baseline_")[1]

    projection = projection.upper()
    if readout == "morph_classes":
        readout = "RPE_classes"
    file_name = os.path.join(
            hyperparameter_dir,
            projection,
            f"best_params_{classifier_name}_{readout}.dict"
    )
    with open(file_name, "rb") as file:
        best_params_ = pickle.load(file)

    return best_params_

def _instantiate_classifier(clf,
                            readout: Readouts,
                            best_params: dict):
    if "RPE_Final" in readout:
        if "n_jobs" not in best_params:
            best_params["n_jobs"] = 16
        return clf(**best_params)
    else:
        return MultiOutputClassifier(clf(**best_params), n_jobs = 16)

def _save_classifier_results(output_dir: str,
                             readout: Union[Readouts, BaselineReadouts],
                             val_experiment_id: str,
                             val_dataset_id: str,
                             eval_set: EvaluationSets,
                             proj: Projections,
                             f1_scores: pd.DataFrame,
                             confusion_matrices: np.ndarray,
                             external_experiment_id: Optional[str] = None) -> None:
    proj = PROJECTION_SAVE_MAP[proj]
    save_dir = os.path.join(output_dir, f"classification_{readout}")
    os.makedirs(save_dir, exist_ok=True)
    file_name = _create_cnn_filename(val_experiment_id = val_experiment_id,
                                     val_dataset_id = external_experiment_id or val_dataset_id,
                                     eval_set = eval_set,
                                     proj = proj)

    f1_scores.to_csv(
        os.path.join(save_dir, f"{file_name}_CLFF1.csv"),
        index = False
    )

    np.save(
        os.path.join(save_dir, f"{file_name}_CLFCM.npy"),
        confusion_matrices
    )
    return

def _read_classifier_results(output_dir: str,
                             readout: Union[Readouts, BaselineReadouts],
                             val_experiment_id: str,
                             val_dataset_id: str,
                             eval_set: EvaluationSets,
                             proj: Projections,
                             external_experiment_id: Optional[str])-> tuple:
    proj = PROJECTION_SAVE_MAP[proj]
    save_dir = os.path.join(output_dir, f"classification_{readout}")
    file_name = _create_cnn_filename(val_experiment_id = val_experiment_id,
                                     val_dataset_id = external_experiment_id or val_dataset_id,
                                     eval_set = eval_set,
                                     proj = proj)
    csv_file = os.path.join(save_dir, f"{file_name}_CLFF1.csv")
    if os.path.isfile(csv_file):
        f1_scores = pd.read_csv(csv_file, index_col = False)
    else:
        return None, None

    numpy_file = os.path.join(save_dir, f"{file_name}_CLFCM.npy")
    if os.path.isfile(numpy_file):
        confusion_matrices = np.load(numpy_file)
    else:
        return None, None

    if f1_scores is not None and confusion_matrices is not None:
        print(f"Reading classifier results for {val_experiment_id}: {readout} ({eval_set})")

    return f1_scores, confusion_matrices
    
def _postprocess_cnn_frame(df: pd.DataFrame,
                           eval_set: EvaluationSets,
                           baseline: bool = False) -> pd.DataFrame:
    _df = df[df["classifier"] == "Ensemble"].copy()
    classifier_name = f"Ensemble_{eval_set}" if not baseline else f"Baseline_Ensemble_{eval_set}"
    _df["classifier"] = classifier_name
    assert isinstance(_df, pd.DataFrame)
    return _df

def convert_cnn_output_to_float(data: pd.DataFrame) -> pd.DataFrame:
    data = cast(pd.DataFrame, data[data["ExperimentID"] != "ExperimentID"].copy())
    float_cols = ["TrainF1", "TestF1", "ValF1", "TrainLoss", "TestLoss", "ValLoss"]
    for col in float_cols:
        data.loc[:, col] = data[col].astype(float)
    data.loc[:, "Epoch"] = data["Epoch"].astype(int)
    return data

def __neural_net_evaluation(val_dataset_id: str,
                            val_experiment_id: str,
                            eval_set: EvaluationSets,
                            readout: Union[BaselineReadouts, Readouts],
                            raw_data_dir: str,
                            experiment_dir: str,
                            output_dir: str,
                            proj: ProjectionIDs = "SL3",
                            weights: Optional[dict] = None,
                            baseline: bool = False) -> pd.DataFrame:

    f1_scores, confusion_matrices = _read_neural_net_results(
        output_dir = output_dir,
        readout = readout,
        val_experiment_id = val_experiment_id,
        val_dataset_id = val_dataset_id,
        eval_set = eval_set,
        proj = proj
    )
    if f1_scores is not None and confusion_matrices is not None:
        return f1_scores, confusion_matrices

    val_dataset_filename = (
        f"M{val_dataset_id}_full_{proj}_fixed.cds"
        if eval_set == "test"
        else f"{val_dataset_id}_full_{proj}_fixed.cds"
    )
    
    if baseline:
        original_readout = BASELINE_READOUT_TO_READOUT_MAP[readout]
    else:
        original_readout = readout

    if eval_set == "test":
        val_dataset = _read_val_dataset_split(raw_data_dir = raw_data_dir,
                                              file_name = val_dataset_filename,
                                              readout = readout if not baseline else original_readout)
    else:
        val_dataset = _read_val_dataset_full(raw_data_dir = raw_data_dir,
                                             file_name = val_dataset_filename)

    val_loader = create_val_loader(val_dataset = val_dataset,
                                   readout = readout if not baseline else original_readout,
                                   eval_set = eval_set)
    assert -1 not in val_dataset.metadata["IMAGE_ARRAY_INDEX"]
    df = val_dataset.metadata
    # in order to keep the X shape, we set for SL003
    df = df[df["slice"] == "SL003"]
    if eval_set == "test":
        df = df[df["set"] == "test"]
    assert pd.Series(df["IMAGE_ARRAY_INDEX"]).is_monotonic_increasing
    assert isinstance(df, pd.DataFrame)
    
    if hasattr(val_dataset, "dataset_metadata"):
        print("Current validation dataset: ", val_dataset.dataset_metadata.dataset_id)
    
    models = generate_neural_net_ensemble(
        val_experiment_id = val_experiment_id,
        readout = readout,
        val_loader = val_loader,
        eval_set = eval_set,
        experiment_dir = experiment_dir,
        output_dir = output_dir,
        output_file_name = f"model_ensemble_{readout}_{val_experiment_id}_{proj}"
    )
    print([model.original_name for model in models])
    truth_arr, ensemble_pred, single_predictions = ensemble_probability_averaging(
        models, val_loader, weights = weights
    )
    single_predictions = {
        model: np.hstack(single_predictions[model]) for model in single_predictions
    }

    truth_array = np.array([np.argmax(el) for el in truth_arr])

    # this happens for some reason in E002 when SUMmed...
    if truth_array.shape[0] > df.index.shape[0] or ensemble_pred.shape[0] > df.index.shape[0]:
        truth_array = truth_array[:df.index.shape[0]]
        ensemble_pred = ensemble_pred[:df.index.shape[0]]
        single_predictions = {
            model: array[:df.index.shape[0]]
            for model, array in single_predictions.items()
        }
        print("\n\nWARNING!!! ARRAY SHAPES DO NOT MATCH!!!\n\n")
    
    truth_values = pd.DataFrame(
        data = truth_array,
        columns = ["truth"],
        index = df.index
    )

    pred_values = pd.DataFrame(
        data = ensemble_pred,
        columns = ["pred"],
        index = df.index
    )

    assert truth_values.shape[0] == pred_values.shape[0]

    df = pd.concat([df, truth_values, pred_values], axis = 1)

    return df


def _neural_net_evaluation(val_dataset_id: str,
                           val_experiment_id: str,
                           eval_set: EvaluationSets,
                           readout: Union[BaselineReadouts, Readouts],
                           raw_data_dir: str,
                           experiment_dir: str,
                           output_dir: str,
                           proj: ProjectionIDs = "SL3",
                           weights: Optional[dict] = None,
                           baseline: bool = False) -> tuple:
    
    # df contains the metadata, the truth values and the pred values
    df = __neural_net_evaluation(**locals())

    f1_dfs = []

    labels = _get_labels(readout)
    conf_matrix_df = df.copy()
    confusion_matrices = conf_matrix_df.groupby("loop").apply(
        lambda group: confusion_matrix(group["truth"], group["pred"], labels = labels)
    )
    confusion_matrices = np.array(confusion_matrices.tolist())

    ensemble_f1 = calculate_f1_scores(df)
    ensemble_f1 = ensemble_f1.rename(columns = {"F1": "Ensemble"}).set_index("loop")
    f1_dfs.append(ensemble_f1)

    for model in single_predictions:
        df.loc[:, "pred"] = single_predictions[model]
        f1 = calculate_f1_scores(df)
        f1 = f1.rename(columns = {"F1": model}).set_index("loop")
        f1_dfs.append(f1)

    neural_net_f1 = pd.concat(f1_dfs, axis=1)
    neural_net_f1 = neural_net_f1.reset_index().melt(id_vars = "loop",
                                                     value_name = "F1",
                                                     var_name = "classifier")
    neural_net_f1["experiment"] = val_dataset_id

    _save_neural_net_results(output_dir = output_dir,
                             readout = readout,
                             val_dataset_id = val_dataset_id,
                             val_experiment_id = val_experiment_id,
                             proj = proj,
                             eval_set = eval_set,
                             f1_scores = neural_net_f1,
                             confusion_matrices = confusion_matrices)

    return neural_net_f1, confusion_matrices

def __classifier_evaluation(val_experiment_id: str,
                            readout: Readouts,
                            eval_set: EvaluationSets,
                            morphometrics_dir: str,
                            hyperparameter_dir: str,
                            proj: Projections = "",
                            output_dir: str = "./figure_data",
                            baseline: bool = False,
                            external_experiment_id: Optional[str] = None) -> pd.DataFrame:

    val_dataset_id = val_experiment_id

    if baseline:
        original_readout = BASELINE_READOUT_TO_READOUT_MAP[readout]
    else:
        original_readout = readout

    f1_scores, confusion_matrices = _read_classifier_results(
        output_dir = output_dir,
        readout = readout,
        val_experiment_id = val_experiment_id,
        val_dataset_id = val_dataset_id,
        eval_set = eval_set,
        proj = proj,
        external_experiment_id = external_experiment_id
    )
    if f1_scores is not None and confusion_matrices is not None:
        return f1_scores, confusion_matrices

    labels = _get_labels(readout)

    suffix = f"_{proj}" if proj else ""
    morphometrics_frame = get_morphometrics_frame(results_dir = morphometrics_dir,
                                                  external_experiment_id = external_experiment_id,
                                                  suffix = suffix)
    data_columns = get_data_columns_morphometrics(morphometrics_frame)

    train_df, test_df, val_df = create_data_dfs(
        df = morphometrics_frame,
        experiment = val_experiment_id,
        readout = readout,
        data_columns = data_columns
    )
    X_train = _get_data_array(train_df, data_columns)
    y_train = _get_labels_array(train_df, readout if not baseline else original_readout)

    if baseline:
        # baseline: we shuffle:
        np.random.seed(187)
        np.random.shuffle(y_train)

    X_test = _get_data_array(test_df, data_columns)
    y_test = _get_labels_array(test_df, readout if not baseline else original_readout)

    X_val = _get_data_array(val_df, data_columns)
    y_val = _get_labels_array(val_df, readout if not baseline else original_readout)

    clf_ = BEST_CLASSIFIERS[readout if not baseline else original_readout]
    clf_name = clf_().__class__.__name__
    best_params = {}

    best_params = _get_best_params(hyperparameter_dir,
                                   readout = readout,
                                   projection = proj,
                                   classifier_name = clf_name)
    clf = _instantiate_classifier(clf_,
                                  readout = readout if not baseline else original_readout,
                                  best_params = best_params)
    print(f"Fitting classifier {clf_name} for {readout}: {eval_set}")
    clf.fit(X_train, y_train)

    result_df = val_df if eval_set == "val" else test_df
    result_df["truth"] = np.argmax(y_val, axis = 1) if eval_set == "val" else np.argmax(y_test, axis = 1)
    predictions = clf.predict(X_val if eval_set == "val" else X_test)
    result_df["pred"] = np.argmax(predictions, axis = 1)

    return result_df


def _classifier_evaluation(val_experiment_id: str,
                           readout: Readouts,
                           eval_set: EvaluationSets,
                           morphometrics_dir: str,
                           hyperparameter_dir: str,
                           proj: Projections = "",
                           output_dir: str = "./figure_data",
                           baseline: bool = False,
                           external_experiment_id: Optional[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
    # result_df contains the metadata, truth values and pred values
    result_df = __classifier_evaluation(**locals())

    f1_scores = calculate_f1_scores(result_df)
    f1_scores["experiment"] = val_dataset_id
    f1_scores["classifier"] = f"Morphometrics_{eval_set}" if not baseline else f"Baseline_Morphometrics_{eval_set}"

    confusion_matrices = result_df.groupby("loop").apply(
        lambda group: confusion_matrix(group["truth"], group["pred"], labels = labels)
    )
    confusion_matrices = np.array(confusion_matrices.tolist())

    _save_classifier_results(output_dir = output_dir,
                             readout = readout,
                             val_dataset_id = val_dataset_id,
                             val_experiment_id = val_experiment_id,
                             proj = proj,
                             eval_set = eval_set,
                             f1_scores = f1_scores,
                             confusion_matrices = confusion_matrices,
                             external_experiment_id = external_experiment_id)

    return f1_scores, confusion_matrices

def neural_net_evaluation_baseline(val_dataset_id: str,
                                   val_experiment_id: str,
                                   eval_set: EvaluationSets,
                                   readout: BaselineReadouts,
                                   raw_data_dir: str,
                                   experiment_dir: str,
                                   output_dir: str,
                                   proj: ProjectionIDs = "SL3",
                                   weights: Optional[dict] = None,
                                   baseline: bool = True) -> tuple:
    return _neural_net_evaluation(**locals())

def neural_net_evaluation_baseline_raw_data(val_dataset_id: str,
                                            val_experiment_id: str,
                                            eval_set: EvaluationSets,
                                            readout: BaselineReadouts,
                                            raw_data_dir: str,
                                            experiment_dir: str,
                                            output_dir: str,
                                            proj: ProjectionIDs = "SL3",
                                            weights: Optional[dict] = None,
                                            baseline: bool = True) -> tuple:
    return __neural_net_evaluation(**locals())

def neural_net_evaluation(val_dataset_id: str,
                          val_experiment_id: str,
                          eval_set: EvaluationSets,
                          readout: Readouts,
                          raw_data_dir: str,
                          experiment_dir: str,
                          output_dir: str,
                          proj: ProjectionIDs = "SL3",
                          weights: Optional[dict] = None,
                          baseline: bool = False) -> tuple:
    return _neural_net_evaluation(**locals())

def neural_net_evaluation_raw_data(val_dataset_id: str,
                                   val_experiment_id: str,
                                   eval_set: EvaluationSets,
                                   readout: Readouts,
                                   raw_data_dir: str,
                                   experiment_dir: str,
                                   output_dir: str,
                                   proj: ProjectionIDs = "SL3",
                                   weights: Optional[dict] = None,
                                   baseline: bool = False) -> tuple:
    return __neural_net_evaluation(**locals())


def classifier_evaluation(val_experiment_id: str,
                          readout: Readouts,
                          eval_set: EvaluationSets,
                          morphometrics_dir: str,
                          hyperparameter_dir: str,
                          proj: Projections = "",
                          output_dir: str = "./figure_data",
                          baseline: bool = False,
                          external_experiment_id: Optional[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
    return _classifier_evaluation(**locals())

def classifier_evaluation_raw_data(val_experiment_id: str,
                                   readout: Readouts,
                                   eval_set: EvaluationSets,
                                   morphometrics_dir: str,
                                   hyperparameter_dir: str,
                                   proj: Projections = "",
                                   output_dir: str = "./figure_data",
                                   baseline: bool = False,
                                   external_experiment_id: Optional[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
    return __classifier_evaluation(**locals())

def classifier_evaluation_baseline(val_experiment_id: str,
                                   readout: BaselineReadouts,
                                   eval_set: EvaluationSets,
                                   morphometrics_dir: str,
                                   hyperparameter_dir: str,
                                   proj: Projections = "",
                                   output_dir: str = "./figure_data",
                                   baseline: bool = True,
                                   external_experiment_id: Optional[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
    return _classifier_evaluation(**locals())

def classifier_evaluation_baseline_raw_data(val_experiment_id: str,
                                            readout: BaselineReadouts,
                                            eval_set: EvaluationSets,
                                            morphometrics_dir: str,
                                            hyperparameter_dir: str,
                                            proj: Projections = "",
                                            output_dir: str = "./figure_data",
                                            baseline: bool = True,
                                            external_experiment_id: Optional[str] = None) -> tuple[pd.DataFrame, np.ndarray]:
    return __classifier_evaluation(**locals())

def calculate_f1_weights(classification_dir: str,
                         readout: Readouts,
                         proj: Projections,
                         experiment: str,
                         output_dir: str,
                         eval_set: Literal["test", "val"]) -> pd.DataFrame:
    data = pd.read_csv(
        os.path.join(classification_dir, f"results/{readout}.txt"),
        index_col = False
    )
    data = convert_cnn_output_to_float(data)
    data = data[data["ValExpID"] == experiment]
    readout_score = "ValF1" if eval_set == "val" else "TestF1"
    raw_scores = data.groupby(["Model", "ValExpID"]).max()[readout_score].reset_index()
    res = {}
    for model in raw_scores["Model"]:
        res[f"{model}_test_{experiment}"] = raw_scores.loc[raw_scores["Model"] == model, readout_score].iloc[0]
    
    return res

def _generate_classification_results_external_experiment(external_experiment_id: str,
                                                         readout: Union[Readouts, BaselineReadouts],
                                                         output_dir: str,
                                                         proj: Projections,
                                                         hyperparameter_dir: str,
                                                         experiment_dir: str,
                                                         morphometrics_dir: str,
                                                         raw_data_dir: str,
                                                         baseline: bool = False):

    experiments = cfg.EXPERIMENTS

    clf_f1s = []
    cnn_f1s = []
    clf_cms = []
    cnn_cms = []

    if baseline:
        nn_eval_func = neural_net_evaluation_baseline
        clf_eval_func = classifier_evaluation_baseline
    else:
        nn_eval_func = neural_net_evaluation
        clf_eval_func = classifier_evaluation

    for experiment in experiments:
        for eval_set in ["val"]:
            if not baseline:
                weights = calculate_f1_weights(classification_dir = experiment_dir,
                                               readout = readout,
                                               experiment = experiment,
                                               proj = proj,
                                               eval_set = eval_set,
                                               output_dir = output_dir)
            else:
                weights = None

            cnn_f1, cnn_cm = nn_eval_func(
                val_dataset_id = external_experiment_id,
                val_experiment_id = experiment,
                eval_set = eval_set,
                readout = readout,
                experiment_dir = experiment_dir,
                output_dir = output_dir,
                proj = PROJECTION_TO_PROJECTION_ID_MAP[proj],
                weights = weights,
                raw_data_dir = raw_data_dir,
            )
            cnn_f1 = _postprocess_cnn_frame(cnn_f1, eval_set, baseline = baseline)
            cnn_f1s.append(cnn_f1)
            cnn_cms.append(cnn_cm)

            clf_f1, clf_cm = clf_eval_func(
                val_experiment_id = experiment,
                readout = readout,
                eval_set = eval_set,
                morphometrics_dir = morphometrics_dir,
                hyperparameter_dir = hyperparameter_dir,
                proj = proj,
                output_dir = output_dir,
                external_experiment_id = external_experiment_id
            )
            clf_f1s.append(clf_f1)
            clf_cms.append(clf_cm)

    f1_scores = pd.concat([*clf_f1s, *cnn_f1s], axis = 0)

    return f1_scores, clf_cms, cnn_cms


def _generate_classification_results(readout: Union[Readouts, BaselineReadouts],
                                     output_dir: str,
                                     proj: Projections,
                                     hyperparameter_dir: str,
                                     experiment_dir: str,
                                     morphometrics_dir: str,
                                     raw_data_dir: str,
                                     baseline: bool = False):
    experiments = cfg.EXPERIMENTS

    clf_f1s = []
    cnn_f1s = []
    clf_cms = []
    cnn_cms = []
    eval_sets: Sequence[EvaluationSets] = ["test", "val"]

    if baseline:
        nn_eval_func = neural_net_evaluation_baseline
        clf_eval_func = classifier_evaluation_baseline
    else:
        nn_eval_func = neural_net_evaluation
        clf_eval_func = classifier_evaluation

    for experiment in experiments:
        for eval_set in eval_sets:
            if not baseline:
                weights = calculate_f1_weights(classification_dir = experiment_dir,
                                               readout = readout,
                                               experiment = experiment,
                                               proj = proj,
                                               eval_set = eval_set,
                                               output_dir = output_dir)
            else:
                weights = None

            cnn_f1, cnn_cm = nn_eval_func(
                val_dataset_id = experiment,
                val_experiment_id = experiment,
                eval_set = eval_set,
                readout = readout,
                experiment_dir = experiment_dir,
                output_dir = output_dir,
                proj = PROJECTION_TO_PROJECTION_ID_MAP[proj],
                weights = weights,
                raw_data_dir = raw_data_dir,
            )
            cnn_f1 = _postprocess_cnn_frame(cnn_f1, eval_set, baseline = baseline)
            cnn_f1s.append(cnn_f1)
            cnn_cms.append(cnn_cm)

            clf_f1, clf_cm = clf_eval_func(
                val_experiment_id = experiment,
                readout = readout,
                eval_set = eval_set,
                morphometrics_dir = morphometrics_dir,
                hyperparameter_dir = hyperparameter_dir,
                proj = proj,
                output_dir = output_dir
            )
            clf_f1s.append(clf_f1)
            clf_cms.append(clf_cm)

    f1_scores = pd.concat([*clf_f1s, *cnn_f1s], axis = 0)

    return f1_scores, clf_cms, cnn_cms

def _generate_classification_results_raw_data(readout: Union[Readouts, BaselineReadouts],
                                              output_dir: str,
                                              proj: Projections,
                                              hyperparameter_dir: str,
                                              experiment_dir: str,
                                              morphometrics_dir: str,
                                              raw_data_dir: str,
                                              baseline: bool = False):
    experiments = cfg.EXPERIMENTS

    clf_f1s = []
    cnn_f1s = []
    eval_sets: Sequence[EvaluationSets] = ["test", "val"]

    if baseline:
        nn_eval_func = neural_net_evaluation_baseline_raw_data
        clf_eval_func = classifier_evaluation_baseline_raw_data
    else:
        nn_eval_func = neural_net_evaluation_raw_data
        clf_eval_func = classifier_evaluation_raw_data

    for experiment in experiments:
        for eval_set in eval_sets:
            if not baseline:
                weights = calculate_f1_weights(classification_dir = experiment_dir,
                                               readout = readout,
                                               experiment = experiment,
                                               proj = proj,
                                               eval_set = eval_set,
                                               output_dir = output_dir)
            else:
                weights = None

            cnn_f1 = nn_eval_func(
                val_dataset_id = experiment,
                val_experiment_id = experiment,
                eval_set = eval_set,
                readout = readout,
                experiment_dir = experiment_dir,
                output_dir = output_dir,
                proj = PROJECTION_TO_PROJECTION_ID_MAP[proj],
                weights = weights,
                raw_data_dir = raw_data_dir,
            )
            cnn_f1s.append(cnn_f1)

            clf_f1 = clf_eval_func(
                val_experiment_id = experiment,
                readout = readout,
                eval_set = eval_set,
                morphometrics_dir = morphometrics_dir,
                hyperparameter_dir = hyperparameter_dir,
                proj = proj,
                output_dir = output_dir
            )
            clf_f1s.append(clf_f1)

    cnn_f1s_df = pd.concat([*cnn_f1s], axis = 0)
    clf_f1s_df = pd.concat([*clf_f1s], axis = 0)

    return cnn_f1s_df, clf_f1s_df
