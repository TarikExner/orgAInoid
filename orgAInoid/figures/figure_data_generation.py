import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.decomposition import PCA

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from sklearn.manifold import TSNE
from umap import UMAP

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from typing import Optional, Literal, NoReturn, Union, Sequence

from ..classification._dataset import OrganoidDataset, OrganoidTrainingDataset
from ..classification.models import DenseNet121, ResNet50, MobileNetV3_Large

from ..classification._utils import create_dataloader
from ..classification._evaluation import ModelWithTemperature
from ..classification._utils import (create_data_dfs,
                                     _get_data_array,
                                     _get_labels_array)

from . import figure_config as cfg


METADATA_COLUMNS = [
    'experiment', 'well', 'file_name', 'position', 'slice', 'loop',
    'Condition', 'RPE_Final', 'RPE_Norin', 'RPE_Cassian',
    'Confidence_score_RPE', 'Total_RPE_amount', 'RPE_classes', 'Lens_Final',
    'Lens_Norin', 'Lens_Cassian', 'Confidence_score_lens', 'Lens_area',
    'Lens_classes', 'label'
]

CLASS_ENTRIES = {
    "RPE_Final": ["No", "Yes"],
    "Lens_Final": ["No", "Yes"],
    "RPE_classes": [0,1,2,3],
    "Lens_classes": [0,1,2,3]
}

EVALUATORS = ["HEAT21", "HEAT22", "HEAT23", "HEAT24", "HEAT25", "HEAT27"]

PC_COLUMNS = [
    f"PC{i}" for i in range(1,21)
]

_CONTAINS_MAP = {
    "RPE_Final_Contains": "RPE_Final",
    "Lens_Final_Contains": "Lens_Final"
}
_STRING_LABEL_COLS = {"RPE_Final", "Lens_Final"}

_BINARY_MAP = {"yes": 1, "no": 0}

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
    "Lens_Final_Contains": 2
}

MODEL_NAMES = ["DenseNet121", "ResNet50", "MobileNetV3_Large"]

ModelNames = Literal["DenseNet121", "ResNet50", "MobileNetV3_Large"]

EvaluationSets = Literal["test", "val"]

Readouts = Literal["RPE_Final", "Lens_Final", "RPE_classes", "Lens_classes"]

HumanReadouts = Literal[
    "RPE_Final_Contains",
    "Lens_Final_Contains",
    "RPE_Final",
    "Lens_Final",
    "RPE_classes",
    "Lens_classes"
]

Projections = Literal["max", "sum", "all_slices", ""]
ProjectionIDs = Literal["ZMAX", "ZSUM", "ALL_SLICES", "SL3"]

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
    "RPE_Final": DecisionTreeClassifier,
    "RPE_classes": DecisionTreeClassifier,
    "Lens_Final": DecisionTreeClassifier,
    "Lens_classes": DecisionTreeClassifier,
    "morph_classes": DecisionTreeClassifier
}

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

def get_morphometrics_frame(results_dir: str,
                            suffix: Optional[str] = None) -> pd.DataFrame:
    if not suffix:
        suffix = ""

    df = pd.DataFrame()
    for i, exp in enumerate(cfg.EXPERIMENTS):
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
    return df

def get_data_columns_morphometrics(df: pd.DataFrame):
    return [col for col in df.columns if col not in METADATA_COLUMNS]

def calculate_organoid_dimensionality_reduction(df: pd.DataFrame,
                                                data_columns: list[str],
                                                dimreds: list[str] = ["UMAP", "TSNE"],
                                                use_pca: bool = False,
                                                n_pcs: int = 20,
                                                timeframe_length: int = 24,
                                                output_dir: str = "./figure_data/",
                                                output_filename: str = "morphometrics_dimred") -> pd.DataFrame:
    save_suffix = "_pca" if use_pca else "_raw"

    output_file = os.path.join(output_dir, f"{output_filename}{save_suffix}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    pc_columns = [f"PC{i}" for i in range(1, n_pcs+1)]

    for experiment in cfg.EXPERIMENTS:
        print(f">>> Calculating dimreds for experiment {experiment} using {save_suffix.strip('_')} data")
        exp_data = df.loc[df["experiment"] == experiment, data_columns].to_numpy()
        exp_data = StandardScaler().fit_transform(exp_data)
        if use_pca:
            _pca = PCA(
                n_components = n_pcs,
                random_state = 187
            ).fit_transform(exp_data)
            df.loc[df["experiment"] == experiment, pc_columns] = _pca
            dimred_input_data = _pca
        else:
            dimred_input_data = exp_data

        for dim_red in dimreds:
            if dim_red == "UMAP":
                print("... calculating UMAP")
                coords = UMAP(init = "pca", random_state = 187).fit_transform(dimred_input_data)
                df.loc[df["experiment"] == experiment, ["UMAP1", "UMAP2"]] = coords
            elif dim_red == "TSNE":
                print("... calculating TSNE")
                coords = TSNE(random_state = 187).fit_transform(dimred_input_data)
                df.loc[df["experiment"] == experiment, ["TSNE1", "TSNE2"]] = coords
            else:
                raise ValueError(f"Unknown DimRed {dim_red}")

    timepoints = [f"LO{i:03d}" for i in range(1, 145)]
    timeframe_dict = {
        tp: str((i // timeframe_length) + 1)
        for i, tp in enumerate(timepoints)
    }

    df["timeframe"] = df["loop"].map(timeframe_dict)
    df["timeframe"] = df["timeframe"].astype(str)
    df.to_csv(output_file, index = False)
    return df

def calculate_organoid_distances(df: pd.DataFrame,
                                 data_columns: list[str],
                                 use_pca: bool = False,
                                 n_pcs: int = 20,
                                 output_dir: str = "./figure_data/organoid_distances",
                                 output_filename: str = "organoid_distances") -> pd.DataFrame:
    """\
    Calculates the distances between organoids and between loops
    Note: n_pcs = 20 keeps ~93% variance of the morphometrics data
    """

    save_suffix = "_pca" if use_pca else "_raw"

    output_file = os.path.join(output_dir, f"{output_filename}{save_suffix}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    dist_dfs = []
    original_data_columns = data_columns
    for experiment in cfg.EXPERIMENTS:
        data_columns = original_data_columns

        print(f">>> Calculating distances for experiment {experiment} using {save_suffix.strip('_')} data")
        exp_data = df[df["experiment"] == experiment].copy()
        time_points = sorted(exp_data["loop"].unique())

        exp_data[data_columns] = StandardScaler().fit_transform(exp_data[data_columns])
        if use_pca:
            _pca = PCA(
                n_components = n_pcs,
                random_state = 187
            ).fit(exp_data[data_columns])
            scores = _pca.transform(exp_data[data_columns])

            pc_columns = [f"PC{i}" for i in range(1,n_pcs+1)]
            pc_frame = pd.DataFrame(data = scores,
                                    columns = pd.Index(pc_columns),
                                    index = exp_data.index)
            exp_data = pd.concat([exp_data, pc_frame], axis = 1)
            data_columns = pc_columns

        distance_data = []

        # Compute the interorganoid distances at each time point
        for time_point in time_points:
            df_time = exp_data[exp_data["loop"] == time_point].sort_values("well")
            data = df_time[data_columns].values
            subjects = df_time["well"].values
            distances = pdist(data, metric = "euclidean")
            dist_matrix = squareform(distances)
            idx_upper = np.triu_indices(len(subjects), k = 1)
            for i, j in zip(*idx_upper):
                distance_data.append({
                    "loop": time_point,
                    "distance_type": "interorganoid",
                    "distance": dist_matrix[i,j]
                })

        # Compute intertimepoint distances between consecutive time points
        for i in range(len(time_points) - 1):
            time_point_n = time_points[i]
            time_point_n1 = time_points[i+1]
            df_n = exp_data[exp_data["loop"] == time_point_n]
            df_n1 = exp_data[exp_data["loop"] == time_point_n1]

            common_subjects = np.intersect1d(df_n["well"].unique(), df_n1["well"].unique())

            if len(common_subjects) == 0:
                continue

            df_n_common = df_n[df_n["well"].isin(common_subjects)].sort_values("well")
            df_n1_common = df_n1[df_n1["well"].isin(common_subjects)].sort_values("well")

            distances = []
            for well in common_subjects:
                _distance = cdist(
                    df_n_common.loc[df_n_common["well"] == well, data_columns].to_numpy(),
                    df_n1_common.loc[df_n1_common["well"] == well, data_columns].to_numpy()
                )
                distances.append(_distance[0][0])

            avg_loop = time_point_n
            for dist in distances:
                distance_data.append({
                    'loop': avg_loop,
                    'distance_type': 'intraorganoid',
                    'distance': dist
                })

        df_distances = pd.DataFrame(distance_data)
        df_distances["loop"] = [int(loop.split("LO")[1]) for loop in df_distances["loop"].tolist()]
        df_distances["experiment"] = experiment
        df_distances['distance'] = df_distances\
            .groupby(['loop', 'distance_type', 'experiment'])['distance']\
            .transform(
                lambda x: np.clip(
                    x, x.quantile(0.025), x.quantile(0.975)
                ) if x.name[1] == "intraorganoid" else x
            )

        dist_dfs.append(df_distances)
    dist_df = pd.concat(dist_dfs, axis = 0)
    dist_df.to_csv(output_file, index = False)
    return dist_df

def compare_neighbors_by_loop(df,
                              dimred,
                              data_cols,
                              loop_col,
                              n_neighbors=30,
                              metric="euclidean",
                              user_suffix="pca"):
    """
    Compute the average Jaccard similarity of n-nearest neighbors between
    the original data space and a 2D embedding, **per loop**.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
          - columns in data_cols (high-dimensional data)
          - f"{dimred}1", f"{dimred}2" (embedding coordinates)
          - loop_col (string grouping variable)
    dimred : str
        Prefix name of the embedding, e.g. "TSNE" or "UMAP" (case-insensitive).
    data_cols : list of str
        Column names for the original high‑dimensional data.
    loop_col : str
        Column name for the loop identifier.
    n_neighbors : int, default=10
        Number of neighbors to compare.
    metric : str, default='euclidean'
        Distance metric for neighbor search.

    Returns
    -------
    loop_jaccard : pandas.Series
        Mean Jaccard index for each loop (indexed by loop).
    sample_jaccard : pandas.Series
        Jaccard index for each sample (indexed by df.index).
    """
    # Extract data
    X = df[data_cols].values
    if user_suffix != "pca":
        X = StandardScaler().fit_transform(X)
    Y = df[[f"{dimred}1", f"{dimred}2"]].values
    loops = df[loop_col]

    # Fit neighbor searchers
    nbrs_X = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric).fit(X)
    nbrs_Y = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric).fit(Y)

    idx_X = nbrs_X.kneighbors(return_distance=False)
    idx_Y = nbrs_Y.kneighbors(return_distance=False)

    # Compute Jaccard per sample
    jacc_list = []
    for i in range(len(df)):
        set_X = set(idx_X[i][1:])  # drop self
        set_Y = set(idx_Y[i][1:])
        inter = len(set_X & set_Y)
        union = len(set_X | set_Y)
        jacc_list.append(inter / union if union > 0 else np.nan)
    sample_jaccard = pd.Series(jacc_list, index=df.index, name='jaccard')

    # Group by loop and average
    loop_jaccard = sample_jaccard.groupby(loops).mean()

    return loop_jaccard, sample_jaccard

def compare_neighbors_by_experiment(df: pd.DataFrame,
                                    dimred: str,
                                    user_suffix: str,
                                    data_cols: list[str],
                                    loop_col: str = "loop",
                                    experiment_col: str = "experiment",
                                    n_neighbors: int = 30,
                                    metric: str = "euclidean",
                                    output_dir: str = "./figure_data",
                                    output_filename: str = "jaccard_neighbors") -> pd.DataFrame:
    """
    For each experiment and each loop, compute the mean Jaccard index
    between high‑D neighbors and dimred neighbors.

    Returns a DataFrame with columns:
      [experiment, loop, mean_jaccard]
    """
    save_suffix = f"_{dimred}_{user_suffix}"

    output_file = os.path.join(output_dir, f"{output_filename}{save_suffix}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    records = []
    exps = df[experiment_col].unique()
    for exp in exps:
        print(f">>> Computing neighbors vs. dimred for experiment {exp}")
        sub = df[df[experiment_col] == exp]
        # this returns a Series indexed by loop
        loop_jacc, _ = compare_neighbors_by_loop(
            sub, dimred, data_cols, loop_col,
            n_neighbors=n_neighbors,
            metric=metric, user_suffix=user_suffix
        )
        for loop_val, jacc in loop_jacc.items():
            records.append({
                experiment_col: exp,
                loop_col: loop_val,
                "mean_jaccard": jacc
            })

    result_df = pd.DataFrame.from_records(records)
    result_df["loop"] = [int(lo.split("LO")[1]) for lo in result_df["loop"]]
    result_df.to_csv(output_file, index = False)
    return result_df

def well_same_well_fraction_by_loop(df: pd.DataFrame,
                                    data_cols: list[str],
                                    well_col: str,
                                    loop_col: str,
                                    n_neighbors: int = 30,
                                    metric: str = "euclidean",
                                    user_suffix = "pca") -> pd.Series:
    """
    For one experiment subset, compute for each loop the average
    fraction of each cell’s n_neighbors (in the original data space)
    that share the same well.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns in data_cols, plus well_col and loop_col.
    data_cols : list of str
        Column names for the original high‑dimensional data.
    well_col : str
        Column name indicating well membership.
    loop_col : str
        Column name indicating loop/timepoint.
    n_neighbors : int
        How many nearest neighbors to consider (default 10).
    metric : str
        Distance metric for NearestNeighbors (default 'euclidean').

    Returns
    -------
    pd.Series
        Indexed by loop value, with the mean fraction of same‑well neighbors.
    """
    # 1) Extract high‑dimensional features and metadata
    X = df[data_cols].values
    if user_suffix != "pca":
        X = StandardScaler().fit_transform(X)
    wells = df[well_col].values
    loops = df[loop_col].values

    # 2) Find n_neighbors in the original data space
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric).fit(X)
    neighbors = nbrs.kneighbors(return_distance=False)[:, 1:]  # drop self

    # 3) Compute, for each cell, the fraction of its neighbors in the same well
    fracs = [(wells[neighbors[i]] == wells[i]).mean() for i in range(len(df))]

    # 4) Group by loop and average
    return pd.Series(fracs, index=df.index, name='obs_frac') \
             .groupby(loops) \
             .mean()

def neighbors_per_well_by_experiment(df: pd.DataFrame,
                                     dimred: str,
                                     user_suffix: str,
                                     data_cols: list[str],
                                     well_col: str = "well",
                                     loop_col: str = "loop",
                                     experiment_col: str = "experiment",
                                     n_neighbors: int = 30,
                                     metric: str = "euclidean",
                                     output_dir: str = "./figure_data",
                                     output_filename: str = "well_enrichment") -> pd.DataFrame:
    """
    For each experiment and loop, compute the same-well neighbor fraction.

    Returns
    -------
    pd.DataFrame with columns:
      [experiment, loop, obs_frac]
    """
    save_suffix = f"_{dimred}_{user_suffix}"

    output_file = os.path.join(output_dir, f"{output_filename}{save_suffix}.csv")

    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    records = []
    for exp in df[experiment_col].unique():
        print(f">>> Computing neighbors per well for experiment {exp}")
        sub = df[df[experiment_col] == exp]
        frac_by_loop = well_same_well_fraction_by_loop(
            sub, data_cols=data_cols, well_col=well_col,
            loop_col=loop_col, n_neighbors=n_neighbors,
            metric=metric, user_suffix=user_suffix
        )
        for loop_val, obs_frac in frac_by_loop.items():
            records.append({
                experiment_col: exp,
                loop_col: loop_val,
                'obs_frac': obs_frac
            })

    df = pd.DataFrame.from_records(records)
    df["loop"] = _loop_to_timepoint(df["loop"].tolist())
    df.to_csv(output_file, index = False)
    return df



def get_ground_truth_annotations(morphometrics_dir: str = "",
                                 n_timeframes: int = 12,
                                 output_dir: str = "./figure_data",
                                 output_filename: str = "human_ground_truth_annotations") -> pd.DataFrame:
    
    output_file = os.path.join(output_dir, f"{output_filename}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    morphometrics = get_morphometrics_frame(morphometrics_dir)

    morphometrics = morphometrics[morphometrics["slice"] == "SL003"]
    morphometrics["timepoint"] = _loop_to_timepoint(morphometrics["loop"].tolist())
    timeframe_dict = _build_timeframe_dict(n_timeframes)
    morphometrics["timeframe"] = morphometrics["loop"].map(timeframe_dict)
    morphometrics.to_csv(output_file, index = False)

    return morphometrics
    
def _rename_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
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

def concat_human_evaluations(results_dir: str = "",
                             output_dir: str = "./figure_data",
                             output_filename: str = "human_evaluations_concat") -> pd.DataFrame:
    output_file = os.path.join(output_dir, f"{output_filename}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file
    
    eval_dfs = []
    for evaluator in EVALUATORS:
        df = pd.read_csv(os.path.join(results_dir, f"{evaluator}_organoid_classification.csv"))
        df = _rename_annotation_columns(df)
        eval_dfs.append(df)

    human_evaluations = pd.concat(eval_dfs, axis = 0)
    human_evaluations = human_evaluations.reset_index(drop = True)
    human_evaluations.to_csv(output_file, index = False)
    return human_evaluations

def create_human_ground_truth_comparison(evaluator_results_dir: str,
                                         morphometrics_dir: str,
                                         output_dir: str = "./figure_data",
                                         output_filename: str = "human_ground_truth_comparison") -> pd.DataFrame:
    
    output_file = os.path.join(output_dir, f"{output_filename}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    human_evaluations = concat_human_evaluations(evaluator_results_dir, output_dir)
    ground_truth = get_ground_truth_annotations(morphometrics_dir,
                                                output_dir = output_dir)
    comparison = ground_truth.merge(human_evaluations, on = "file_name", how = "inner")

    comparison.to_csv(output_file, index = False)
    return comparison


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

def human_f1_per_evaluator(evaluator_results_dir: str,
                           morphometrics_dir: str,
                           average: str = "weighted",
                           output_dir: str = "./figure_data",
                           output_filename: str = "human_f1_per_evaluator") -> pd.DataFrame:
    """
    Weighted F1 scores per timeframe and evaluator for all human_eval_ columns.
    """
    output_file = os.path.join(output_dir, f"{output_filename}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    df = create_human_ground_truth_comparison(evaluator_results_dir,
                                              morphometrics_dir,
                                              output_dir = output_dir)
    f1_frame = f1_scores(df,
                         group_keys=["timeframe", "Evaluator_ID"],
                         average=average)

    f1_frame.to_csv(output_file, index = False)
    return f1_frame

def human_f1_per_experiment(evaluator_results_dir: str,
                            morphometrics_dir: str,
                            average: str = "weighted",
                            output_dir: str = "./figure_data",
                            output_filename: str = "human_f1_per_experiment") -> pd.DataFrame:
    """
    Weighted F1 scores per timeframe and experiment for all human_eval_ columns.
    """
    output_file = os.path.join(output_dir, f"{output_filename}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file
    df = create_human_ground_truth_comparison(evaluator_results_dir,
                                              morphometrics_dir,
                                              output_dir = output_dir)
    f1_frame = f1_scores(df,
                         group_keys=["timeframe", "experiment"],
                         average=average)

    f1_frame.to_csv(output_file, index = False)
    return f1_frame

def get_dataset_annotations(annotations_dir: str,
                            output_dir: str = "./figure_data",
                            output_filename: str = "dataset_annotations") -> pd.DataFrame:
    output_file = os.path.join(output_dir, f"{output_filename}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    frames = []
    for experiment in cfg.EXPERIMENTS:
        frames.append(
            pd.read_csv(os.path.join(annotations_dir, f"{experiment}_annotations.csv"), sep = ";")
        )
    annotations = pd.concat(frames, axis = 0)
    annotations = annotations.dropna()
    annotations[["experiment", "well"]] = pd.DataFrame(
        data = [
            [ID[:4], ID[4:]] for ID in annotations["ID"].tolist()
        ]
    ).to_numpy()
    annotations.to_csv(output_file, index = False)
    return annotations

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

def confusion_matrix_last_timeframe_all(df: pd.DataFrame,
                                        truth_col: str,
                                        pred_col: str,
                                        timeframe_col: str = "timeframe") -> np.ndarray:
    """
    Compute a single confusion matrix across all evaluators for the last timeframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing truth, prediction, and timeframe columns.
    truth_col : str
        Name of the ground-truth column (e.g. "RPE_Final").
    pred_col : str
        Name of the prediction column (e.g. "RPE_Final_Contains").
    timeframe_col : str, default="timeframe"
        Column indicating the timeframe (int or str).

    Returns
    -------
    np.ndarray
        The confusion matrix summed over all evaluators for the last timeframe.
    """
    sub = df.copy()
    sub[timeframe_col] = sub[timeframe_col].astype(int)
    last = sub[timeframe_col].max()
    sub = sub[sub[timeframe_col] == last]

    if truth_col in _STRING_LABEL_COLS:
        y_true = sub[truth_col].astype(str).str.lower().map(_BINARY_MAP).astype(int).to_numpy()
        y_pred = sub[pred_col].astype(str).str.lower().map(_BINARY_MAP).astype(int).to_numpy()
    else:
        y_true = sub[truth_col].astype(int).to_numpy()
        y_pred = sub[pred_col].astype(int).to_numpy()

    return confusion_matrix(y_true, y_pred)

def human_f1_RPE_visibility_conf_matrix(evaluator_results_dir: str,
                                        morphometrics_dir: str,
                                        output_dir: str = "./figure_data",
                                        output_filename: str = "RPE_vis_conf_matrix"):
    output_filename = os.path.join(output_dir, f"{output_filename}.npy")
    if os.path.isfile(output_filename):
        return np.load(output_filename)


    df = create_human_ground_truth_comparison(evaluator_results_dir,
                                              morphometrics_dir,
                                              output_dir = output_dir)

    conf_matrix = confusion_matrix_last_timeframe_all(df,
                                                      truth_col = "RPE_Final",
                                                      pred_col = "human_eval_RPE_Final_Contains")
    np.save(output_filename, conf_matrix)
    return conf_matrix

## neural net evaluation
def _preprocess_results_file(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["ExperimentID"] != "ExperimentID"].copy()
    assert isinstance(df, pd.DataFrame)
    df["ValF1"] = df["ValF1"].astype(float)
    df["TestF1"] = df["TestF1"].astype(float)
    return df

def _create_f1_weights(results: pd.DataFrame) -> pd.DataFrame:
    """calculates the max f1 score by model"""
    return results.groupby(["ValExpID", "Model"]).max(["TestF1", "ValF1"]).reset_index()

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
                       eval_set: Literal["test", "val"],
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
                                 readout: Readouts,
                                 val_loader: DataLoader,
                                 eval_set: EvaluationSets,
                                 experiment_dir: str,
                                 output_dir: str = "./figure_data",
                                 output_file_name: str = "model_ensemble") -> list[torch.nn.Module, ...]:
    output_file = os.path.join(output_dir, f"{output_file_name}.models")
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

def create_val_loader(val_dataset: OrganoidDataset,
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
                             readout: Readouts,
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
                             readout: Readouts,
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

    return f1_scores, confusion_matrices

def _get_labels(readout: Readouts) -> list[int]:
    return [0, 1] if readout in ["RPE_Final", "Lens_Final"] else [0, 1, 2, 3]

def neural_net_evaluation(val_dataset_id: str,
                          val_experiment_id: str,
                          eval_set: EvaluationSets,
                          readout: Readouts,
                          raw_data_dir: str,
                          experiment_dir: str,
                          output_dir: str,
                          proj: ProjectionIDs = "SL3",
                          weights: Optional[dict] = None) -> tuple:

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
    if eval_set == "test":
        val_dataset = _read_val_dataset_split(raw_data_dir = raw_data_dir,
                                              file_name = val_dataset_filename,
                                              readout = readout)
    else:
        val_dataset = _read_val_dataset_full(raw_data_dir = raw_data_dir,
                                             file_name = val_dataset_filename)

    val_loader = create_val_loader(val_dataset = val_dataset,
                                   readout = readout,
                                   eval_set = eval_set)
    assert -1 not in val_dataset.metadata["IMAGE_ARRAY_INDEX"]
    df = val_dataset.metadata
    # in order to keep the X shape, we set for SL003
    df = df[df["slice"] == "SL003"]
    if eval_set == "test":
        df = df[df["set"] == "test"]
    assert pd.Series(df["IMAGE_ARRAY_INDEX"]).is_monotonic_increasing
    assert isinstance(df, pd.DataFrame)

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
    truth_arr, ensemble_pred, single_predictions = ensemble_probability_averaging(
        models, val_loader, weights = weights
    )
    single_predictions = {
        model: np.hstack(single_predictions[model]) for model in single_predictions
    }

    truth_values = pd.DataFrame(
        data = np.array([np.argmax(el) for el in truth_arr]),
        columns = ["truth"],
        index = df.index
    )

    pred_values = pd.DataFrame(
        data = ensemble_pred,
        columns = ["pred"],
        index = df.index
    )

    df = pd.concat([df, truth_values, pred_values], axis = 1)

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

def _get_best_params(hyperparameter_dir: str,
                     projection: str,
                     classifier_name: str,
                     readout: str) -> dict:
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
    # if readout == "RPE_Final":
    #     if "n_jobs" not in best_params:
    #         best_params["n_jobs"] = 16
    #     return clf(**best_params)
    # else:
    #     return MultiOutputClassifier(clf(**best_params), n_jobs = 16)
    return clf(**best_params)

def _save_classifier_results(output_dir: str,
                             readout: Readouts,
                             val_experiment_id: str,
                             val_dataset_id: str,
                             eval_set: EvaluationSets,
                             proj: Projections,
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
        os.path.join(save_dir, f"{file_name}_CLFF1.csv"),
        index = False
    )

    np.save(
        os.path.join(save_dir, f"{file_name}_CLFCM.npy"),
        confusion_matrices
    )
    return

def _read_classifier_results(output_dir: str,
                             readout: Readouts,
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

    return f1_scores, confusion_matrices
    
def classifier_evaluation(val_experiment_id: str,
                          readout: Readouts,
                          eval_set: EvaluationSets,
                          morphometrics_dir: str,
                          hyperparameter_dir: str,
                          proj: Projections = "",
                          output_dir: str = "./figure_data") -> tuple[pd.DataFrame, np.ndarray]:
    val_dataset_id = val_experiment_id

    f1_scores, confusion_matrices = _read_classifier_results(
        output_dir = output_dir,
        readout = readout,
        val_experiment_id = val_experiment_id,
        val_dataset_id = val_dataset_id,
        eval_set = eval_set,
        proj = proj
    )
    if f1_scores is not None and confusion_matrices is not None:
        return f1_scores, confusion_matrices

    labels = _get_labels(readout)

    suffix = f"_{proj}" if proj else ""
    morphometrics_frame = get_morphometrics_frame(results_dir = morphometrics_dir,
                                                  suffix = suffix)
    data_columns = get_data_columns_morphometrics(morphometrics_frame)

    train_df, test_df, val_df = create_data_dfs(
        df = morphometrics_frame,
        experiment = val_experiment_id,
        readout = readout,
        data_columns = data_columns
    )
    X_train = _get_data_array(train_df, data_columns)
    y_train = _get_labels_array(train_df, readout)

    X_test = _get_data_array(test_df, data_columns)
    y_test = _get_labels_array(test_df, readout)

    X_val = _get_data_array(val_df, data_columns)
    y_val = _get_labels_array(val_df, readout)

    clf_ = BEST_CLASSIFIERS[readout]
    clf_name = clf_().__class__.__name__
    best_params = {}
    # best_params = _get_best_params(hyperparameter_dir,
    #                                readout = readout,
    #                                projection = proj,
    #                                classifier_name = clf_name)
    clf = _instantiate_classifier(clf_, readout = readout, best_params = best_params)
    clf.fit(X_train, y_train)

    result_df = val_df if eval_set == "val" else test_df
    result_df["truth"] = np.argmax(y_val, axis = 1) if eval_set == "val" else np.argmax(y_test, axis = 1)
    predictions = clf.predict(X_val if eval_set == "val" else X_test)
    result_df["pred"] = np.argmax(predictions, axis = 1)

    f1_scores = calculate_f1_scores(result_df)
    f1_scores["experiment"] = val_dataset_id
    f1_scores["classifier"] = "Morphometrics"

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
                             confusion_matrices = confusion_matrices)

    return f1_scores, confusion_matrices

def generate_classification_results(readout: Readouts,
                                    output_dir: str,
                                    proj: Projections,
                                    hyperparameter_dir: str,
                                    experiment_dir: str,
                                    morphometrics_dir: str,
                                    raw_data_dir: str):

    experiments = cfg.EXPERIMENTS

    clf_f1s = []
    cnn_f1s = []

    eval_sets: Sequence[EvaluationSets] = ["test", "val"]

    for experiment in experiments:
        # TODO: weights are the F1 scores!
        weights = None
        for eval_set in eval_sets:
            cnn_f1, cnn_cm = neural_net_evaluation(
                val_dataset_id = experiment,
                val_experiment_id = experiment,
                eval_set = eval_set,
                readout = readout,
                experiment_dir = experiment_dir,
                output_dir = output_dir,
                proj = PROJECTION_TO_PROJECTION_ID_MAP[proj],
                weights = weights,
                raw_data_dir = raw_data_dir
            )
            clf_f1, clf_cm = classifier_evaluation(
                val_experiment_id = experiment,
                readout = readout,
                eval_set = eval_set,
                morphometrics_dir = morphometrics_dir,
                hyperparameter_dir = hyperparameter_dir,
                proj = proj,
                output_dir = output_dir
            )
            cnn_f1s.append(cnn_f1)
            clf_f1s.append(clf_f1)

    return pd.concat(clf_f1s, axis = 0), pd.concat(cnn_f1s, axis = 0)
