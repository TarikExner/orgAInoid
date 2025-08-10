import warnings
import os
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


from sklearn.manifold import TSNE
from umap import UMAP


from typing import Literal, Optional, cast

from . import figure_config as cfg

from .figure_data_utils import (
    _loop_to_timepoint,
    _build_timeframe_dict,
    check_for_file,
    rename_annotation_columns,
    add_loop_from_timeframe,
    f1_scores,
    convert_cnn_output_to_float,
    _generate_classification_results,
    _generate_classification_results_external_experiment,
    get_morphometrics_frame,

    READOUT_BASELINE_READOUT_MAP,
    _STRING_LABEL_COLS,
    _BINARY_MAP,

    Readouts,
    BaselineReadouts,
    Projections,
    ProjectionIDs
)

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

    df["timeframe"] = df["loop"].map(lambda x, m=timeframe_dict: m.get(x))
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
        exp_data = cast(pd.DataFrame, df[df["experiment"] == experiment].copy())
        exp_data: pd.DataFrame
        time_points = sorted(np.unique(exp_data["loop"].to_numpy()))

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
            df_time = cast(pd.DataFrame, exp_data[exp_data["loop"] == time_point].sort_values("well"))
            data: np.ndarray = df_time[data_columns].values
            subjects: np.ndarray = df_time["well"].values
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

    return cast(pd.DataFrame, morphometrics)
    
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
        df = rename_annotation_columns(df)
        eval_dfs.append(df)

    human_evaluations = pd.concat(eval_dfs, axis = 0)
    human_evaluations = human_evaluations.reset_index(drop = True)
    human_evaluations.to_csv(output_file, index = False)
    return human_evaluations

def create_human_ground_truth_comparison(evaluator_results_dir: str,
                                         morphometrics_dir: str,
                                         n_timeframes: int = 12,
                                         output_dir: str = "./figure_data",
                                         output_filename: str = "human_ground_truth_comparison") -> pd.DataFrame:
    
    output_file = os.path.join(output_dir, f"{output_filename}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    human_evaluations = concat_human_evaluations(evaluator_results_dir, output_dir)
    ground_truth = get_ground_truth_annotations(morphometrics_dir,
                                                output_dir = output_dir,
                                                n_timeframes = n_timeframes)
    comparison = ground_truth.merge(human_evaluations, on = "file_name", how = "inner")

    comparison.to_csv(output_file, index = False)
    return comparison


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
                            n_timeframes: int = 12,
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
                                              n_timeframes = n_timeframes,
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

def generate_classification_results(readout: Readouts,
                                    output_dir: str,
                                    proj: Projections,
                                    hyperparameter_dir: str,
                                    experiment_dir: str,
                                    morphometrics_dir: str,
                                    raw_data_dir: str,
                                    baseline: bool = False):
    return _generate_classification_results(**locals())

def generate_baseline_results(readout: BaselineReadouts,
                              output_dir: str,
                              proj: Projections,
                              hyperparameter_dir: str,
                              experiment_dir: str,
                              morphometrics_dir: str,
                              raw_data_dir: str,
                              baseline: bool = True):
    return _generate_classification_results(**locals())

def generate_classification_results_external_experiment(external_experiment_id: str,
                                                        readout: BaselineReadouts,
                                                        output_dir: str,
                                                        proj: Projections,
                                                        hyperparameter_dir: str,
                                                        experiment_dir: str,
                                                        morphometrics_dir: str,
                                                        raw_data_dir: str,
                                                        baseline: bool = False):
    return _generate_classification_results_external_experiment(**locals())

def generate_baseline_results_external_experiment(external_experiment_id: str,
                                                  readout: BaselineReadouts,
                                                  output_dir: str,
                                                  proj: Projections,
                                                  hyperparameter_dir: str,
                                                  experiment_dir: str,
                                                  morphometrics_dir: str,
                                                  raw_data_dir: str,
                                                  baseline: bool = True):
    return _generate_classification_results_external_experiment(**locals())

def get_classification_f1_data_external_experiment(external_experiment_id: str,
                                                   readout: Readouts,
                                                   output_dir: str,
                                                   proj: Projections,
                                                   hyperparameter_dir: str,
                                                   classification_dir: str,
                                                   baseline_dir: Optional[str],
                                                   morphometrics_dir: str,
                                                   raw_data_dir: str,
                                                   evaluator_results_dir: str) -> pd.DataFrame:
    classifier_f1s, _, _ = generate_classification_results_external_experiment(
        external_experiment_id = external_experiment_id,
        readout = readout,
        output_dir = output_dir,
        proj = proj,
        hyperparameter_dir = hyperparameter_dir,
        experiment_dir = classification_dir,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir
    )
    if baseline_dir is not None:
        baseline_f1s, _, _ = generate_baseline_results_external_experiment(
            external_experiment_id = external_experiment_id,
            readout = READOUT_BASELINE_READOUT_MAP[readout],
            output_dir = output_dir,
            proj = proj,
            hyperparameter_dir = hyperparameter_dir,
            experiment_dir = baseline_dir,
            morphometrics_dir = morphometrics_dir,
            raw_data_dir = raw_data_dir
        )
    else:
        baseline_f1s = pd.DataFrame()

    return pd.concat([classifier_f1s, baseline_f1s], axis = 0)

def get_classification_f1_data(readout: Readouts,
                               output_dir: str,
                               proj: Projections,
                               hyperparameter_dir: str,
                               classification_dir: str,
                               baseline_dir: Optional[str],
                               morphometrics_dir: str,
                               raw_data_dir: str,
                               evaluator_results_dir: str) -> pd.DataFrame:
    classifier_f1s, _, _ = generate_classification_results(
        readout = readout,
        output_dir = output_dir,
        proj = proj,
        hyperparameter_dir = hyperparameter_dir,
        experiment_dir = classification_dir,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir
    )
    if baseline_dir is not None:
        baseline_f1s, _, _ = generate_baseline_results(
            readout = READOUT_BASELINE_READOUT_MAP[readout],
            output_dir = output_dir,
            proj = proj,
            hyperparameter_dir = hyperparameter_dir,
            experiment_dir = baseline_dir,
            morphometrics_dir = morphometrics_dir,
            raw_data_dir = raw_data_dir
        )
    else:
        baseline_f1s = pd.DataFrame()

    human_data = human_f1_per_experiment(evaluator_results_dir = evaluator_results_dir,
                                         morphometrics_dir = morphometrics_dir,
                                         n_timeframes = 48,
                                         output_dir = output_dir)
    human_data = add_loop_from_timeframe(human_data,
                                         n_timeframes = 48)
    human_data["classifier"] = "human"
    cols_to_choose = ["experiment", "loop", "F1", "classifier"]
    human_data = human_data.rename(columns = {f"F1_{readout}": "F1"})
    human_data = human_data[cols_to_choose]

    return pd.concat([classifier_f1s, baseline_f1s, human_data], axis = 0)

def get_classifier_comparison(classifier_results_dir: str,
                              readout: Readouts,
                              proj: ProjectionIDs,
                              output_dir: str = "./figure_data",
                              output_filename: str = "classifier_benchmark") -> pd.DataFrame:

    # CLFCOMP_SLICE3
    # CLFCOMP_MAX
    # HYPERPARAM_SUM
    save_suffix = f"_{readout}_{proj}"

    output_file = os.path.join(output_dir, f"{output_filename}{save_suffix}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    if proj == "SL3":
        proj = "SLICE3"

    if "Z" in proj:
        proj = proj.split("Z")[1]

    benchmark = pd.read_csv(
        os.path.join(classifier_results_dir, f"CLFCOMP_{proj}.log"),
        index_col = False
    )
    hyperparam_tuning = pd.read_csv(
        os.path.join(classifier_results_dir, f"HYPERPARAM_{proj}.log"),
        index_col = False
    )
    benchmark["tuning"] = "not tuned"
    hyperparam_tuning["tuning"] = "tuned"
    data = pd.concat([benchmark, hyperparam_tuning], axis = 0)
    data = data[data["readout"] == readout].copy()
    data["ALGORITHM"] = [f"{algorithm}: {tuning}" for algorithm, tuning in zip(data["algorithm"].tolist(), data["tuning"].tolist())]

    data.to_csv(output_file, index = False)
    return cast(pd.DataFrame, data)

def get_cnn_output(classification_dir: str,
                   readout: Readouts,
                   proj: ProjectionIDs,
                   output_dir: str = "./figure_data",
                   output_filename: str = "cnn_output") -> pd.DataFrame:
    save_suffix = f"_{readout}_{proj}"

    output_file = os.path.join(output_dir, f"{output_filename}{save_suffix}.csv")
    existing_file = check_for_file(output_file)
    if existing_file is not None:
        return existing_file

    data = pd.read_csv(
        os.path.join(classification_dir, f"results/{readout}.txt"),
        index_col = False
    )
    data = convert_cnn_output_to_float(data)

    data.to_csv(output_file, index = False)
    return data

def load_and_aggregate_matrices(readout,
                                classifier: Literal["neural_net", "classifier"],
                                eval_set: Literal["test", "val"],
                                proj,
                                figure_data_dir: str,
                                morphometrics_dir: str) -> pd.DataFrame:
    """
    Load confusion matrices for each experiment and each loop, compute the average
    confusion matrix per loop across experiments, and return a DataFrame indexed
    by loop (string "LO###") whose column 'mean_matrix' holds the raw mean arrays.
    """
    experiments = cfg.EXPERIMENTS

    suffix = f"_{proj}" if proj else ""
    morpho = get_morphometrics_frame(morphometrics_dir, suffix=suffix)
    morpho = morpho[["experiment","well","loop"]].drop_duplicates()
    morpho = morpho.sort_values(["experiment","well","loop"], ascending=[True,True,True])

    # hacky, but it's late...
    if not proj:
        proj = "SL3"

    conf_dir = os.path.join(figure_data_dir, f"classification_{readout}")
    data: dict[str, list[np.ndarray]] = {}
    for exp in cfg.EXPERIMENTS:
        clf_tag = "CNN" if classifier == "neural_net" else "CLF"
        fname = f"{exp}_{exp}_{eval_set}_{proj}_{clf_tag}CM.npy"
        mats = np.load(os.path.join(conf_dir, fname))
        if eval_set == "test":
            loops = morpho.loc[morpho["experiment"] != exp, "loop"].unique().tolist()
        else:
            loops = morpho.loc[morpho["experiment"] == exp, "loop"].unique().tolist()
        assert len(loops) == mats.shape[0], f"{len(loops)} loops, {mats.shape[0]} mats"
        for i, lo in enumerate(loops):
            data.setdefault(lo, []).append(mats[i])

    mean_dict = {lo: np.mean(ms, axis=0) for lo, ms in data.items()}
    df = pd.DataFrame({
        'loop': list(mean_dict.keys()),
        'mean_matrix': list(mean_dict.values())
    }).set_index('loop')
    df.index.name = 'loop'
    return df

def classwise_confusion_matrix(matrix):
    """
    Compute 2x2 numpy arrays for all classes from a 4x4 confusion matrix.

    Args:
        matrix (ndarray): A 4x4 percentage confusion matrix.

    Returns:
        tuple: A tuple of 4 numpy arrays, each representing a 2x2 confusion matrix for a class.
    """
    n_classes = matrix.shape[0]
    results = []

    for focus_class in range(n_classes):
        tp = matrix[focus_class, focus_class]
        fn = np.sum(matrix[focus_class, :]) - tp
        fp = np.sum(matrix[:, focus_class]) - tp
        tn = np.sum(matrix) - tp - fn - fp
        results.append(np.array([[tn, fp], [fn, tp]]))
    return tuple(results)

def convert_to_percentages(df: pd.DataFrame,
                           matrix_col: str = 'mean_matrix') -> pd.DataFrame:
    """
    Given a DataFrame with an array column of confusion matrices, compute percentage
    normalization per matrix and store in 'percentage_matrix'.
    """
    df['percentage_matrix'] = df[matrix_col].apply(lambda m: (m / m.sum()) * 100)
    return df

def flatten_for_plotting(df: pd.DataFrame,
                         classes: int) -> pd.DataFrame:
    """
    Flatten percentage_matrix for plotting.
    - For 2 classes: return DataFrame with columns ['tn','tp','fn','fp'].
    - For N>2 classes: return dict mapping 'class0'..'class{N-1}' to DataFrames
      each with ['tn','tp','fn','fp'].
    Index is loop divided by 2 (float).
    """
    """
    Flatten percentage_matrix for plotting.
    - For 2 classes: return DataFrame with columns ['tn','tp','fn','fp'].
    - For N>2 classes: return dict mapping 'class0'..'class{N-1}' to DataFrames
      each with ['tn','tp','fn','fp'].
    Index is loop divided by 2 (float).
    """
    # dynamic class count from matrix shape
    sample_mat = df['percentage_matrix'].iloc[0]
    n_classes = sample_mat.shape[0]
    # compute numeric loop index
    loops = df.index.to_series().str.replace('LO','').astype(int) / 2

    if n_classes == 2:
        labels = ['tn','fp','fn','tp']
        records: list[pd.DataFrame] = []
        for lo, pct in df['percentage_matrix'].items():
            values = pct.reshape(4).astype(float)
            rec = pd.DataFrame({
                'loop': loops.loc[lo],
                'component': labels,
                'value': values
            })
            records.append(rec)
        long = pd.concat(records, ignore_index=True)
        plot_df = long.pivot(index='loop', columns='component', values='value')
        # order: tn, tp, fn, fp
        return plot_df[['tn','tp','fn','fp']]
    else:
        # multiclass: build per-class 2x2 frames
        cmaps = df['percentage_matrix'].apply(classwise_confusion_matrix)
        out: dict[str, list[pd.DataFrame]] = {f'class{i}': [] for i in range(n_classes)}
        for lo, tups in cmaps.items():
            for i, cm2 in enumerate(tups):
                values = cm2.reshape(4).astype(float)
                labels = ['tn','fp','fn','tp']
                rec = pd.DataFrame({
                    'loop': loops.loc[lo],
                    'component': labels,
                    'value': values
                })
                out[f'class{i}'].append(rec)
        result: dict[str, pd.DataFrame] = {}
        for _cls, recs in out.items():
            long = pd.concat(recs, ignore_index=True)
            df_cls = long.pivot(index='loop', columns='component', values='value')
            result[_cls] = df_cls[['tn','tp','fn','fp']]
        return result

def create_confusion_matrix_frame(readout: Readouts,
                                  classifier: Literal["neural_net", "classifier"],
                                  eval_set: Literal["test", "val"],
                                  proj: Projections, # = ""!
                                  figure_data_dir: str,
                                  morphometrics_dir: str) -> pd.DataFrame:
    """
    High-level function returning a DataFrame ready for plotting:
    - Binary readouts (RPE_classification, Lens_classification): 2 classes.
    - Multi-class readouts (RPE_classes_classification, Lens_classes_classification): 4 classes.
    Columns reflect confusion matrix components, rows are loop hours.
    """
    output_dir = os.path.join(figure_data_dir, f"classification_{readout}")
    output_filename = os.path.join(output_dir, f"conf_matrices_{classifier}_{eval_set}.data")

    if os.path.isfile(output_filename):
        return pd.read_pickle(output_filename)

    base_df = load_and_aggregate_matrices(
        readout = readout,
        classifier = classifier,
        eval_set = eval_set,
        proj = proj,
        figure_data_dir = figure_data_dir,
        morphometrics_dir = morphometrics_dir
    )
    pct_df = convert_to_percentages(base_df)
    _cls = 2 if 'classification' in readout and 'classes' not in readout else 4
    plot_df = flatten_for_plotting(pct_df, classes=_cls)
    plot_df.to_pickle(output_filename)
    return plot_df


def run_f1_statistics(df: pd.DataFrame,
                      readout: Readouts,
                      p_adjust: str = "bonferroni") -> pd.DataFrame:

    from scipy.stats import shapiro, f_oneway, ttest_ind

    group_col = "classifier"
    value_col = "F1"
    loop_col = "loop"
    records = []

    for loop in sorted(df[loop_col].unique()):
        sub = df[df[loop_col] == loop]
        

        # 1) Normality check
        for clf, grp in sub.groupby(group_col):
            W, p_sw = shapiro(grp[value_col])
            if p_sw < 0.05:
                warnings.warn(
                    f'Loop {loop}, classifier "{clf}" fails normality (Shapiro p={p_sw:.3f})',
                    UserWarning
                )
        
        # 2) ANOVA
        groups = [grp[value_col].values
                  for _, grp in sub.groupby(group_col)]
        F, p_anova = f_oneway(*groups)
        records.append({
            'loop':      loop,
            'test':      'anova',
            'group1':    None,
            'group2':    None,
            'statistic': F,
            'p_raw':     p_anova,
            'p_adj':     p_anova
        })
        
        # pairwise ttest
        classes = sorted(sub[group_col].unique())
        pair_results = []
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                g1, g2 = classes[i], classes[j]
                a = sub.loc[sub[group_col]==g1, value_col].values
                b = sub.loc[sub[group_col]==g2, value_col].values
                t_stat, p_val = ttest_ind(a, b, equal_var=True)
                pair_results.append((g1, g2, t_stat, p_val))

        m = len(pair_results)
        raw_ps = [pr[3] for pr in pair_results]
        if p_adjust == 'bonferroni':
            adj_ps = [min(p*m, 1.0) for p in raw_ps]
        elif p_adjust == 'holm':
            order = np.argsort(raw_ps)
            adj = np.empty(m)
            for rank, idx in enumerate(order):
                adj[idx] = min((m-rank)*raw_ps[idx], 1.0)
            adj_ps = list(adj)
        else:
            raise ValueError(f'Unknown p_adjust: {p_adjust}')

        for (g1, g2, t_stat, p_val), p_a in zip(pair_results, adj_ps):
            records.append({
                'loop':      loop,
                'test':      't-test',
                'group1':    g1,
                'group2':    g2,
                'statistic': t_stat,
                'p_raw':     p_val,
                'p_adj':     p_a
            })

    res = pd.DataFrame.from_records(records)
    res["readout"] = readout
    return res
