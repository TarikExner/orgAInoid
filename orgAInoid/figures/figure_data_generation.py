import os
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from typing import Optional, Literal

from . import figure_config as cfg


from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.decomposition import PCA

METADATA_COLUMNS = [
    'experiment', 'well', 'file_name', 'position', 'slice', 'loop',
    'Condition', 'RPE_Final', 'RPE_Norin', 'RPE_Cassian',
    'Confidence_score_RPE', 'Total_RPE_amount', 'RPE_classes', 'Lens_Final',
    'Lens_Norin', 'Lens_Cassian', 'Confidence_score_lens', 'Lens_area',
    'Lens_classes', 'label'
]

def get_morphometrics_frame(results_dir: str,
                            suffix: Optional[Literal["max", "sum", "all_slices"]] = None) -> pd.DataFrame:
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
    if os.path.isfile(output_file):
        return pd.read_csv(output_file, index_col = False)

    pc_columns = [f"PC{i}" for i in range(1, n_pcs+1)]

    for experiment in cfg.EXPERIMENTS:
        print(f"Calculating experiment {experiment}")
        exp_data = df.loc[df["experiment"] == experiment, data_columns].to_numpy()
        exp_data = StandardScaler().fit_transform(exp_data)
        if use_pca:
            _pca = PCA(
                n_components = n_pcs
            ).fit_transform(exp_data)
            df.loc[df["experiment"] == experiment, pc_columns] = _pca
            dimred_input_data = _pca
        else:
            dimred_input_data = exp_data

        for dim_red in dimreds:
            if dim_red == "UMAP":
                print("... calculating UMAP")
                coords = UMAP().fit_transform(dimred_input_data)
                df.loc[df["experiment"] == experiment, ["UMAP1", "UMAP2"]] = coords
            elif dim_red == "TSNE":
                print("... calculating TSNE")
                coords = TSNE().fit_transform(dimred_input_data)
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
    if os.path.isfile(output_file):
        return pd.read_csv(output_file, index_col = False)

    dist_dfs = []
    original_data_columns = data_columns
    for experiment in cfg.EXPERIMENTS:
        data_columns = original_data_columns

        print(f"Calculating distances for experiment {experiment}")
        exp_data = df[df["experiment"] == experiment].copy()
        time_points = sorted(exp_data["loop"].unique())

        exp_data[data_columns] = StandardScaler().fit_transform(exp_data[data_columns])
        if use_pca:
            _pca = PCA(
                n_components = n_pcs
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
                              n_neighbors=10,
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
                                    n_neighbors: int = 50,
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
    if os.path.isfile(output_file):
        return pd.read_csv(output_file, index_col = False)

    records = []
    exps = df[experiment_col].unique()
    for exp in exps:
        print(f">>> Computing neighbors for experiment {exp}")
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
                                    n_neighbors: int = 10,
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
                                     n_neighbors: int = 10,
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
    if os.path.isfile(output_file):
        return pd.read_csv(output_file, index_col = False)

    records = []
    for exp in df[experiment_col].unique():
        print(f">>> Computing neighbors for experiment {exp}")
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
    df["loop"] = [int(lo.split("LO")[1]) for lo in df["loop"]]
    df.to_csv(output_file, index = False)
    return df

