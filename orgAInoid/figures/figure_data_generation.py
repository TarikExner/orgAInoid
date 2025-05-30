import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score

from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

import pickle
from . import figure_config as cfg

from typing import Literal

from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.decomposition import PCA

def calculate_organoid_dimensionality_reduction(df: pd.DataFrame,
                                                data_columns: list[str],
                                                dimreds: list[str] = ["UMAP", "TSNE"],
                                                use_pca: bool = False,
                                                n_pcs: int = 15,
                                                timeframe_length: int = 24,
                                                output_dir: str = "./figure_data/",
                                                output_filename: str = "dataset_morphometrics.csv") -> pd.DataFrame:
    
    if os.path.isfile(os.path.join(output_dir, output_filename)):
        return pd.read_csv(os.path.join(output_dir, output_filename))

    for experiment in cfg.EXPERIMENTS:
        exp_data = df.loc[df["experiment"] == experiment, data_columns].to_numpy()
        exp_data[data_columns] = StandardScaler().fit_transform(exp_data[data_columns])
        if use_pca:
            _pca = PCA(
                n_components = n_pcs
            ).fit_transform(exp_data[data_columns])
            dimred_input_data = _pca
        else:
            dimred_input_data = exp_data[data_columns].to_numpy()

        for dim_red in dimreds:
            if dim_red == "UMAP":
                coords = UMAP().fit_transform(dimred_input_data)
                df.loc[df["experiment"] == experiment, ["UMAP1", "UMAP2"]] = coords
            elif dim_red == "TSNE":
                coords = TSNE().fit_transform(dimred_input_data)
                df.loc[df["experiment"] == experiment, ["TSNE1", "TSNE2"]] = coords
            else:
                raise ValueError(f"Unknown DimRed {dim_red}")
    timepoints = [f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}" for i in range(1,145)]
    timeframe_dict = {}
    j = 0
    for i, timepoint in enumerate(timepoints):
        if i%timeframe_length== 0:
            j += 1
        timeframe_dict[timepoint] = str(j)

    df["timeframe"] = df["loop"].map(timeframe_dict)
    df["timeframe"] = df["timeframe"].astype(str)

    df.to_csv(
        os.path.join(
            output_dir,
            output_filename
        )
    )
    return df

def calculate_organoid_distances(df: pd.DataFrame,
                                 data_columns: list[str],
                                 use_pca: bool = False,
                                 n_pcs: int = 15,
                                 output_dir: str = "./figure_data/organoid_distances",
                                 output_filename: str = "organoid_distances.csv") -> pd.DataFrame:
    """Calculates the distances between organoids and between loops"""

    if os.path.isfile(os.path.join(output_dir, output_filename)):
        return pd.read_csv(os.path.join(output_dir, output_filename))

    df[data_columns] = StandardScaler().fit_transform(df[data_columns])

    if use_pca:
        _pca = PCA(
            n_components = n_pcs
        ).fit_transform(df[data_columns])
        pc_columns = [f"PC{i}" for i in range(1,n_pcs+1)]
        df[pc_columns] = _pca
        data_columns = pc_columns
    else:
        df[data_columns] = StandardScaler().fit_transform(df[data_columns])

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
            ).fit_transform(df[data_columns])
            pc_columns = [f"PC{i}" for i in range(1,n_pcs+1)]
            df[pc_columns] = _pca
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
        df_distances['distance'] = df_distances.groupby(['loop', 'distance_type', 'experiment'])['distance']\
                .transform(lambda x: np.clip(x, x.quantile(0.025), x.quantile(0.975)) if x.name[1] == "intraorganoid" else x)
        df_distances.to_csv(
            os.path.join(
                output_dir,
                f"{experiment}_distances.csv"
            )
        )
        dist_dfs.append(df_distances)
    dist_df = pd.concat(dist_dfs, axis = 0)
    dist_df.to_csv(
        os.path.join(
            output_dir,
            output_filename
        )
    )
    return dist_df

def main():
    # if True, will generate data again
    REDO_ANALYSIS = True
    annotations_filename = "./figure_data/dataset_annotations.csv"

    if REDO_ANALYSIS or not os.path.isfile(annotations_filename):
        experiments = cfg.EXPERIMENTS
        
        frames = []
        for exp in experiments:
            frames.append(pd.read_csv(f"../metadata/{exp}_annotations.csv", sep = ";"))
        
        df = pd.concat(frames, axis = 0)
        df = df.dropna()
        df[["experiment", "well"]] = pd.DataFrame(data = [[ID[:4], ID[4:]] for ID in df["ID"].tolist()]).to_numpy()
        df.to_csv(annotations_filename)
    else:
        df = pd.read_csv(annotations_filename)
    print(df["experiment"].unique().tolist())
    df.head()

##############
# FIGURE S1C #
##############

    FIGURE_S1_B_FILENAME = "./figure_data/RPE_visibility.csv"
    FIGURE_S1_B_CONFMATRIX_FILENAME = "./figure_data/RPE_visibility_conf_matrix.npy"

    EVALUATORS = ["HEAT1", "HEAT2", "HEAT3", "HEAT4", "HEAT5", "HEAT6", "HEAT7"]
    TIMEFRAME_LENGTH = 12

    REDO_ANALYSIS = True

    def _rename_columns(df) -> pd.DataFrame:
        eval_id = df["Evaluator_ID"].unique()[0]
        df = df.rename(columns = {"FileName": "file_name",
                                  "ContainsRPE": f"{eval_id}_RPE_Final_Contains",
                                  "WillDevelopRPE": f"{eval_id}_RPE_Final",
                                  "RPESize": f"{eval_id}_RPE_classes",
                                  "ContainsLens": f"{eval_id}_Lens_Final_Contains",
                                  "WillDevelopLens": f"{eval_id}_Lens_Final",
                                  "LensSize": f"{eval_id}_Lens_classes"})
        return df

    def _add_loop_from_timeframe(df):
        timeframes = df["timeframe"].unique()
        n_timeframes = len(timeframes)
        loops_per_timeframe = 144 // n_timeframes
        half_timeframe = loops_per_timeframe // 2
        corr_loops = np.arange(half_timeframe, 144 + half_timeframe, loops_per_timeframe)
        loop_map = {timeframe: corr_loops[i] for i, timeframe in enumerate(timeframes)}
        df["loop"] = df["timeframe"].map(loop_map)
        return df

    def _one_hot_encode_labels(labels_array: np.ndarray,
                               readout: str) -> np.ndarray:
        n_classes_dict = {
            "RPE_Final": 2,
            "Lens_Final": 2,
            "RPE_classes": 4,
            "Lens_classes": 4,
            "RPE_Final_vis": 2,
            "Lens_Final_vis": 2
        }
        n_classes = n_classes_dict[readout]
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

            labels_array = np.hstack([labels_array, full_class_spectrum])
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(labels_array)
        
        onehot_encoder = OneHotEncoder()
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        classification = onehot_encoder.fit_transform(integer_encoded).toarray()

        if n_appended != 0:
            classification = classification[:-n_appended]

        return classification

    def calculate_f1_scores(df, by: str = "loop"):
        loops = df.sort_values([by], ascending = True)["loop"].unique().tolist()
        f1_scores = pd.DataFrame(df.groupby(by).apply(lambda x: f1_score(x['truth'], x['pred'], average = "weighted")), columns = ["F1"]).reset_index()
        if by == "loop":
            f1_scores[by] = _loop_to_timepoint(loops)
        return f1_scores

    def get_raw_evaluations(readout, rename: bool = True) -> pd.DataFrame:
        if "vis" in readout:
            ground_truth_readout = readout.split("_vis")[0]
            eval_readout = readout.split("_vis")[0] + "_Contains"
        else:
            ground_truth_readout = readout
            eval_readout = readout
        evaluators = EVALUATORS
        eval_dfs = []
        for evaluator in evaluators:
            df = pd.read_csv(f"../human_evaluation/evaluations/{evaluator}_organoid_classification.csv")
            if rename:
                df = _rename_columns(df)
            eval_dfs.append(df)

        return eval_dfs

    def produce_scores(readout: str,
                       ground_truth: pd.DataFrame,
                       timepoints: list[str]) -> pd.DataFrame:
        if "vis" in readout:
            ground_truth_readout = readout.split("_vis")[0]
            eval_readout = readout.split("_vis")[0] + "_Contains"
        else:
            ground_truth_readout = readout
            eval_readout = readout
        eval_dfs = get_raw_evaluations(readout, rename = True)
        f1_scores = pd.DataFrame(data = list(range(int(len(timepoints)/timeframe_length) + 1)), columns = ["timeframe"])
        for i, eval_df in enumerate(eval_dfs):
            merged = ground_truth.merge(eval_dfs[i], on = "file_name", how = "inner")
            merged = merged.dropna(how = "any")
            eval_id = eval_dfs[i]["Evaluator_ID"].unique()[0]
            merged = merged.replace({"Yes": "yes", "No": "no"})
            merged["truth"] = np.argmax(_one_hot_encode_labels(merged[ground_truth_readout], readout), axis = 1)
            merged["pred"] =  np.argmax(_one_hot_encode_labels(merged[f"{eval_id}_{eval_readout}"], readout), axis = 1)
            scores = calculate_f1_scores(merged, by = "timeframe")
            scores["timeframe"] = scores["timeframe"].astype(np.int8)
            scores = scores.rename(columns = {"F1": eval_id})
            f1_scores = f1_scores.merge(scores, on = "timeframe")
        f1_scores = f1_scores.set_index("timeframe")
        f1_scores["mean"] = f1_scores.mean(axis = 1)
        f1_scores = f1_scores.reset_index()
        f1_scores["timeframe"] = f1_scores["timeframe"].astype(np.int8)
        f1_scores = f1_scores.melt(id_vars = "timeframe", var_name = "evaluator", value_name = "F1_score")

        return f1_scores

    if REDO_ANALYSIS or not os.path.isfile(FIGURE_S1_B_CONFMATRIX_FILENAME):
        timeframe_length = TIMEFRAME_LENGTH
        timepoints = [f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}" for i in range(1,145)]
        timeframe_dict = {}
        j = 0
        for i, timepoint in enumerate(timepoints):
            if i%timeframe_length == 0:
                j += 1
            timeframe_dict[timepoint] = str(j)
        
        ground_truth = pd.read_csv("../human_evaluation/human_evaluation_ground_truth.csv")
        ground_truth["timeframe"] = ground_truth["loop"].map(timeframe_dict)

        eval_dfs = get_raw_evaluations("RPE_Final_vis")

        conf_matrices = []
        for i, eval_df in enumerate(eval_dfs):
            merged = ground_truth.merge(eval_dfs[i], on = "file_name", how = "inner")
            merged = merged.dropna(how = "any")
            merged = merged.replace({"Yes": "yes", "No": "no"})
            eval_id = merged["Evaluator_ID"].unique()[0]
            last_frame = max(merged["timeframe"].astype(int))
            merged = merged[merged["timeframe"] == str(last_frame)]
            conf_matrix = confusion_matrix(merged["RPE_Final"].to_numpy(), merged[f"{eval_id}_RPE_Final_Contains"].to_numpy())
            conf_matrices.append(conf_matrix)
        
        conf_matrix = np.array(conf_matrices).sum(axis = 0)

        np.save(FIGURE_S1_B_CONFMATRIX_FILENAME, conf_matrix)
    else:
        conf_matrix = np.load(FIGURE_S1_B_CONFMATRIX_FILENAME)

    print(conf_matrix)

    if REDO_ANALYSIS or not os.path.isfile(FIGURE_S1_B_FILENAME):
        timeframe_length = TIMEFRAME_LENGTH
        timepoints = [f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}" for i in range(1,145)]
        timeframe_dict = {}
        j = 0
        for i, timepoint in enumerate(timepoints):
            if i%timeframe_length == 0:
                j += 1
            timeframe_dict[timepoint] = str(j)
        
        ground_truth = pd.read_csv("../human_evaluation/human_evaluation_ground_truth.csv")
        ground_truth["timeframe"] = ground_truth["loop"].map(timeframe_dict)
        
        rpe_vis = _add_loop_from_timeframe(produce_scores("RPE_Final_vis", ground_truth, timepoints))
        
        rpe_vis.to_csv(FIGURE_S1_B_FILENAME, index = False)
        df = rpe_vis
    else:
        df = pd.read_csv(FIGURE_S1_B_FILENAME)

    print(df.head())

    FIGURE_S1_D_FILENAME = "./figure_data/Lens_visibility.csv"

    if REDO_ANALYSIS or not os.path.isfile(FIGURE_S1_D_FILENAME):
        timeframe_length = TIMEFRAME_LENGTH
        timepoints = [f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}" for i in range(1,145)]
        timeframe_dict = {}
        j = 0
        for i, timepoint in enumerate(timepoints):
            if i%timeframe_length == 0:
                j += 1
            timeframe_dict[timepoint] = str(j)
        
        ground_truth = pd.read_csv("../human_evaluation/human_evaluation_ground_truth.csv")
        ground_truth["timeframe"] = ground_truth["loop"].map(timeframe_dict)
        
        rpe_vis = _add_loop_from_timeframe(produce_scores("Lens_Final_vis", ground_truth, timepoints))
        
        rpe_vis.to_csv(FIGURE_S1_D_FILENAME, index = False)
        df = rpe_vis
    else:
        df = pd.read_csv(FIGURE_S1_D_FILENAME)

    print(df.head())

#############
# FIGURE 2B #
#############

    FIGURE_2_B_FILENAME = "./figure_data/dataset_morphometrics.csv"
    DIMRED = "TSNE"
    TIMEFRAME_LENGTH = 24

    def _calculate_TSNE(df: pd.DataFrame,
                        data_columns: list[str]) -> pd.DataFrame:
        for experiment in df["experiment"].unique():
            data = df.loc[df["experiment"] == experiment, data_columns].to_numpy()
            scaled_data = StandardScaler().fit_transform(data)
            print(f"Calculating TSNE for {experiment}")
            coords = TSNE().fit_transform(scaled_data)
            df.loc[df["experiment"] == experiment, ["TSNE1", "TSNE2"]] = coords

        return df

    def _calculate_UMAP(df: pd.DataFrame,
                        data_columns: list[str]) -> pd.DataFrame:
        for experiment in df["experiment"].unique():
            data = df.loc[df["experiment"] == experiment, data_columns].to_numpy()
            scaled_data = StandardScaler().fit_transform(data)
            print(f"Calculating UMAP for {experiment}")
            coords = UMAP(init = "pca").fit_transform(scaled_data)
            df.loc[df["experiment"] == experiment, ["UMAP1", "UMAP2"]] = coords

        return df

    REDO_ANALYSIS = False
    metadata_columns = [
        'experiment', 'well', 'file_name', 'position', 'slice', 'loop',
        'Condition', 'RPE_Final', 'RPE_Norin', 'RPE_Cassian',
        'Confidence_score_RPE', 'Total_RPE_amount', 'RPE_classes', 'Lens_Final',
        'Lens_Norin', 'Lens_Cassian', 'Confidence_score_lens', 'Lens_area',
        'Lens_classes', 'label'
    ]

    if REDO_ANALYSIS or not os.path.isfile(FIGURE_2_B_FILENAME):
        experiments = cfg.EXPERIMENTS
        for i, exp in enumerate(experiments):
            data = pd.read_csv(f"../shape_analysis/results/{exp}_morphometrics.csv")
            if i == 0:
                frame = data
            else:
                frame = pd.concat([frame, data], axis = 0)
        # we dont need label
        frame = frame.drop(["label"], axis = 1)          
        # we remove columns and rows that only contain NA
        frame = frame.dropna(how = "all", axis = 1)
        
        data_columns = [col for col in frame.columns if col not in metadata_columns]  
        frame = frame.dropna(how = "all", axis = 0, subset = data_columns)

        frame["RPE_classes"] = frame["RPE_classes"].fillna(0)
        
        assert not frame.isna().any().sum()
        
        data_columns = [col for col in frame.columns if col not in metadata_columns]

        frame = _calculate_TSNE(frame, data_columns)
        frame = _calculate_UMAP(frame, data_columns)

        timepoints = [f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}" for i in range(1,145)]
        timeframe_dict = {}
        j = 0
        for i, timepoint in enumerate(timepoints):
            if i%TIMEFRAME_LENGTH == 0:
                j += 1
            timeframe_dict[timepoint] = str(j)
        
        frame["timeframe"] = frame["loop"].map(timeframe_dict)
        frame["timeframe"] = frame["timeframe"].astype(str)
        
        frame.to_csv(FIGURE_2_B_FILENAME)
        df = frame
    else:
        df = pd.read_csv(FIGURE_2_B_FILENAME, index_col = [0])
        data_columns = [col for col in df.columns if col not in metadata_columns]

    print(df.head())

    FIGURE_2_C_FILENAME = "./figure_data/organoid_distances/organoid_distances.csv"

    from scipy.spatial.distance import pdist, squareform, cdist

    if REDO_ANALYSIS or not os.path.isfile(FIGURE_2_C_FILENAME):
        df = pd.read_csv(FIGURE_2_B_FILENAME, index_col = [0])
        df[data_columns] = StandardScaler().fit_transform(df[data_columns])
        dist_dfs = []
        for experiment in cfg.EXPERIMENTS:
            print(f"Calculating distances for {experiment}")
            exp_df = df[df["experiment"] == experiment]
            
            # Get sorted list of unique time points
            time_points = sorted(exp_df['loop'].unique())
            
            # Initialize list to store distances
            distance_data = []
            
            # Compute interorganoid distances at each time point
            for time_point in time_points:
                df_time = exp_df[exp_df['loop'] == time_point].sort_values('well')
                data = df_time[data_columns].values
                subjects = df_time['well'].values
                distances = pdist(data, metric='euclidean')
                dist_matrix = squareform(distances)
                idx_upper = np.triu_indices(len(subjects), k=1)
                for i, j in zip(*idx_upper):
                    distance_data.append({
                        'loop': time_point,
                        'distance_type': 'interorganoid',
                        'distance': dist_matrix[i, j]
                    })
            
            # Compute intertimepoint distances between consecutive time points
            for i in range(len(time_points) - 1):
                time_point_n = time_points[i]
                time_point_n1 = time_points[i + 1]
                df_n = exp_df[exp_df['loop'] == time_point_n]
                df_n1 = exp_df[exp_df['loop'] == time_point_n1]
                common_subjects = np.intersect1d(df_n['well'].unique(), df_n1['well'].unique())
                if len(common_subjects) == 0:
                    continue
                df_n_common = df_n[df_n['well'].isin(common_subjects)].sort_values('well')
                df_n1_common = df_n1[df_n1['well'].isin(common_subjects)].sort_values('well')
                data_n = df_n_common[data_columns].values
                data_n1 = df_n1_common[data_columns].values
                distances = []
                for well in common_subjects:
                    _distance = cdist(df_n_common.loc[df_n_common["well"] == well, data_columns].to_numpy(),
                                      df_n1_common.loc[df_n1_common["well"] == well, data_columns].to_numpy())
                    distances.append(_distance[0][0])
                #distances = np.linalg.norm(data_n - data_n1, axis=1)
                avg_loop = time_point_n
                for dist in distances:
                    distance_data.append({
                        'loop': avg_loop,
                        'distance_type': 'intraorganoid',
                        'distance': dist
                    })
            
            # Create dataframe
            df_distances = pd.DataFrame(distance_data)
            df_distances["loop"] = [int(loop.split("LO")[1]) for loop in df_distances["loop"].tolist()]
            df_distances["experiment"] = experiment
            df_distances['distance'] = df_distances.groupby(['loop', 'distance_type', 'experiment'])['distance']\
                    .transform(lambda x: np.clip(x, x.quantile(0.025), x.quantile(0.975)) if x.name[1] == "intraorganoid" else x)
            df_distances.to_csv(f"./figure_data/organoid_distances/{experiment}_distances.csv")
            dist_dfs.append(df_distances)
        dist_df = pd.concat(dist_dfs, axis = 0)
        dist_df.to_csv(FIGURE_2_C_FILENAME)
        df_distances = dist_df
        
    else:
        df_distances = pd.read_csv(FIGURE_2_C_FILENAME)

    print(df_distances.head())

    org_distances_no_wnt_filename = "./figure_data/organoid_distances/organoid_distances_no_wnt.csv"

    from scipy.spatial.distance import pdist, squareform, cdist
    REDO_ANALYSIS = True
    if REDO_ANALYSIS or not os.path.isfile(org_distances_no_wnt_filename):
        df = pd.read_csv(FIGURE_2_B_FILENAME, index_col = [0])
        df[data_columns] = StandardScaler().fit_transform(df[data_columns])
        df = df[df["Condition"] == "0nM"]
        dist_dfs = []
        for experiment in cfg.EXPERIMENTS:
            print(f"Calculating distances for {experiment}")
            exp_df = df[df["experiment"] == experiment]
            
            # Get sorted list of unique time points
            time_points = sorted(exp_df['loop'].unique())
            
            # Initialize list to store distances
            distance_data = []
            
            # Compute interorganoid distances at each time point
            for time_point in time_points:
                df_time = exp_df[exp_df['loop'] == time_point].sort_values('well')
                data = df_time[data_columns].values
                subjects = df_time['well'].values
                distances = pdist(data, metric='euclidean')
                dist_matrix = squareform(distances)
                idx_upper = np.triu_indices(len(subjects), k=1)
                for i, j in zip(*idx_upper):
                    distance_data.append({
                        'loop': time_point,
                        'distance_type': 'interorganoid',
                        'distance': dist_matrix[i, j]
                    })
            
            # Compute intertimepoint distances between consecutive time points
            for i in range(len(time_points) - 1):
                time_point_n = time_points[i]
                time_point_n1 = time_points[i + 1]
                df_n = exp_df[exp_df['loop'] == time_point_n]
                df_n1 = exp_df[exp_df['loop'] == time_point_n1]
                common_subjects = np.intersect1d(df_n['well'].unique(), df_n1['well'].unique())
                if len(common_subjects) == 0:
                    continue
                df_n_common = df_n[df_n['well'].isin(common_subjects)].sort_values('well')
                df_n1_common = df_n1[df_n1['well'].isin(common_subjects)].sort_values('well')
                data_n = df_n_common[data_columns].values
                data_n1 = df_n1_common[data_columns].values
                distances = []
                for well in common_subjects:
                    _distance = cdist(df_n_common.loc[df_n_common["well"] == well, data_columns].to_numpy(),
                                      df_n1_common.loc[df_n1_common["well"] == well, data_columns].to_numpy())
                    distances.append(_distance[0][0])
                #distances = np.linalg.norm(data_n - data_n1, axis=1)
                avg_loop = time_point_n
                for dist in distances:
                    distance_data.append({
                        'loop': avg_loop,
                        'distance_type': 'intraorganoid',
                        'distance': dist
                    })
            
            # Create dataframe
            df_distances = pd.DataFrame(distance_data)
            df_distances["loop"] = [int(loop.split("LO")[1]) for loop in df_distances["loop"].tolist()]
            df_distances["experiment"] = experiment
            df_distances['distance'] = df_distances.groupby(['loop', 'distance_type', 'experiment'])['distance']\
                    .transform(lambda x: np.clip(x, x.quantile(0.025), x.quantile(0.975)) if x.name[1] == "intraorganoid" else x)
            df_distances.to_csv(f"./figure_data/organoid_distances/{experiment}_distances_no_wnt.csv")
            dist_dfs.append(df_distances)
        dist_df = pd.concat(dist_dfs, axis = 0)
        dist_df.to_csv(org_distances_no_wnt_filename)
        df_distances_no_wnt = dist_df
        
    else:
        df_distances_no_wnt = pd.read_csv(org_distances_no_wnt_filename)

    print(df_distances.head())

#############
# FIGURE 3B #
#############

    FIGURE_3_A_HUMAN_FILENAME = "./figure_data/RPE_human.csv"
    TIMEFRAME_LENGTH = 6

    if REDO_ANALYSIS or not os.path.isfile(FIGURE_3_A_HUMAN_FILENAME):
        timeframe_length = TIMEFRAME_LENGTH
        timepoints = [f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}" for i in range(1,145)]
        timeframe_dict = {}
        j = 0
        for i, timepoint in enumerate(timepoints):
            if i%timeframe_length == 0:
                j += 1
            timeframe_dict[timepoint] = str(j)
        
        ground_truth = pd.read_csv("../human_evaluation/human_evaluation_ground_truth.csv")
        ground_truth["timeframe"] = ground_truth["loop"].map(timeframe_dict)
        
        rpe_final = _add_loop_from_timeframe(produce_scores("RPE_Final", ground_truth, timepoints))
        rpe_final = rpe_final[rpe_final["evaluator"] != "mean"]
        rpe_final = rpe_final.rename(columns = {"F1_score": "F1", "evaluator": "experiment"})
        rpe_final["classifier"] = "human"
        
        rpe_final.to_csv(FIGURE_3_A_HUMAN_FILENAME, index = False)
        human_eval = rpe_final
    else:
        human_eval = pd.read_csv(FIGURE_3_A_HUMAN_FILENAME)

    print(human_eval.head())

    def generate_classification_dataframe(assay: Literal["RPE_classfication",
                                                         "Lens_classification",
                                                         "RPE_classes_classification",
                                                         "Lens_classes_classification"]):
        experiments = cfg.EXPERIMENTS
        
        val_data = pd.DataFrame()
        for exp in experiments:
            _data = pd.read_csv(f"./figure_data/{assay}/{exp}_neural_net_results_eval_set_val.csv", index_col = [0])
            _data["experiment"] = exp
            val_data = pd.concat([val_data, _data], axis = 0)
        
        val_data = val_data[val_data["Neural Net"] == "Ensemble"]
        val_data["Neural Net"] = "Ensemble_val"
        val_data = val_data.rename(columns = {"Neural Net": "classifier"})
        
        test_data = pd.DataFrame()
        for exp in experiments:
            _data = pd.read_csv(f"./figure_data/{assay}/{exp}_neural_net_results_eval_set_test.csv", index_col = [0])
            _data["experiment"] = exp
            test_data = pd.concat([test_data, _data], axis = 0)
        
        test_data = test_data[test_data["Neural Net"] == "Ensemble"]
        test_data["Neural Net"] = "Ensemble_test"
        test_data = test_data.rename(columns = {"Neural Net": "classifier"})
        
        classifier_data_val = pd.read_csv(f"./figure_data/{assay}/classifier_results_eval_set_val.csv")
        classifier_data_val = classifier_data_val.rename(columns = {"classifier": "algorithm"})
        classifier_data_val["classifier"] = "Morphometrics_val"
        classifier_data_test = pd.read_csv(f"./figure_data/{assay}/classifier_results_eval_set_test.csv")
        classifier_data_test = classifier_data_test.rename(columns = {"classifier": "algorithm"})
        classifier_data_test["classifier"] = "Morphometrics_test"

        human_eval_key = assay.split("_classification")[0]
        human_eval = pd.read_csv(f"./figure_data/{human_eval_key}_human.csv")
        
        full_data = pd.concat([val_data, test_data, classifier_data_val, classifier_data_test, human_eval])
        full_data.to_csv(f"./figure_data/{assay.lower()}.csv")

        return full_data

    def _generate_confusion_matrix_frame(assay: Literal["RPE_classfication",
                                                        "Lens_classification",
                                                        "RPE_classes_classification",
                                                        "Lens_classes_classification"],
                                         classifier: Literal["neural_net", "classifier"],
                                         eval_set: Literal["test", "val"]):
        loops = [
            f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}"
            for i in range(1, 144)
        ]
        results = pd.DataFrame(data = {"loop": loops})
        experiments = cfg.EXPERIMENTS
        for experiment in experiments: 
            conf_matrix_filename = f"./figure_data/{assay}/{experiment}_{classifier}_conf_matrices_{eval_set}.npy"
            _conf_matrix = np.load(conf_matrix_filename)
            file_overview = pd.read_csv(f"../metadata/{experiment}_file_overview.csv")
            file_overview = file_overview.sort_values(["experiment", "well", "loop", "slice"], ascending = [True, True, True, True])
            loops = file_overview["loop"].unique().tolist()
            if "neural_net" in conf_matrix_filename: # remove that later once we have the correct val sets :D
                loops = loops[:-1]
            exp_conf_df = pd.DataFrame(columns=["loop", experiment])
            for i, loop in enumerate(loops):
                new_row = pd.DataFrame({"loop": [loop], experiment: [_conf_matrix[i]]})
                exp_conf_df = pd.concat([exp_conf_df, new_row], ignore_index=True)
            results = results.merge(exp_conf_df, on = "loop", how = "outer")
        
        zero_matrix = np.zeros((2,2)) if assay in ["RPE_classification", "Lens_classification"] else np.zeros((4,4))
        for col in results.columns:
            results[col] = results[col].apply(
                lambda x: zero_matrix if x is None or (isinstance(x, float) and pd.isna(x)) else x
            )
        results = results.set_index("loop")
        
        results["summed_confusion_matrix"] = results.apply(
            lambda row: sum(row), axis=1
        )
        
        # Step 2: Calculate percentages for individual cells per loop
        results["percentage_matrix"] = results["summed_confusion_matrix"].apply(
            lambda matrix: (matrix / matrix.sum()) * 100
        )
        
        with open(f"./figure_data/{assay}_{eval_set}_{classifier}_confusion_matrices.data", "wb") as file:
            pickle.dump(results, file)

        return results

    def generate_human_classification_dataframe(by: Literal["RPE_Final", "Lens_Final", "RPE_classes", "Lens_classes"]):
        TIMEFRAME_LENGTH = 6

        timeframe_length = TIMEFRAME_LENGTH
        timepoints = [f"LO{i}" if i >= 100 else f"LO0{i}" if i>= 10 else f"LO00{i}" for i in range(1,145)]
        timeframe_dict = {}
        j = 0
        for i, timepoint in enumerate(timepoints):
            if i%timeframe_length == 0:
                j += 1
            timeframe_dict[timepoint] = str(j)
        
        ground_truth = pd.read_csv("../human_evaluation/human_evaluation_ground_truth.csv")
        ground_truth["timeframe"] = ground_truth["loop"].map(timeframe_dict)
        
        df = _add_loop_from_timeframe(produce_scores(by, ground_truth, timepoints))
        df = df[df["evaluator"] != "mean"]
        df = df.rename(columns = {"F1_score": "F1", "evaluator": "experiment"})
        df["classifier"] = "human"
        by_names = {"RPE_Final": "RPE_human", "Lens_Final": "Lens_human", "RPE_classes": "RPE_classes_human", "Lens_classes": "Lens_classes_human"}
        filename = f"./figure_data/{by_names[by]}.csv"
        df.to_csv(filename, index = False)

        return df

    import warnings

# Example with temporary suppression
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        _ = generate_human_classification_dataframe("RPE_Final")
        _ = generate_human_classification_dataframe("Lens_Final")
        _ = generate_human_classification_dataframe("RPE_classes")
        _ = generate_human_classification_dataframe("Lens_classes")

    _ = generate_classification_dataframe("RPE_classification")
    _ = generate_classification_dataframe("Lens_classification")
    _ = generate_classification_dataframe("RPE_classes_classification")
    _ = generate_classification_dataframe("Lens_classes_classification")

    _ = _generate_confusion_matrix_frame("RPE_classification", "neural_net", "val")
    _ = _generate_confusion_matrix_frame("RPE_classification", "neural_net", "test")
    _ = _generate_confusion_matrix_frame("RPE_classification", "classifier", "val")
    _ = _generate_confusion_matrix_frame("RPE_classification", "classifier", "test")

    _ = _generate_confusion_matrix_frame("RPE_classes_classification", "neural_net", "val")
    _ = _generate_confusion_matrix_frame("RPE_classes_classification", "neural_net", "test")
    _ = _generate_confusion_matrix_frame("RPE_classes_classification", "classifier", "val")
    _ = _generate_confusion_matrix_frame("RPE_classes_classification", "classifier", "test")

    _ = _generate_confusion_matrix_frame("Lens_classification", "neural_net", "val")
    _ = _generate_confusion_matrix_frame("Lens_classification", "neural_net", "test")
    _ = _generate_confusion_matrix_frame("Lens_classification", "classifier", "val")
    _ = _generate_confusion_matrix_frame("Lens_classification", "classifier", "test")

    _ = _generate_confusion_matrix_frame("Lens_classes_classification", "neural_net", "val")
    _ = _generate_confusion_matrix_frame("Lens_classes_classification", "neural_net", "test")
    _ = _generate_confusion_matrix_frame("Lens_classes_classification", "classifier", "val")
    _ = _generate_confusion_matrix_frame("Lens_classes_classification", "classifier", "test")


if __name__ == "__main__":
    main()
