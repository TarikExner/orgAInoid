import os
import pandas as pd
import gc
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from typing import Optional
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GroupKFold

from ._classifier_scoring import SCORES_TO_USE, write_to_scores
from .models import (CLASSIFIERS_TO_TEST_FULL,
                     CLASSIFIERS_TO_TEST_RPE,
                     CLASSIFIERS_TO_TEST_LENS,
                     CLASSIFIERS_TO_TEST_RPE_CLASSES,
                     CLASSIFIERS_TO_TEST_LENS_CLASSES)
from ._utils import (conduct_hyperparameter_search,
                     _get_data_array,
                     _get_labels_array,
                     _run_classifier_fit_and_val)

def run_hyperparameter_tuning(df: pd.DataFrame,
                              output_dir: str,
                              data_columns: list[str],
                              readout: str,
                              analysis_id: Optional[str] = None):
    """\
    Runs the Hyperparameter tuning based on the readout.
    """
    if readout == "RPE_Final":
        for classifier in CLASSIFIERS_TO_TEST_RPE:
            _run_hyperparameter_tuning(df, output_dir, classifier, data_columns, readout, analysis_id)
    elif readout == "Lens_Final":
        for classifier in CLASSIFIERS_TO_TEST_LENS:
            _run_hyperparameter_tuning(df, output_dir, classifier, data_columns, readout, analysis_id)
    elif readout == "RPE_classes":
        for classifier in CLASSIFIERS_TO_TEST_RPE_CLASSES:
            _run_hyperparameter_tuning(df, output_dir, classifier, data_columns, readout, analysis_id)
    else:
        assert readout == "Lens_classes", "Unknown readout"
        for classifier in CLASSIFIERS_TO_TEST_LENS_CLASSES:
            _run_hyperparameter_tuning(df, output_dir, classifier, data_columns, readout, analysis_id)

def _get_classifier(classifier_name,
                    params: Optional[dict] = None,
                    hyperparameter: bool = False):

    if params is None:
        params = {}

    if CLASSIFIERS_TO_TEST_FULL[classifier_name]["allows_multi_class"]:
        if CLASSIFIERS_TO_TEST_FULL[classifier_name]["multiprocessing"] and not hyperparameter:
            params["n_jobs"] = 16
            clf = CLASSIFIERS_TO_TEST_FULL[classifier_name]["classifier"](**params)
        else:
            clf = CLASSIFIERS_TO_TEST_FULL[classifier_name]["classifier"](**params)
    else:
        if CLASSIFIERS_TO_TEST_FULL[classifier_name]["scalable"] is False:
            clf = MultiOutputClassifier(CLASSIFIERS_TO_TEST_FULL[classifier_name]["classifier"](**params))
        else:
            clf = MultiOutputClassifier(CLASSIFIERS_TO_TEST_FULL[classifier_name]["classifier"](**params), n_jobs = 16)

    return clf

def _run_hyperparameter_tuning(df: pd.DataFrame,
                               output_dir: str,
                               classifier: str,
                               data_columns: list[str],
                               readout: str,
                               analysis_id: Optional[str] = None):

    """\
    Function to run the classifier hyperparameter tuning.

    df is the dataframe with data_columns consisting of unscaled, raw data.

    The function will run through the classifiers defined in .models.

    Classifiers that have been calculated already will be skipped.

    """
    df = df.sort_values(["experiment", "well", "loop", "slice"])
    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = (
        "algorithm,readout,score_on,experiment,train_time," +
        f"pred_time_train,pred_time_test,pred_time_val,{scores}\n"
    )
    if analysis_id is not None:
        score_key = f"HYPERPARAM_{analysis_id}"
    else:
        score_key = "Hyperparameter_Tuning"
    score_file = os.path.join(output_dir, f"{score_key}.log")
    if not os.path.isfile(score_file):
        write_to_scores(resource_metrics,
                        output_dir = output_dir,
                        key = score_key,
                        init = True)
        already_calculated = {}
    else:
        scores = pd.read_csv(score_file, index_col = False)
        scores = scores[["algorithm", "readout"]].drop_duplicates()
        already_calculated = {}
        for _readout in scores["readout"].unique():
            already_calculated[_readout] = scores.loc[
                scores["readout"] == readout,
                "algorithm"
            ].tolist()
    
    try:
        if classifier in already_calculated[readout]:
            print(f"Skipping {classifier}, as it has already been calculated for readout {readout}")
            return
    except KeyError:
        pass

    readouts = [readout]

    experiments = df["experiment"].unique().tolist()
    
    if analysis_id is not None:
        param_dir = os.path.join(output_dir, "best_params/", analysis_id)
    else:
        param_dir = os.path.join(output_dir, "best_params/")

    if not os.path.exists(param_dir):
        os.makedirs(param_dir, exist_ok = True)

    hyper_df = df.copy()

    for readout in readouts:
        param_file_name = f"best_params_{classifier}_{readout}.dict"
        param_file_dir = os.path.join(param_dir, param_file_name)

        if not os.path.isfile(param_file_dir):
            print(f"Calculating hyperparameters of {classifier} for {readout}.")
            clf = _get_classifier(classifier_name = classifier,
                                  hyperparameter = True)
            pipe = Pipeline([
                ('standardscaler', StandardScaler()),
                ('minmaxscaler', MinMaxScaler()),
                ('clf', clf)
            ])
            X = _get_data_array(hyper_df, data_columns)
            y = _get_labels_array(hyper_df, readout)
            group_kfold = GroupKFold(n_splits = 5)
            hyper_df["group"] = [
                f"{experiment}_{well}"
                for experiment, well in
                zip(
                    hyper_df["experiment"].tolist(),
                    hyper_df["well"].tolist()
                )
            ]
            groups = hyper_df["group"].tolist()
            grid = CLASSIFIERS_TO_TEST_FULL[classifier]["grid"]
            new_grid = {}
            for key in grid:
                new_grid[f"clf__{key}"] = grid[key]
            grid = new_grid
            print(grid)
            hyperparameter_search = conduct_hyperparameter_search(
                pipe,
                grid = CLASSIFIERS_TO_TEST_FULL[classifier]["grid"],
                method = "HalvingRandomSearchCV",
                X_train = X,
                y_train = y,
                group_kfold = group_kfold,
                groups = groups
            )
            best_params = hyperparameter_search.best_params_
            cleaned_best_params = {}

            for key, value in best_params.items():
                if key.startswith("estimator__"):
                    cleaned_best_params[key.split("estimator__")[1]] = value
                else:
                    cleaned_best_params[key] = value

            with open(param_file_dir, "wb") as file:
                pickle.dump(cleaned_best_params, file)   
            del X, y
            gc.collect()
        else:
            print(f"Loading {classifier} for {readout} as it is already calculated.")
            with open(param_file_dir, "rb") as file:
                cleaned_best_params = pickle.load(file)

        for experiment in experiments:
            clf = _get_classifier(classifier_name = classifier,
                                  params = cleaned_best_params)

            _run_classifier_fit_and_val(df = df,
                                        experiment = experiment,
                                        data_columns = data_columns,
                                        readout = readout,
                                        clf = clf,
                                        classifier_name = classifier,
                                        output_dir = output_dir,
                                        score_key = score_key)

    return
