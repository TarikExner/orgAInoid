import os
import numpy as np
import pandas as pd
import time
import gc
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GroupKFold

from ._classifier_scoring import SCORES_TO_USE, write_to_scores, score_classifier
from .models import (CLASSIFIERS_TO_TEST_FULL,
                     CLASSIFIERS_TO_TEST_RPE,
                     CLASSIFIERS_TO_TEST_LENS,
                     CLASSIFIERS_TO_TEST_RPE_CLASSES,
                     CLASSIFIERS_TO_TEST_LENS_CLASSES)
from ._utils import _one_hot_encode_labels, _apply_train_test_split, conduct_hyperparameter_search

def run_hyperparameter_tuning(df: pd.DataFrame,
                              output_dir: str,
                              data_columns: list[str],
                              readout: str):
    """\
    Runs the Hyperparameter tuning based on the readout.
    """
    if readout == "RPE_Final":
        for classifier in CLASSIFIERS_TO_TEST_RPE:
            _run_hyperparameter_tuning(df, output_dir, classifier, data_columns, readout)
    elif readout == "Lens_Final":
        for classifier in CLASSIFIERS_TO_TEST_LENS:
            _run_hyperparameter_tuning(df, output_dir, classifier, data_columns, readout)
    elif readout == "RPE_classes":
        for classifier in CLASSIFIERS_TO_TEST_RPE_CLASSES:
            _run_hyperparameter_tuning(df, output_dir, classifier, data_columns, readout)
    else:
        assert readout == "Lens_classes", "Unknown readout"
        for classifier in CLASSIFIERS_TO_TEST_LENS_CLASSES:
            _run_hyperparameter_tuning(df, output_dir, classifier, data_columns, readout)

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
                               readout: str):

    """\
    Function to run the classifier hyperparameter tuning.

    df is the dataframe with data_columns consisting of unscaled, raw data.

    The function will run through the classifiers defined in .models.

    Classifiers that have been calculated already will be skipped.

    """

    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = (
        "algorithm,readout,score_on,experiment,train_time," +
        f"pred_time_train,pred_time_test,pred_time_val,{scores}\n"
    )
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

    param_dir = os.path.join(output_dir, "best_params/")

    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    scaler = StandardScaler()
    second_scaler = MinMaxScaler()

    hyper_df = df.copy()
    hyper_df["group"] = [
        f"{experiment}_{well}"
        for experiment, well in
        zip(
            hyper_df["experiment"].tolist(),
            hyper_df["well"].tolist()
        )
    ]

    hyper_df[data_columns] = scaler.fit_transform(hyper_df[data_columns])
    hyper_df[data_columns] = second_scaler.fit_transform(hyper_df[data_columns])

    for readout in readouts:
        param_file_name = f"best_params_{classifier}_{readout}.dict"
        param_file_dir = os.path.join(param_dir, param_file_name)

        if not os.path.isfile(param_file_dir):
            print(f"Calculating hyperparameters of {classifier} for {readout}.")
            clf = _get_classifier(classifier_name = classifier,
                                  hyperparameter = True)
            X = hyper_df[data_columns].to_numpy()
            y = _one_hot_encode_labels(hyper_df[readout].to_numpy(),
                                       readout = readout)
            group_kfold = GroupKFold(n_splits = 5)
            groups = hyper_df["group"].tolist()
            hyperparameter_search = conduct_hyperparameter_search(
                clf,
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

            scaler = StandardScaler()

            non_val_df = df[df["experiment"] != experiment].copy()
            assert isinstance(non_val_df, pd.DataFrame)

            train_df, test_df = _apply_train_test_split(non_val_df)
            val_df = df[df["experiment"] == experiment].copy()
            assert isinstance(val_df, pd.DataFrame)

            scaler.fit(non_val_df[data_columns])

            train_df[data_columns] = scaler.transform(train_df[data_columns])
            test_df[data_columns] = scaler.transform(test_df[data_columns])
            val_df[data_columns] = scaler.transform(val_df[data_columns])

            train_test_df = pd.concat([train_df, test_df], axis = 0)

            second_scaler = MinMaxScaler()
            second_scaler.fit(train_test_df[data_columns])

            train_df[data_columns] = second_scaler.transform(train_df[data_columns])
            test_df[data_columns] = second_scaler.transform(test_df[data_columns])
            val_df[data_columns] = second_scaler.transform(val_df[data_columns])

            X_train = train_df[data_columns]
            y_train = _one_hot_encode_labels(train_df[readout].to_numpy(),
                                             readout = readout)

            X_test = test_df[data_columns]
            y_test = _one_hot_encode_labels(test_df[readout].to_numpy(),
                                            readout = readout)

            X_val = val_df[data_columns]
            y_val = _one_hot_encode_labels(val_df[readout].to_numpy(),
                                           readout = readout)
            
            start = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - start

            y_train_argmax = np.argmax(y_train, axis = 1)
            y_test_argmax = np.argmax(y_test, axis = 1)
            y_val_argmax = np.argmax(y_val, axis = 1)
            
            start = time.time()
            pred_train = clf.predict(X_train)
            pred_train_argmax = np.argmax(pred_train, axis = 1)
            pred_time_train = time.time() - start

            start = time.time()
            pred_test = clf.predict(X_test)
            pred_test_argmax = np.argmax(pred_test, axis = 1)
            pred_time_test = time.time() - start

            start = time.time()
            pred_val = clf.predict(X_val)
            pred_val_argmax = np.argmax(pred_val, axis = 1)
            pred_time_val= time.time() - start

            scores = score_classifier(true_arr = y_train_argmax,
                                      pred_arr = pred_train_argmax,
                                      readout = readout)
            score_string = ",".join(scores)
            write_to_scores(f"{classifier},{readout},train,{experiment},{train_time},{pred_time_train},{pred_time_test},{pred_time_val},{score_string}",
                            output_dir = output_dir,
                            key = score_key)

            scores = score_classifier(true_arr = y_test_argmax,
                                      pred_arr = pred_test_argmax,
                                      readout = readout)
            score_string = ",".join(scores)
            write_to_scores(f"{classifier},{readout},test,{experiment},{train_time},{pred_time_train},{pred_time_test},{pred_time_val},{score_string}",
                            output_dir = output_dir,
                            key = score_key)
            scores = score_classifier(true_arr = y_val_argmax,
                                      pred_arr = pred_val_argmax,
                                      readout = readout)
            score_string = ",".join(scores)
            write_to_scores(f"{classifier},{readout},val,{experiment},{train_time},{pred_time_train},{pred_time_test},{pred_time_val},{score_string}",
                            output_dir = output_dir,
                            key = score_key)

            del clf, X_train, X_test, X_val, y_train, y_test, y_val
            gc.collect()






 
