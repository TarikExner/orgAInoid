import os
import numpy as np
import pandas as pd
import time
import gc
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional

from ._classifier_scoring import SCORES_TO_USE, write_to_scores, score_classifier
from .models import CLASSIFIERS_TO_TEST, CLASSIFIERS_TO_TEST_2
from ._utils import _one_hot_encode_labels, _apply_train_test_split, conduct_hyperparameter_search


def _get_classifier(classifier_name,
                    params: Optional[dict] = None,
                    hyperparameter: bool = False):
    if params is None:
        params = {}
    if classifier_name in ["RandomForestClassifier", "ExtraTreesClassifier"] and not hyperparameter:
        params["n_jobs"] = 16
    return CLASSIFIERS_TO_TEST[classifier_name]["classifier"](**params)


def run_hyperparameter_tuning(df: pd.DataFrame,
                              output_dir: str,
                              data_columns: list[str]):
    for classifier in CLASSIFIERS_TO_TEST:
        _run_hyperparameter_tuning(df, output_dir, classifier, data_columns)

def run_hyperparameter_tuning_2(df: pd.DataFrame,
                                output_dir: str,
                                data_columns: list[str]):
    for classifier in CLASSIFIERS_TO_TEST_2:
        _run_hyperparameter_tuning(df, output_dir, classifier, data_columns)



def _run_hyperparameter_tuning(df: pd.DataFrame,
                               output_dir: str,
                               classifier: str,
                               data_columns: list[str]):

    """\
    The function expects unscaled raw data.

    """

    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = (
        "algorithm,readout,score_on,experiment,train_time," +
        f"pred_time_train,pred_time_test,pred_time_val,{scores}\n"
    )
    score_key = "Hyperparameter_Tuning"

    write_to_scores(resource_metrics,
                    output_dir = output_dir,
                    key = score_key,
                    init = True)

    readouts = ["RPE_Final", "Lens_Final", "RPE_classes", "Lens_classes"]

    experiments = df["experiment"].unique().tolist()

    param_dir = os.path.join(output_dir, "best_params/")
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    scaler = StandardScaler()
    hyper_df = df.copy()
    hyper_df[data_columns] = scaler.fit_transform(hyper_df[data_columns])

    if classifier.endswith("NB"):
        second_scaler = MinMaxScaler()
        hyper_df[data_columns] = second_scaler.fit_transform(hyper_df[data_columns])

    for readout in readouts:
        clf = _get_classifier(classifier_name = classifier,
                              hyperparameter = True)
        X = hyper_df[data_columns].to_numpy()
        y = _one_hot_encode_labels(hyper_df[readout].to_numpy())
        hyperparameter_search = conduct_hyperparameter_search(
            clf,
            grid = CLASSIFIERS_TO_TEST[classifier]["grid"],
            method = "HalvingRandomSearchCV",
            X_train = X,
            y_train = y
        )
        best_params = hyperparameter_search.best_params_
        param_file_name = f"best_params_{classifier}_{readout}.dict" 
        with open(f"{os.path.join(param_dir, param_file_name)}", "wb") as file:
            pickle.dump(best_params, file)   
        del X, y
        gc.collect()


        for experiment in experiments:
            clf = _get_classifier(classifier_name = classifier,
                                  params = best_params)

            scaler = StandardScaler()

            non_val_df = df[df["experiment"] != experiment].copy()
            assert isinstance(non_val_df, pd.DataFrame)

            scaler.fit(non_val_df[data_columns])

            val_df = df[df["experiment"] == experiment].copy()
            assert isinstance(val_df, pd.DataFrame)
            train_df, test_df = _apply_train_test_split(non_val_df)

            train_df[data_columns] = scaler.transform(train_df[data_columns])
            test_df[data_columns] = scaler.transform(test_df[data_columns])
            val_df[data_columns] = scaler.transform(val_df[data_columns])

            if classifier.endswith("NB"):
                # naive bayes methods do not allow negative values
                train_test_df = pd.concat([train_df, test_df], axis = 0)
                second_scaler = MinMaxScaler()
                second_scaler.fit(train_test_df[data_columns])

                train_df[data_columns] = second_scaler.transform(train_df[data_columns])
                test_df[data_columns] = second_scaler.transform(test_df[data_columns])
                val_df[data_columns] = second_scaler.transform(val_df[data_columns])

            X_train = train_df[data_columns]
            y_train = _one_hot_encode_labels(train_df[readout].to_numpy())

            X_test = test_df[data_columns]
            y_test = _one_hot_encode_labels(test_df[readout].to_numpy())

            X_val = val_df[data_columns]
            y_val = _one_hot_encode_labels(val_df[readout].to_numpy())
            
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
                                      pred_arr = pred_train_argmax)
            score_string = ",".join(scores)
            write_to_scores(f"{classifier},{readout},train,{experiment},{train_time},{pred_time_train},{pred_time_test},{pred_time_val},{score_string}",
                            output_dir = output_dir,
                            key = score_key)

            scores = score_classifier(true_arr = y_test_argmax,
                                      pred_arr = pred_test_argmax)
            score_string = ",".join(scores)
            write_to_scores(f"{classifier},{readout},test,{experiment},{train_time},{pred_time_train},{pred_time_test},{pred_time_val},{score_string}",
                            output_dir = output_dir,
                            key = score_key)
            scores = score_classifier(true_arr = y_val_argmax,
                                      pred_arr = pred_val_argmax)
            score_string = ",".join(scores)
            write_to_scores(f"{classifier},{readout},val,{experiment},{train_time},{pred_time_train},{pred_time_test},{pred_time_val},{score_string}",
                            output_dir = output_dir,
                            key = score_key)

            del clf, X_train, X_test, X_val, y_train, y_test, y_val
            gc.collect()






 
