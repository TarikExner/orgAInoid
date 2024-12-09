import os
import numpy as np
import pandas as pd
import time
import gc
import pickle
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multioutput import MultiOutputClassifier


import itertools


from ._classifier_scoring import SCORES_TO_USE, write_to_scores, score_classifier
from .models import (CLASSIFIERS_TO_TEST_FULL,
                     FINAL_CLASSIFIER_RPE,
                     FINAL_CLASSIFIER_LENS,
                     FINAL_CLASSIFIER_RPE_CLASSES,
                     FINAL_CLASSIFIER_LENS_CLASSES)
from ._utils import _one_hot_encode_labels, _apply_train_test_split

from typing import Optional

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

def test_for_n_experiments(df: pd.DataFrame,
                            output_dir: str,
                            classifier: str,
                            use_tuned_classifier: bool,
                            data_columns: list[str],
                            readout: str):

    """\
    The function expects unscaled raw data.

    This function is supposed to test how many experiments are necessary
    for a given F1-score.

    """

    readouts = [readout]

    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = (
        "algorithm,readout,n_experiments,permutation,score_on,experiment,train_time," +
        f"pred_time_train,pred_time_test,pred_time_val,{scores}\n"
    )
    score_key = "n_experiments"
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
        assert isinstance(scores, pd.DataFrame)
        already_calculated = {}
        for _readout in scores["readout"].unique():
            already_calculated[_readout] = scores.loc[
                scores["readout"] == readout,
                "algorithm"
            ].tolist()

    experiments = df["experiment"].unique().tolist()
    total_experiments = experiments

    n_experiments_to_test = list(range(1, len(total_experiments)))
    
    if readout == "RPE_Final":
        classifiers_to_test = FINAL_CLASSIFIER_RPE
    elif readout == "Lens_Final":
        classifiers_to_test = FINAL_CLASSIFIER_LENS
    elif readout == "RPE_classes":
        classifiers_to_test = FINAL_CLASSIFIER_RPE_CLASSES
    elif readout == "Lens_classes":
        classifiers_to_test = FINAL_CLASSIFIER_LENS_CLASSES
    else:
        raise ValueError("Unknown readout")

    param_dir = os.path.join(output_dir, "best_params/")

    for readout in readouts:
        param_file_name = f"best_params_{classifier}_{readout}.dict"
        param_file_dir = os.path.join(param_dir, param_file_name)

        for classifier in classifiers_to_test:

            if readout in already_calculated:
                if classifier in already_calculated[readout]:
                    print(f"Skipping {classifier} for {readout} as it is already calculated")
                    continue

            print(f"... running {classifier} on readout {readout}")
            for val_experiment in experiments:
                print(f"CURRENT VALIDATION EXPERIMENT: {val_experiment}")

                for n_exp in n_experiments_to_test:


                    non_val_experiments = [exp for exp in total_experiments if exp != val_experiment]

                    all_combinations = list(itertools.combinations(non_val_experiments, n_exp))
                    random.shuffle(all_combinations)
                    all_combinations = all_combinations[:50] if len(all_combinations) >= 50 else all_combinations

                    for permutation, experiments_to_test in enumerate(all_combinations):
                        assert len(experiments_to_test) == len(set(experiments_to_test))
                        if use_tuned_classifier:
                            with open(param_file_dir, "rb") as file:
                                cleaned_best_params = pickle.load(file)
                        else:
                            cleaned_best_params = {}
                        
                        clf = _get_classifier(classifier_name = classifier,
                                              params = cleaned_best_params)


                        non_val_df = df[df["experiment"] != val_experiment].copy()
                        assert isinstance(non_val_df, pd.DataFrame)

                        non_val_df = non_val_df[non_val_df["experiment"].isin(experiments_to_test)].copy()
                        assert isinstance(non_val_df, pd.DataFrame)

                        train_df, test_df = _apply_train_test_split(non_val_df)

                        y_train = _one_hot_encode_labels(train_df[readout].to_numpy(),
                                                         readout = readout)

                        if np.unique(y_train, axis = 0).shape[0] == 1:
                            print(f"Skipping combination {experiments_to_test} because only one class is present")
                            continue

                        val_df = df[df["experiment"] == val_experiment].copy()
                        assert isinstance(val_df, pd.DataFrame)

                        scaler = StandardScaler()
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

                        X_train = train_df[data_columns].to_numpy()

                        X_test = test_df[data_columns].to_numpy()
                        y_test = _one_hot_encode_labels(test_df[readout].to_numpy(),
                                                        readout = readout)

                        X_val = val_df[data_columns].to_numpy()
                        y_val = _one_hot_encode_labels(val_df[readout].to_numpy(),
                                                       readout = readout)


                        print(f"     Calculating... {n_exp}: {experiments_to_test}")
                        
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
                        write_to_scores(f"{classifier},{readout},{n_exp},{permutation},train,{val_experiment},{train_time},{pred_time_train},{pred_time_test},{pred_time_val},{score_string}",
                                        output_dir = output_dir,
                                        key = score_key)

                        scores = score_classifier(true_arr = y_test_argmax,
                                                  pred_arr = pred_test_argmax,
                                                  readout = readout)
                        score_string = ",".join(scores)
                        write_to_scores(f"{classifier},{readout},{n_exp},{permutation},test,{val_experiment},{train_time},{pred_time_train},{pred_time_test},{pred_time_val},{score_string}",
                                        output_dir = output_dir,
                                        key = score_key)
                        scores = score_classifier(true_arr = y_val_argmax,
                                                  pred_arr = pred_val_argmax,
                                                  readout = readout)
                        score_string = ",".join(scores)
                        write_to_scores(f"{classifier},{readout},{n_exp},{permutation},val,{val_experiment},{train_time},{pred_time_train},{pred_time_test},{pred_time_val},{score_string}",
                                        output_dir = output_dir,
                                        key = score_key)

                        del clf, X_train, X_test, X_val, y_train, y_test, y_val
                        gc.collect()

    return




    

