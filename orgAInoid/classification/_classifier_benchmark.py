import os
import numpy as np
import pandas as pd
import time
import gc

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multioutput import MultiOutputClassifier

from ._classifier_scoring import SCORES_TO_USE, write_to_scores, score_classifier
from .models import CLASSIFIERS_TO_TEST_FULL

from ._utils import _one_hot_encode_labels, _apply_train_test_split



def run_classifier_comparison(df: pd.DataFrame,
                              output_dir: str,
                              data_columns: list[str]):
    """\
    The function expects unscaled raw data.

    """

    scores = ",".join([score for score in SCORES_TO_USE])
    resource_metrics = (
        "algorithm,readout,score_on,experiment,train_time," +
        f"pred_time_train,pred_time_test,pred_time_val,{scores}\n"
    )
    score_key = "Scores"
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
        for readout in scores["readout"].unique():
            already_calculated[readout] = scores.loc[
                scores["readout"] == readout,
                "algorithm"
            ].tolist()

    readouts = ["RPE_Final", "Lens_Final", "RPE_classes", "Lens_classes"]

    experiments = df["experiment"].unique().tolist()
    
    for readout in readouts:
        for classifier in CLASSIFIERS_TO_TEST_FULL:
            if readout in already_calculated:
                if classifier in already_calculated[readout]:
                    print(f"Skipping {classifier} for {readout} as it is already calculated")
                    continue
            if classifier in ["LabelPropagation", "LabelSpreading", "CategoricalNB"]:
                print(f"Skipping {classifier} due to memory reasons!")
                continue
            if readout == "RPE_classes" and classifier == "NuSVC":
                print("Skipping NuSVC for RPE classes")
                continue
            print(f"... running {classifier} on readout {readout}")
            for experiment in experiments:

                if CLASSIFIERS_TO_TEST_FULL[classifier]["allows_multi_class"]:
                    if CLASSIFIERS_TO_TEST_FULL[classifier]["multiprocessing"]:
                        clf = CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"](n_jobs = 16)
                    else:
                        clf = CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"]()
                else:
                    if CLASSIFIERS_TO_TEST_FULL[classifier]["scalable"] is False:
                        clf = MultiOutputClassifier(CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"]())
                    else:
                        clf = MultiOutputClassifier(CLASSIFIERS_TO_TEST_FULL[classifier]["classifier"](), n_jobs = 16)

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




