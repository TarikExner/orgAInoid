import os
import numpy as np
import pandas as pd
import time
import gc

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from ._classifier_scoring import SCORES_TO_USE, write_to_scores, score_classifier
from .models import CLASSIFIERS_TO_TEST_FULL


def _one_hot_encode_labels(labels_array: np.ndarray) -> np.ndarray:
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels_array)
    
    onehot_encoder = OneHotEncoder()
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    classification = onehot_encoder.fit_transform(integer_encoded).toarray()
    return classification

def _filter_wells(df: pd.DataFrame,
                  combinations: np.ndarray,
                  columns: list[str]):
    combinations_df = pd.DataFrame(combinations, columns=columns)
    return df.merge(combinations_df, on=columns, how='inner')

def _apply_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_to_match = ["experiment", "well"]
    unique_well_per_exp = df[cols_to_match].drop_duplicates().reset_index(drop = True).to_numpy()
    train_wells, test_wells = train_test_split(unique_well_per_exp, test_size = 0.1, random_state = 187)

    assert isinstance(train_wells, np.ndarray)
    assert isinstance(test_wells, np.ndarray)

    train_df = _filter_wells(df, train_wells, ["experiment", "well"])
    test_df = _filter_wells(df, test_wells, ["experiment", "well"])

    return train_df, test_df

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

    write_to_scores(resource_metrics,
                    output_dir = output_dir,
                    key = score_key,
                    init = True)

    readouts = ["RPE_Final", "Lens_Final", "RPE_classes", "Lens_classes"]

    experiments = df["experiment"].unique().tolist()
    
    for readout in readouts:
        for classifier in CLASSIFIERS_TO_TEST_FULL:
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




