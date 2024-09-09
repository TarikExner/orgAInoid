from sklearn.metrics import confusion_matrix
import os
import numpy as np
from sklearn.metrics import (balanced_accuracy_score,
                             average_precision_score,
                             brier_score_loss,
                             f1_score,
                             jaccard_score,
                             confusion_matrix,
                             cohen_kappa_score,
                             matthews_corrcoef)
from typing import Optional

def _get_accuracy(conf_matrix: np.ndarray) -> float:
     tp = _get_tp(conf_matrix)
     tn = _get_tn(conf_matrix)
     fp = _get_fp(conf_matrix)
     fn = _get_fn(conf_matrix)
     return (tp + tn) / (tp + tn + fp + fn)

def _get_f1_score(conf_matrix):
     precision = _get_precision(conf_matrix)
     recall = _get_recall(conf_matrix)
     return (2*(precision * recall)) / (precision + recall)

def _get_precision(conf_matrix: np.ndarray) -> float:
     tp = _get_tp(conf_matrix)
     return tp / (tp+_get_fp(conf_matrix))

def _get_recall(conf_matrix: np.ndarray) -> float:
     tp = _get_tp(conf_matrix)
     return tp / (tp+_get_fn(conf_matrix))

def _get_tn(conf_matrix: np.ndarray) -> float:
     return conf_matrix[0]

def _get_fp(conf_matrix: np.ndarray) -> float:
     return conf_matrix[1]

def _get_fn(conf_matrix: np.ndarray) -> float:
     return conf_matrix[2]

def _get_tp(conf_matrix: np.ndarray) -> float:
     return conf_matrix[3]

def _get_fpr(conf_matrix: np.ndarray) -> float:
    t_n, f_p, f_n, t_p = conf_matrix
    return f_p / (f_p + t_n)

def _get_fnr(conf_matrix: np.ndarray) -> float:
    t_n, f_p, f_n, t_p = conf_matrix
    return f_n / (t_p + f_n)

def _get_tnr(conf_matrix: np.ndarray) -> float:
    t_n, f_p, f_n, t_p = conf_matrix
    return t_n / (t_n + f_p)

def _get_fpv(conf_matrix: np.ndarray) -> float:
    t_n, f_p, f_n, t_p = conf_matrix
    return t_n / (t_n + f_n)

def _get_fdr(conf_matrix: np.ndarray) -> float:
    t_n, f_p, f_n, t_p = conf_matrix
    return f_p / (t_p + f_p) 

SCORES_TO_USE = {
    "accuracy_score": _get_accuracy,
    "balanced_accuracy_score": balanced_accuracy_score,
    #"top_k_accuracy_score": top_k_accuracy_score,
    "average_precision_score": average_precision_score,
    "brier_score_loss": brier_score_loss,
    "f1_score": _get_f1_score,
    #"log_loss": log_loss, ## if only one class is present it fails
    "precision_score": _get_precision, # ppv
    "recall_score": _get_recall, # tpr
    "jaccard_score": jaccard_score,
    "tn": _get_tn,
    "fp": _get_fp,
    "fn": _get_fn,
    "tp": _get_tp,
    "fpr": _get_fpr,
    "fnr": _get_fnr,
    "tnr": _get_tnr,
    "fpv": _get_fpv,
    "fdr": _get_fdr,
    #"fbeta": fbeta_score,
    "cohen_kappa_test": cohen_kappa_score,
    "matthews_test": matthews_corrcoef
}

def score_classifier(true_arr: np.ndarray,
                     pred_arr: np.ndarray):
    if true_arr.shape[0] != 0:
        conf_matrix = confusion_matrix(true_arr, pred_arr, labels = [0, 1]).ravel()
        test_scores = [str(SCORES_TO_USE[score](true_arr, pred_arr))
                       if score not in ["tn", "fp", "fn", "tp",
                                        "fpr", "fnr", "tnr", "fpv", "fdr",
                                        "precision_score", "recall_score", "f1_score",
                                        "accuracy_score"] else
                       str(SCORES_TO_USE[score](conf_matrix))
                       for score in SCORES_TO_USE]
    else: # y_test is empty
        test_scores = ["-1.0" for _ in SCORES_TO_USE]
    new_line = ["\n"]
    return test_scores + new_line 

def write_to_scores(string_to_add,
                    output_dir: str,
                    key: str,
                    init: bool = False):
    if init:
        with open(os.path.join(output_dir, f"{key}.log"), "w") as score_file:
                score_file.write(string_to_add)
 
    else:
        with open(os.path.join(output_dir, f"{key}.log"), "a") as score_file:
                  score_file.write(string_to_add)
    return
