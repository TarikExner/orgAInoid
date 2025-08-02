import os
import pandas as pd

from .figure_data_generation import run_f1_statistics, get_classification_f1_data


def supplementary_file_S1_generation(figure_output_dir: str,
                                     raw_data_dir: str,
                                     morphometrics_dir: str,
                                     hyperparameter_dir: str,
                                     rpe_classification_dir: str,
                                     lens_classification_dir: str,
                                     rpe_baseline_dir: str,
                                     lens_baseline_dir: str,
                                     rpe_classes_classification_dir: str,
                                     lens_classes_classification_dir: str,
                                     rpe_classes_baseline_dir: str,
                                     lens_classes_baseline_dir: str,
                                     figure_data_dir: str,
                                     evaluator_results_dir: str,
                                     **kwargs) -> pd.DataFrame:
    rpe_final_f1s = get_classification_f1_data(
        readout = "RPE_Final",
        output_dir = figure_data_dir,
        proj = "",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = rpe_classification_dir,
        baseline_dir = rpe_baseline_dir,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    lens_final_f1s = get_classification_f1_data(
        readout = "Lens_Final",
        output_dir = figure_data_dir,
        proj = "",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = lens_classification_dir,
        baseline_dir = lens_baseline_dir,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    rpe_classes_f1s = get_classification_f1_data(
        readout = "RPE_classes",
        output_dir = figure_data_dir,
        proj = "",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = rpe_classes_classification_dir,
        baseline_dir = rpe_classes_baseline_dir,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    lens_classes_f1s = get_classification_f1_data(
        readout = "Lens_classes",
        output_dir = figure_data_dir,
        proj = "",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = lens_classes_classification_dir,
        baseline_dir = lens_classes_baseline_dir,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )

    raw_data = {
        "RPE_Final": rpe_final_f1s,
        "Lens_Final": lens_final_f1s,
        "RPE_classes": rpe_classes_f1s,
        "Lens_classes": lens_classes_f1s
    }

    res = pd.concat(
        [run_f1_statistics(df, readout) for readout, df in raw_data.items()],
        axis = 0
    )
    classifier_dict = {
        # we switch nomenclature for test and val sets
        "Morphometrics_test": "Morphometrics_val",
        "Morphometrics_val": "Morphometrics_test",
        "Ensemble_test": "Ensemble_val",
        "Ensemble_val": "Ensemble_test",
        "human": "Expert_prediction",
        "Baseline_Morphometrics": "Baseline_Morphometrics",
        "Baseline_Ensemble": "Baseline_Ensemble"
    }
    res["group1"] = res["group1"].map(classifier_dict)
    res["group2"] = res["group2"].map(classifier_dict)

    res.to_csv(
        os.path.join(figure_output_dir, "Supplementary_File_1.csv"),
        index = False
    )
    return res
