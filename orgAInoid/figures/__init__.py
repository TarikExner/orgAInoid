from .figure_1 import figure_1_generation
from .figure_2 import figure_2_generation
from .figure_2_reviewer import figure_2_reviewer_generation
from .figure_3 import figure_3_generation
from .figure_3_reviewer import figure_3_reviewer_generation

from .figure_S1 import figure_S1_generation
from .figure_S2 import figure_S2_generation



__all__ = [
    "figure_1_generation",
    "figure_2_generation",
    "figure_2_reviewer_generation",
    "figure_3_generation"
    "figure_3_reviewer_generation"

    "figure_S1_generation",
    "figure_S2_generation"
]

DIRECTORIES = {
    "annotations_dir": "../metadata",
    "morphometrics_dir": "../shape_analysis/results",
    "evaluator_results_dir": "../human_evaluation/evaluations",
    "figure_output_dir": "./final_figures",
    "figure_data_dir": "./figure_data",
    "sketch_dir": "./sketches",
    "microscopy_dir": "./microscopy_images",
    "raw_data_dir": "../classification/raw_data",
    "hyperparameter_dir": "../shape_analysis/results/best_params",

    "rpe_classification_dir": "../classification/experiment_32",
    "lens_classification_dir": "../classification/experiment_33",
    "rpe_classes_classification_dir": "../classification/experiment_34",
    "rpe_classes_classification_dir": "../classification/experiment_35"

    # "rpe_baseline_dir": "../classification/experiment_32",
    # "lens_classification_dir": "../classification/experiment_33",
    # "rpe_classes_classification_dir": "../classification/experiment_34",
    # "rpe_classes_classification_dir": "../classification/experiment_35"
}

def generate_final_figures():
    figure_1_generation(**DIRECTORIES)
    figure_2_generation(**DIRECTORIES)
    figure_2_reviewer_generation(**DIRECTORIES)
    figure_3_generation(**DIRECTORIES)
    figure_3_reviewer_generation(**DIRECTORIES)

    figure_S1_generation(**DIRECTORIES)
    figure_S2_generation(**DIRECTORIES)


