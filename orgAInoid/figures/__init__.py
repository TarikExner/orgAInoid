from .figure_1 import figure_1_generation
from .figure_2 import figure_2_generation
from .figure_3 import figure_3_generation
from .figure_4 import figure_4_generation
from .figure_5 import figure_5_generation

from .figure_S1 import figure_S1_generation
from .figure_S2 import figure_S2_generation
from .figure_S3 import figure_S3_generation
from .figure_S4 import figure_S4_generation
from .figure_S5 import figure_S5_generation
from .figure_S6 import figure_S6_generation
from .figure_S7 import figure_S7_generation
from .figure_S8 import figure_S8_generation
from .figure_S9 import figure_S9_generation
from .figure_S10 import figure_S10_generation
from .figure_S11 import figure_S11_generation
from .figure_S12 import figure_S12_generation
from .figure_S13 import figure_S13_generation
from .figure_S14 import figure_S14_generation
from .figure_S15 import figure_S15_generation
from .figure_S16 import figure_S16_generation
from .figure_S17 import figure_S17_generation
from .figure_S18 import figure_S18_generation
from .figure_S19 import figure_S19_generation
from .figure_S20 import figure_S20_generation
from .figure_S21 import figure_S21_generation
from .figure_S22 import figure_S22_generation
from .figure_S23 import figure_S23_generation
from .figure_S24 import figure_S24_generation
from .figure_S25 import figure_S25_generation
from .figure_S26 import figure_S26_generation
from .figure_S27 import figure_S27_generation
from .figure_S28 import figure_S28_generation
from .figure_S29 import figure_S29_generation
# from .figure_S30 import figure_S30_generation
from .figure_S31 import figure_S31_generation
from .figure_S32 import figure_S32_generation
from .figure_S33 import figure_S33_generation
from .figure_S34 import figure_S34_generation
from .figure_S35 import figure_S35_generation
from .figure_S36 import figure_S36_generation
from .figure_S37 import figure_S37_generation
from .figure_S38 import figure_S38_generation
from .figure_S39 import figure_S39_generation

from .supp_file_S1 import supplementary_file_S1_generation

__all__ = [
    "figure_1_generation",
    "figure_2_generation",
    "figure_3_generation",
    "figure_4_generation",
    "figure_5_generation",

    "figure_S1_generation",
    "figure_S2_generation",
    "figure_S3_generation",
    "figure_S4_generation",
    "figure_S5_generation",
    "figure_S6_generation",
    "figure_S7_generation",
    "figure_S8_generation",
    "figure_S9_generation",
    "figure_S10_generation",
    "figure_S11_generation",
    "figure_S12_generation",
    "figure_S13_generation",
    "figure_S14_generation",
    "figure_S15_generation",
    "figure_S16_generation",
    "figure_S17_generation",
    "figure_S18_generation",
    "figure_S19_generation",
    "figure_S20_generation",
    "figure_S21_generation",
    "figure_S22_generation",
    "figure_S23_generation",
    "figure_S24_generation",
    "figure_S25_generation",
    "figure_S26_generation",
    "figure_S27_generation",
    "figure_S28_generation",
    "figure_S29_generation",
    # "figure_S30_generation",
    "figure_S31_generation",
    "figure_S32_generation",
    "figure_S33_generation",
    "figure_S34_generation",
    "figure_S35_generation",
    "figure_S36_generation",
    "figure_S37_generation",
    "figure_S38_generation",
    "figure_S39_generation",

    "supplementary_file_S1_generation"
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
    "classifier_results_dir": "../shape_analysis/results",

    "rpe_classification_dir": "../classification/experiment_32",
    "lens_classification_dir": "../classification/experiment_33",
    "rpe_classes_classification_dir": "../classification/experiment_34",
    "lens_classes_classification_dir": "../classification/experiment_35",

    "rpe_baseline_dir": "../classification/experiment_36_RPE_baseline",
    "lens_baseline_dir": "../classification/experiment_37_Lens_baseline",
    "rpe_classes_baseline_dir": "../classification/experiment_38_RPEC_baseline",
    "lens_classes_baseline_dir": "../classification/experiment_39_LensC_baseline",
    "morph_classes_experiment_dir": "../classification/experiment_48_Morph_classes",

    "rpe_classification_dir_sum": "../classification/experiment_40_RPE_Final_ZSUM",
    "lens_classification_dir_sum": "../classification/experiment_41_Lens_Final_ZSUM",
    "rpe_classes_classification_dir_sum": "../classification/experiment_42_RPE_classes_Final_ZSUM",
    "lens_classes_classification_dir_sum": "../classification/experiment_43_Lens_classes_Final_ZSUM",
    "morph_classes_experiment_dir_sum": "../classification/experiment_49_Morph_classes_ZSUM",

    "rpe_classification_dir_max": "../classification/experiment_44_RPE_Final_ZMAX",
    "lens_classification_dir_max": "../classification/experiment_45_Lens_Final_ZMAX",
    "rpe_classes_classification_dir_max": "../classification/experiment_46_RPE_classes_ZMAX",
    "lens_classes_classification_dir_max": "../classification/experiment_47_Lens_classes_ZMAX",
    "morph_classes_experiment_dir_max": "../classification/experiment_50_Morph_classes_ZMAX",

    "saliency_input_dir": "../classification/saliencies/results",
}

DESCRIPTIONS = {
    "Supplementary_Figure_S1":
        "original Supp Fig S1 showing Supplementary data for Figure 1",
    "Supplementary_Figure_S2":
        "original Supp Fig S2 showing Supplementary data for Figure 2, distances of no-Wnt-organoids",
    "Supplementary_Figure_S3":
        "jaccard neighbor data on organoid distances",
    "Supplementary_Figure_S4":
        "original Supplementary Figure S3 showing the classifier comp. for RPE Final and Lens final",
    "Supplementary_Figure_S5":
        "SUM IMAGES: Supplementary Figure S3 showing the classifier comp. for RPE Final and Lens final",
    "Supplementary_Figure_S6":
        "MAX IMAGES: Supplementary Figure S3 showing the classifier comp. for RPE Final and Lens final",




        
}

def generate_final_figures():
    figure_1_generation(**DIRECTORIES)
    figure_2_generation(**DIRECTORIES)
    figure_3_generation(**DIRECTORIES)
    figure_4_generation(**DIRECTORIES)
    figure_5_generation(**DIRECTORIES)

    figure_S1_generation(**DIRECTORIES)
    figure_S2_generation(**DIRECTORIES)
    figure_S3_generation(**DIRECTORIES)
    figure_S4_generation(**DIRECTORIES)
    figure_S5_generation(**DIRECTORIES)
    figure_S6_generation(**DIRECTORIES)
    figure_S7_generation(**DIRECTORIES)
    figure_S8_generation(**DIRECTORIES)
    figure_S9_generation(**DIRECTORIES)
    figure_S10_generation(**DIRECTORIES)
    figure_S11_generation(**DIRECTORIES)
    figure_S12_generation(**DIRECTORIES)
    figure_S13_generation(**DIRECTORIES)
    figure_S14_generation(**DIRECTORIES)
    figure_S15_generation(**DIRECTORIES)
    figure_S16_generation(**DIRECTORIES)
    figure_S17_generation(**DIRECTORIES)
    figure_S18_generation(**DIRECTORIES)
    figure_S19_generation(**DIRECTORIES)
    figure_S20_generation(**DIRECTORIES)
    figure_S21_generation(**DIRECTORIES)
    figure_S22_generation(**DIRECTORIES)
    figure_S23_generation(**DIRECTORIES)
    figure_S24_generation(**DIRECTORIES)
    figure_S25_generation(**DIRECTORIES)
    figure_S26_generation(**DIRECTORIES)
    figure_S27_generation(**DIRECTORIES)
    figure_S28_generation(**DIRECTORIES)
    figure_S29_generation(**DIRECTORIES)
    # figure_S30_generation(**DIRECTORIES)
    figure_S31_generation(**DIRECTORIES)
    figure_S32_generation(**DIRECTORIES)
    figure_S33_generation(**DIRECTORIES)
    figure_S34_generation(**DIRECTORIES)
    figure_S35_generation(**DIRECTORIES)
    figure_S36_generation(**DIRECTORIES)
    figure_S37_generation(**DIRECTORIES)
    figure_S38_generation(**DIRECTORIES)
    figure_S39_generation(**DIRECTORIES)

    supplementary_file_S1_generation(**DIRECTORIES)



