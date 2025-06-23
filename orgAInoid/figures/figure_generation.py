from . import (figure_1_generation,
               figure_2_generation,

               figure_S1_generation,
               figure_S2_generation,

               )

DIRECTORIES = {
    "annotations_dir": "../metadata",
    "morphometrics_dir": "../shape_analysis/results",
    "evaluator_results_dir": "../human_evaluation/evaluations",
    "figure_output_dir": "./final_figures",
    "figure_data_dir": "./figure_data",
    "sketch_dir": "./sketches",
    "microscopy_dir": "./microscopy_images",
}

def generate_final_figures():
    figure_1_generation(**DIRECTORIES)
    figure_2_generation(**DIRECTORIES)

    figure_S1_generation(**DIRECTORIES)
    figure_S2_generation(**DIRECTORIES)


if __name__ == "__main__":
    generate_final_figures()


