import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib.ticker import MultipleLocator

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_utils import _generate_classification_results

def _generate_main_figure(morph_classes_normal: pd.DataFrame,
                          morph_classes_sum: pd.DataFrame,
                          morph_classes_max: pd.DataFrame,
                          figure_output_dir: str = "",
                          sketch_dir: str = "",
                          figure_name: str = ""):

    def generate_subfigure_a(fig: Figure,
                             ax: Axes,
                             gs: SubplotSpec,
                             subfigure_label) -> None:
        """Contains the raw values of RPE/Lens over all organoids"""
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.4)

        data = morph_classes_normal

        # preprocessing:
        data.loc[data["classifier"].str.contains("Baseline_Morphometrics"), "classifier"] = "Baseline_Morphometrics"
        data.loc[data["classifier"].str.contains("Baseline_Ensemble"), "classifier"] = "Baseline_Ensemble"

        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1,1)

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data = data,
            x = "hours",
            y = "F1",
            hue = "classifier",
            ax = accuracy_plot,
            errorbar = "se",
        )

        accuracy_plot.axhline(y = 0.25, xmin = 0.03, xmax = 0.30, linestyle = "--", color = "black")
        accuracy_plot.text(x = 0, y = 0.27, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")

        
        handles, labels = accuracy_plot.get_legend_handles_labels()
        projection = "single slice"
        labels_dict = {
            # we switch nomenclature for test and val sets
            "Morphometrics_test": "Decision Tree (morphometrics): Validation",
            "Morphometrics_val": "Decision Tree (morphometrics): Test",
            "Ensemble_test": "CNN (image data): Validation",
            "Ensemble_val": "CNN (image data): Test",
            "human": "Expert prediction",
            "Baseline_Morphometrics": "Decision Tree (morphometrics): Baseline",
            "Baseline_Ensemble": "CNN (image data): Baseline"
        }
        labels = [labels_dict[label] for label in labels]
        accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_title(f"Prediction accuracy: Morph classes\non image projection: {projection}", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.set_ylim(0.18, 1.01)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
        return

    fig = plt.figure(layout = "constrained",
                     figsize = (cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL / 2))
    gs = GridSpec(ncols = 6,
                  nrows = 1,
                  figure = fig)
    a_coords = gs[0,:]

    fig_a = fig.add_subplot(a_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.png")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

    return

def figure_3_reviewer_morph_classes_generation(sketch_dir: str,
                                               figure_output_dir: str,
                                               raw_data_dir: str,
                                               morphometrics_dir: str,
                                               hyperparameter_dir: str,
                                               morph_classes_experiment_dir: str,
                                               morph_classes_experiment_dir_sum: str,
                                               morph_classes_experiment_dir_max: str,
                                               figure_data_dir: str,
                                               evaluator_results_dir: str,
                                               **kwargs) -> None:
    morph_classes_normal, _, _ = _generate_classification_results(
        readout = "morph_classes",
        output_dir = figure_data_dir,
        proj = "",
        hyperparameter_dir = hyperparameter_dir,
        experiment_dir = morph_classes_experiment_dir,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
    )
    morph_classes_sum, _, _ = _generate_classification_results(
        readout = "morph_classes",
        output_dir = figure_data_dir,
        proj = "sum",
        hyperparameter_dir = hyperparameter_dir,
        experiment_dir = morph_classes_experiment_dir_sum,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
    )
    morph_classes_max, _, _ = _generate_classification_results(
        readout = "morph_classes",
        output_dir = figure_data_dir,
        proj = "max",
        hyperparameter_dir = hyperparameter_dir,
        experiment_dir = morph_classes_experiment_dir_max,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
    )
    _generate_main_figure(morph_classes_normal = morph_classes_normal,
                          morph_classes_sum = morph_classes_sum,
                          morph_classes_max = morph_classes_max,
                          figure_output_dir = figure_output_dir,
                          sketch_dir = sketch_dir,
                          figure_name = "Reviewer_Figure_5")
