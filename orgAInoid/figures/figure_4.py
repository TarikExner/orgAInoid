import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib.ticker import MultipleLocator

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import get_classification_f1_data


def _generate_main_figure(rpe_classes_f1: pd.DataFrame,
                          lens_classes_f1: pd.DataFrame,
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

        data = rpe_classes_f1

        # preprocessing:
        data.loc[data["classifier"].str.contains("Baseline_Morphometrics"), "classifier"] = "Baseline_Morphometrics"
        data.loc[data["classifier"].str.contains("Baseline_Ensemble"), "classifier"] = "Baseline_Ensemble"

        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1,1)

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(data = data, x = "hours", y = "F1", hue = "classifier", ax = accuracy_plot, errorbar = "se")

        accuracy_plot.axhline(y = 0.25, xmin = 0.03, xmax = 0.62, linestyle = "--", color = "black")
        accuracy_plot.text(x = 32, y = 0.27, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")

        RPE_prediction_cutoff = 26/2
        RPE_visibility_cutoff = 96/2
        accuracy_plot.annotate(
            "Confident Deep\nLearning Predictions",
            xy=(RPE_prediction_cutoff, 0.77),
            xytext=(RPE_prediction_cutoff, 0.87),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha='center'
        )

        accuracy_plot.annotate(
            "Confident RPE visibility",
            xy=(RPE_visibility_cutoff, 0.77),
            xytext=(RPE_visibility_cutoff, 0.87),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha='center'
        )

        
        handles, labels = accuracy_plot.get_legend_handles_labels()
        labels_dict = {
            # we switch nomenclature for test and val sets
            "Morphometrics_test": "HGBC (morphometrics): Validation",
            "Morphometrics_val": "HGBC (morphometrics): Test",
            "Ensemble_test": "CNN (image data): Validation",
            "Ensemble_val": "CNN (image data): Test",
            "human": "Expert prediction",
            "Baseline_Morphometrics": "HGBC (morphometrics): Baseline",
            "Baseline_Ensemble": "CNN (image data): Baseline"
        }
        labels = [labels_dict[label] for label in labels]

        accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE, ncols = 2)
        accuracy_plot.set_title("Prediction accuracy: RPE area", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylim(0.03, 0.99)
        accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
        return

    def generate_subfigure_b(fig: Figure,
                             ax: Axes,
                             gs: SubplotSpec,
                             subfigure_label) -> None:
        """Contains the raw values of RPE/Lens over all organoids"""
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.4)

        data = lens_classes_f1

        # preprocessing:
        data.loc[data["classifier"].str.contains("Baseline_Morphometrics"), "classifier"] = "Baseline_Morphometrics"
        data.loc[data["classifier"].str.contains("Baseline_Ensemble"), "classifier"] = "Baseline_Ensemble"

        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1,1)

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(data = data, x = "hours", y = "F1", hue = "classifier", ax = accuracy_plot, errorbar = "se")

        accuracy_plot.axhline(y = 0.25, xmin = 0.03, xmax = 0.62, linestyle = "--", color = "black")
        accuracy_plot.text(x = 32, y = 0.27, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")

        lens_prediction_cutoff = 14/2
        lens_visibility_cutoff = 86/2
        accuracy_plot.annotate(
            "Confident Deep\nLearning Predictions",
            xy=(lens_prediction_cutoff, 0.77),
            xytext=(lens_prediction_cutoff, 0.87),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha='center'
        )

        accuracy_plot.annotate(
            "Confident Lens visibility",
            xy=(lens_visibility_cutoff, 0.77),
            xytext=(lens_visibility_cutoff, 0.87),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha='center'
        )

        handles, labels = accuracy_plot.get_legend_handles_labels()
        labels_dict = {
            # we switch nomenclature for test and val sets
            "Morphometrics_test": "QDA (morphometrics): Validation",
            "Morphometrics_val": "QDA (morphometrics): Test",
            "Ensemble_test": "CNN (image data): Validation",
            "Ensemble_val": "CNN (image data): Test",
            "human": "Expert prediction",
            "Baseline_Morphometrics": "QDA (morphometrics): Baseline",
            "Baseline_Ensemble": "CNN (image data): Baseline"
        }
        labels = [labels_dict[label] for label in labels]

        accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE, ncols = 2)
        accuracy_plot.set_title("Prediction accuracy: Lens sizes", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylim(0.03, 0.99)
        accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
        return

    def _crop_array(arr):
        return arr[:,40:180, 40:180]

    def _clip_to_percentile(arr, lower: float = 0.05, upper: float = 0.995):
        lower_q = np.quantile(arr, lower)
        upper_q = np.quantile(arr, upper)
        return np.clip(arr, lower_q, upper_q)

    def generate_subfigure_c(fig: Figure,
                             ax: Axes,
                             gs: SubplotSpec,
                             subfigure_label) -> None:
        """Contains the raw values of RPE/Lens over all organoids"""
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.4)
        images = _crop_array(np.load("./figure_data/relevance_propagation/images.npy"))
        rpe_attr = _crop_array(np.load("./figure_data/relevance_propagation/attributions_RPE_Final.npy"))
        lens_attr = _crop_array(np.load("./figure_data/relevance_propagation/attributions_Lens_Final.npy"))
        rpe_classes_attr = _crop_array(np.load("./figure_data/relevance_propagation/attributions_RPE_classes.npy"))
        lens_classes_attr = _crop_array(np.load("./figure_data/relevance_propagation/attributions_Lens_classes.npy"))

        fig_sgs = gs.subgridspec(3,7)
        loops_to_show = [0,1,2]
        loop_idxs = [0, 30, 143]

        for loop in loops_to_show:
            img_plot = fig.add_subplot(fig_sgs[loop, 1])
            img_plot.imshow(images[loop_idxs[loop]], cmap = "Greys_r")
            img_plot.set_ylabel(f"hours: {int(np.ceil(loop_idxs[loop]/2))}", fontsize = cfg.TITLE_SIZE, labelpad = 0)
            
            rpe_attr_plot = fig.add_subplot(fig_sgs[loop, 2])
            rpe_attr_plot.imshow(
                _clip_to_percentile(rpe_attr[loop_idxs[loop]]), cmap = "hot"
            )
            
            lens_attr_plot = fig.add_subplot(fig_sgs[loop, 3])
            lens_attr_plot.imshow(
                _clip_to_percentile(lens_attr[loop_idxs[loop]]), cmap = "hot"
            )
            
            rpe_classes_attr_plot = fig.add_subplot(fig_sgs[loop, 4])
            rpe_classes_attr_plot.imshow(
                _clip_to_percentile(rpe_classes_attr[loop_idxs[loop]]), cmap = "hot"
            )
            
            lens_classes_attr_plot = fig.add_subplot(fig_sgs[loop, 5])
            lens_classes_attr_plot.imshow(
                _clip_to_percentile(lens_classes_attr[loop_idxs[loop]]), cmap = "hot"
            )

            for axis in [img_plot, rpe_attr_plot, lens_attr_plot, rpe_classes_attr_plot, lens_classes_attr_plot]:
                axis.set_xticklabels([])
                axis.set_yticklabels([])
                axis.tick_params(left = False, bottom = False)

            if loop == 0:
                img_plot.set_title("Brightfield", fontsize = cfg.TITLE_SIZE)
                rpe_attr_plot.set_title("RPE emergence", fontsize = cfg.TITLE_SIZE)
                lens_attr_plot.set_title("Lens emergence", fontsize = cfg.TITLE_SIZE)
                rpe_classes_attr_plot.set_title("RPE area", fontsize = cfg.TITLE_SIZE)
                lens_classes_attr_plot.set_title("Lens sizes", fontsize = cfg.TITLE_SIZE)
        return

    fig = plt.figure(layout = "constrained",
                     figsize = (cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 6,
                  nrows = 3,
                  figure = fig,
                  height_ratios = [1,1,0.8])
    a_coords = gs[0,:]
    b_coords = gs[1,:]
    c_coords = gs[2,:]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.png")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

def figure_4_generation(sketch_dir: str,
                        figure_output_dir: str,
                        raw_data_dir: str,
                        morphometrics_dir: str,
                        hyperparameter_dir: str,
                        rpe_classes_classification_dir: str,
                        lens_classes_classification_dir: str,
                        rpe_classes_baseline_dir: str,
                        lens_classes_baseline_dir: str,
                        figure_data_dir: str,
                        evaluator_results_dir: str,
                        **kwargs) -> None:
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
    _generate_main_figure(rpe_classes_f1 = rpe_classes_f1s,
                          lens_classes_f1 = lens_classes_f1s,
                          figure_output_dir = figure_output_dir,
                          sketch_dir = sketch_dir,
                          figure_name = "Figure_4")

