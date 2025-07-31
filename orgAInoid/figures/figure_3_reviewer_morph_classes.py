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
        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1,1)

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(data = data, x = "hours", y = "F1", hue = "classifier", ax = accuracy_plot, errorbar = "se")

        accuracy_plot.axhline(y = 0.5, xmin = 0.03, xmax = 0.97, linestyle = "--", color = "black")
        accuracy_plot.text(x = 120/2, y = 0.52, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")

        RPE_prediction_cutoff = 22/2
        RPE_visibility_cutoff = 96/2
        accuracy_plot.annotate(
            "Confident Deep Learning Predictions",
            xy=(RPE_prediction_cutoff, 0.95),
            xytext=(RPE_prediction_cutoff, 1.05),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha='center'
        )

        accuracy_plot.annotate(
            "Confident RPE visibility",
            xy=(RPE_visibility_cutoff, 0.95),
            xytext=(RPE_visibility_cutoff, 1.05),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha='center'
        )

        
        handles, labels = accuracy_plot.get_legend_handles_labels()
        projection = "single slice"
        # labels = ["CNN (image data): Validation", "CNN (image data): Test", "Random Forest (morphometrics): Validation", "Random Forest (morphometrics): Test", "Expert prediction"]
        accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_title(f"Prediction accuracy: Morph classes\non image projection {projection}", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.set_ylim(0.18, 1.099)
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

        data = morph_classes_sum
        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1,1)

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(data = data, x = "hours", y = "F1", hue = "classifier", ax = accuracy_plot, errorbar = "se")

        accuracy_plot.axhline(y = 0.5, xmin = 0.03, xmax = 0.97, linestyle = "--", color = "black")
        accuracy_plot.text(x = 120/2, y = 0.52, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")

        lens_prediction_cutoff = 14/2
        lens_visibility_cutoff = 86/2
        accuracy_plot.annotate(
            "Confident Deep\nLearning Predictions",
            xy=(lens_prediction_cutoff, 0.95),
            xytext=(lens_prediction_cutoff, 1.05),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha='center'
        )

        accuracy_plot.annotate(
            "Confident Lens visibility",
            xy=(lens_visibility_cutoff, 0.95),
            xytext=(lens_visibility_cutoff, 1.05),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha='center'
        )

        handles, labels = accuracy_plot.get_legend_handles_labels()
        projection = "sum"
        # labels = ["CNN (image data): Validation", "CNN (image data): Test", "QDA (morphometrics): Validation", "QDA (morphometrics): Test", "Expert prediction"]
        accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_title(f"Prediction accuracy: Emergence of Lenses\non image projection {projection}", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.set_ylim(0.18, 1.149)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
        return

    def generate_subfigure_c(fig: Figure,
                             ax: Axes,
                             gs: SubplotSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.4)

        data = morph_classes_sum
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
        projection = "max"
        # labels = ["CNN (image data): Validation", "CNN (image data): Test", "HGBC (morphometrics): Validation", "HGBC (morphometrics): Test", "Expert prediction"]
        accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_title(f"Prediction accuracy: RPE area\non image projection {projection}", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylim(0.03, 0.99)
        accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
        return


    fig = plt.figure(layout = "constrained",
                     figsize = (cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 6,
                  nrows = 4,
                  figure = fig,
                  height_ratios = [1,1,1,1])
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
