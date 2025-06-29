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

from .figure_data_generation import get_classification_f1_data

def _generate_main_figure(rpe_f1: pd.DataFrame,
                          lens_f1: pd.DataFrame,
                          rpe_classes_f1: pd.DataFrame,
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

        data = rpe_f1
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
        # labels = ["CNN (image data): Validation", "CNN (image data): Test", "Random Forest (morphometrics): Validation", "Random Forest (morphometrics): Test", "Expert prediction"]
        accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_title("Prediction accuracy: Emergence of RPE", fontsize = cfg.TITLE_SIZE)
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

        data = lens_f1
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
        # labels = ["CNN (image data): Validation", "CNN (image data): Test", "QDA (morphometrics): Validation", "QDA (morphometrics): Test", "Expert prediction"]
        accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_title("Prediction accuracy: Emergence of Lenses", fontsize = cfg.TITLE_SIZE)
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

        data = rpe_classes_f1
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
        # labels = ["CNN (image data): Validation", "CNN (image data): Test", "HGBC (morphometrics): Validation", "HGBC (morphometrics): Test", "Expert prediction"]
        accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_title("Prediction accuracy: RPE area", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_ylim(0.03, 0.99)
        accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
        return

    def generate_subfigure_d(fig: Figure,
                             ax: Axes,
                             gs: SubplotSpec,
                             subfigure_label) -> None:
        """Contains the raw values of RPE/Lens over all organoids"""
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.4)

        data = lens_classes_f1
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
        # labels = ["CNN (image data): Validation", "CNN (image data): Test", "QDA (morphometrics): Validation", "QDA (morphometrics): Test", "Expert prediction"]
        accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE)
        accuracy_plot.set_title("Prediction accuracy: Lens sizes", fontsize = cfg.TITLE_SIZE)
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
    d_coords = gs[3,:]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")
    generate_subfigure_d(fig, fig_d, d_coords, "D")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.png")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

    return

def figure_3_reviewer_generation(sketch_dir: str,
                                 figure_output_dir: str,
                                 raw_data_dir: str,
                                 morphometrics_dir: str,
                                 hyperparameter_dir: str,
                                 rpe_classification_dir_sum: str,
                                 lens_classification_dir_sum: str,
                                 rpe_classes_classification_dir_sum: str,
                                 lens_classes_classification_dir_sum: str,
                                 rpe_classification_dir_max: str,
                                 lens_classification_dir_max: str,
                                 rpe_classes_classification_dir_max: str,
                                 lens_classes_classification_dir_max: str,
                                 figure_data_dir: str,
                                 evaluator_results_dir: str,
                                 **kwargs) -> None:
    rpe_final_f1s = get_classification_f1_data(
        readout = "RPE_Final",
        output_dir = figure_data_dir,
        proj = "sum",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = rpe_classification_dir_sum,
        baseline_dir = None,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    lens_final_f1s = get_classification_f1_data(
        readout = "Lens_Final",
        output_dir = figure_data_dir,
        proj = "sum",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = lens_classification_dir_sum,
        baseline_dir = None,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    rpe_classes_f1 = get_classification_f1_data(
        readout = "RPE_classes",
        output_dir = figure_data_dir,
        proj = "sum",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = rpe_classes_classification_dir_sum,
        baseline_dir = None,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    lens_classes_f1 = get_classification_f1_data(
        readout = "Lens_classes",
        output_dir = figure_data_dir,
        proj = "sum",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = lens_classes_classification_dir_sum,
        baseline_dir = None,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    _generate_main_figure(rpe_f1 = rpe_final_f1s,
                          lens_f1 = lens_final_f1s,
                          rpe_classes_f1 = rpe_classes_f1,
                          lens_classes_f1 = lens_classes_f1,
                          figure_output_dir = figure_output_dir,
                          sketch_dir = sketch_dir,
                          figure_name = "Reviewer_Figure_3")

    rpe_final_f1s = get_classification_f1_data(
        readout = "RPE_Final",
        output_dir = figure_data_dir,
        proj = "max",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = rpe_classification_dir_max,
        baseline_dir = None,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    lens_final_f1s = get_classification_f1_data(
        readout = "Lens_Final",
        output_dir = figure_data_dir,
        proj = "max",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = lens_classification_dir_max,
        baseline_dir = None,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    rpe_classes_f1 = get_classification_f1_data(
        readout = "RPE_classes",
        output_dir = figure_data_dir,
        proj = "max",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = rpe_classes_classification_dir_max,
        baseline_dir = None,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    lens_classes_f1 = get_classification_f1_data(
        readout = "Lens_classes",
        output_dir = figure_data_dir,
        proj = "max",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = lens_classes_classification_dir_max,
        baseline_dir = None,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    _generate_main_figure(rpe_f1 = rpe_final_f1s,
                          lens_f1 = lens_final_f1s,
                          rpe_classes_f1 = rpe_classes_f1,
                          lens_classes_f1 = lens_classes_f1,
                          figure_output_dir = figure_output_dir,
                          sketch_dir = sketch_dir,
                          figure_name = "Reviewer_Figure_4")
