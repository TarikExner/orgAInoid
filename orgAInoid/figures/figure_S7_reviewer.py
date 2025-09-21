import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import get_classifier_comparison

def _generate_main_figure(rpe_classes_res: pd.DataFrame,
                          lens_classes_res: pd.DataFrame,
                          proj: str,
                          figure_output_dir: str,
                          figure_name: str) -> None:

    def generate_subfigure_a(fig: Figure,
                             ax: Axes,
                             gs: SubplotSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.3)
        fig_sgs = gs.subgridspec(1,1)

        readout = "RPE_classes"

        data = rpe_classes_res
        data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
        order = data[
            (data["score_on"] == "val") &
            (data["readout"] == readout)
        ].groupby("ALGORITHM").median("f1_score").sort_values("f1_score", ascending = False).index.tolist()

        clf_comp_plot = fig.add_subplot(fig_sgs[0])

        plot_params = {
            "data":  data[(data["score_on"] == "val") & (data["readout"] == readout)],
            "x": "ALGORITHM",
            "y": "f1_score",
            "order": order,
            "ax": clf_comp_plot
        }
        sns.stripplot(**plot_params,
                      hue = "experiment",
                      jitter = 0.05,
                      s = 4,
                      linewidth = 0.4,
                      edgecolor = "black",
                      palette = cfg.EXPERIMENT_LEGEND_CMAP)
        sns.boxplot(**plot_params,
                    boxprops = dict(facecolor = "white"),
                    whis = (0,100),
                    linewidth = 1,
                    dodge = False)
        clf_comp_plot.set_title(f"Classifier performance for RPE classes\non image projection {proj}", fontsize = cfg.TITLE_SIZE)
        clf_comp_plot.set_ylabel("F1 Score", fontsize = cfg.AXIS_LABEL_SIZE)
        clf_comp_plot.set_xticklabels(clf_comp_plot.get_xticklabels(), ha = "right", rotation = 45, fontsize = cfg.AXIS_LABEL_SIZE)
        clf_comp_plot.set_xlabel("", fontsize = cfg.AXIS_LABEL_SIZE)
        clf_comp_plot.set_ylim(-0.03, 1.03)
        clf_comp_plot.legend(bbox_to_anchor = (1.01, 0.37), loc = "center left", fontsize = cfg.TITLE_SIZE)
        clf_comp_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        return

    def generate_subfigure_b(fig: Figure,
                             ax: Axes,
                             gs: SubplotSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.3)
        fig_sgs = gs.subgridspec(1,1)

        readout = "Lens_classes"
        data = lens_classes_res
        data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
        order = data[
            (data["score_on"] == "val") &
            (data["readout"] == readout)
        ].groupby("ALGORITHM").median("f1_score").sort_values("f1_score", ascending = False).index.tolist()

        clf_comp_plot = fig.add_subplot(fig_sgs[0])

        plot_params = {
            "data":  data[(data["score_on"] == "val") & (data["readout"] == readout)],
            "x": "ALGORITHM",
            "y": "f1_score",
            "order": order,
            "ax": clf_comp_plot
        }
        sns.stripplot(**plot_params,
                      hue = "experiment",
                      jitter = 0.05,
                      s = 4,
                      linewidth = 0.4,
                      edgecolor = "black",
                      palette = cfg.EXPERIMENT_LEGEND_CMAP)
        sns.boxplot(**plot_params,
                    boxprops = dict(facecolor = "white"),
                    whis = (0,100),
                    linewidth = 1,
                    dodge = False)
        clf_comp_plot.set_title(f"Classifier performance for lens classes\non image projection {proj}", fontsize = cfg.TITLE_SIZE)
        clf_comp_plot.set_ylabel("F1 Score", fontsize = cfg.AXIS_LABEL_SIZE)
        clf_comp_plot.set_xticklabels(clf_comp_plot.get_xticklabels(), ha = "right", rotation = 45, fontsize = cfg.AXIS_LABEL_SIZE)
        clf_comp_plot.set_xlabel("", fontsize = cfg.AXIS_LABEL_SIZE)
        clf_comp_plot.set_ylim(-0.03, 1.03)
        clf_comp_plot.legend(bbox_to_anchor = (1.01, 0.37), loc = "center left", fontsize = cfg.TITLE_SIZE)
        clf_comp_plot.tick_params(**cfg.TICKPARAMS_PARAMS)

        return

    fig = plt.figure(layout = "constrained",
                     figsize = (cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL*0.65))
    gs = GridSpec(ncols = 6,
                  nrows = 2,
                  figure = fig,
                  height_ratios = [1,1])
    a_coords = gs[0,:]
    b_coords = gs[1,:]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.png")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

    return

def figure_S7_reviewer_generation(classifier_results_dir: str,
                                  figure_data_dir: str,
                                  figure_output_dir: str,
                                  **kwargs) -> None:

    rpe_classes_clf = get_classifier_comparison(classifier_results_dir = classifier_results_dir,
                                                readout = "RPE_classes",
                                                proj = "ZSUM",
                                                output_dir = figure_data_dir)
    lens_classes_clf = get_classifier_comparison(classifier_results_dir = classifier_results_dir,
                                                 readout = "Lens_classes",
                                                 proj = "ZSUM",
                                                 output_dir = figure_data_dir)
    _generate_main_figure(rpe_classes_res = rpe_classes_clf,
                          lens_classes_res = lens_classes_clf,
                          proj = "SUM",
                          figure_output_dir = figure_output_dir,
                          figure_name = "Reviewer_Figure_9")

    rpe_classes_clf = get_classifier_comparison(classifier_results_dir = classifier_results_dir,
                                                readout = "RPE_classes",
                                                proj = "ZMAX",
                                                output_dir = figure_data_dir)
    lens_classes_clf = get_classifier_comparison(classifier_results_dir = classifier_results_dir,
                                                 readout = "Lens_classes",
                                                 proj = "ZMAX",
                                                 output_dir = figure_data_dir)
    _generate_main_figure(rpe_classes_res = rpe_classes_clf,
                          lens_classes_res = lens_classes_clf,
                          proj = "MAX",
                          figure_output_dir = figure_output_dir,
                          figure_name = "Reviewer_Figure_10")
