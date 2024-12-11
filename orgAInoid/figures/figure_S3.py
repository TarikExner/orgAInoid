import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import cv2

import .figure_config as cfg
import .figure_utils as utils

def generate_subfigure_a(fig: Figure,
                         ax: Axes,
                         gs: GridSpec,
                         subfigure_label) -> None:
    """Will contain the experimental overview sketch"""
    ax.axis("off")
    utils._figure_label(ax, subfigure_label, x = -0.3)
    fig_sgs = gs.subgridspec(1,1)

    readout = "RPE_Final"

    benchmark = pd.read_csv("./figure_data/classifier_comparison.log", index_col = False)
    benchmark["tuning"] = "not tuned"
    hyperparam_tuning = pd.read_csv("./figure_data/classifier_hyperparameter_tuning.log", index_col = False)
    hyperparam_tuning["tuning"] = "tuned"
    data = pd.concat([benchmark, hyperparam_tuning], axis = 0)
    data = data[data["readout"] == readout].copy()
    data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
    data["ALGORITHM"] = [f"{algorithm}: {tuning}" for algorithm, tuning in zip(data["algorithm"].tolist(), data["tuning"].tolist())]
    order = data[(data["score_on"] == "val") & (data["readout"] == readout)].groupby("ALGORITHM").median("f1_score").sort_values("f1_score", ascending = False).index.tolist()

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
    clf_comp_plot.set_title(f"Classifier performance for RPE emergence", fontsize = cfg.TITLE_SIZE)
    clf_comp_plot.set_ylabel("F1 Score", fontsize = cfg.AXIS_LABEL_SIZE)
    clf_comp_plot.set_xticklabels(clf_comp_plot.get_xticklabels(), ha = "right", rotation = 45, fontsize = cfg.AXIS_LABEL_SIZE)
    clf_comp_plot.set_xlabel("", fontsize = cfg.AXIS_LABEL_SIZE)
    clf_comp_plot.set_ylim(-0.03, 1.03)
    clf_comp_plot.legend(bbox_to_anchor = (1.01, 0.37), loc = "center left", fontsize = cfg.TITLE_SIZE)
    clf_comp_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
    return

def generate_subfigure_b(fig: Figure,
                         ax: Axes,
                         gs: GridSpec,
                         subfigure_label) -> None:
    """Will contain the experimental overview sketch"""
    ax.axis("off")
    utils._figure_label(ax, subfigure_label, x = -0.3)
    fig_sgs = gs.subgridspec(1,1)

    readout = "Lens_Final"

    benchmark = pd.read_csv("./figure_data/classifier_comparison.log", index_col = False)
    benchmark["tuning"] = "not tuned"
    hyperparam_tuning = pd.read_csv("./figure_data/classifier_hyperparameter_tuning.log", index_col = False)
    hyperparam_tuning["tuning"] = "tuned"
    data = pd.concat([benchmark, hyperparam_tuning], axis = 0)
    data = data[data["readout"] == readout].copy()
    data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
    data["ALGORITHM"] = [f"{algorithm}: {tuning}" for algorithm, tuning in zip(data["algorithm"].tolist(), data["tuning"].tolist())]
    order = data[(data["score_on"] == "val") & (data["readout"] == readout)].groupby("ALGORITHM").median("f1_score").sort_values("f1_score", ascending = False).index.tolist()

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
    clf_comp_plot.set_title(f"Classifier performance for lens emergence", fontsize = cfg.TITLE_SIZE)
    clf_comp_plot.set_ylabel("F1 Score", fontsize = cfg.AXIS_LABEL_SIZE)
    clf_comp_plot.set_xticklabels(clf_comp_plot.get_xticklabels(), ha = "right", rotation = 45, fontsize = cfg.AXIS_LABEL_SIZE)
    clf_comp_plot.set_xlabel("", fontsize = cfg.AXIS_LABEL_SIZE)
    clf_comp_plot.set_ylim(-0.03, 1.03)
    clf_comp_plot.legend(bbox_to_anchor = (1.01, 0.37), loc = "center left", fontsize = cfg.TITLE_SIZE)
    clf_comp_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
    return

if __name__ == "__main__":
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


    plt.savefig("./prefinal_figures/FigureS3.pdf", dpi = 300, bbox_inches = "tight")
    plt.savefig("./prefinal_figures/FigureS3.png", dpi = 300, bbox_inches = "tight")

    plt.show()
