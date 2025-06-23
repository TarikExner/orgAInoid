import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import cv2

from matplotlib.ticker import MultipleLocator

import .figure_config as cfg
import .figure_utils as utils

def generate_subfigure_a(fig: Figure,
                         ax: Axes,
                         gs: GridSpec,
                         subfigure_label) -> None:
    """Will contain the experimental overview sketch"""
    ax.axis("off")
    utils._figure_label(ax, subfigure_label, x = -0.4)

    fig_sgs = gs.subgridspec(1,1)

    sketch = fig.add_subplot(fig_sgs[0])
    utils._prep_image_axis(sketch)
    img = cv2.imread('./sketches/Figure_3.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sketch.imshow(img)
    return

def generate_subfigure_b(fig: Figure,
                         ax: Axes,
                         gs: GridSpec,
                         subfigure_label) -> None:
    """Contains the raw values of RPE/Lens over all organoids"""
    ax.axis("off")
    utils._figure_label(ax, subfigure_label, x = -0.4)

    data = pd.read_csv("./figure_data/rpe_classification.csv", index_col = [0])
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
    labels = ["CNN (image data): Validation", "CNN (image data): Test", "Random Forest (morphometrics): Validation", "Random Forest (morphometrics): Test", "Expert prediction"]
    accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE)
    accuracy_plot.set_title("Prediction accuracy: Emergence of RPE", fontsize = cfg.TITLE_SIZE)
    accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
    accuracy_plot.set_ylim(0.18, 1.099)
    accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
    accuracy_plot.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)
    accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
    return

def generate_subfigure_c(fig: Figure,
                         ax: Axes,
                         gs: GridSpec,
                         subfigure_label) -> None:
    """Contains the raw values of RPE/Lens over all organoids"""
    ax.axis("off")
    utils._figure_label(ax, subfigure_label, x = -0.4)

    data = pd.read_csv("./figure_data/lens_classification.csv", index_col = [0])
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
    labels = ["CNN (image data): Validation", "CNN (image data): Test", "QDA (morphometrics): Validation", "QDA (morphometrics): Test", "Expert prediction"]
    accuracy_plot.legend(handles, labels, loc = "lower right", fontsize = cfg.TITLE_SIZE)
    accuracy_plot.set_title("Prediction accuracy: Emergence of Lenses", fontsize = cfg.TITLE_SIZE)
    accuracy_plot.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
    accuracy_plot.set_ylim(0.18, 1.149)
    accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
    accuracy_plot.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)
    accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
    return

if __name__ == "__main__":
    fig = plt.figure(layout = "constrained",
                     figsize = (cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 6,
                  nrows = 3,
                  figure = fig,
                  height_ratios = [0.6,1,1])
    a_coords = gs[0,:]
    b_coords = gs[1,:]
    c_coords = gs[2,:]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")

    plt.savefig("./prefinal_figures/Figure3.pdf", dpi = 300, bbox_inches = "tight")
    plt.savefig("./prefinal_figures/Figure3.png", dpi = 300, bbox_inches = "tight")
    plt.show()


def figure_3_generation():
    pass
