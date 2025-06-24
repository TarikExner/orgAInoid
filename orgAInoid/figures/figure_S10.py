import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import cv2

import pickle

import .figure_config as cfg
import .figure_utils as utils

def generate_subfigure_a(fig: Figure,
                         ax: Axes,
                         gs: SubplotSpec,
                         subfigure_label) -> None:
    ax.axis("off")
    utils._figure_label(ax, subfigure_label, x = -0.4)

    data = pd.read_csv("./figure_data/rpe_classes_classification.csv", index_col = [0])
    data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
    data["hours"] = data["loop"] / 2

    fig_sgs = gs.subgridspec(1,2)

    accuracy_plot_test = fig.add_subplot(fig_sgs[0])
    sns.lineplot(data = data[data["classifier"] == "Ensemble_test"], x = "hours", y = "F1", hue = "experiment", ax = accuracy_plot_test, errorbar = "se", palette = cfg.EXPERIMENT_LEGEND_CMAP)
    accuracy_plot_test.axhline(y = 0.25, xmin = 0.03, xmax = 0.97, linestyle = "--", color = "black")
    accuracy_plot_test.text(x = 40, y = 0.27, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")
    accuracy_plot_test.set_title("Prediction accuracy: RPE area\nin test organoids by deep learning", fontsize = cfg.TITLE_SIZE)
    accuracy_plot_test.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
    accuracy_plot_test.set_ylim(-0.1, 1.1)
    accuracy_plot_test.tick_params(**cfg.TICKPARAMS_PARAMS)
    accuracy_plot_test.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)    
    accuracy_plot_test.legend().remove()

    accuracy_plot_val = fig.add_subplot(fig_sgs[1])
    sns.lineplot(data = data[data["classifier"] == "Ensemble_val"], x = "hours", y = "F1", hue = "experiment", ax = accuracy_plot_val, errorbar = "se", palette = "tab20")
    accuracy_plot_val.axhline(y = 0.25, xmin = 0.03, xmax = 0.97, linestyle = "--", color = "black")
    accuracy_plot_val.text(x = 40, y = 0.27, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")
    accuracy_plot_val.set_title("Prediction accuracy: RPE area\nin validation organoids by deep learning", fontsize = cfg.TITLE_SIZE)
    accuracy_plot_val.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
    accuracy_plot_val.set_ylim(-0.1, 1.1)
    accuracy_plot_val.tick_params(**cfg.TICKPARAMS_PARAMS)
    accuracy_plot_val.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)    
    accuracy_plot_val.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left", fontsize = cfg.TITLE_SIZE)

    return

def generate_subfigure_b(fig: Figure,
                         ax: Axes,
                         gs: SubplotSpec,
                         subfigure_label) -> None:
    ax.axis("off")
    utils._figure_label(ax, subfigure_label, x = -0.4)

    fig_sgs = gs.subgridspec(4,2, hspace = 0, wspace = 0)

    with open("./figure_data/RPE_classes_classification_test_neural_net_confusion_matrices.data", "rb") as file:
        test = pickle.load(file)
        test[["class0", "class1", "class2", "class3"]] = test["percentage_matrix"].apply(
            lambda matrix: pd.Series(utils.classwise_confusion_matrix(matrix))
        )
    with open("./figure_data/RPE_classes_classification_val_neural_net_confusion_matrices.data", "rb") as file:
        validation = pickle.load(file)
        validation[["class0", "class1", "class2", "class3"]] = validation["percentage_matrix"].apply(
            lambda matrix: pd.Series(utils.classwise_confusion_matrix(matrix))
        )
    
    colors = cfg.CONF_MATRIX_COLORS

    for i, class_label in enumerate(["class0", "class1", "class2", "class3"]):
        c = i
        test_conf_matrix = fig.add_subplot(fig_sgs[i, 0])
        test_data = utils._preprocess_four_class_results(test, class_label)
        cumulative_base = np.zeros_like(test_data.values)
        
        for i in range(1, len(test_data.columns)):
            cumulative_base[:, i] = cumulative_base[:, i - 1] + test_data.iloc[:, i - 1].values
    
        for i, component in enumerate(test_data.columns):
            test_conf_matrix.fill_between(
                test_data.index,
                cumulative_base[:, i],
                cumulative_base[:, i] + test_data.iloc[:, i],
                color=colors[i],
                label=component,
                alpha=0.8,
            )
        
        handles, labels = test_conf_matrix.get_legend_handles_labels()
        handles, labels = handles[::-1], labels[::-1]
        labels = [cfg.CONF_MATRIX_LABEL_DICT[label] for label in labels]
        
        test_conf_matrix.set_ylabel(class_label, fontsize=cfg.AXIS_LABEL_SIZE)
        test_conf_matrix.legend(handles, labels, fontsize=cfg.AXIS_LABEL_SIZE-1, bbox_to_anchor=(1.01, 0.5), loc="center left")
        test_conf_matrix.set_ylim(0, 100)
        test_conf_matrix.set_xlim(0, 72)
        if c != 3:
            test_conf_matrix.set_xticklabels([])
            test_conf_matrix.set_yticklabels([])
            test_conf_matrix.tick_params(bottom = False, left = False)
        else:
            test_conf_matrix.tick_params(**cfg.TICKPARAMS_PARAMS)
            test_conf_matrix.set_yticklabels([])
            test_conf_matrix.tick_params(left = False)

        if c == 0:
            test_conf_matrix.set_title("Confusion matrices: RPE area\nin test organoids by deep learning", fontsize = cfg.TITLE_SIZE)
    
    for i, class_label in enumerate(["class0", "class1", "class2", "class3"]):
        c = i
        val_conf_matrix = fig.add_subplot(fig_sgs[i, 1])
        val_data = utils._preprocess_four_class_results(validation, class_label)
        cumulative_base = np.zeros_like(val_data.values)
        
        for i in range(1, len(val_data.columns)):
            cumulative_base[:, i] = cumulative_base[:, i - 1] + val_data.iloc[:, i - 1].values
    
        for i, component in enumerate(val_data.columns):
            val_conf_matrix.fill_between(
                val_data.index,
                cumulative_base[:, i],
                cumulative_base[:, i] + val_data.iloc[:, i],
                color=colors[i],
                label=component,
                alpha=0.8,
            )
        
        handles, labels = val_conf_matrix.get_legend_handles_labels()
        handles, labels = handles[::-1], labels[::-1]
        labels = [cfg.CONF_MATRIX_LABEL_DICT[label] for label in labels]
        
        val_conf_matrix.set_ylabel(class_label, fontsize=cfg.AXIS_LABEL_SIZE)
        val_conf_matrix.legend(handles, labels, fontsize=cfg.AXIS_LABEL_SIZE-1, bbox_to_anchor=(1.01, 0.5), loc="center left")
        val_conf_matrix.set_ylim(0, 100)
        val_conf_matrix.set_xlim(0, 72)
        if c != 3:
            val_conf_matrix.set_xticklabels([])
            val_conf_matrix.set_yticklabels([])
            val_conf_matrix.tick_params(bottom = False, left = False)
        else:
            val_conf_matrix.tick_params(**cfg.TICKPARAMS_PARAMS)
            val_conf_matrix.set_yticklabels([])
            val_conf_matrix.tick_params(left = False)

        if c == 0:
            val_conf_matrix.set_title("Confusion matrices: RPE area\nin validation organoids by deep learning", fontsize = cfg.TITLE_SIZE)

    return

def generate_subfigure_c(fig: Figure,
                         ax: Axes,
                         gs: SubplotSpec,
                         subfigure_label) -> None:
    ax.axis("off")
    utils._figure_label(ax, subfigure_label, x = -0.4)

    data = pd.read_csv("./figure_data/lens_classes_classification.csv", index_col = [0])
    data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
    data["hours"] = data["loop"] / 2

    fig_sgs = gs.subgridspec(1,2)

    accuracy_plot_test = fig.add_subplot(fig_sgs[0])
    sns.lineplot(data = data[data["classifier"] == "Ensemble_test"], x = "hours", y = "F1", hue = "experiment", ax = accuracy_plot_test, errorbar = "se", palette = cfg.EXPERIMENT_LEGEND_CMAP)
    accuracy_plot_test.axhline(y = 0.25, xmin = 0.03, xmax = 0.97, linestyle = "--", color = "black")
    accuracy_plot_test.text(x = 40, y = 0.27, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")
    accuracy_plot_test.set_title("Prediction accuracy: Lens sizes\nin test organoids by deep learning", fontsize = cfg.TITLE_SIZE)
    accuracy_plot_test.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
    accuracy_plot_test.set_ylim(-0.1, 1.1)
    accuracy_plot_test.tick_params(**cfg.TICKPARAMS_PARAMS)
    accuracy_plot_test.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)    
    accuracy_plot_test.legend().remove()

    accuracy_plot_val = fig.add_subplot(fig_sgs[1])
    sns.lineplot(data = data[data["classifier"] == "Ensemble_val"], x = "hours", y = "F1", hue = "experiment", ax = accuracy_plot_val, errorbar = "se", palette = "tab20")
    accuracy_plot_val.axhline(y = 0.25, xmin = 0.03, xmax = 0.97, linestyle = "--", color = "black")
    accuracy_plot_val.text(x = 40, y = 0.27, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")
    accuracy_plot_val.set_title("Prediction accuracy: Lens sizes\nin validation organoids by deep learning", fontsize = cfg.TITLE_SIZE)
    accuracy_plot_val.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
    accuracy_plot_val.set_ylim(-0.1, 1.1)
    accuracy_plot_val.tick_params(**cfg.TICKPARAMS_PARAMS)
    accuracy_plot_val.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)    
    accuracy_plot_val.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left", fontsize = cfg.TITLE_SIZE)

    return


def generate_subfigure_d(fig: Figure,
                         ax: Axes,
                         gs: SubplotSpec,
                         subfigure_label) -> None:
    ax.axis("off")
    utils._figure_label(ax, subfigure_label, x = -0.4)

    fig_sgs = gs.subgridspec(4,2, hspace = 0, wspace = 0)

    with open("./figure_data/Lens_classes_classification_test_neural_net_confusion_matrices.data", "rb") as file:
        test = pickle.load(file)
        test[["class0", "class1", "class2", "class3"]] = test["percentage_matrix"].apply(
            lambda matrix: pd.Series(utils.classwise_confusion_matrix(matrix))
        )
    with open("./figure_data/Lens_classes_classification_val_neural_net_confusion_matrices.data", "rb") as file:
        validation = pickle.load(file)
        validation[["class0", "class1", "class2", "class3"]] = validation["percentage_matrix"].apply(
            lambda matrix: pd.Series(utils.classwise_confusion_matrix(matrix))
        )
    
    colors = cfg.CONF_MATRIX_COLORS

    for i, class_label in enumerate(["class0", "class1", "class2", "class3"]):
        c = i
        test_conf_matrix = fig.add_subplot(fig_sgs[i, 0])
        test_data = utils._preprocess_four_class_results(test, class_label)
        cumulative_base = np.zeros_like(test_data.values)
        
        for i in range(1, len(test_data.columns)):
            cumulative_base[:, i] = cumulative_base[:, i - 1] + test_data.iloc[:, i - 1].values
    
        for i, component in enumerate(test_data.columns):
            test_conf_matrix.fill_between(
                test_data.index,
                cumulative_base[:, i],
                cumulative_base[:, i] + test_data.iloc[:, i],
                color=colors[i],
                label=component,
                alpha=0.8,
            )
        
        handles, labels = test_conf_matrix.get_legend_handles_labels()
        handles, labels = handles[::-1], labels[::-1]
        labels = [cfg.CONF_MATRIX_LABEL_DICT[label] for label in labels]
        
        test_conf_matrix.set_ylabel(class_label, fontsize=cfg.AXIS_LABEL_SIZE)
        test_conf_matrix.legend(handles, labels, fontsize=cfg.AXIS_LABEL_SIZE-1, bbox_to_anchor=(1.01, 0.5), loc="center left")
        test_conf_matrix.set_ylim(0, 100)
        test_conf_matrix.set_xlim(0, 72)
        if c != 3:
            test_conf_matrix.set_xticklabels([])
            test_conf_matrix.set_yticklabels([])
            test_conf_matrix.tick_params(bottom = False, left = False)
        else:
            test_conf_matrix.tick_params(**cfg.TICKPARAMS_PARAMS)
            test_conf_matrix.set_yticklabels([])
            test_conf_matrix.tick_params(left = False)

        if c == 0:
            test_conf_matrix.set_title("Confusion matrices: Lens sizes\nin test organoids by deep learning", fontsize = cfg.TITLE_SIZE)
    
    for i, class_label in enumerate(["class0", "class1", "class2", "class3"]):
        c = i
        val_conf_matrix = fig.add_subplot(fig_sgs[i, 1])
        val_data = utils._preprocess_four_class_results(validation, class_label)
        cumulative_base = np.zeros_like(val_data.values)
        
        for i in range(1, len(val_data.columns)):
            cumulative_base[:, i] = cumulative_base[:, i - 1] + val_data.iloc[:, i - 1].values
    
        for i, component in enumerate(val_data.columns):
            val_conf_matrix.fill_between(
                val_data.index,
                cumulative_base[:, i],
                cumulative_base[:, i] + val_data.iloc[:, i],
                color=colors[i],
                label=component,
                alpha=0.8,
            )
        
        handles, labels = val_conf_matrix.get_legend_handles_labels()
        handles, labels = handles[::-1], labels[::-1]
        labels = [cfg.CONF_MATRIX_LABEL_DICT[label] for label in labels]
        
        val_conf_matrix.set_ylabel(class_label, fontsize=cfg.AXIS_LABEL_SIZE)
        val_conf_matrix.legend(handles, labels, fontsize=cfg.AXIS_LABEL_SIZE-1, bbox_to_anchor=(1.01, 0.5), loc="center left")
        val_conf_matrix.set_ylim(0, 100)
        val_conf_matrix.set_xlim(0, 72)
        if c != 3:
            val_conf_matrix.set_xticklabels([])
            val_conf_matrix.set_yticklabels([])
            val_conf_matrix.tick_params(bottom = False, left = False)
        else:
            val_conf_matrix.tick_params(**cfg.TICKPARAMS_PARAMS)
            val_conf_matrix.set_yticklabels([])
            val_conf_matrix.tick_params(left = False)

        if c == 0:
            val_conf_matrix.set_title("Confusion matrices: Lens sizes\nin validation organoids by deep learning", fontsize = cfg.TITLE_SIZE)
    
        

    return

if __name__ == "__main__":
    fig = plt.figure(layout = "constrained",
                     figsize = (cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 6,
                  nrows = 4,
                  figure = fig,
                  height_ratios = [1,1.3,1,1.3])
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

    plt.savefig("./prefinal_figures/FigureS10.pdf", dpi = 300, bbox_inches = "tight")
    plt.savefig("./prefinal_figures/FigureS10.png", dpi = 300, bbox_inches = "tight")
    plt.show()
