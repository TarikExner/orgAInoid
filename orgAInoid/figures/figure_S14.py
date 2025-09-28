import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import (get_classification_f1_data,
                                     create_confusion_matrix_frame)


def _generate_main_figure(rpe_f1_data: pd.DataFrame,
                          lens_f1_data: pd.DataFrame,
                          rpe_cnn_test_cm: pd.DataFrame,
                          rpe_cnn_val_cm: pd.DataFrame,
                          lens_cnn_test_cm: pd.DataFrame,
                          lens_cnn_val_cm: pd.DataFrame,
                          figure_output_dir: str,
                          figure_name: str):

    def generate_subfigure_a(fig: Figure,
                             ax: Axes,
                             gs: SubplotSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.4)

        data = rpe_f1_data
        data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1,2)

        accuracy_plot_test = fig.add_subplot(fig_sgs[0])
        sns.lineplot(data = data[data["classifier"] == "Ensemble_test"], x = "hours", y = "F1", hue = "experiment", ax = accuracy_plot_test, errorbar = "se", palette = cfg.EXPERIMENT_LEGEND_CMAP)
        accuracy_plot_test.axhline(y = 0.5, xmin = 0.03, xmax = 0.97, linestyle = "--", color = "black")
        accuracy_plot_test.text(x = 40, y = 0.52, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")
        accuracy_plot_test.set_title("Prediction accuracy: Emergence of RPE\nin validation organoids by deep learning", fontsize = cfg.TITLE_SIZE)
        accuracy_plot_test.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot_test.set_ylim(-0.1, 1.1)
        accuracy_plot_test.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot_test.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)    
        accuracy_plot_test.legend().remove()

        accuracy_plot_val = fig.add_subplot(fig_sgs[1])
        sns.lineplot(data = data[data["classifier"] == "Ensemble_val"], x = "hours", y = "F1", hue = "experiment", ax = accuracy_plot_val, errorbar = "se", palette = "tab20")
        accuracy_plot_val.axhline(y = 0.5, xmin = 0.03, xmax = 0.97, linestyle = "--", color = "black")
        accuracy_plot_val.text(x = 40, y = 0.52, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")
        accuracy_plot_val.set_title("Prediction accuracy: Emergence of RPE\nin test organoids by deep learning", fontsize = cfg.TITLE_SIZE)
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

        test_data = rpe_cnn_test_cm
        val_data = rpe_cnn_val_cm

        colors = cfg.CONF_MATRIX_COLORS
        
        fig_sgs = gs.subgridspec(1,2)

        test_conf_matrix = fig.add_subplot(fig_sgs[0])
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
        
        test_conf_matrix.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        test_conf_matrix.set_ylabel("Percentage", fontsize=cfg.AXIS_LABEL_SIZE)
        test_conf_matrix.legend(handles, labels, fontsize=cfg.AXIS_LABEL_SIZE, bbox_to_anchor=(1.01, 0.5), loc="center left")
        test_conf_matrix.set_ylim(0, 100)
        test_conf_matrix.set_xlim(0, 72)
        test_conf_matrix.tick_params(**cfg.TICKPARAMS_PARAMS)
        test_conf_matrix.set_title("Confusion matrix: Emergence of RPE\nin validation organoids by deep learning", fontsize = cfg.TITLE_SIZE)

        val_conf_matrix = fig.add_subplot(fig_sgs[1])
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
        
        val_conf_matrix.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        val_conf_matrix.set_ylabel("Percentage", fontsize=cfg.AXIS_LABEL_SIZE)
        val_conf_matrix.legend(handles, labels, fontsize=cfg.AXIS_LABEL_SIZE, bbox_to_anchor=(1.01, 0.5), loc="center left")
        val_conf_matrix.set_ylim(0, 100)
        val_conf_matrix.set_xlim(0, 72)
        val_conf_matrix.tick_params(**cfg.TICKPARAMS_PARAMS)
        val_conf_matrix.set_title("Confusion matrix: Emergence of RPE\nin test organoids by deep learning", fontsize = cfg.TITLE_SIZE)

        return

    def generate_subfigure_c(fig: Figure,
                             ax: Axes,
                             gs: SubplotSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.4)

        data = lens_f1_data
        data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1,2)

        accuracy_plot_test = fig.add_subplot(fig_sgs[0])
        sns.lineplot(data = data[data["classifier"] == "Ensemble_test"], x = "hours", y = "F1", hue = "experiment", ax = accuracy_plot_test, errorbar = "se", palette = cfg.EXPERIMENT_LEGEND_CMAP)
        accuracy_plot_test.axhline(y = 0.5, xmin = 0.03, xmax = 0.97, linestyle = "--", color = "black")
        accuracy_plot_test.text(x = 40, y = 0.52, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")
        accuracy_plot_test.set_title("Prediction accuracy: Emergence of lenses\nin validation organoids by deep learning", fontsize = cfg.TITLE_SIZE)
        accuracy_plot_test.set_ylabel("F1 score", fontsize = cfg.AXIS_LABEL_SIZE)
        accuracy_plot_test.set_ylim(-0.1, 1.1)
        accuracy_plot_test.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot_test.set_xlabel("hours", fontsize = cfg.AXIS_LABEL_SIZE)    
        accuracy_plot_test.legend().remove()

        accuracy_plot_val = fig.add_subplot(fig_sgs[1])
        sns.lineplot(data = data[data["classifier"] == "Ensemble_val"], x = "hours", y = "F1", hue = "experiment", ax = accuracy_plot_val, errorbar = "se", palette = "tab20")
        accuracy_plot_val.axhline(y = 0.5, xmin = 0.03, xmax = 0.97, linestyle = "--", color = "black")
        accuracy_plot_val.text(x = 40, y = 0.52, s = "Random Prediction", fontsize = cfg.TITLE_SIZE, color = "black")
        accuracy_plot_val.set_title("Prediction accuracy: Emergence of lenses\nin test organoids by deep learning", fontsize = cfg.TITLE_SIZE)
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

        test_data = lens_cnn_test_cm
        val_data = lens_cnn_val_cm

        colors = cfg.CONF_MATRIX_COLORS
        
        fig_sgs = gs.subgridspec(1,2)

        test_conf_matrix = fig.add_subplot(fig_sgs[0])
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
        
        test_conf_matrix.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        test_conf_matrix.set_ylabel("Percentage", fontsize=cfg.AXIS_LABEL_SIZE)
        test_conf_matrix.legend(handles, labels, fontsize=cfg.AXIS_LABEL_SIZE, bbox_to_anchor=(1.01, 0.5), loc="center left")
        test_conf_matrix.set_ylim(0, 100)
        test_conf_matrix.set_xlim(0, 72)
        test_conf_matrix.tick_params(**cfg.TICKPARAMS_PARAMS)
        test_conf_matrix.set_title("Confusion matrix: Emergence of lenses\nin validation organoids by deep learning", fontsize = cfg.TITLE_SIZE)

        val_conf_matrix = fig.add_subplot(fig_sgs[1])
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
        
        val_conf_matrix.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        val_conf_matrix.set_ylabel("Percentage", fontsize=cfg.AXIS_LABEL_SIZE)
        val_conf_matrix.legend(handles, labels, fontsize=cfg.AXIS_LABEL_SIZE, bbox_to_anchor=(1.01, 0.5), loc="center left")
        val_conf_matrix.set_ylim(0, 100)
        val_conf_matrix.set_xlim(0, 72)
        val_conf_matrix.tick_params(**cfg.TICKPARAMS_PARAMS)
        val_conf_matrix.set_title("Confusion matrix: Emergence of lenses\nin test organoids by deep learning", fontsize = cfg.TITLE_SIZE)

        return

    fig = plt.figure(layout = "constrained",
                     figsize = (cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 6,
                  nrows = 4,
                  figure = fig,
                  height_ratios = [1,0.7,1, 0.7])
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

def figure_S14_generation(sketch_dir: str,
                          figure_output_dir: str,
                          raw_data_dir: str,
                          morphometrics_dir: str,
                          hyperparameter_dir: str,
                          rpe_classification_dir: str,
                          lens_classification_dir: str,
                          rpe_baseline_dir: str,
                          lens_baseline_dir: str,
                          figure_data_dir: str,
                          evaluator_results_dir: str,
                          **kwargs) -> None:

    rpe_final_f1s = get_classification_f1_data(
        readout = "RPE_Final",
        output_dir = figure_data_dir,
        proj = "max",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = rpe_classification_dir,
        baseline_dir = None,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )
    rpe_cnn_test_cm = create_confusion_matrix_frame(readout = "RPE_Final",
                                                    classifier = "neural_net",
                                                    eval_set = "test",
                                                    proj = "max",
                                                    figure_data_dir = figure_data_dir,
                                                    morphometrics_dir = morphometrics_dir)
    rpe_cnn_val_cm = create_confusion_matrix_frame(readout = "RPE_Final",
                                                    classifier = "neural_net",
                                                    eval_set = "val",
                                                    proj = "max",
                                                    figure_data_dir = figure_data_dir,
                                                    morphometrics_dir = morphometrics_dir)

    lens_final_f1s = get_classification_f1_data(
        readout = "Lens_Final",
        output_dir = figure_data_dir,
        proj = "max",
        hyperparameter_dir = hyperparameter_dir,
        classification_dir = lens_classification_dir,
        baseline_dir = None,
        morphometrics_dir = morphometrics_dir,
        raw_data_dir = raw_data_dir,
        evaluator_results_dir = evaluator_results_dir
    )

    lens_cnn_test_cm = create_confusion_matrix_frame(readout = "Lens_Final",
                                                     classifier = "neural_net",
                                                     eval_set = "test",
                                                     proj = "max",
                                                     figure_data_dir = figure_data_dir,
                                                     morphometrics_dir = morphometrics_dir)
    lens_cnn_val_cm = create_confusion_matrix_frame(readout = "Lens_Final",
                                                    classifier = "neural_net",
                                                    eval_set = "val",
                                                    proj = "max",
                                                    figure_data_dir = figure_data_dir,
                                                    morphometrics_dir = morphometrics_dir)

    _generate_main_figure(rpe_f1_data = rpe_final_f1s,
                          lens_f1_data = lens_final_f1s,
                          rpe_cnn_test_cm = rpe_cnn_test_cm,
                          rpe_cnn_val_cm = rpe_cnn_val_cm,
                          lens_cnn_test_cm = lens_cnn_test_cm,
                          lens_cnn_val_cm = lens_cnn_val_cm,
                          figure_output_dir = figure_output_dir,
                          figure_name = "Supplementary_Figure_S14")

