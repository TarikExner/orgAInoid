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

from .figure_data_generation import (
    get_classification_f1_data,
    create_confusion_matrix_frame,
)


def _stacked_area(
    ax: Axes,
    data4: pd.DataFrame,
    colors,
    label_dict,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
):
    """
    Plot one stacked area (columns must be ['tn','tp','fn','fp'] in that order).
    """
    """
    data4 columns must be ['tn','tp','fn','fp'].
    Scales to percent if needed and fills NaNs->0 to avoid gaps.
    """
    cols = ["tn", "tp", "fn", "fp"]
    df = data4[cols].astype(float).copy()

    # fix NaNs
    df = df.fillna(0.0)

    # if totals look like fractions, scale to %
    totals = df.sum(axis=1).values
    if np.nanmax(totals) <= 1.5:  # 1.0 (± rounding) -> convert to %
        df *= 100.0

    X = df.index.values
    Y = df.values

    cumulative_base = np.zeros_like(Y)
    for i in range(1, Y.shape[1]):
        cumulative_base[:, i] = cumulative_base[:, i - 1] + Y[:, i - 1]

    for i, comp in enumerate(cols):
        ax.fill_between(
            X,
            cumulative_base[:, i],
            cumulative_base[:, i] + Y[:, i],
            color=colors[i],
            label=comp,
            alpha=0.8,
        )

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    labels = [label_dict.get(l, l) for l in labels]
    return handles, labels


def _generate_main_figure(
    rpe_classes_f1_data: pd.DataFrame,
    lens_classes_f1_data: pd.DataFrame,
    rpe_classes_clf_test_cm: pd.DataFrame,
    rpe_classes_clf_val_cm: pd.DataFrame,
    lens_classes_clf_test_cm: pd.DataFrame,
    lens_classes_clf_val_cm: pd.DataFrame,
    figure_output_dir: str,
    figure_name: str,
):
    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        data = rpe_classes_f1_data
        data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1, 2)

        accuracy_plot_test = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=data[data["classifier"] == "Ensemble_test"],
            x="hours",
            y="F1",
            hue="experiment",
            ax=accuracy_plot_test,
            errorbar="se",
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
        )
        accuracy_plot_test.axhline(
            y=0.25, xmin=0.03, xmax=0.97, linestyle="--", color="black"
        )
        accuracy_plot_test.text(
            x=40, y=0.27, s="Random Prediction", fontsize=cfg.TITLE_SIZE, color="black"
        )
        accuracy_plot_test.set_title(
            "Prediction accuracy: RPE area\nin validation organoids by deep learning",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot_test.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot_test.set_ylim(-0.1, 1.1)
        accuracy_plot_test.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot_test.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot_test.legend().remove()

        accuracy_plot_val = fig.add_subplot(fig_sgs[1])
        sns.lineplot(
            data=data[data["classifier"] == "Ensemble_val"],
            x="hours",
            y="F1",
            hue="experiment",
            ax=accuracy_plot_val,
            errorbar="se",
            palette="tab20",
        )
        accuracy_plot_val.axhline(
            y=0.25, xmin=0.03, xmax=0.97, linestyle="--", color="black"
        )
        accuracy_plot_val.text(
            x=40, y=0.27, s="Random Prediction", fontsize=cfg.TITLE_SIZE, color="black"
        )
        accuracy_plot_val.set_title(
            "Prediction accuracy: RPE area\nin test organoids by deep learning",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot_val.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot_val.set_ylim(-0.1, 1.1)
        accuracy_plot_val.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot_val.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot_val.legend(
            bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=cfg.TITLE_SIZE
        )

        return

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        test_wide = rpe_classes_clf_test_cm
        val_wide = rpe_classes_clf_val_cm

        colors = cfg.CONF_MATRIX_COLORS
        label_dict = cfg.CONF_MATRIX_LABEL_DICT

        fig_sgs = gs.subgridspec(4, 2, hspace=0, wspace=0)

        for i in range(4):
            sub_ax = fig.add_subplot(fig_sgs[i, 0])
            cls = f"class{i}"
            df_cls = test_wide.xs(cls, axis=1, level=0)[["tn", "tp", "fn", "fp"]]

            handles, labels = _stacked_area(sub_ax, df_cls, colors, label_dict)

            if i == 0:
                sub_ax.set_title(
                    "Confusion matrices: RPE area\nin validation organoids by deep learning",
                    fontsize=cfg.TITLE_SIZE,
                )
            sub_ax.set_ylabel(cls, fontsize=cfg.AXIS_LABEL_SIZE)
            sub_ax.set_ylim(0, 100)
            sub_ax.set_xlim(0, 72)

            if i != 3:
                sub_ax.set_xticklabels([])
                sub_ax.set_yticklabels([])
                sub_ax.tick_params(bottom=False, left=False)
            else:
                sub_ax.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
                sub_ax.tick_params(**cfg.TICKPARAMS_PARAMS)
                sub_ax.set_yticklabels([])
                sub_ax.tick_params(left=False)

            sub_ax.legend(
                handles,
                labels,
                fontsize=cfg.AXIS_LABEL_SIZE,
                bbox_to_anchor=(1.01, 0.5),
                loc="center left",
            )

        for i in range(4):
            sub_ax = fig.add_subplot(fig_sgs[i, 1])
            cls = f"class{i}"
            df_cls = val_wide.xs(cls, axis=1, level=0)[["tn", "tp", "fn", "fp"]]

            handles, labels = _stacked_area(sub_ax, df_cls, colors, label_dict)

            if i == 0:
                sub_ax.set_title(
                    "Confusion matrices: RPE area\nin test organoids by deep learning",
                    fontsize=cfg.TITLE_SIZE,
                )
            sub_ax.set_ylabel(cls, fontsize=cfg.AXIS_LABEL_SIZE)
            sub_ax.set_ylim(0, 100)
            sub_ax.set_xlim(0, 72)

            if i != 3:
                sub_ax.set_xticklabels([])
                sub_ax.set_yticklabels([])
                sub_ax.tick_params(bottom=False, left=False)
            else:
                sub_ax.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
                sub_ax.tick_params(**cfg.TICKPARAMS_PARAMS)
                sub_ax.set_yticklabels([])
                sub_ax.tick_params(left=False)

            sub_ax.legend(
                handles,
                labels,
                fontsize=cfg.AXIS_LABEL_SIZE,
                bbox_to_anchor=(1.01, 0.5),
                loc="center left",
            )

        return

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        data = lens_classes_f1_data

        data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1, 2)

        accuracy_plot_test = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=data[data["classifier"] == "Ensemble_test"],
            x="hours",
            y="F1",
            hue="experiment",
            ax=accuracy_plot_test,
            errorbar="se",
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
        )

        accuracy_plot_test.axhline(
            y=0.25, xmin=0.03, xmax=0.97, linestyle="--", color="black"
        )
        accuracy_plot_test.text(
            x=40, y=0.27, s="Random Prediction", fontsize=cfg.TITLE_SIZE, color="black"
        )
        accuracy_plot_test.set_title(
            "Prediction accuracy: Lens sizes\nin validation organoids by deep learning",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot_test.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot_test.set_ylim(-0.1, 1.1)
        accuracy_plot_test.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot_test.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot_test.legend().remove()

        accuracy_plot_val = fig.add_subplot(fig_sgs[1])
        sns.lineplot(
            data=data[data["classifier"] == "Ensemble_val"],
            x="hours",
            y="F1",
            hue="experiment",
            ax=accuracy_plot_val,
            errorbar="se",
            palette="tab20",
        )

        accuracy_plot_val.axhline(
            y=0.25, xmin=0.03, xmax=0.97, linestyle="--", color="black"
        )
        accuracy_plot_val.text(
            x=40, y=0.27, s="Random Prediction", fontsize=cfg.TITLE_SIZE, color="black"
        )
        accuracy_plot_val.set_title(
            "Prediction accuracy: Lens sizes\nin test organoids by deep learning",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot_val.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot_val.set_ylim(-0.1, 1.1)
        accuracy_plot_val.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot_val.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot_val.legend(
            bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=cfg.TITLE_SIZE
        )

        return

    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        test_wide = lens_classes_clf_test_cm
        val_wide = lens_classes_clf_val_cm

        colors = cfg.CONF_MATRIX_COLORS
        label_dict = cfg.CONF_MATRIX_LABEL_DICT

        fig_sgs = gs.subgridspec(4, 2, hspace=0, wspace=0)

        for i in range(4):
            sub_ax = fig.add_subplot(fig_sgs[i, 0])
            cls = f"class{i}"
            df_cls = test_wide.xs(cls, axis=1, level=0)[["tn", "tp", "fn", "fp"]]

            handles, labels = _stacked_area(sub_ax, df_cls, colors, label_dict)

            if i == 0:
                sub_ax.set_title(
                    "Confusion matrices: Lens sizes\nin validation organoids by deep learning",
                    fontsize=cfg.TITLE_SIZE,
                )
            sub_ax.set_ylabel(cls, fontsize=cfg.AXIS_LABEL_SIZE)
            sub_ax.set_ylim(0, 100)
            sub_ax.set_xlim(0, 72)

            if i != 3:
                sub_ax.set_xticklabels([])
                sub_ax.set_yticklabels([])
                sub_ax.tick_params(bottom=False, left=False)
            else:
                sub_ax.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
                sub_ax.tick_params(**cfg.TICKPARAMS_PARAMS)
                sub_ax.set_yticklabels([])
                sub_ax.tick_params(left=False)

            sub_ax.legend(
                handles,
                labels,
                fontsize=cfg.AXIS_LABEL_SIZE,
                bbox_to_anchor=(1.01, 0.5),
                loc="center left",
            )

        # right column: validation
        for i in range(4):
            sub_ax = fig.add_subplot(fig_sgs[i, 1])
            cls = f"class{i}"
            df_cls = val_wide.xs(cls, axis=1, level=0)[["tn", "tp", "fn", "fp"]]

            handles, labels = _stacked_area(sub_ax, df_cls, colors, label_dict)

            if i == 0:
                sub_ax.set_title(
                    "Confusion matrices: Lens sizes\nin test organoids by deep learning",
                    fontsize=cfg.TITLE_SIZE,
                )
            sub_ax.set_ylabel(cls, fontsize=cfg.AXIS_LABEL_SIZE)
            sub_ax.set_ylim(0, 100)
            sub_ax.set_xlim(0, 72)

            if i != 3:
                sub_ax.set_xticklabels([])
                sub_ax.set_yticklabels([])
                sub_ax.tick_params(bottom=False, left=False)
            else:
                sub_ax.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
                sub_ax.tick_params(**cfg.TICKPARAMS_PARAMS)
                sub_ax.set_yticklabels([])
                sub_ax.tick_params(left=False)

            sub_ax.legend(
                handles,
                labels,
                fontsize=cfg.AXIS_LABEL_SIZE,
                bbox_to_anchor=(1.01, 0.5),
                loc="center left",
            )

        return

    fig = plt.figure(
        layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL)
    )
    gs = GridSpec(ncols=6, nrows=4, figure=fig, height_ratios=[1, 1.3, 1, 1.3])
    a_coords = gs[0, :]
    b_coords = gs[1, :]
    c_coords = gs[2, :]
    d_coords = gs[3, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")
    generate_subfigure_d(fig, fig_d, d_coords, "D")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.png")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.tif")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight", transparent = True)

    return


def figure_S23_generation(
    sketch_dir: str,
    figure_output_dir: str,
    raw_data_dir: str,
    morphometrics_dir: str,
    hyperparameter_dir: str,
    rpe_classes_classification_dir: str,
    lens_classes_classification_dir: str,
    figure_data_dir: str,
    evaluator_results_dir: str,
    **kwargs,
) -> None:
    rpe_classes_f1s = get_classification_f1_data(
        readout="RPE_classes",
        output_dir=figure_data_dir,
        proj="",
        hyperparameter_dir=hyperparameter_dir,
        classification_dir=rpe_classes_classification_dir,
        baseline_dir=None,
        morphometrics_dir=morphometrics_dir,
        raw_data_dir=raw_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )
    rpe_classes_cnn_test_cm = create_confusion_matrix_frame(
        readout="RPE_classes",
        classifier="neural_net",
        eval_set="test",
        proj="",
        figure_data_dir=figure_data_dir,
        morphometrics_dir=morphometrics_dir,
    )
    rpe_classes_cnn_val_cm = create_confusion_matrix_frame(
        readout="RPE_classes",
        classifier="neural_net",
        eval_set="val",
        proj="",
        figure_data_dir=figure_data_dir,
        morphometrics_dir=morphometrics_dir,
    )

    lens_classes_f1s = get_classification_f1_data(
        readout="Lens_classes",
        output_dir=figure_data_dir,
        proj="",
        hyperparameter_dir=hyperparameter_dir,
        classification_dir=lens_classes_classification_dir,
        baseline_dir=None,
        morphometrics_dir=morphometrics_dir,
        raw_data_dir=raw_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )
    lens_classes_cnn_test_cm = create_confusion_matrix_frame(
        readout="Lens_classes",
        classifier="neural_net",
        eval_set="test",
        proj="",
        figure_data_dir=figure_data_dir,
        morphometrics_dir=morphometrics_dir,
    )
    lens_classes_cnn_val_cm = create_confusion_matrix_frame(
        readout="Lens_classes",
        classifier="neural_net",
        eval_set="val",
        proj="",
        figure_data_dir=figure_data_dir,
        morphometrics_dir=morphometrics_dir,
    )

    _generate_main_figure(
        rpe_classes_f1_data=rpe_classes_f1s,
        lens_classes_f1_data=lens_classes_f1s,
        rpe_classes_clf_test_cm=rpe_classes_cnn_test_cm,
        rpe_classes_clf_val_cm=rpe_classes_cnn_val_cm,
        lens_classes_clf_test_cm=lens_classes_cnn_test_cm,
        lens_classes_clf_val_cm=lens_classes_cnn_val_cm,
        figure_output_dir=figure_output_dir,
        figure_name="S23_Fig",
    )

    rpe_final_output_dir = os.path.join(figure_output_dir, "S73_Data.csv")
    rpe_classes_f1s.to_csv(rpe_final_output_dir, index = False)

    rpe_cnn_output_dir = os.path.join(figure_output_dir, "S74_Data.csv")
    rpe_classes_cnn_test_cm["eval_set"] = "val"
    rpe_classes_cnn_val_cm["eval_set"] = "test"
    rpe_classes_cnn = pd.concat([rpe_classes_cnn_test_cm, rpe_classes_cnn_val_cm], axis = 0)
    rpe_classes_cnn.to_csv(rpe_cnn_output_dir, index = False)

    lens_final_output_dir = os.path.join(figure_output_dir, "S75_Data.csv")
    lens_classes_f1s.to_csv(lens_final_output_dir, index = False)

    lens_cnn_output_dir = os.path.join(figure_output_dir, "S76_Data.csv")
    lens_classes_cnn_test_cm["eval_set"] = "val"
    lens_classes_cnn_val_cm["eval_set"] = "test"
    lens_classes_cnn = pd.concat([lens_classes_cnn_test_cm, lens_classes_cnn_val_cm], axis = 0)
    lens_classes_cnn.to_csv(lens_cnn_output_dir, index = False)

    return
