import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Literal

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import get_classification_f1_data_raw, f1_vs_distance_plot


def _generate_main_figure(
    rpe_classes_test: pd.DataFrame,
    rpe_classes_val: pd.DataFrame,
    lens_classes_test: pd.DataFrame,
    lens_classes_val: pd.DataFrame,
    classifier: Literal["CNN", "CLF"],
    figure_output_dir: str = "",
    figure_name: str = "",
):
    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        readout = "RPE area"
        eval_set = "validation"

        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        fig_sgs = gs.subgridspec(1, 1)

        data = rpe_classes_test
        data["val_experiment"] = data["val_experiment"].map(cfg.EXPERIMENT_MAP)
        data = data.sort_values("val_experiment", ascending=True)
        data["val_experiment"] = data["val_experiment"].astype("category")

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=data,
            x="dist_center",
            y="f1_weighted",
            hue="val_experiment",
            palette="tab20",
            ax=accuracy_plot,
        )

        accuracy_plot.legend(
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.AXIS_LABEL_SIZE,
            ncols=2,
        )
        accuracy_plot.set_title(
            f"{classifier} F1 distribution over bin center distance for\n{readout} in {eval_set} organoids",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot.set_xlabel(
            "bin center distance [0=center, 1=edge]", fontsize=cfg.AXIS_LABEL_SIZE
        )
        accuracy_plot.set_ylabel("F1-score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)

        return

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        readout = "RPE area"
        eval_set = "test"

        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        fig_sgs = gs.subgridspec(1, 1)

        data = rpe_classes_val
        data["val_experiment"] = data["val_experiment"].map(cfg.EXPERIMENT_MAP)
        data = data.sort_values("val_experiment", ascending=True)
        data["val_experiment"] = data["val_experiment"].astype("category")

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=data,
            x="dist_center",
            y="f1_weighted",
            hue="val_experiment",
            palette="tab20",
            ax=accuracy_plot,
        )

        accuracy_plot.legend(
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.AXIS_LABEL_SIZE,
            ncols=2,
        )
        accuracy_plot.set_title(
            f"{classifier} F1 distribution over bin center distance for\n{readout} in {eval_set} organoids",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot.set_xlabel(
            "bin center distance [0=center, 1=edge]", fontsize=cfg.AXIS_LABEL_SIZE
        )
        accuracy_plot.set_ylabel("F1-score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)

        return

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        readout = "Lens sizes"
        eval_set = "validation"

        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        fig_sgs = gs.subgridspec(1, 1)

        data = lens_classes_test
        data["val_experiment"] = data["val_experiment"].map(cfg.EXPERIMENT_MAP)
        data = data.sort_values("val_experiment", ascending=True)
        data["val_experiment"] = data["val_experiment"].astype("category")

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=data,
            x="dist_center",
            y="f1_weighted",
            hue="val_experiment",
            palette="tab20",
            ax=accuracy_plot,
        )

        accuracy_plot.legend(
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.AXIS_LABEL_SIZE,
            ncols=2,
        )
        accuracy_plot.set_title(
            f"{classifier} F1 distribution over bin center distance for\n{readout} in {eval_set} organoids",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot.set_xlabel(
            "bin center distance [0=center, 1=edge]", fontsize=cfg.AXIS_LABEL_SIZE
        )
        accuracy_plot.set_ylabel("F1-score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)

        return

    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        readout = "Lens sizes"
        eval_set = "test"

        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        fig_sgs = gs.subgridspec(1, 1)

        data = lens_classes_val
        data["val_experiment"] = data["val_experiment"].map(cfg.EXPERIMENT_MAP)
        data = data.sort_values("val_experiment", ascending=True)
        data["val_experiment"] = data["val_experiment"].astype("category")

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=data,
            x="dist_center",
            y="f1_weighted",
            hue="val_experiment",
            palette="tab20",
            ax=accuracy_plot,
        )

        accuracy_plot.legend(
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.AXIS_LABEL_SIZE,
            ncols=2,
        )
        accuracy_plot.set_title(
            f"{classifier} F1 distribution over bin center distance for\n{readout} in {eval_set} organoids",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot.set_xlabel(
            "bin center distance [0=center, 1=edge]", fontsize=cfg.AXIS_LABEL_SIZE
        )
        accuracy_plot.set_ylabel("F1-score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)

        return

    fig = plt.figure(
        layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL)
    )
    gs = GridSpec(ncols=6, nrows=4, figure=fig, height_ratios=[1, 1, 1, 1])
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

    return


def figure_S29_generation(
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
    rpe_classes = get_classification_f1_data_raw(
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
    lens_classes = get_classification_f1_data_raw(
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

    BIN_EDGES_RPE = [0, 956.9, 1590.4]
    BIN_EDGES_LENS = [0, 16324.85763, 29083.23]

    def subset_for_classifier(
        df: pd.DataFrame, classifier: Literal["CNN", "CLF"]
    ) -> pd.DataFrame:
        return df[df["classifier"] == classifier].copy()

    f1d_rpe_classes_val = f1_vs_distance_plot(
        subset_for_classifier(rpe_classes, "CLF"),
        bin_edges=BIN_EDGES_RPE,
        set_name="val",
        n_distance_bins=15,
    )
    f1d_rpe_classes_test = f1_vs_distance_plot(
        subset_for_classifier(rpe_classes, "CLF"),
        bin_edges=BIN_EDGES_RPE,
        set_name="test",
        n_distance_bins=15,
    )
    f1d_lens_classes_val = f1_vs_distance_plot(
        subset_for_classifier(lens_classes, "CLF"),
        bin_edges=BIN_EDGES_LENS,
        value_col="Lens_area",
        set_name="val",
        n_distance_bins=15,
    )
    f1d_lens_classes_test = f1_vs_distance_plot(
        subset_for_classifier(lens_classes, "CLF"),
        bin_edges=BIN_EDGES_LENS,
        value_col="Lens_area",
        set_name="test",
        n_distance_bins=15,
    )

    _generate_main_figure(
        rpe_classes_test=f1d_rpe_classes_test,
        rpe_classes_val=f1d_rpe_classes_val,
        lens_classes_test=f1d_lens_classes_test,
        lens_classes_val=f1d_lens_classes_val,
        figure_output_dir=figure_output_dir,
        classifier="CLF",
        figure_name="Supplementary_Figure_S29",
    )
