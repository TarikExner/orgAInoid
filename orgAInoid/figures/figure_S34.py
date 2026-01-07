import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes


from . import figure_config as cfg
from . import figure_utils as utils

from .saliency_figures_generation import get_saliency_results


def _generate_main_figure(
    rpe_sal: pd.DataFrame,
    lens_sal: pd.DataFrame,
    rpe_classes_sal: pd.DataFrame,
    lens_classes_sal: pd.DataFrame,
    figure_output_dir: str = "",
    figure_name: str = "",
):
    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "RPE emergence"

        data = rpe_sal.copy()
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2

        agg = (
            data.groupby(["method", "hours"])["rank_corr"]
            .agg(["mean", "sem"])
            .reset_index()
        )

        sub = gs.subgridspec(1, 1)
        axm = fig.add_subplot(sub[0, 0])
        sns.lineplot(
            data=agg,
            x="hours",
            y="mean",
            hue="method",
            marker="",
            errorbar="se",
            ax=axm,
        )

        axm.set_title(
            f"{readout}: Cross-model consistency per saliency method",
            fontsize=cfg.TITLE_SIZE,
        )
        axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        axm.set_ylabel("Rank correlation (mean ± SEM)", fontsize=cfg.AXIS_LABEL_SIZE)
        axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)
        axm.legend(
            bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=cfg.AXIS_LABEL_SIZE
        )

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "Lens emergence"

        data = lens_sal.copy()
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2

        agg = (
            data.groupby(["method", "hours"])["rank_corr"]
            .agg(["mean", "sem"])
            .reset_index()
        )

        sub = gs.subgridspec(1, 1)
        axm = fig.add_subplot(sub[0, 0])
        sns.lineplot(
            data=agg,
            x="hours",
            y="mean",
            hue="method",
            marker="",
            errorbar="se",
            ax=axm,
        )

        axm.set_title(
            f"{readout}: Cross-model consistency per saliency method",
            fontsize=cfg.TITLE_SIZE,
        )
        axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        axm.set_ylabel("Rank correlation (mean ± SEM)", fontsize=cfg.AXIS_LABEL_SIZE)
        axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)
        axm.legend(
            bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=cfg.AXIS_LABEL_SIZE
        )
        return

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "RPE area"

        data = rpe_classes_sal.copy()
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2

        agg = (
            data.groupby(["method", "hours"])["rank_corr"]
            .agg(["mean", "sem"])
            .reset_index()
        )

        sub = gs.subgridspec(1, 1)
        axm = fig.add_subplot(sub[0, 0])
        sns.lineplot(
            data=agg,
            x="hours",
            y="mean",
            hue="method",
            marker="",
            errorbar="se",
            ax=axm,
        )

        axm.set_title(
            f"{readout}: Cross-model consistency per saliency method",
            fontsize=cfg.TITLE_SIZE,
        )
        axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        axm.set_ylabel("Rank correlation (mean ± SEM)", fontsize=cfg.AXIS_LABEL_SIZE)
        axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)
        axm.legend(
            bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=cfg.AXIS_LABEL_SIZE
        )

        return

    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "Lens sizes"

        data = lens_classes_sal.copy()
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2

        agg = (
            data.groupby(["method", "hours"])["rank_corr"]
            .agg(["mean", "sem"])
            .reset_index()
        )

        sub = gs.subgridspec(1, 1)
        axm = fig.add_subplot(sub[0, 0])
        sns.lineplot(
            data=agg,
            x="hours",
            y="mean",
            hue="method",
            marker="",
            errorbar="se",
            ax=axm,
        )

        axm.set_title(
            f"{readout}: Cross-model consistency per saliency method",
            fontsize=cfg.TITLE_SIZE,
        )
        axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        axm.set_ylabel("Rank correlation (mean ± SEM)", fontsize=cfg.AXIS_LABEL_SIZE)
        axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)
        axm.legend(
            bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=cfg.AXIS_LABEL_SIZE
        )

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

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.tif")
    plt.savefig(
        output_dir,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        transparent=False,
        pil_kwargs={"compression": "tiff_lzw"}
    )

    return


def figure_S34_generation(
    sketch_dir: str,
    figure_output_dir: str,
    saliency_input_dir: str,
    raw_data_dir: str,
    morphometrics_dir: str,
    hyperparameter_dir: str,
    rpe_classification_dir: str,
    lens_classification_dir: str,
    rpe_classes_classification_dir: str,
    lens_classes_classification_dir: str,
    annotations_dir: str,
    figure_data_dir: str,
    evaluator_results_dir: str,
    **kwargs,
) -> None:
    rpe_saliency_results = get_saliency_results(
        result="cross_model_correlation",
        readout="RPE_Final",
        saliency_input_dir=saliency_input_dir,
        raw_data_dir=raw_data_dir,
        morphometrics_dir=morphometrics_dir,
        hyperparameter_dir=hyperparameter_dir,
        rpe_classification_dir=rpe_classification_dir,
        lens_classification_dir=lens_classification_dir,
        rpe_classes_classification_dir=rpe_classes_classification_dir,
        lens_classes_classification_dir=lens_classes_classification_dir,
        annotations_dir=annotations_dir,
        figure_data_dir=figure_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )
    lens_saliency_results = get_saliency_results(
        result="cross_model_correlation",
        readout="Lens_Final",
        saliency_input_dir=saliency_input_dir,
        raw_data_dir=raw_data_dir,
        morphometrics_dir=morphometrics_dir,
        hyperparameter_dir=hyperparameter_dir,
        rpe_classification_dir=rpe_classification_dir,
        lens_classification_dir=lens_classification_dir,
        rpe_classes_classification_dir=rpe_classes_classification_dir,
        lens_classes_classification_dir=lens_classes_classification_dir,
        annotations_dir=annotations_dir,
        figure_data_dir=figure_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )
    rpe_classes_saliency_results = get_saliency_results(
        result="cross_model_correlation",
        readout="RPE_classes",
        saliency_input_dir=saliency_input_dir,
        raw_data_dir=raw_data_dir,
        morphometrics_dir=morphometrics_dir,
        hyperparameter_dir=hyperparameter_dir,
        rpe_classification_dir=rpe_classification_dir,
        lens_classification_dir=lens_classification_dir,
        rpe_classes_classification_dir=rpe_classes_classification_dir,
        lens_classes_classification_dir=lens_classes_classification_dir,
        annotations_dir=annotations_dir,
        figure_data_dir=figure_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )
    lens_classes_saliency_results = get_saliency_results(
        result="cross_model_correlation",
        readout="Lens_classes",
        saliency_input_dir=saliency_input_dir,
        raw_data_dir=raw_data_dir,
        morphometrics_dir=morphometrics_dir,
        hyperparameter_dir=hyperparameter_dir,
        rpe_classification_dir=rpe_classification_dir,
        lens_classification_dir=lens_classification_dir,
        rpe_classes_classification_dir=rpe_classes_classification_dir,
        lens_classes_classification_dir=lens_classes_classification_dir,
        annotations_dir=annotations_dir,
        figure_data_dir=figure_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )

    assert rpe_saliency_results is not None
    assert lens_saliency_results is not None
    assert rpe_classes_saliency_results is not None
    assert lens_classes_saliency_results is not None

    _generate_main_figure(
        rpe_sal=rpe_saliency_results,
        lens_sal=lens_saliency_results,
        rpe_classes_sal=rpe_classes_saliency_results,
        lens_classes_sal=lens_classes_saliency_results,
        figure_output_dir=figure_output_dir,
        figure_name="S34_Fig",
    )

    rpe_saliency_results["readout"] = "RPE_Final"
    lens_saliency_results["readout"] = "Lens_Final"
    rpe_classes_saliency_results["readout"] = "RPE_classes"
    lens_classes_saliency_results["readout"] = "Lens_classes"

    final_frame_output_dir = os.path.join(figure_output_dir, "S90_Data.csv")
    final_frame = pd.concat([
        rpe_saliency_results,
        lens_saliency_results,
        rpe_classes_saliency_results,
        lens_classes_saliency_results
    ], axis = 0)
    final_frame.to_csv(final_frame_output_dir, index = False)

    return
