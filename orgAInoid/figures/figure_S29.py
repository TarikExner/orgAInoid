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
    ) -> tuple:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "RPE emergence"

        data = rpe_sal.copy()
        data["pair"] = [f"{a}::{b}" for a, b in zip(data["method_a"], data["method_b"])]
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2

        # aggregate: mean across experiments+wells
        agg = data.groupby(["model", "hours", "pair"], as_index=False)[
            "dice_avg"
        ].mean()

        models = sorted(agg["model"].unique())

        # 1×4 grid, shared y-axis, no horizontal spacing
        sub = gs.subgridspec(1, 3, wspace=0)
        palette = sns.color_palette("husl", n_colors=agg["pair"].nunique())

        axes = []
        for i, m in enumerate(models):
            ax = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            dat = agg[agg["model"] == m]
            sns.lineplot(
                data=dat,
                x="hours",
                y="dice_avg",
                hue="pair",
                marker="",
                palette=palette,
                errorbar=None,
                ax=ax,
                linewidth=0.5,
            )
            ax.set_title(f"{readout}: {m}", fontsize=cfg.TITLE_SIZE)
            ax.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)
            if i == 0:
                ax.set_ylabel("Dice coefficient", fontsize=cfg.AXIS_LABEL_SIZE)
            else:
                ax.set_ylabel("")
            ax.get_legend().remove()
            axes.append(ax)

        handles, labels = axes[0].get_legend_handles_labels()
        return handles, labels

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "Lens emergence"

        data = lens_sal.copy()
        data["pair"] = [f"{a}::{b}" for a, b in zip(data["method_a"], data["method_b"])]
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2

        # aggregate: mean across experiments+wells
        agg = data.groupby(["model", "hours", "pair"], as_index=False)[
            "dice_avg"
        ].mean()

        models = sorted(agg["model"].unique())

        # 1×4 grid, shared y-axis, no horizontal spacing
        sub = gs.subgridspec(1, 3, wspace=0)
        palette = sns.color_palette("husl", n_colors=agg["pair"].nunique())

        axes = []
        for i, m in enumerate(models):
            ax = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            dat = agg[agg["model"] == m]
            sns.lineplot(
                data=dat,
                x="hours",
                y="dice_avg",
                hue="pair",
                marker="",
                palette=palette,
                errorbar=None,
                ax=ax,
                linewidth=0.5,
            )
            ax.set_title(f"{readout}: {m}", fontsize=cfg.TITLE_SIZE)
            ax.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)
            if i == 0:
                ax.set_ylabel("Dice coefficient", fontsize=cfg.AXIS_LABEL_SIZE)
            else:
                ax.set_ylabel("")
            ax.get_legend().remove()
            axes.append(ax)

        return

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "RPE area"

        data = rpe_classes_sal.copy()
        data["pair"] = [f"{a}::{b}" for a, b in zip(data["method_a"], data["method_b"])]
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2

        # aggregate: mean across experiments+wells
        agg = data.groupby(["model", "hours", "pair"], as_index=False)[
            "dice_avg"
        ].mean()

        models = sorted(agg["model"].unique())

        # 1×4 grid, shared y-axis, no horizontal spacing
        sub = gs.subgridspec(1, 3, wspace=0)
        palette = sns.color_palette("husl", n_colors=agg["pair"].nunique())

        axes = []
        for i, m in enumerate(models):
            ax = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            dat = agg[agg["model"] == m]
            sns.lineplot(
                data=dat,
                x="hours",
                y="dice_avg",
                hue="pair",
                marker="",
                palette=palette,
                errorbar=None,
                ax=ax,
                linewidth=0.5,
            )
            ax.set_title(f"{readout}: {m}", fontsize=cfg.TITLE_SIZE)
            ax.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)
            if i == 0:
                ax.set_ylabel("Dice coefficient", fontsize=cfg.AXIS_LABEL_SIZE)
            else:
                ax.set_ylabel("")
            ax.get_legend().remove()
            axes.append(ax)

        return

    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "Lens sizes"

        data = lens_classes_sal.copy()
        data["pair"] = [f"{a}::{b}" for a, b in zip(data["method_a"], data["method_b"])]
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2

        # aggregate: mean across experiments+wells
        agg = data.groupby(["model", "hours", "pair"], as_index=False)[
            "dice_avg"
        ].mean()

        models = sorted(agg["model"].unique())

        # 1×4 grid, shared y-axis, no horizontal spacing
        sub = gs.subgridspec(1, 3, wspace=0)
        palette = sns.color_palette("husl", n_colors=agg["pair"].nunique())

        axes = []
        for i, m in enumerate(models):
            ax = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            dat = agg[agg["model"] == m]
            sns.lineplot(
                data=dat,
                x="hours",
                y="dice_avg",
                hue="pair",
                marker="",
                palette=palette,
                errorbar=None,
                ax=ax,
                linewidth=0.5,
            )
            ax.set_title(f"{readout}: {m}", fontsize=cfg.TITLE_SIZE)
            ax.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)
            if i == 0:
                ax.set_ylabel("Dice coefficient", fontsize=cfg.AXIS_LABEL_SIZE)
            else:
                ax.set_ylabel("")
            ax.get_legend().remove()
            axes.append(ax)

        return

    def generate_subfigure_e(
        fig: Figure, ax: Axes, gs: SubplotSpec, handels, labels, subfigure_label
    ) -> None:
        ax.axis("off")

        ax = fig.add_subplot(gs)
        ax.axis("off")
        leg = ax.legend(
            handles,
            labels,
            loc="center",
            frameon=False,
            ncol=6,
            title="Method pair",
            fontsize=cfg.AXIS_LABEL_SIZE,
            markerscale=4,
        )
        for line in leg.get_lines():
            line.set_linewidth(2.5)

        return

    fig = plt.figure(
        layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL)
    )
    gs = GridSpec(ncols=6, nrows=5, figure=fig, height_ratios=[1, 1, 1, 1, 1])
    a_coords = gs[0, :]
    b_coords = gs[1, :]
    c_coords = gs[2, :]
    d_coords = gs[3, :]
    e_coords = gs[4, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)
    fig_e = fig.add_subplot(e_coords)

    handles, labels = generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")
    generate_subfigure_d(fig, fig_d, d_coords, "D")
    generate_subfigure_e(fig, fig_e, e_coords, handles, labels, "")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.png")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.tif")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight", transparent = True)

    return


def figure_S29_generation(
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
        result="agreement_method_pairwise",
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
        result="agreement_method_pairwise",
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
        result="agreement_method_pairwise",
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
        result="agreement_method_pairwise",
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
        figure_name="S29_Fig",
    )

    rpe_saliency_results["readout"] = "RPE_Final"
    lens_saliency_results["readout"] = "Lens_Final"
    rpe_classes_saliency_results["readout"] = "RPE_classes"
    lens_classes_saliency_results["readout"] = "Lens_classes"

    final_frame_output_dir = os.path.join(figure_output_dir, "S89_Data.csv")
    final_frame = pd.concat([
        rpe_saliency_results,
        lens_saliency_results,
        rpe_classes_saliency_results,
        lens_classes_saliency_results
    ], axis = 0)
    final_frame.to_csv(final_frame_output_dir, index = False)

    return
