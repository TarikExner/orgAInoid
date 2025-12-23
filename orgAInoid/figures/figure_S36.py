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
    thresholds = (2, 3, 4)

    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "RPE emergence"

        data = rpe_sal.copy()
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2.0
        data["voted"] = data["voted"].astype(int)

        vote_counts = (
            data.groupby(
                ["experiment", "well", "loop", "hours", "model", "region"],
                as_index=False,
            )["voted"]
            .sum()
            .rename(columns={"voted": "vote_count"})
        )

        models = sorted(vote_counts["model"].unique())
        sub = gs.subgridspec(1, 4, wspace=0)
        palette = "Set1"

        axes = []
        handles, labels = None, None
        x_min, x_max = vote_counts["hours"].min(), vote_counts["hours"].max()

        for i, m in enumerate(models[:3]):
            axm = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            df_m = vote_counts[vote_counts["model"] == m]

            curves = []
            for t in thresholds:
                per_well = (
                    df_m.assign(hit=(df_m["vote_count"] >= t).astype(int))
                    .groupby(["experiment", "well", "hours"], as_index=False)
                    .agg(frac=("hit", lambda x: x.sum() / len(x)))
                )
                avg = (
                    per_well.groupby("hours", as_index=False)["frac"]
                    .mean()
                    .assign(threshold=f"≥{t}")
                )
                curves.append(avg)

            plotdf = pd.concat(curves, ignore_index=True)

            sns.lineplot(
                data=plotdf,
                x="hours",
                y="frac",
                hue="threshold",
                errorbar=None,
                marker="",
                linewidth=0.9,
                palette=palette,
                ax=axm,
            )

            axm.set_title(
                f"Fraction of labeled regions\n{readout}\n{m}", fontsize=cfg.TITLE_SIZE
            )
            axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            axm.set_xlim(x_min, x_max)
            axm.set_ylim(0, 1)
            axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)

            if i == 0:
                axm.set_ylabel(
                    "Fraction of regions (per well)", fontsize=cfg.AXIS_LABEL_SIZE
                )
                handles, labels = axm.get_legend_handles_labels()
            else:
                axm.set_ylabel("")

            axm.get_legend().remove()
            axes.append(axm)

        ax_leg = fig.add_subplot(sub[0, 3])
        ax_leg.axis("off")
        if handles and labels:
            leg = ax_leg.legend(
                handles,
                labels,
                loc="center",
                frameon=False,
                title="Number of methods",
                fontsize=cfg.AXIS_LABEL_SIZE,
                markerscale=4,
            )
            for line in leg.get_lines():
                line.set_linewidth(2.5)

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "Lens emergence"

        data = lens_sal.copy()
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2.0
        data["voted"] = data["voted"].astype(int)

        vote_counts = (
            data.groupby(
                ["experiment", "well", "loop", "hours", "model", "region"],
                as_index=False,
            )["voted"]
            .sum()
            .rename(columns={"voted": "vote_count"})
        )

        models = sorted(vote_counts["model"].unique())
        sub = gs.subgridspec(1, 4, wspace=0)
        palette = "Set1"

        axes = []
        handles, labels = None, None
        x_min, x_max = vote_counts["hours"].min(), vote_counts["hours"].max()

        for i, m in enumerate(models[:3]):
            axm = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            df_m = vote_counts[vote_counts["model"] == m]

            curves = []
            for t in thresholds:
                per_well = (
                    df_m.assign(hit=(df_m["vote_count"] >= t).astype(int))
                    .groupby(["experiment", "well", "hours"], as_index=False)
                    .agg(frac=("hit", lambda x: x.sum() / len(x)))
                )
                avg = (
                    per_well.groupby("hours", as_index=False)["frac"]
                    .mean()
                    .assign(threshold=f"≥{t}")
                )
                curves.append(avg)

            plotdf = pd.concat(curves, ignore_index=True)

            sns.lineplot(
                data=plotdf,
                x="hours",
                y="frac",
                hue="threshold",
                errorbar=None,
                marker="",
                linewidth=0.9,
                palette=palette,
                ax=axm,
            )

            axm.set_title(
                f"Fraction of labeled regions\n{readout}\n{m}", fontsize=cfg.TITLE_SIZE
            )
            axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            axm.set_xlim(x_min, x_max)
            axm.set_ylim(0, 1)
            axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)

            if i == 0:
                axm.set_ylabel(
                    "Fraction of regions (per well)", fontsize=cfg.AXIS_LABEL_SIZE
                )
                handles, labels = axm.get_legend_handles_labels()
            else:
                axm.set_ylabel("")

            axm.get_legend().remove()
            axes.append(axm)

        ax_leg = fig.add_subplot(sub[0, 3])
        ax_leg.axis("off")
        if handles and labels:
            leg = ax_leg.legend(
                handles,
                labels,
                loc="center",
                frameon=False,
                title="Number of methods",
                fontsize=cfg.AXIS_LABEL_SIZE,
                markerscale=4,
            )
            for line in leg.get_lines():
                line.set_linewidth(2.5)

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "RPE area"

        data = rpe_classes_sal.copy()
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2.0
        data["voted"] = data["voted"].astype(int)

        vote_counts = (
            data.groupby(
                ["experiment", "well", "loop", "hours", "model", "region"],
                as_index=False,
            )["voted"]
            .sum()
            .rename(columns={"voted": "vote_count"})
        )

        models = sorted(vote_counts["model"].unique())
        sub = gs.subgridspec(1, 4, wspace=0)
        palette = "Set1"

        axes = []
        handles, labels = None, None
        x_min, x_max = vote_counts["hours"].min(), vote_counts["hours"].max()

        for i, m in enumerate(models[:3]):
            axm = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            df_m = vote_counts[vote_counts["model"] == m]

            curves = []
            for t in thresholds:
                per_well = (
                    df_m.assign(hit=(df_m["vote_count"] >= t).astype(int))
                    .groupby(["experiment", "well", "hours"], as_index=False)
                    .agg(frac=("hit", lambda x: x.sum() / len(x)))
                )
                avg = (
                    per_well.groupby("hours", as_index=False)["frac"]
                    .mean()
                    .assign(threshold=f"≥{t}")
                )
                curves.append(avg)

            plotdf = pd.concat(curves, ignore_index=True)

            sns.lineplot(
                data=plotdf,
                x="hours",
                y="frac",
                hue="threshold",
                errorbar=None,
                marker="",
                linewidth=0.9,
                palette=palette,
                ax=axm,
            )

            axm.set_title(
                f"Fraction of labeled regions\n{readout}\n{m}", fontsize=cfg.TITLE_SIZE
            )
            axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            axm.set_xlim(x_min, x_max)
            axm.set_ylim(0, 1)
            axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)

            if i == 0:
                axm.set_ylabel(
                    "Fraction of regions (per well)", fontsize=cfg.AXIS_LABEL_SIZE
                )
                handles, labels = axm.get_legend_handles_labels()
            else:
                axm.set_ylabel("")

            axm.get_legend().remove()
            axes.append(axm)

        ax_leg = fig.add_subplot(sub[0, 3])
        ax_leg.axis("off")
        if handles and labels:
            leg = ax_leg.legend(
                handles,
                labels,
                loc="center",
                frameon=False,
                title="Number of methods",
                fontsize=cfg.AXIS_LABEL_SIZE,
                markerscale=4,
            )
            for line in leg.get_lines():
                line.set_linewidth(2.5)

        return

    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        readout = "Lens sizes"

        data = lens_classes_sal.copy()
        data["loop"] = data["loop"].astype(int)
        data["hours"] = data["loop"] / 2.0
        data["voted"] = data["voted"].astype(int)

        vote_counts = (
            data.groupby(
                ["experiment", "well", "loop", "hours", "model", "region"],
                as_index=False,
            )["voted"]
            .sum()
            .rename(columns={"voted": "vote_count"})
        )

        models = sorted(vote_counts["model"].unique())
        sub = gs.subgridspec(1, 4, wspace=0)
        palette = "Set1"

        axes = []
        handles, labels = None, None
        x_min, x_max = vote_counts["hours"].min(), vote_counts["hours"].max()

        for i, m in enumerate(models[:3]):
            axm = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            df_m = vote_counts[vote_counts["model"] == m]

            curves = []
            for t in thresholds:
                per_well = (
                    df_m.assign(hit=(df_m["vote_count"] >= t).astype(int))
                    .groupby(["experiment", "well", "hours"], as_index=False)
                    .agg(frac=("hit", lambda x: x.sum() / len(x)))
                )
                avg = (
                    per_well.groupby("hours", as_index=False)["frac"]
                    .mean()
                    .assign(threshold=f"≥{t}")
                )
                curves.append(avg)

            plotdf = pd.concat(curves, ignore_index=True)

            sns.lineplot(
                data=plotdf,
                x="hours",
                y="frac",
                hue="threshold",
                errorbar=None,
                marker="",
                linewidth=0.9,
                palette=palette,
                ax=axm,
            )

            axm.set_title(
                f"Fraction of labeled regions\n{readout}\n{m}", fontsize=cfg.TITLE_SIZE
            )
            axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            axm.set_xlim(x_min, x_max)
            axm.set_ylim(0, 1)
            axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)

            if i == 0:
                axm.set_ylabel(
                    "Fraction of regions (per well)", fontsize=cfg.AXIS_LABEL_SIZE
                )
                handles, labels = axm.get_legend_handles_labels()
            else:
                axm.set_ylabel("")

            axm.get_legend().remove()
            axes.append(axm)

        ax_leg = fig.add_subplot(sub[0, 3])
        ax_leg.axis("off")
        if handles and labels:
            leg = ax_leg.legend(
                handles,
                labels,
                loc="center",
                frameon=False,
                title="Number of methods",
                fontsize=cfg.AXIS_LABEL_SIZE,
                markerscale=4,
            )
            for line in leg.get_lines():
                line.set_linewidth(2.5)

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


def figure_S36_generation(
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
        result="region_votes_summary",
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
        result="region_votes_summary",
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
        result="region_votes_summary",
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
        result="region_votes_summary",
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
        figure_name="Supplementary_Figure_S36",
    )

    rpe_saliency_results["readout"] = "RPE_Final"
    lens_saliency_results["readout"] = "Lens_Final"
    rpe_classes_saliency_results["readout"] = "RPE_classes"
    lens_classes_saliency_results["readout"] = "Lens_classes"

    final_frame_output_dir = os.path.join(figure_output_dir, "Data_S92.csv")
    final_frame = pd.concat([
        rpe_saliency_results,
        lens_saliency_results,
        rpe_classes_saliency_results,
        lens_classes_saliency_results
    ], axis = 0)
    final_frame.to_csv(final_frame_output_dir, index = False)

    return
