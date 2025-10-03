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
        data = data[data["time"] != 1].copy()
        data["loop"] = data["time"].astype(int)
        data["hours"] = data["loop"] / 2.0

        # mean across experiments + wells
        agg = data.groupby(["model", "hours", "method"], as_index=False)["drift"].mean()

        models = sorted(agg["model"].unique())

        sub = gs.subgridspec(1, 4, wspace=0)

        n_methods = agg["method"].nunique()
        palette = "tab10"

        axes = []
        handles, labels = None, None

        for i, m in enumerate(models):
            if i >= 3:
                break  # only first 3 models shown here
            axm = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            dat = agg[agg["model"] == m]
            sns.lineplot(
                data=dat,
                x="hours",
                y="drift",
                hue="method",
                errorbar="se",
                marker="",
                linewidth=0.9,
                palette=palette,
                ax=axm,
            )
            axm.set_title(
                f"Spatial drift of saliency\n{readout}\n{m}", fontsize=cfg.TITLE_SIZE
            )
            axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)

            if i == 0:
                axm.set_ylabel(
                    "center of mass displacement [pixels]", fontsize=cfg.AXIS_LABEL_SIZE
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
                ncol=1,
                title="Method",
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
        data = data[data["time"] != 1].copy()
        data["loop"] = data["time"].astype(int)
        data["hours"] = data["loop"] / 2.0

        # mean across experiments + wells
        agg = data.groupby(["model", "hours", "method"], as_index=False)["drift"].mean()

        models = sorted(agg["model"].unique())

        sub = gs.subgridspec(1, 4, wspace=0)

        n_methods = agg["method"].nunique()
        palette = "tab10"

        axes = []
        handles, labels = None, None

        for i, m in enumerate(models):
            if i >= 3:
                break  # only first 3 models shown here
            axm = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            dat = agg[agg["model"] == m]
            sns.lineplot(
                data=dat,
                x="hours",
                y="drift",
                hue="method",
                errorbar="se",
                marker="",
                linewidth=0.9,
                palette=palette,
                ax=axm,
            )
            axm.set_title(
                f"Spatial drift of saliency\n{readout}\n{m}", fontsize=cfg.TITLE_SIZE
            )
            axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)

            if i == 0:
                axm.set_ylabel(
                    "center of mass displacement [pixels]", fontsize=cfg.AXIS_LABEL_SIZE
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
                ncol=1,
                title="Method",
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
        data = data[data["time"] != 1].copy()
        data["loop"] = data["time"].astype(int)
        data["hours"] = data["loop"] / 2.0

        # mean across experiments + wells
        agg = data.groupby(["model", "hours", "method"], as_index=False)["drift"].mean()

        models = sorted(agg["model"].unique())

        sub = gs.subgridspec(1, 4, wspace=0)

        n_methods = agg["method"].nunique()
        palette = "tab10"

        axes = []
        handles, labels = None, None

        for i, m in enumerate(models):
            if i >= 3:
                break  # only first 3 models shown here
            axm = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            dat = agg[agg["model"] == m]
            sns.lineplot(
                data=dat,
                x="hours",
                y="drift",
                hue="method",
                errorbar="se",
                marker="",
                linewidth=0.9,
                palette=palette,
                ax=axm,
            )
            axm.set_title(
                f"Spatial drift of saliency\n{readout}\n{m}", fontsize=cfg.TITLE_SIZE
            )
            axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)

            if i == 0:
                axm.set_ylabel(
                    "center of mass displacement [pixels]", fontsize=cfg.AXIS_LABEL_SIZE
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
                ncol=1,
                title="Method",
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
        data = data[data["time"] != 1].copy()
        data["loop"] = data["time"].astype(int)
        data["hours"] = data["loop"] / 2.0

        # mean across experiments + wells
        agg = data.groupby(["model", "hours", "method"], as_index=False)["drift"].mean()

        models = sorted(agg["model"].unique())

        sub = gs.subgridspec(1, 4, wspace=0)

        n_methods = agg["method"].nunique()
        palette = "tab10"

        axes = []
        handles, labels = None, None

        for i, m in enumerate(models):
            if i >= 3:
                break  # only first 3 models shown here
            axm = fig.add_subplot(sub[0, i], sharey=axes[0] if axes else None)
            dat = agg[agg["model"] == m]
            sns.lineplot(
                data=dat,
                x="hours",
                y="drift",
                hue="method",
                errorbar="se",
                marker="",
                linewidth=0.9,
                palette=palette,
                ax=axm,
            )
            axm.set_title(
                f"Spatial drift of saliency\n{readout}\n{m}", fontsize=cfg.TITLE_SIZE
            )
            axm.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
            axm.tick_params(labelsize=cfg.AXIS_LABEL_SIZE)

            if i == 0:
                axm.set_ylabel(
                    "center of mass displacement [pixels]", fontsize=cfg.AXIS_LABEL_SIZE
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
                ncol=1,
                title="Method",
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


def figure_S37_generation(
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
        result="entropy_drift_timeseries",
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
        result="entropy_drift_timeseries",
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
        result="entropy_drift_timeseries",
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
        result="entropy_drift_timeseries",
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
    _generate_main_figure(
        rpe_sal=rpe_saliency_results,
        lens_sal=lens_saliency_results,
        rpe_classes_sal=rpe_classes_saliency_results,
        lens_classes_sal=lens_classes_saliency_results,
        figure_output_dir=figure_output_dir,
        figure_name="Supplementary_Figure_S37",
    )
