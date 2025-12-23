import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import get_cnn_output


def _generate_main_figure(
    rpe_output: pd.DataFrame,
    lens_output: pd.DataFrame,
    figure_output_dir: str = "",
    sketch_dir: str = "",
    figure_name: str = "",
):
    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.3)
        fig_sgs = gs.subgridspec(1, 4)

        readout = "RPE_Final"
        model = "DenseNet121"

        data = rpe_output
        data = data[data["Readout"] == readout]
        data["ValExpID"] = data["ValExpID"].map(cfg.EXPERIMENT_MAP)

        train_f1 = fig.add_subplot(fig_sgs[0])
        test_f1 = fig.add_subplot(fig_sgs[1])
        val_f1 = fig.add_subplot(fig_sgs[2])

        legend = fig.add_subplot(fig_sgs[3])

        plot_data = data[data["Model"] == model]
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TrainF1",
            hue="ValExpID",
            ax=train_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TestF1",
            hue="ValExpID",
            ax=test_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="ValF1",
            hue="ValExpID",
            ax=val_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )

        f1_ylim = (0.3, 1.01)
        handles, labels = train_f1.get_legend_handles_labels()
        for axis in [train_f1, test_f1, val_f1]:
            axis.legend().remove()
            axis.tick_params(**cfg.TICKPARAMS_PARAMS)
            axis.set_xlabel(axis.get_xlabel(), fontsize=cfg.AXIS_LABEL_SIZE)
            axis.set_ylabel(axis.get_ylabel(), fontsize=cfg.AXIS_LABEL_SIZE)

        for axis in [train_f1, test_f1, val_f1]:
            axis.set_ylim(f1_ylim)
            axis.set_xlabel("Epoch")

        legend.axis("off")
        legend.legend(
            handles,
            labels,
            loc="center left",
            fontsize=cfg.TITLE_SIZE,
            **cfg.TWO_COL_LEGEND,
        )

        test_f1.set_title(f"{model}: RPE emergence", fontsize=cfg.TITLE_SIZE)

        return

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.3)
        fig_sgs = gs.subgridspec(1, 4)

        readout = "RPE_Final"
        model = "ResNet50"

        data = rpe_output
        data = data[data["Readout"] == readout]
        data["ValExpID"] = data["ValExpID"].map(cfg.EXPERIMENT_MAP)

        train_f1 = fig.add_subplot(fig_sgs[0])
        test_f1 = fig.add_subplot(fig_sgs[1])
        val_f1 = fig.add_subplot(fig_sgs[2])

        legend = fig.add_subplot(fig_sgs[3])

        plot_data = data[data["Model"] == model]
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TrainF1",
            hue="ValExpID",
            ax=train_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TestF1",
            hue="ValExpID",
            ax=test_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="ValF1",
            hue="ValExpID",
            ax=val_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )

        f1_ylim = (0.3, 1.01)
        handles, labels = train_f1.get_legend_handles_labels()
        for axis in [train_f1, test_f1, val_f1]:
            axis.legend().remove()
            axis.tick_params(**cfg.TICKPARAMS_PARAMS)
            axis.set_xlabel(axis.get_xlabel(), fontsize=cfg.AXIS_LABEL_SIZE)
            axis.set_ylabel(axis.get_ylabel(), fontsize=cfg.AXIS_LABEL_SIZE)

        for axis in [train_f1, test_f1, val_f1]:
            axis.set_ylim(f1_ylim)
            axis.set_xlabel("Epoch")

        legend.axis("off")
        legend.legend(
            handles,
            labels,
            loc="center left",
            fontsize=cfg.TITLE_SIZE,
            **cfg.TWO_COL_LEGEND,
        )

        test_f1.set_title(f"{model}: RPE emergence", fontsize=cfg.TITLE_SIZE)

        return

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.3)
        fig_sgs = gs.subgridspec(1, 4)

        readout = "RPE_Final"
        model = "MobileNetV3_Large"

        data = rpe_output
        data = data[data["Readout"] == readout]
        data["ValExpID"] = data["ValExpID"].map(cfg.EXPERIMENT_MAP)

        train_f1 = fig.add_subplot(fig_sgs[0])
        test_f1 = fig.add_subplot(fig_sgs[1])
        val_f1 = fig.add_subplot(fig_sgs[2])

        legend = fig.add_subplot(fig_sgs[3])

        plot_data = data[data["Model"] == model]
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TrainF1",
            hue="ValExpID",
            ax=train_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TestF1",
            hue="ValExpID",
            ax=test_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="ValF1",
            hue="ValExpID",
            ax=val_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )

        f1_ylim = (0.3, 1.01)
        handles, labels = train_f1.get_legend_handles_labels()
        for axis in [train_f1, test_f1, val_f1]:
            axis.legend().remove()
            axis.tick_params(**cfg.TICKPARAMS_PARAMS)
            axis.set_xlabel(axis.get_xlabel(), fontsize=cfg.AXIS_LABEL_SIZE)
            axis.set_ylabel(axis.get_ylabel(), fontsize=cfg.AXIS_LABEL_SIZE)

        for axis in [train_f1, test_f1, val_f1]:
            axis.set_ylim(f1_ylim)
            axis.set_xlabel("Epoch")

        legend.axis("off")
        legend.legend(
            handles,
            labels,
            loc="center left",
            fontsize=cfg.TITLE_SIZE,
            **cfg.TWO_COL_LEGEND,
        )

        test_f1.set_title(f"{model}: RPE emergence", fontsize=cfg.TITLE_SIZE)

        return

    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.3)
        fig_sgs = gs.subgridspec(1, 4)

        readout = "Lens_Final"
        model = "DenseNet121"

        data = lens_output
        data = data[data["Readout"] == readout]
        data["ValExpID"] = data["ValExpID"].map(cfg.EXPERIMENT_MAP)

        train_f1 = fig.add_subplot(fig_sgs[0])
        test_f1 = fig.add_subplot(fig_sgs[1])
        val_f1 = fig.add_subplot(fig_sgs[2])

        legend = fig.add_subplot(fig_sgs[3])

        plot_data = data[data["Model"] == model]
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TrainF1",
            hue="ValExpID",
            ax=train_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TestF1",
            hue="ValExpID",
            ax=test_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="ValF1",
            hue="ValExpID",
            ax=val_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )

        f1_ylim = (0.3, 1.01)
        handles, labels = train_f1.get_legend_handles_labels()
        for axis in [train_f1, test_f1, val_f1]:
            axis.legend().remove()
            axis.tick_params(**cfg.TICKPARAMS_PARAMS)
            axis.set_xlabel(axis.get_xlabel(), fontsize=cfg.AXIS_LABEL_SIZE)
            axis.set_ylabel(axis.get_ylabel(), fontsize=cfg.AXIS_LABEL_SIZE)

        for axis in [train_f1, test_f1, val_f1]:
            axis.set_ylim(f1_ylim)
            axis.set_xlabel("Epoch")

        legend.axis("off")
        legend.legend(
            handles,
            labels,
            loc="center left",
            fontsize=cfg.TITLE_SIZE,
            **cfg.TWO_COL_LEGEND,
        )

        test_f1.set_title(f"{model}: lens emergence", fontsize=cfg.TITLE_SIZE)

        return

    def generate_subfigure_e(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.3)
        fig_sgs = gs.subgridspec(1, 4)

        readout = "Lens_Final"
        model = "ResNet50"

        data = lens_output
        data = data[data["Readout"] == readout]
        data["ValExpID"] = data["ValExpID"].map(cfg.EXPERIMENT_MAP)

        train_f1 = fig.add_subplot(fig_sgs[0])
        test_f1 = fig.add_subplot(fig_sgs[1])
        val_f1 = fig.add_subplot(fig_sgs[2])

        legend = fig.add_subplot(fig_sgs[3])

        plot_data = data[data["Model"] == model]
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TrainF1",
            hue="ValExpID",
            ax=train_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TestF1",
            hue="ValExpID",
            ax=test_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="ValF1",
            hue="ValExpID",
            ax=val_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )

        f1_ylim = (0.3, 1.01)
        handles, labels = train_f1.get_legend_handles_labels()
        for axis in [train_f1, test_f1, val_f1]:
            axis.legend().remove()
            axis.tick_params(**cfg.TICKPARAMS_PARAMS)
            axis.set_xlabel(axis.get_xlabel(), fontsize=cfg.AXIS_LABEL_SIZE)
            axis.set_ylabel(axis.get_ylabel(), fontsize=cfg.AXIS_LABEL_SIZE)

        for axis in [train_f1, test_f1, val_f1]:
            axis.set_ylim(f1_ylim)
            axis.set_xlabel("Epoch")

        legend.axis("off")
        legend.legend(
            handles,
            labels,
            loc="center left",
            fontsize=cfg.TITLE_SIZE,
            **cfg.TWO_COL_LEGEND,
        )

        test_f1.set_title(f"{model}: lens emergence", fontsize=cfg.TITLE_SIZE)

        return

    def generate_subfigure_f(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.3)
        fig_sgs = gs.subgridspec(1, 4)

        readout = "Lens_Final"
        model = "MobileNetV3_Large"

        data = lens_output
        data = data[data["Readout"] == readout]
        data["ValExpID"] = data["ValExpID"].map(cfg.EXPERIMENT_MAP)

        train_f1 = fig.add_subplot(fig_sgs[0])
        test_f1 = fig.add_subplot(fig_sgs[1])
        val_f1 = fig.add_subplot(fig_sgs[2])

        legend = fig.add_subplot(fig_sgs[3])

        plot_data = data[data["Model"] == model]
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TrainF1",
            hue="ValExpID",
            ax=train_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="TestF1",
            hue="ValExpID",
            ax=test_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )
        sns.lineplot(
            data=plot_data,
            x="Epoch",
            y="ValF1",
            hue="ValExpID",
            ax=val_f1,
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
            linewidth=0.75,
        )

        f1_ylim = (0.3, 1.01)
        handles, labels = train_f1.get_legend_handles_labels()
        for axis in [train_f1, test_f1, val_f1]:
            axis.legend().remove()
            axis.tick_params(**cfg.TICKPARAMS_PARAMS)
            axis.set_xlabel(axis.get_xlabel(), fontsize=cfg.AXIS_LABEL_SIZE)
            axis.set_ylabel(axis.get_ylabel(), fontsize=cfg.AXIS_LABEL_SIZE)

        for axis in [train_f1, test_f1, val_f1]:
            axis.set_ylim(f1_ylim)
            axis.set_xlabel("Epoch")

        legend.axis("off")
        legend.legend(
            handles,
            labels,
            loc="center left",
            fontsize=cfg.TITLE_SIZE,
            **cfg.TWO_COL_LEGEND,
        )

        test_f1.set_title(f"{model}: lens emergence", fontsize=cfg.TITLE_SIZE)

        return

    fig = plt.figure(
        layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL)
    )
    gs = GridSpec(ncols=6, nrows=6, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1])
    a_coords = gs[0, :]
    b_coords = gs[1, :]
    c_coords = gs[2, :]
    d_coords = gs[3, :]
    e_coords = gs[4, :]
    f_coords = gs[5, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)
    fig_e = fig.add_subplot(e_coords)
    fig_f = fig.add_subplot(f_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")
    generate_subfigure_d(fig, fig_d, d_coords, "D")
    generate_subfigure_e(fig, fig_e, e_coords, "E")
    generate_subfigure_f(fig, fig_f, f_coords, "F")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.png")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")

    return


def figure_S11_generation(
    rpe_classification_dir: str,
    lens_classification_dir: str,
    figure_data_dir: str,
    figure_output_dir: str,
    **kwargs,
) -> None:
    rpe_final_output = get_cnn_output(
        classification_dir=rpe_classification_dir,
        readout="RPE_Final",
        proj="SL3",
        output_dir=figure_data_dir,
    )
    lens_final_output = get_cnn_output(
        classification_dir=lens_classification_dir,
        readout="Lens_Final",
        proj="SL3",
        output_dir=figure_data_dir,
    )

    _generate_main_figure(
        rpe_output=rpe_final_output,
        lens_output=lens_final_output,
        figure_output_dir=figure_output_dir,
        figure_name="Supplementary_Figure_S11",
    )

    final_frame_output_dir = os.path.join(figure_output_dir, "Data_S38.csv")
    final_frame = pd.concat([rpe_final_output, lens_final_output], axis = 0)
    final_frame.to_csv(final_frame_output_dir, index = False)

    return
