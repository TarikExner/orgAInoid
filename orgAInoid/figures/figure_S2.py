import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import (
    get_morphometrics_frame,
    calculate_organoid_dimensionality_reduction,
    calculate_organoid_distances,
)
from .figure_data_utils import get_data_columns_morphometrics


def _generate_main_figure(
    dimred_data: pd.DataFrame,
    distance_data: pd.DataFrame,
    figure_output_dir: str = "",
    sketch_dir: str = "",
    figure_name: str = "",
):
    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.45)

        data = dimred_data.copy()
        data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
        data["timeframe"] = data["timeframe"].astype(str)
        data["timeframe"] = data["timeframe"].astype("category")
        data = data.sort_values(["experiment", "timeframe"], ascending=[True, True])

        experiments = data["experiment"].unique()
        label_padding = 0

        n_experiments = len(experiments)
        if n_experiments % 2 == 0:
            fig_sgs = gs.subgridspec(int(n_experiments / 3) + 2, 6)
        else:
            fig_sgs = gs.subgridspec(int(n_experiments / 3) + 1, 6)

        point_kwargs = {"s": 3, "edgecolor": "black", "linewidth": 0.1}

        row_counter = 0
        col_counter = 0
        for i, exp in enumerate(experiments):
            if i % 3 == 0 and i != 0:
                row_counter += 1

            tsne_plot_well = fig.add_subplot(fig_sgs[row_counter, col_counter])
            tsne_plot_timeframe = fig.add_subplot(fig_sgs[row_counter, col_counter + 1])

            plot_data = data[data["experiment"] == exp]
            sns.scatterplot(
                data=plot_data,
                x="TSNE1",
                y="TSNE2",
                hue="well",
                ax=tsne_plot_well,
                **point_kwargs,
                legend=False,
                rasterized=True,
            )
            sns.scatterplot(
                data=plot_data,
                x="TSNE1",
                y="TSNE2",
                hue="timeframe",
                ax=tsne_plot_timeframe,
                **point_kwargs,
                rasterized=True,
            )
            tsne_plot_timeframe.legend().remove()

            utils._remove_axis_labels(tsne_plot_well)
            tsne_plot_well.xaxis.labelpad = label_padding
            tsne_plot_well.yaxis.labelpad = label_padding
            tsne_plot_well.set_xlabel(
                "TSNE1", fontsize=cfg.AXIS_LABEL_SIZE, labelpad=label_padding
            )
            tsne_plot_well.set_ylabel(
                "TSNE2", fontsize=cfg.AXIS_LABEL_SIZE, labelpad=label_padding
            )
            tsne_plot_well.set_title(
                f"{exp}: Colored\nby organoid", fontsize=cfg.AXIS_LABEL_SIZE
            )

            utils._remove_axis_labels(tsne_plot_timeframe)
            tsne_plot_timeframe.xaxis.labelpad = label_padding
            tsne_plot_timeframe.yaxis.labelpad = label_padding
            tsne_plot_timeframe.set_xlabel(
                "TSNE1", fontsize=cfg.AXIS_LABEL_SIZE, labelpad=label_padding
            )
            tsne_plot_timeframe.set_ylabel(
                "TSNE2", fontsize=cfg.AXIS_LABEL_SIZE, labelpad=label_padding
            )
            tsne_plot_timeframe.set_title(
                f"{exp}: Colored\nby timeframe", fontsize=cfg.AXIS_LABEL_SIZE
            )

            if col_counter in [0, 2]:
                col_counter += 2
            else:
                col_counter = 0

        timeframe_legend = fig.add_subplot(fig_sgs[row_counter, col_counter])

        handles, labels = tsne_plot_timeframe.get_legend_handles_labels()
        labels = [
            " 0 - 12h",
            "13 - 24h",
            "25 - 36h",
            "37 - 48h",
            "49 - 60h",
            "61 - 72h",
        ]
        timeframe_legend.legend(
            handles,
            labels,
            loc="center left",
            title="timeframe",
            markerscale=3,
            title_fontsize=cfg.TITLE_SIZE,
            fontsize=cfg.AXIS_LABEL_SIZE,
        )
        utils._prep_image_axis(timeframe_legend)
        return

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.45)

        data = distance_data.copy()
        data["hours"] = data["loop"] / 2
        data["experiment"] = data["experiment"].map(cfg.EXPERIMENT_MAP)
        data = data.sort_values("experiment", ascending=True)
        data["experiment"] = data["experiment"].astype("category")
        fig_sgs = gs.subgridspec(1, 1)
        distance_plot = fig.add_subplot(fig_sgs[0])

        sns.lineplot(
            data=data[data["distance_type"] == "interorganoid"],
            x="hours",
            y="distance",
            hue="experiment",
            palette="tab20",
            errorbar="se",
            ax=distance_plot,
        )
        distance_plot.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        distance_plot.set_ylabel(
            "Interorganoid Distance (euclidean)", fontsize=cfg.AXIS_LABEL_SIZE
        )
        distance_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        distance_plot.set_title(
            "Interorganoid distances over time", fontsize=cfg.TITLE_SIZE
        )
        distance_plot.set_xlim(
            distance_plot.get_xlim()[0], distance_plot.get_xlim()[1] + 17
        )
        distance_plot.legend(
            loc="center right",
            fontsize=cfg.TITLE_SIZE,
            title="",
            **cfg.TWO_COL_LEGEND,
            markerscale=0.5,
        )

        return

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: GridSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.45)

        data = distance_data.copy()
        data["hours"] = data["loop"] / 2
        data["experiment"] = data["experiment"].astype("category")
        data = data[~data["experiment"].isin(["E002", "E007", "E012"])]

        fig_sgs = gs.subgridspec(1, 1)
        distance_plot = fig.add_subplot(fig_sgs[0])

        sns.lineplot(
            data=data[data["distance_type"] == "intraorganoid"],
            x="hours",
            y="distance",
            errorbar="se",
            ax=distance_plot,
        )
        distance_plot.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        distance_plot.set_ylabel(
            "Intraorganoid Distance (euclidean)", fontsize=cfg.AXIS_LABEL_SIZE
        )
        distance_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        distance_plot.set_title(
            "Intraorganoid distances over time", fontsize=cfg.TITLE_SIZE
        )

        return

    fig = plt.figure(
        layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL)
    )
    gs = GridSpec(ncols=1, nrows=3, figure=fig, height_ratios=[1.4, 0.5, 0.5])
    a_coords = gs[0, :]
    b_coords = gs[1, :]
    c_coords = gs[2, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.png")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")

    return


def figure_S2_generation(
    morphometrics_dir: str, figure_data_dir: str, figure_output_dir: str, **kwargs
):
    morphometrics = get_morphometrics_frame(morphometrics_dir)
    data_columns = get_data_columns_morphometrics(morphometrics)

    dimreds = calculate_organoid_dimensionality_reduction(
        morphometrics, data_columns, use_pca=True, output_dir=figure_data_dir
    )
    morphometrics = morphometrics[morphometrics["Condition"] == "0nM"]
    assert isinstance(morphometrics, pd.DataFrame)
    organoid_distances = calculate_organoid_distances(
        morphometrics,
        data_columns,
        use_pca=True,
        output_dir=figure_data_dir,
        output_filename="organoid_distances_no_Wnt",
    )

    _generate_main_figure(
        dimred_data=dimreds,
        distance_data=organoid_distances,
        figure_output_dir=figure_output_dir,
        figure_name="Supplementary_Figure_S2",
    )

    dist_output_dir = os.path.join(figure_output_dir, "Data_S1_SF2a.csv")
    organoid_distances["experiment"] = organoid_distances["experiment"].map(cfg.EXPERIMENT_MAP)
    organoid_distances.to_csv(dist_output_dir, index = False)

    dimred_output_dir = os.path.join(figure_output_dir, "Data_S1_SF2bc.csv")
    dimreds["experiment"] = dimreds["experiment"].map(cfg.EXPERIMENT_MAP)
    dimreds.to_csv(dimred_output_dir, index = False)

    return
