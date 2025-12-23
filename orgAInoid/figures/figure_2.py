import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib.patches import Rectangle

from typing import cast

import cv2

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import (
    get_morphometrics_frame,
    calculate_organoid_dimensionality_reduction,
    calculate_organoid_distances,
    compare_neighbors_by_experiment,
    neighbors_per_well_by_experiment,
    PC_COLUMNS,
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
        """Will contain the experimental overview sketch"""
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.45)

        fig_sgs = gs.subgridspec(1, 1)

        sketch = fig.add_subplot(fig_sgs[0])
        utils._prep_image_axis(sketch)
        img = cv2.imread(os.path.join(sketch_dir, "Figure_2.png"), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sketch.imshow(img)
        return

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.45)

        data = dimred_data.copy()
        data = data[data["experiment"] == "E001"]
        data["timeframe"] = data["timeframe"].astype(str)
        data = data.sort_values("timeframe", ascending=True)

        fig_sgs = gs.subgridspec(1, 5)

        point_kwargs = {"s": 5, "edgecolor": "black", "linewidth": 0.05}
        point_kwargs_inset = {"s": 10, "edgecolor": "black", "linewidth": 0.1}

        tsne_well_overview = fig.add_subplot(fig_sgs[0])
        tsne_well_inset = fig.add_subplot(fig_sgs[1])
        tsne_timeframe_overview = fig.add_subplot(fig_sgs[2])
        tsne_timeframe_inset = fig.add_subplot(fig_sgs[3])
        timeframe_legend = fig.add_subplot(fig_sgs[4])

        rect_coords = {"x": -1, "y": -1, "width": 50, "height": 50}
        rect1 = Rectangle(
            (rect_coords["x"], rect_coords["y"]),
            rect_coords["width"],
            rect_coords["height"],
            linewidth=3,
            edgecolor="black",
            facecolor="none",
        )
        rect2 = Rectangle(
            (rect_coords["x"], rect_coords["y"]),
            rect_coords["width"],
            rect_coords["height"],
            linewidth=3,
            edgecolor="black",
            facecolor="none",
        )
        inset_mask = (
            (data["TSNE1"] >= rect_coords["x"])
            & (data["TSNE1"] <= rect_coords["x"] + rect_coords["width"])
            & (data["TSNE2"] >= rect_coords["y"])
            & (data["TSNE2"] <= rect_coords["y"] + rect_coords["height"])
        )
        inset_data = data[inset_mask]

        sns.scatterplot(
            data=data,
            x="TSNE1",
            y="TSNE2",
            hue="well",
            ax=tsne_well_overview,
            **point_kwargs,
            rasterized=True,
        )
        utils._remove_axis_labels(tsne_well_overview)
        tsne_well_overview.legend().remove()
        tsne_well_overview.add_patch(rect1)
        tsne_well_overview.set_title("Colored by organoid\n", fontsize=cfg.TITLE_SIZE)
        tsne_well_overview.set_xlabel(
            tsne_well_overview.get_xlabel(), fontsize=cfg.AXIS_LABEL_SIZE
        )
        tsne_well_overview.set_ylabel(
            tsne_well_overview.get_ylabel(), fontsize=cfg.AXIS_LABEL_SIZE
        )

        sns.scatterplot(
            data=inset_data,
            x="TSNE1",
            y="TSNE2",
            hue="well",
            ax=tsne_well_inset,
            **point_kwargs_inset,
            legend=False,
            rasterized=True,
        )
        utils._remove_axis_labels(tsne_well_inset)
        tsne_well_inset.set_xlabel(
            tsne_well_inset.get_xlabel(), fontsize=cfg.AXIS_LABEL_SIZE
        )
        tsne_well_inset.set_ylabel(
            tsne_well_inset.get_ylabel(), fontsize=cfg.AXIS_LABEL_SIZE
        )
        tsne_well_inset.set_title(
            "Colored by organoid\nenlarged inset", fontsize=cfg.TITLE_SIZE
        )

        sns.scatterplot(
            data=data,
            x="TSNE1",
            y="TSNE2",
            hue="timeframe",
            ax=tsne_timeframe_overview,
            **point_kwargs,
            rasterized=True,
        )
        utils._remove_axis_labels(tsne_timeframe_overview)
        tsne_timeframe_overview.legend().remove()
        tsne_timeframe_overview.add_patch(rect2)
        tsne_timeframe_overview.set_title(
            "Colored by timeframe\n", fontsize=cfg.TITLE_SIZE
        )
        tsne_timeframe_overview.set_xlabel(
            tsne_timeframe_overview.get_xlabel(), fontsize=cfg.AXIS_LABEL_SIZE
        )
        tsne_timeframe_overview.set_ylabel(
            tsne_timeframe_overview.get_ylabel(), fontsize=cfg.AXIS_LABEL_SIZE
        )

        sns.scatterplot(
            data=inset_data,
            x="TSNE1",
            y="TSNE2",
            hue="timeframe",
            ax=tsne_timeframe_inset,
            **point_kwargs_inset,
            legend=False,
            rasterized=True,
        )
        utils._remove_axis_labels(tsne_timeframe_inset)
        tsne_timeframe_inset.set_title(
            "Colored by timeframe\nenlarged inset", fontsize=cfg.TITLE_SIZE
        )
        tsne_timeframe_inset.set_xlabel(
            tsne_timeframe_inset.get_xlabel(), fontsize=cfg.AXIS_LABEL_SIZE
        )
        tsne_timeframe_inset.set_ylabel(
            tsne_timeframe_inset.get_ylabel(), fontsize=cfg.AXIS_LABEL_SIZE
        )

        handles, labels = tsne_timeframe_overview.get_legend_handles_labels()
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

    def generate_subfigure_c(
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
            data=cast(pd.DataFrame, data[data["distance_type"] == "interorganoid"]),
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
            loc="center right", fontsize=cfg.TITLE_SIZE, title="", **cfg.TWO_COL_LEGEND
        )

        return

    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
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
            data=cast(pd.DataFrame, data[data["distance_type"] == "intraorganoid"]),
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
    gs = GridSpec(ncols=6, nrows=4, figure=fig, height_ratios=[0.8, 0.6, 1, 1])

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
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")

    return


def figure_2_generation(
    morphometrics_dir: str,
    figure_data_dir: str,
    sketch_dir: str,
    figure_output_dir: str,
    **kwargs,
):
    morphometrics = get_morphometrics_frame(morphometrics_dir)
    data_columns = get_data_columns_morphometrics(morphometrics)

    # we use n_pcs 20 for everything
    pc_columns = PC_COLUMNS

    organoid_distances_pca = calculate_organoid_distances(
        morphometrics, data_columns, use_pca=True, output_dir=figure_data_dir
    )
    organoid_distances = calculate_organoid_distances(
        morphometrics, data_columns, use_pca=False, output_dir=figure_data_dir
    )

    dimreds_pca = calculate_organoid_dimensionality_reduction(
        morphometrics, data_columns, use_pca=True, output_dir=figure_data_dir
    )
    dimreds = calculate_organoid_dimensionality_reduction(
        morphometrics, data_columns, use_pca=False, output_dir=figure_data_dir
    )

    jaccard_pca_umap = compare_neighbors_by_experiment(
        dimreds_pca,
        dimred="UMAP",
        user_suffix="pca",
        data_cols=pc_columns,
        output_dir=figure_data_dir,
    )
    jaccard_pca_tsne = compare_neighbors_by_experiment(
        dimreds_pca,
        dimred="TSNE",
        user_suffix="pca",
        data_cols=pc_columns,
        output_dir=figure_data_dir,
    )
    jaccard_raw_umap = compare_neighbors_by_experiment(
        dimreds,
        dimred="UMAP",
        user_suffix="raw",
        data_cols=data_columns,
        output_dir=figure_data_dir,
    )
    jaccard_raw_tsne = compare_neighbors_by_experiment(
        dimreds,
        dimred="TSNE",
        user_suffix="raw",
        data_cols=data_columns,
        output_dir=figure_data_dir,
    )

    well_frac_pca_umap = neighbors_per_well_by_experiment(
        dimreds_pca,
        dimred="UMAP",
        user_suffix="pca",
        data_cols=pc_columns,
        output_dir=figure_data_dir,
    )
    well_frac_pca_tsne = neighbors_per_well_by_experiment(
        dimreds_pca,
        dimred="TSNE",
        user_suffix="pca",
        data_cols=pc_columns,
        output_dir=figure_data_dir,
    )
    well_frac_raw_umap = neighbors_per_well_by_experiment(
        dimreds,
        dimred="UMAP",
        user_suffix="raw",
        data_cols=data_columns,
        output_dir=figure_data_dir,
    )
    well_frac_raw_umap = neighbors_per_well_by_experiment(
        dimreds,
        dimred="TSNE",
        user_suffix="raw",
        data_cols=data_columns,
        output_dir=figure_data_dir,
    )

    _generate_main_figure(
        dimred_data=dimreds_pca,
        distance_data=organoid_distances_pca,
        figure_output_dir=figure_output_dir,
        figure_name="Figure_2",
        sketch_dir=sketch_dir,
    )

    dimred_output_dir = os.path.join(figure_output_dir, "Data_S2.csv")
    dimreds_pca.to_csv(dimred_output_dir, index = False)

    dist_output_dir = os.path.join(figure_output_dir, "Data_S3.csv")
    organoid_distances_pca.to_csv(dist_output_dir, index = False)

    return
