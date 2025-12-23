import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase

from typing import cast

import cv2

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import get_dataset_annotations


def _generate_main_figure(
    annotation_data: pd.DataFrame,
    figure_output_dir: str = "",
    microscopy_dir: str = "",
    sketch_dir: str = "",
    figure_name: str = "",
):
    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        """Will contain the experimental overview sketch"""
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.3)
        fig_sgs = gs.subgridspec(1, 1)

        sketch = fig.add_subplot(fig_sgs[0])
        utils._prep_image_axis(sketch)
        img = cv2.imread(os.path.join(sketch_dir, "Figure_1.png"), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sketch.imshow(img)
        return

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        """Contains the raw values of RPE/Lens over all organoids"""
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.3)

        y_label = "number of organoids"
        x_label = "experiment"

        fig_sgs = gs.subgridspec(1, 2)
        data = annotation_data.copy()
        data["experiment"] = [cfg.EXPERIMENT_MAP[exp] for exp in data["experiment"]]
        data = data.sort_values("experiment", ascending=True)

        rpe_crosstab = pd.crosstab(data["experiment"], data["RPE_Final"])
        lens_crosstab = pd.crosstab(data["experiment"], data["Lens_Final"])

        rpe_plot = fig.add_subplot(fig_sgs[0])
        lens_plot = fig.add_subplot(fig_sgs[1])

        n_control_organoids = (
            data[data["Condition"] == "0nM"].groupby("experiment").size().tolist()
        )
        rpe_crosstab.plot(
            kind="bar",
            stacked=True,
            ax=rpe_plot,
            color=["white", "black"],
            edgecolor="black",
            align="center",
        )

        x_positions = []
        widths = []
        for i, bar in enumerate(rpe_plot.patches):
            x_positions.append(bar.get_x())
            widths.append(bar.get_width())
        x_positions = x_positions[:11]
        widths = widths[:11]
        x_positions = [x + width / 2 for x, width in zip(x_positions, widths)]
        rpe_plot.bar(
            x_positions,
            n_control_organoids,
            widths,
            align="center",
            label="No\nWnt-\ntreat-\nment",
            color="white",
            edgecolor="black",
            hatch="////",
            alpha=0.7,
        )

        rpe_plot.set_xlabel(x_label, fontsize=cfg.AXIS_LABEL_SIZE)
        rpe_plot.set_ylabel(y_label, fontsize=cfg.AXIS_LABEL_SIZE)
        rpe_plot.set_title("RPE presence at last timepoint", fontsize=cfg.TITLE_SIZE)
        rpe_plot.legend().remove()
        rpe_plot.tick_params(**cfg.TICKPARAMS_PARAMS)

        lens_crosstab.plot(
            kind="bar",
            stacked=True,
            ax=lens_plot,
            color=["white", "cyan"],
            edgecolor="black",
        )
        lens_plot.set_xlabel(x_label, fontsize=cfg.AXIS_LABEL_SIZE)
        lens_plot.set_ylabel(y_label, fontsize=cfg.AXIS_LABEL_SIZE)
        lens_plot.set_title("Lens presence at last timepoint", fontsize=cfg.TITLE_SIZE)

        class HandlerDiagonalSplitRectangle(HandlerBase):
            def create_artists(
                self,
                legend,
                orig_handle,
                xdescent,
                ydescent,
                width,
                height,
                fontsize,
                trans,
            ):
                coords_left = [
                    (xdescent, ydescent),
                    (xdescent + width, ydescent),
                    (xdescent, ydescent + height),
                ]

                coords_right = [
                    (xdescent + width, ydescent),
                    (xdescent + width, ydescent + height),
                    (xdescent, ydescent + height),
                ]

                triangle_left = mpatches.Polygon(
                    coords_left,
                    facecolor="black",
                    edgecolor="none",
                    linewidth=0,
                    transform=trans,
                )
                triangle_right = mpatches.Polygon(
                    coords_right,
                    facecolor="cyan",
                    edgecolor="none",
                    linewidth=0,
                    transform=trans,
                )

                rect_outline = mpatches.Rectangle(
                    (xdescent, ydescent),
                    width,
                    height,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1,
                    transform=trans,
                )
                return [triangle_left, triangle_right, rect_outline]

        handles, labels = rpe_plot.get_legend_handles_labels()

        custom_handle = mpatches.Rectangle(
            (0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=1
        )
        hatched_handle = mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            hatch="////",
            label="No Wnt-treatment",
            alpha=0.7,
        )
        lens_plot.legend(
            [handles[0], custom_handle, hatched_handle],
            labels,
            handler_map={custom_handle: HandlerDiagonalSplitRectangle()},
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            title="contains\ntissue",
            fontsize=cfg.TITLE_SIZE,
            title_fontsize=cfg.TITLE_SIZE,
        )
        lens_plot.tick_params(**cfg.TICKPARAMS_PARAMS)

        return

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.3)
        data = annotation_data.copy()

        data["Total_RPE_amount"] = (
            data["Total_RPE_amount"] * cfg.RPE_UM_CONVERSION_FACTOR
        )

        fig_sgs = gs.subgridspec(3, 4)

        y_label = "proportion"

        left_histogram_plot = fig.add_subplot(fig_sgs[0, 0:2])
        right_histogram_plot = fig.add_subplot(fig_sgs[0, 2:])

        histkwargs = {
            "stat": "proportion",
            "bins": 20,
            "kde": True,
            "fill": False,
            "thresh": None,
        }
        sns.histplot(
            data=cast(pd.DataFrame, data[data["RPE_Final"] == "yes"]),
            x="Total_RPE_amount",
            **histkwargs,
            ax=left_histogram_plot,
            color="black",
        )
        sns.histplot(
            data=cast(pd.DataFrame, data[data["Lens_Final"] == "yes"]),
            x="Lens_area",
            **histkwargs,
            ax=right_histogram_plot,
            color="black",
        )

        left_histogram_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        left_histogram_plot.set_title("RPE area distribution", fontsize=cfg.TITLE_SIZE)
        left_histogram_plot.set_xlabel("area [µm²]", fontsize=cfg.AXIS_LABEL_SIZE)
        left_histogram_plot.set_ylabel(y_label, fontsize=cfg.AXIS_LABEL_SIZE)
        for cutoff in cfg.RPE_CUTOFFS:
            left_histogram_plot.axvline(
                x=cutoff * cfg.RPE_UM_CONVERSION_FACTOR, color="black"
            )

        right_histogram_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        right_histogram_plot.set_title(
            "Lens size distribution", fontsize=cfg.TITLE_SIZE
        )
        right_histogram_plot.set_xlabel("size [µm²]", fontsize=cfg.AXIS_LABEL_SIZE)
        right_histogram_plot.set_ylabel(y_label, fontsize=cfg.AXIS_LABEL_SIZE)

        for cutoff in cfg.LENS_CUTOFFS:
            right_histogram_plot.axvline(x=cutoff, color="black")

        class0_rpe_image = fig.add_subplot(fig_sgs[1, 0])
        class1_rpe_image = fig.add_subplot(fig_sgs[1, 1])
        class2_rpe_image = fig.add_subplot(fig_sgs[2, 0])
        class3_rpe_image = fig.add_subplot(fig_sgs[2, 1])

        class0_rpe_tif = cv2.imread(
            os.path.join(microscopy_dir, "E001E0001_cropped_adjusted_scaled.tif"),
            cv2.IMREAD_UNCHANGED,
        )
        class0_rpe_tif = cv2.cvtColor(class0_rpe_tif, cv2.COLOR_BGR2RGB)
        class0_rpe_image.imshow(class0_rpe_tif)
        class0_rpe_image.set_title("RPE: class 0", fontsize=cfg.TITLE_SIZE)
        utils.remove_ticks_and_labels(class0_rpe_image)
        class1_rpe_tif = cv2.imread(
            os.path.join(microscopy_dir, "E001C0002_cropped_adjusted_scaled.tif"),
            cv2.IMREAD_UNCHANGED,
        )
        class1_rpe_tif = cv2.cvtColor(class1_rpe_tif, cv2.COLOR_BGR2RGB)
        class1_rpe_image.imshow(class1_rpe_tif)
        class1_rpe_image.set_title("RPE: class 1", fontsize=cfg.TITLE_SIZE)
        utils.remove_ticks_and_labels(class1_rpe_image)
        class2_rpe_tif = cv2.imread(
            os.path.join(microscopy_dir, "E001A0010_cropped_adjusted_scaled.tif"),
            cv2.IMREAD_UNCHANGED,
        )
        class2_rpe_tif = cv2.cvtColor(class2_rpe_tif, cv2.COLOR_BGR2RGB)
        class2_rpe_image.imshow(class2_rpe_tif)
        class2_rpe_image.set_title("RPE: class 2", fontsize=cfg.TITLE_SIZE)
        utils.remove_ticks_and_labels(class2_rpe_image)
        class3_rpe_tif = cv2.imread(
            os.path.join(microscopy_dir, "E001F0008_cropped_adjusted_scaled.tif"),
            cv2.IMREAD_UNCHANGED,
        )
        class3_rpe_tif = cv2.cvtColor(class3_rpe_tif, cv2.COLOR_BGR2RGB)
        class3_rpe_image.imshow(class3_rpe_tif)
        class3_rpe_image.set_title("RPE: class 3", fontsize=cfg.TITLE_SIZE)
        utils.remove_ticks_and_labels(class3_rpe_image)

        class0_lens_image = fig.add_subplot(fig_sgs[1, 2])
        class1_lens_image = fig.add_subplot(fig_sgs[1, 3])
        class2_lens_image = fig.add_subplot(fig_sgs[2, 2])
        class3_lens_image = fig.add_subplot(fig_sgs[2, 3])

        class0_lens_tif = cv2.imread(
            os.path.join(
                microscopy_dir,
                "E008_Plate_Montage_loop_144_downsampled1_B010_cropped_adjusted_scaled.tif",
            ),
            cv2.IMREAD_UNCHANGED,
        )
        class0_lens_tif = cv2.cvtColor(class0_lens_tif, cv2.COLOR_BGR2RGB)
        class0_lens_image.imshow(class0_lens_tif)
        class0_lens_image.set_title("Lens: class 0", fontsize=cfg.TITLE_SIZE)
        utils.remove_ticks_and_labels(class0_lens_image)
        class1_lens_tif = cv2.imread(
            os.path.join(
                microscopy_dir,
                "E008_Plate_Montage_loop_144_downsampled1_A010_cropped_adjusted_scaled.tif",
            ),
            cv2.IMREAD_UNCHANGED,
        )
        class1_lens_tif = cv2.cvtColor(class1_lens_tif, cv2.COLOR_BGR2RGB)
        class1_lens_image.imshow(class1_lens_tif)
        class1_lens_image.set_title("Lens: class 1", fontsize=cfg.TITLE_SIZE)
        utils.remove_ticks_and_labels(class1_lens_image)
        class2_lens_tif = cv2.imread(
            os.path.join(
                microscopy_dir,
                "E008_Plate_Montage_loop_144_downsampled1_D009_cropped_adjusted_scaled.tif",
            ),
            cv2.IMREAD_UNCHANGED,
        )
        class2_lens_tif = cv2.cvtColor(class2_lens_tif, cv2.COLOR_BGR2RGB)
        class2_lens_image.imshow(class2_lens_tif)
        class2_lens_image.set_title("Lens: class 2", fontsize=cfg.TITLE_SIZE)
        utils.remove_ticks_and_labels(class2_lens_image)
        class3_lens_tif = cv2.imread(
            os.path.join(
                microscopy_dir,
                "E001_Plate_Montage_loop_144_downsampled1_E001_cropped_adjusted_scaled.tif",
            ),
            cv2.IMREAD_UNCHANGED,
        )
        class3_lens_tif = cv2.cvtColor(class3_lens_tif, cv2.COLOR_BGR2RGB)
        class3_lens_image.imshow(class3_lens_tif)
        class3_lens_image.set_title("Lens: class 3", fontsize=cfg.TITLE_SIZE)
        utils.remove_ticks_and_labels(class3_lens_image)

        return

    fig = plt.figure(
        layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL)
    )
    gs = GridSpec(ncols=6, nrows=3, figure=fig, height_ratios=[0.5, 0.4, 1])
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

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.tif")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")

    return


def figure_1_generation(
    annotations_dir: str,
    figure_data_dir: str,
    sketch_dir: str,
    microscopy_dir: str,
    figure_output_dir: str,
    **kwargs,
):
    dataset_annotations = get_dataset_annotations(
        annotations_dir, output_dir=figure_data_dir
    )

    _generate_main_figure(
        annotation_data=dataset_annotations,
        figure_output_dir=figure_output_dir,
        microscopy_dir=microscopy_dir,
        figure_name="Figure_1",
        sketch_dir=sketch_dir,
    )

    data_output_dir = os.path.join(figure_output_dir, "Data_S1.csv")
    dataset_annotations["experiment"] = dataset_annotations["experiment"].map(cfg.EXPERIMENT_MAP)
    dataset_annotations.to_csv(data_output_dir, index = False)

