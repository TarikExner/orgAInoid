import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from scipy.stats import f_oneway

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import (
    get_dataset_annotations,
    human_f1_RPE_visibility_conf_matrix,
    human_f1_per_evaluator,
    add_loop_from_timeframe,
)


def _generate_main_figure(
    annotation_data: pd.DataFrame,
    human_f1_data: pd.DataFrame,
    conf_matrix: np.ndarray,
    figure_output_dir: str = "./",
    figure_name: str = "",
):
    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.25)
        data = annotation_data.copy()
        data["experiment"] = [cfg.EXPERIMENT_MAP[exp] for exp in data["experiment"]]
        data = data.sort_values("experiment", ascending=True)

        fig_sgs = gs.subgridspec(1, 2)

        stripplot_params = cfg.STRIPPLOT_PARAMS.copy()
        stripplot_params["s"] = 5
        stripplot_params["dodge"] = False
        stripplot_params["jitter"] = 0.1
        stripplot_params["palette"] = "tab20"

        percentage_df = (
            data.groupby(["Condition", "experiment"])["RPE_Final"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .reset_index(name="RPE_Percentage")
        )
        area_df = (
            data.groupby(["Condition", "experiment"])["Total_RPE_amount"]
            .mean()
            .reset_index(name="RPE_area")
        )
        area_df["RPE_area"] = area_df["RPE_area"] * cfg.RPE_UM_CONVERSION_FACTOR

        conditions = area_df["Condition"].unique()

        grouped_data = [
            area_df[area_df["Condition"] == condition]["RPE_area"]
            for condition in conditions
            if condition != "0nM"
        ]
        assert len(grouped_data) == 3
        _, p_value_area = f_oneway(*grouped_data)

        grouped_data = [
            percentage_df[percentage_df["Condition"] == condition]["RPE_Percentage"]
            for condition in conditions
            if condition != "0nM"
        ]
        assert len(grouped_data) == 3
        _, p_value_percentage = f_oneway(*grouped_data)

        rpe_percentage = fig.add_subplot(fig_sgs[0])
        percentage_plot_kwargs = {
            "data": percentage_df,
            "x": "Condition",
            "y": "RPE_Percentage",
            "ax": rpe_percentage,
        }
        sns.stripplot(**percentage_plot_kwargs, hue="experiment", **stripplot_params)
        sns.boxplot(**percentage_plot_kwargs, **cfg.BOXPLOT_PARAMS)
        rpe_percentage.legend(
            bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=cfg.AXIS_LABEL_SIZE
        )
        rpe_percentage.set_title(
            "Presence of RPE at last time point\nby Wnt-surrogate",
            fontsize=cfg.TITLE_SIZE,
        )
        rpe_percentage.set_ylabel(
            "organoids with RPE [%]", fontsize=cfg.AXIS_LABEL_SIZE
        )
        rpe_percentage.set_xticklabels(["0 nM", "1 nM", "2 nM", "4 nM"])
        rpe_percentage.tick_params(**cfg.TICKPARAMS_PARAMS)
        rpe_percentage.set_xlabel(
            "Wnt-surrogate concentration [nM]", fontsize=cfg.AXIS_LABEL_SIZE
        )
        rpe_percentage.set_ylim(
            rpe_percentage.get_ylim()[0], rpe_percentage.get_ylim()[1] * 1.2
        )
        rpe_percentage.text(
            x=rpe_percentage.get_xlim()[0] + 0.1,
            y=rpe_percentage.get_ylim()[1] * 0.8,
            s=f"ANOVA(Wnt-surrogate > 0nM)\np = {round(p_value_percentage, 2)}",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )

        rpe_area = fig.add_subplot(fig_sgs[1])
        area_plot_kwargs = {
            "data": area_df,
            "x": "Condition",
            "y": "RPE_area",
            "ax": rpe_area,
        }
        sns.stripplot(**area_plot_kwargs, hue="experiment", **stripplot_params)
        sns.boxplot(**area_plot_kwargs, **cfg.BOXPLOT_PARAMS)
        rpe_area.legend(
            bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=cfg.AXIS_LABEL_SIZE
        )
        rpe_area.set_title(
            "RPE amount at last time point\nby Wnt-surrogate", fontsize=cfg.TITLE_SIZE
        )
        rpe_area.set_ylabel("RPE area [µm²]", fontsize=cfg.AXIS_LABEL_SIZE)
        rpe_area.tick_params(**cfg.TICKPARAMS_PARAMS)
        rpe_area.set_xlabel(
            "Wnt-surrogate concentration [nM]", fontsize=cfg.AXIS_LABEL_SIZE
        )
        rpe_area.set_xticklabels(["0 nM", "1 nM", "2 nM", "4 nM"])
        rpe_area.set_ylim(rpe_area.get_ylim()[0], rpe_area.get_ylim()[1] * 1.2)
        rpe_area.text(
            x=rpe_area.get_xlim()[0] + 0.1,
            y=rpe_area.get_ylim()[1] * 0.8,
            s=f"ANOVA(Wnt-surrogate > 0nM)\np = {round(p_value_area, 2)}",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )

        return

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.25)
        data = annotation_data.copy()
        data["experiment"] = [cfg.EXPERIMENT_MAP[exp] for exp in data["experiment"]]
        data = data.sort_values("experiment", ascending=True)

        fig_sgs = gs.subgridspec(1, 2)

        stripplot_params = cfg.STRIPPLOT_PARAMS.copy()
        stripplot_params["s"] = 5
        stripplot_params["dodge"] = False
        stripplot_params["jitter"] = 0.1
        stripplot_params["palette"] = "tab20"

        percentage_df = (
            data.groupby(["Condition", "experiment"])["Lens_Final"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .reset_index(name="Lens_Percentage")
        )
        area_df = (
            data.groupby(["Condition", "experiment"])["Lens_area"]
            .mean()
            .reset_index(name="Lens_area")
        )

        conditions = area_df["Condition"].unique()

        grouped_data = [
            area_df[area_df["Condition"] == condition]["Lens_area"]
            for condition in conditions
        ]
        _, p_value_area = f_oneway(*grouped_data)

        grouped_data = [
            percentage_df[percentage_df["Condition"] == condition]["Lens_Percentage"]
            for condition in conditions
        ]
        _, p_value_percentage = f_oneway(*grouped_data)

        lens_percentage = fig.add_subplot(fig_sgs[0])
        percentage_plot_kwargs = {
            "data": percentage_df,
            "x": "Condition",
            "y": "Lens_Percentage",
            "ax": lens_percentage,
        }
        sns.stripplot(**percentage_plot_kwargs, hue="experiment", **stripplot_params)
        sns.boxplot(**percentage_plot_kwargs, **cfg.BOXPLOT_PARAMS)
        handles, labels = lens_percentage.get_legend_handles_labels()
        lens_percentage.legend(
            handles,
            labels,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )
        lens_percentage.set_title(
            "Presence of lenses at last time point\nby Wnt-surrogate concentration",
            fontsize=cfg.TITLE_SIZE,
        )
        lens_percentage.set_ylabel(
            "organoids with lens [%]", fontsize=cfg.AXIS_LABEL_SIZE
        )
        lens_percentage.tick_params(**cfg.TICKPARAMS_PARAMS)
        lens_percentage.set_xlabel(
            "Wnt-surrogate concentration [nM]", fontsize=cfg.AXIS_LABEL_SIZE
        )
        lens_percentage.set_xticklabels(["0 nM", "1 nM", "2 nM", "4 nM"])
        lens_percentage.set_ylim(
            lens_percentage.get_ylim()[0], lens_percentage.get_ylim()[1] * 1.1
        )
        lens_percentage.text(
            x=lens_percentage.get_xlim()[0] + 0.1,
            y=lens_percentage.get_ylim()[1] * 0.9,
            s=f"ANOVA p = {round(p_value_percentage, 2)}",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )

        lens_area = fig.add_subplot(fig_sgs[1])
        area_plot_kwargs = {
            "data": area_df,
            "x": "Condition",
            "y": "Lens_area",
            "ax": lens_area,
        }
        sns.stripplot(**area_plot_kwargs, hue="experiment", **stripplot_params)
        sns.boxplot(**area_plot_kwargs, **cfg.BOXPLOT_PARAMS)
        handles, labels = lens_area.get_legend_handles_labels()
        lens_area.legend(
            handles,
            labels,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )
        lens_area.set_title(
            "Lens sizes at last time point\nby Wnt-surrogate", fontsize=cfg.TITLE_SIZE
        )
        lens_area.set_ylabel("lens area [µm²]", fontsize=cfg.AXIS_LABEL_SIZE)
        lens_area.tick_params(**cfg.TICKPARAMS_PARAMS)
        lens_area.set_xlabel(
            "Wnt-surrogate concentration [nM]", fontsize=cfg.AXIS_LABEL_SIZE
        )
        lens_area.set_xticklabels(["0 nM", "1 nM", "2 nM", "4 nM"])
        lens_area.set_ylim(lens_area.get_ylim()[0], lens_area.get_ylim()[1] * 1.1)
        lens_area.text(
            x=lens_area.get_xlim()[0] + 0.1,
            y=lens_area.get_ylim()[1] * 0.9,
            s=f"ANOVA p = {round(p_value_area, 2)}",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )

        return

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.25)
        data = human_f1_data.copy()
        data = data[["F1_RPE_Final", "Evaluator_ID", "loop"]]
        data["hours"] = data["loop"] / 2
        data = data[data["Evaluator_ID"] != "mean"]

        fig_sgs = gs.subgridspec(1, 8)

        plot_kwargs = {"x": "hours", "y": "F1_RPE_Final", "hue": "Evaluator_ID"}

        vis_over_time = fig.add_subplot(fig_sgs[:6])
        sns.lineplot(data=data, ax=vis_over_time, **plot_kwargs)
        vis_over_time.set_xlim(-1, 75)
        vis_over_time.set_ylim(-0.05, 1.05)
        handles, labels = vis_over_time.get_legend_handles_labels()
        labels = [label.replace("HEAT2", "Expert") for label in labels]
        vis_over_time.legend(
            handles,
            labels,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )
        vis_over_time.tick_params(**cfg.TICKPARAMS_PARAMS)
        vis_over_time.set_title("RPE visibility over time", fontsize=cfg.TITLE_SIZE)
        vis_over_time.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        vis_over_time.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        vis_over_time.axhline(y=0.9, color="black", linestyle="--")
        vis_over_time.text(x=1, y=0.94, s="F1 = 0.9", fontsize=cfg.TITLE_SIZE)
        vis_over_time.axhline(y=0.8, color="black", linestyle="--")
        vis_over_time.text(x=1, y=0.70, s="F1 = 0.8", fontsize=cfg.TITLE_SIZE)

        conf_matrix_plot = fig.add_subplot(fig_sgs[6:])
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt=".0f",
            ax=conf_matrix_plot,
            cmap="Reds",
            cbar=False,
            annot_kws={"fontsize": cfg.TITLE_SIZE},
        )
        conf_matrix_plot.set_ylabel("stereomicroscopy", fontsize=cfg.AXIS_LABEL_SIZE)
        conf_matrix_plot.set_yticklabels(["No", "Yes"], fontsize=cfg.AXIS_LABEL_SIZE)
        conf_matrix_plot.set_xlabel("expert annotation", fontsize=cfg.AXIS_LABEL_SIZE)
        conf_matrix_plot.set_xticklabels(["No", "Yes"], fontsize=cfg.AXIS_LABEL_SIZE)
        conf_matrix_plot.set_title(
            "Confusion matrix\nfor last timeframe", fontsize=cfg.TITLE_SIZE
        )

        return

    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.25)
        data = human_f1_data.copy()
        data["hours"] = data["loop"] / 2
        data = data[data["Evaluator_ID"].isin(["HEAT21", "HEAT27"])]

        fig_sgs = gs.subgridspec(1, 1)

        plot_kwargs = {"x": "hours", "y": "F1_Lens_Final", "hue": "Evaluator_ID"}

        vis_over_time = fig.add_subplot(fig_sgs[0])
        sns.lineplot(data=data, ax=vis_over_time, **plot_kwargs)
        vis_over_time.set_xlim(-1, 75)
        vis_over_time.set_ylim(-0.05, 1.05)
        handles, labels = vis_over_time.get_legend_handles_labels()
        labels = [label.replace("HEAT", "Annotator") for label in labels]
        labels = ["Annotator1", "Annotator2"]
        vis_over_time.legend(
            handles,
            labels,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.AXIS_LABEL_SIZE,
        )
        vis_over_time.tick_params(**cfg.TICKPARAMS_PARAMS)
        vis_over_time.set_title("Lens visibility over time", fontsize=cfg.TITLE_SIZE)
        vis_over_time.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        vis_over_time.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        vis_over_time.axhline(y=0.9, color="black", linestyle="--")
        vis_over_time.text(x=1, y=0.94, s="F1 = 0.9", fontsize=cfg.TITLE_SIZE)
        vis_over_time.axhline(y=0.8, color="black", linestyle="--")
        vis_over_time.text(x=1, y=0.70, s="F1 = 0.8", fontsize=cfg.TITLE_SIZE)

        return

    def generate_subfigure_e_and_f(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.25)
        data = annotation_data.copy()
        data["experiment"] = [cfg.EXPERIMENT_MAP[exp] for exp in data["experiment"]]
        data = data.sort_values("experiment", ascending=True)

        ax.text(x=0.36, y=1, s="F", fontsize=12)

        rpe_crosstab = pd.crosstab(data["experiment"], data["RPE_classes"])
        lens_crosstab = pd.crosstab(data["experiment"], data["Lens_classes"])

        x_label = "experiment"
        y_label = "n organoids"

        fig_sgs = gs.subgridspec(1, 2)

        rpe_plot = fig.add_subplot(fig_sgs[0])

        rpe_crosstab.plot(kind="bar", stacked=True, ax=rpe_plot)
        rpe_plot.set_xlabel(x_label, fontsize=cfg.AXIS_LABEL_SIZE)
        rpe_plot.set_ylabel(y_label, fontsize=cfg.AXIS_LABEL_SIZE)
        rpe_plot.set_title("RPE class distribution", fontsize=cfg.TITLE_SIZE)
        rpe_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        handles, labels = rpe_plot.get_legend_handles_labels()
        handles = handles[::-1]
        labels = labels[::-1]
        rpe_plot.legend(
            handles,
            labels,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.AXIS_LABEL_SIZE,
            title="class",
            title_fontsize=cfg.AXIS_LABEL_SIZE,
        )

        lens_plot = fig.add_subplot(fig_sgs[1])

        lens_crosstab.plot(kind="bar", stacked=True, ax=lens_plot)
        lens_plot.set_xlabel(x_label, fontsize=cfg.AXIS_LABEL_SIZE)
        lens_plot.set_ylabel(y_label, fontsize=cfg.AXIS_LABEL_SIZE)
        lens_plot.set_title("Lens class distribution", fontsize=cfg.TITLE_SIZE)
        lens_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        handles, labels = rpe_plot.get_legend_handles_labels()
        handles = handles[::-1]
        labels = labels[::-1]
        lens_plot.legend(
            handles,
            labels,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.AXIS_LABEL_SIZE,
            title="class",
            title_fontsize=cfg.AXIS_LABEL_SIZE,
        )
        return

    fig = plt.figure(
        layout="constrained", figsize=(cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL)
    )
    gs = GridSpec(ncols=6, nrows=5, figure=fig, height_ratios=[1, 1, 0.9, 0.9, 0.7])
    a_coords = gs[0, :]
    b_coords = gs[1, :]
    c_coords = gs[2, :]
    d_coords = gs[3, :]
    e_f_coords = gs[4, :]

    fig_a = fig.add_subplot(a_coords)
    fig_b = fig.add_subplot(b_coords)
    fig_c = fig.add_subplot(c_coords)
    fig_d = fig.add_subplot(d_coords)
    fig_e_f = fig.add_subplot(e_f_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "A")
    generate_subfigure_b(fig, fig_b, b_coords, "B")
    generate_subfigure_c(fig, fig_c, c_coords, "C")
    generate_subfigure_d(fig, fig_d, d_coords, "D")
    generate_subfigure_e_and_f(fig, fig_e_f, e_f_coords, "E")

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


def figure_S1_generation(
    annotations_dir: str,
    evaluator_results_dir: str,
    morphometrics_dir: str,
    figure_data_dir: str,
    figure_output_dir: str,
    **kwargs,
):
    dataset_annotations = get_dataset_annotations(
        annotations_dir, output_dir=figure_data_dir
    )

    human_evaluator_f1 = human_f1_per_evaluator(
        evaluator_results_dir=evaluator_results_dir,
        morphometrics_dir=morphometrics_dir,
        n_timeframes=12,
        average="weighted",
        output_dir=figure_data_dir,
    )
    human_evaluator_f1 = add_loop_from_timeframe(human_evaluator_f1, n_timeframes=12)

    conf_matrix = human_f1_RPE_visibility_conf_matrix(
        evaluator_results_dir=evaluator_results_dir,
        morphometrics_dir=morphometrics_dir,
        output_dir=figure_data_dir,
    )

    _generate_main_figure(
        annotation_data=dataset_annotations,
        human_f1_data=human_evaluator_f1,
        conf_matrix=conf_matrix,
        figure_output_dir=figure_output_dir,
        figure_name="S1_Fig",
    )

    data_output_dir = os.path.join(figure_output_dir, "S9_Data.csv")
    dataset_annotations.to_csv(data_output_dir, index = False)

    human_eval_output_dir = os.path.join(figure_output_dir, "S10_Data.csv")
    human_evaluator_f1.to_csv(human_eval_output_dir, index = False)

    return
