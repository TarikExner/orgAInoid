import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib.ticker import MultipleLocator

from typing import Literal

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import get_classification_f1_data


def _generate_main_figure(
    plot1_sum: pd.DataFrame,
    plot2_sum: pd.DataFrame,
    plot1_max: pd.DataFrame,
    plot2_max: pd.DataFrame,
    readout: Literal["emergence", "area"],
    figure_output_dir: str = "",
    sketch_dir: str = "",
    figure_name: str = "",
):
    rpe_classifier = "Random Forest"
    lens_classifier = "QDA"
    rpe_classes_classifier = "HGBC"
    lens_classes_classifier = "QDA"

    labels_dict_rpe = {
        # we switch nomenclature for test and val sets
        "Morphometrics_test": f"{rpe_classifier if readout == 'emergence' else rpe_classes_classifier} (morphometrics): Validation",
        "Morphometrics_val": f"{rpe_classifier if readout == 'emergence' else rpe_classes_classifier} (morphometrics): Test",
        "Ensemble_test": "CNN (image data): Validation",
        "Ensemble_val": "CNN (image data): Test",
        "human": "Expert prediction",
        "Baseline_Morphometrics": f"{rpe_classifier if readout == 'emergence' else rpe_classes_classifier} (morphometrics): Baseline",
        "Baseline_Ensemble": "CNN (image data): Baseline",
    }

    labels_dict_lens = {
        # we switch nomenclature for test and val sets
        "Morphometrics_test": f"{lens_classifier if readout == 'emergence' else lens_classes_classifier} (morphometrics): Validation",
        "Morphometrics_val": f"{lens_classifier if readout == 'emergence' else lens_classes_classifier} (morphometrics): Test",
        "Ensemble_test": "CNN (image data): Validation",
        "Ensemble_val": "CNN (image data): Test",
        "human": "Expert prediction",
        "Baseline_Morphometrics": f"{lens_classifier if readout == 'emergence' else lens_classes_classifier} (morphometrics): Baseline",
        "Baseline_Ensemble": "CNN (image data): Baseline",
    }

    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        projection = "sum-intensity z-projection"
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        data = plot1_sum

        data.loc[
            data["classifier"].str.contains("Baseline_Morphometrics"), "classifier"
        ] = "Baseline_Morphometrics"
        data.loc[data["classifier"].str.contains("Baseline_Ensemble"), "classifier"] = (
            "Baseline_Ensemble"
        )

        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1, 1)

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=data,
            x="hours",
            y="F1",
            hue="classifier",
            ax=accuracy_plot,
            errorbar="se",
        )

        if readout == "emergence":
            accuracy_plot.axhline(
                y=0.5, xmin=0.03, xmax=0.97, linestyle="--", color="black"
            )
            accuracy_plot.text(
                x=120 / 2,
                y=0.52,
                s="Random Prediction",
                fontsize=cfg.TITLE_SIZE,
                color="black",
            )
        else:
            accuracy_plot.axhline(
                y=0.25, xmin=0.03, xmax=0.30, linestyle="--", color="black"
            )
            accuracy_plot.text(
                x=0,
                y=0.27,
                s="Random Prediction",
                fontsize=cfg.TITLE_SIZE,
                color="black",
            )

        RPE_prediction_cutoff = 22 / 2
        RPE_visibility_cutoff = 96 / 2
        accuracy_plot.annotate(
            "Confident Deep\nLearning Predictions",
            xy=(RPE_prediction_cutoff, 0.95),
            xytext=(RPE_prediction_cutoff, 1.05),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha="center",
        )

        accuracy_plot.annotate(
            "Confident RPE visibility",
            xy=(RPE_visibility_cutoff, 0.95),
            xytext=(RPE_visibility_cutoff, 1.05),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha="center",
        )

        handles, labels = accuracy_plot.get_legend_handles_labels()
        labels = [labels_dict_rpe[label] for label in labels]
        accuracy_plot.legend(
            handles, labels, loc="lower right", fontsize=cfg.TITLE_SIZE, ncols=2
        )
        readout_title = "Emergence of RPE" if readout == "emergence" else "RPE area"
        accuracy_plot.set_title(
            f"Prediction accuracy: {readout_title}\non image projection: {projection}",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.set_ylim(0.01, 1.24)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
        return

    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        projection = "sum-intensity z-projection"
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        data = plot2_sum

        # preprocessing:
        data.loc[
            data["classifier"].str.contains("Baseline_Morphometrics"), "classifier"
        ] = "Baseline_Morphometrics"
        data.loc[data["classifier"].str.contains("Baseline_Ensemble"), "classifier"] = (
            "Baseline_Ensemble"
        )

        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1, 1)

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=data,
            x="hours",
            y="F1",
            hue="classifier",
            ax=accuracy_plot,
            errorbar="se",
        )

        if readout == "emergence":
            accuracy_plot.axhline(
                y=0.5, xmin=0.03, xmax=0.97, linestyle="--", color="black"
            )
            accuracy_plot.text(
                x=120 / 2,
                y=0.52,
                s="Random Prediction",
                fontsize=cfg.TITLE_SIZE,
                color="black",
            )
        else:
            accuracy_plot.axhline(
                y=0.25, xmin=0.03, xmax=0.30, linestyle="--", color="black"
            )
            accuracy_plot.text(
                x=0,
                y=0.27,
                s="Random Prediction",
                fontsize=cfg.TITLE_SIZE,
                color="black",
            )

        lens_prediction_cutoff = 14 / 2
        lens_visibility_cutoff = 86 / 2
        accuracy_plot.annotate(
            "Confident Deep\nLearning Predictions",
            xy=(lens_prediction_cutoff, 0.95),
            xytext=(lens_prediction_cutoff, 1.05),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha="center",
        )

        accuracy_plot.annotate(
            "Confident Lens visibility",
            xy=(lens_visibility_cutoff, 0.95),
            xytext=(lens_visibility_cutoff, 1.05),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha="center",
        )

        handles, labels = accuracy_plot.get_legend_handles_labels()

        labels = [labels_dict_lens[label] for label in labels]
        accuracy_plot.legend(
            handles, labels, loc="lower right", fontsize=cfg.TITLE_SIZE, ncols=2
        )
        readout_title = (
            "Emergence of Lenses" if readout == "emergence" else "Lens sizes"
        )
        accuracy_plot.set_title(
            f"Prediction accuracy: {readout_title}\non image projection: {projection}",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.set_ylim(0.01, 1.24)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
        return

    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        projection = "max-intensity z-projection"
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        data = plot1_max

        # preprocessing:
        data.loc[
            data["classifier"].str.contains("Baseline_Morphometrics"), "classifier"
        ] = "Baseline_Morphometrics"
        data.loc[data["classifier"].str.contains("Baseline_Ensemble"), "classifier"] = (
            "Baseline_Ensemble"
        )

        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1, 1)

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=data,
            x="hours",
            y="F1",
            hue="classifier",
            ax=accuracy_plot,
            errorbar="se",
        )

        if readout == "emergence":
            accuracy_plot.axhline(
                y=0.5, xmin=0.03, xmax=0.97, linestyle="--", color="black"
            )
            accuracy_plot.text(
                x=120 / 2,
                y=0.52,
                s="Random Prediction",
                fontsize=cfg.TITLE_SIZE,
                color="black",
            )
        else:
            accuracy_plot.axhline(
                y=0.25, xmin=0.03, xmax=0.30, linestyle="--", color="black"
            )
            accuracy_plot.text(
                x=0,
                y=0.27,
                s="Random Prediction",
                fontsize=cfg.TITLE_SIZE,
                color="black",
            )

        RPE_prediction_cutoff = 26 / 2
        RPE_visibility_cutoff = 96 / 2
        accuracy_plot.annotate(
            "Confident Deep\nLearning Predictions",
            xy=(RPE_prediction_cutoff, 0.95),
            xytext=(RPE_prediction_cutoff, 1.05),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha="center",
        )

        accuracy_plot.annotate(
            "Confident RPE visibility",
            xy=(RPE_visibility_cutoff, 0.95),
            xytext=(RPE_visibility_cutoff, 1.05),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha="center",
        )

        handles, labels = accuracy_plot.get_legend_handles_labels()
        labels = [labels_dict_rpe[label] for label in labels]
        accuracy_plot.legend(
            handles, labels, loc="lower right", fontsize=cfg.TITLE_SIZE, ncols=2
        )
        readout_title = "Emergence of RPE" if readout == "emergence" else "RPE area"
        accuracy_plot.set_title(
            f"Prediction accuracy: {readout_title}\non image projection: {projection}",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot.set_ylim(0.01, 1.24)
        accuracy_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
        return

    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        projection = "max-intensity z-projection"
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        data = plot2_max

        # preprocessing:
        data.loc[
            data["classifier"].str.contains("Baseline_Morphometrics"), "classifier"
        ] = "Baseline_Morphometrics"
        data.loc[data["classifier"].str.contains("Baseline_Ensemble"), "classifier"] = (
            "Baseline_Ensemble"
        )

        data["hours"] = data["loop"] / 2

        fig_sgs = gs.subgridspec(1, 1)

        accuracy_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=data,
            x="hours",
            y="F1",
            hue="classifier",
            ax=accuracy_plot,
            errorbar="se",
        )

        if readout == "emergence":
            accuracy_plot.axhline(
                y=0.5, xmin=0.03, xmax=0.97, linestyle="--", color="black"
            )
            accuracy_plot.text(
                x=120 / 2,
                y=0.52,
                s="Random Prediction",
                fontsize=cfg.TITLE_SIZE,
                color="black",
            )
        else:
            accuracy_plot.axhline(
                y=0.25, xmin=0.03, xmax=0.30, linestyle="--", color="black"
            )
            accuracy_plot.text(
                x=0,
                y=0.27,
                s="Random Prediction",
                fontsize=cfg.TITLE_SIZE,
                color="black",
            )

        lens_prediction_cutoff = 14 / 2
        lens_visibility_cutoff = 86 / 2
        accuracy_plot.annotate(
            "Confident Deep\nLearning Predictions",
            xy=(lens_prediction_cutoff, 0.95),
            xytext=(lens_prediction_cutoff, 1.05),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha="center",
        )

        accuracy_plot.annotate(
            "Confident Lens visibility",
            xy=(lens_visibility_cutoff, 0.95),
            xytext=(lens_visibility_cutoff, 1.05),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=cfg.TITLE_SIZE,
            ha="center",
        )

        handles, labels = accuracy_plot.get_legend_handles_labels()
        labels = [labels_dict_lens[label] for label in labels]

        accuracy_plot.legend(
            handles, labels, loc="lower right", fontsize=cfg.TITLE_SIZE, ncols=2
        )
        readout_title = (
            "Emergence of lenses" if readout == "emergence" else "Lens sizes"
        )
        accuracy_plot.set_title(
            f"Prediction accuracy: {readout_title}\non image projection: {projection}",
            fontsize=cfg.TITLE_SIZE,
        )
        accuracy_plot.set_ylim(0.01, 1.24)
        accuracy_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        accuracy_plot.set_xlabel("hours", fontsize=cfg.AXIS_LABEL_SIZE)
        accuracy_plot.yaxis.set_major_locator(MultipleLocator(0.1))
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


def figure_S19_generation(
    sketch_dir: str,
    figure_output_dir: str,
    raw_data_dir: str,
    morphometrics_dir: str,
    hyperparameter_dir: str,
    rpe_classification_dir_sum: str,
    lens_classification_dir_sum: str,
    rpe_classes_classification_dir_sum: str,
    lens_classes_classification_dir_sum: str,
    rpe_classification_dir_max: str,
    lens_classification_dir_max: str,
    rpe_classes_classification_dir_max: str,
    lens_classes_classification_dir_max: str,
    figure_data_dir: str,
    evaluator_results_dir: str,
    **kwargs,
) -> None:
    rpe_classes_f1_sum = get_classification_f1_data(
        readout="RPE_classes",
        output_dir=figure_data_dir,
        proj="sum",
        hyperparameter_dir=hyperparameter_dir,
        classification_dir=rpe_classes_classification_dir_sum,
        baseline_dir=None,
        morphometrics_dir=morphometrics_dir,
        raw_data_dir=raw_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )
    lens_classes_f1_sum = get_classification_f1_data(
        readout="Lens_classes",
        output_dir=figure_data_dir,
        proj="sum",
        hyperparameter_dir=hyperparameter_dir,
        classification_dir=lens_classes_classification_dir_sum,
        baseline_dir=None,
        morphometrics_dir=morphometrics_dir,
        raw_data_dir=raw_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )
    rpe_classes_f1_max = get_classification_f1_data(
        readout="RPE_classes",
        output_dir=figure_data_dir,
        proj="max",
        hyperparameter_dir=hyperparameter_dir,
        classification_dir=rpe_classes_classification_dir_max,
        baseline_dir=None,
        morphometrics_dir=morphometrics_dir,
        raw_data_dir=raw_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )
    lens_classes_f1_max = get_classification_f1_data(
        readout="Lens_classes",
        output_dir=figure_data_dir,
        proj="max",
        hyperparameter_dir=hyperparameter_dir,
        classification_dir=lens_classes_classification_dir_max,
        baseline_dir=None,
        morphometrics_dir=morphometrics_dir,
        raw_data_dir=raw_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )
    _generate_main_figure(
        plot1_sum=rpe_classes_f1_sum,
        plot2_sum=lens_classes_f1_sum,
        plot1_max=rpe_classes_f1_max,
        plot2_max=lens_classes_f1_max,
        readout="area",
        figure_output_dir=figure_output_dir,
        sketch_dir=sketch_dir,
        figure_name="Supplementary_Figure_S19",
    )

    rpe_output_dir = os.path.join(figure_output_dir, "S60_Data.csv")
    rpe_classes_f1_sum.to_csv(rpe_output_dir, index = False)

    lens_output_dir = os.path.join(figure_output_dir, "S61_Data.csv")
    lens_classes_f1_sum.to_csv(lens_output_dir, index = False)


    rpe_output_dir = os.path.join(figure_output_dir, "S62_Data.csv")
    rpe_classes_f1_max.to_csv(rpe_output_dir, index = False)

    lens_output_dir = os.path.join(figure_output_dir, "S63_Data.csv")
    lens_classes_f1_max.to_csv(lens_output_dir, index = False)

    return
