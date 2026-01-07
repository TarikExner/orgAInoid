import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import figure_config as cfg
from . import figure_utils as utils


def _prep_data(df):
    df = df.copy()
    df = df[df["ExperimentID"] != "ExperimentID"]
    df["n_experiments"] = df["n_experiments"].astype(int)
    df["ValF1"] = df["ValF1"].astype(float)
    return df

def _generate_main_figure(figure_output_dir: str,
                          figure_name: str):


    def generate_subfigure_a(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        neural_net_data = pd.read_csv(
            "./0_initial_submission_files/figure_data/RPE_classification/RPE_Final_nexp.txt", index_col=False
        )
        neural_net_data = _prep_data(neural_net_data)
        neural_net_data = neural_net_data.sort_values(
            ["ValExpID", "n_experiments"], ascending=[True, True]
        )
        neural_net_data["ValExpID"] = neural_net_data["ValExpID"].map(cfg.EXPERIMENT_MAP)

        classifier_data = pd.read_csv(
            "./0_initial_submission_files/figure_data/classifier_n_experiments.log", index_col=False
        )
        classifier_data = classifier_data[classifier_data["readout"] == "RPE_Final"]
        classifier_data = classifier_data.sort_values(
            ["experiment", "n_experiments"], ascending=[True, True]
        )
        classifier_data["experiment"] = classifier_data["experiment"].map(
            cfg.EXPERIMENT_MAP
        )

        fig_sgs = gs.subgridspec(1, 2)

        nn_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=neural_net_data,
            x="n_experiments",
            y="ValF1",
            hue="ValExpID",
            ax=nn_plot,
            errorbar="se",
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
        )
        nn_plot.set_title(
            "Prediction accuracy: Emergence of RPE\nin test organoids by deep learning",
            fontsize=cfg.TITLE_SIZE,
        )
        nn_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        nn_plot.set_ylim(-0.1, 1.1)
        nn_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        nn_plot.set_xlabel("n experiments used for training", fontsize=cfg.AXIS_LABEL_SIZE)
        nn_plot.legend().remove()

        clf_plot = fig.add_subplot(fig_sgs[1])
        sns.lineplot(
            data=classifier_data,
            x="n_experiments",
            y="f1_score",
            hue="experiment",
            ax=clf_plot,
            errorbar="se",
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
        )

        clf_plot.set_title(
            "Prediction accuracy: Emergence of RPE\nin test organoids by morphometrics",
            fontsize=cfg.TITLE_SIZE,
        )
        clf_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        clf_plot.set_ylim(-0.1, 1.1)
        clf_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        clf_plot.set_xlabel("n experiments used for training", fontsize=cfg.AXIS_LABEL_SIZE)
        clf_plot.legend(
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.TITLE_SIZE,
            **cfg.TWO_COL_LEGEND,
        )

        return


    def generate_subfigure_b(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        neural_net_data = pd.read_csv(
            "./0_initial_submission_files/figure_data/Lens_classification/Lens_Final_nexp.txt", index_col=False
        )
        neural_net_data = _prep_data(neural_net_data)
        neural_net_data = neural_net_data.sort_values(
            ["ValExpID", "n_experiments"], ascending=[True, True]
        )
        neural_net_data["ValExpID"] = neural_net_data["ValExpID"].map(cfg.EXPERIMENT_MAP)

        classifier_data = pd.read_csv(
            "./0_initial_submission_files/figure_data/classifier_n_experiments.log", index_col=False
        )
        classifier_data = classifier_data[classifier_data["readout"] == "Lens_Final"]
        classifier_data = classifier_data.sort_values(
            ["experiment", "n_experiments"], ascending=[True, True]
        )
        classifier_data["experiment"] = classifier_data["experiment"].map(
            cfg.EXPERIMENT_MAP
        )

        fig_sgs = gs.subgridspec(1, 2)

        nn_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=neural_net_data,
            x="n_experiments",
            y="ValF1",
            hue="ValExpID",
            ax=nn_plot,
            errorbar="se",
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
        )
        nn_plot.set_title(
            "Prediction accuracy: Emergence of lenses\nin test organoids by deep learning",
            fontsize=cfg.TITLE_SIZE,
        )
        nn_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        nn_plot.set_ylim(-0.1, 1.1)
        nn_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        nn_plot.set_xlabel("n experiments used for training", fontsize=cfg.AXIS_LABEL_SIZE)
        nn_plot.legend().remove()

        clf_plot = fig.add_subplot(fig_sgs[1])
        sns.lineplot(
            data=classifier_data,
            x="n_experiments",
            y="f1_score",
            hue="experiment",
            ax=clf_plot,
            errorbar="se",
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
        )

        clf_plot.set_title(
            "Prediction accuracy: Emergence of lenses\nin test organoids by morphometrics",
            fontsize=cfg.TITLE_SIZE,
        )
        clf_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        clf_plot.set_ylim(-0.1, 1.1)
        clf_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        clf_plot.set_xlabel("n experiments used for training", fontsize=cfg.AXIS_LABEL_SIZE)
        clf_plot.legend(
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.TITLE_SIZE,
            **cfg.TWO_COL_LEGEND,
        )

        return


    def generate_subfigure_c(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        neural_net_data = pd.read_csv(
            "./0_initial_submission_files/figure_data/RPE_classes_classification/RPE_classes_nexp.txt", index_col=False
        )
        neural_net_data = _prep_data(neural_net_data)
        neural_net_data = neural_net_data.sort_values(
            ["ValExpID", "n_experiments"], ascending=[True, True]
        )
        neural_net_data["ValExpID"] = neural_net_data["ValExpID"].map(cfg.EXPERIMENT_MAP)

        classifier_data = pd.read_csv(
            "./0_initial_submission_files/figure_data/classifier_n_experiments.log", index_col=False
        )
        classifier_data = classifier_data[classifier_data["readout"] == "RPE_classes"]
        classifier_data = classifier_data.sort_values(
            ["experiment", "n_experiments"], ascending=[True, True]
        )
        classifier_data["experiment"] = classifier_data["experiment"].map(
            cfg.EXPERIMENT_MAP
        )

        fig_sgs = gs.subgridspec(1, 2)

        nn_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=neural_net_data,
            x="n_experiments",
            y="ValF1",
            hue="ValExpID",
            ax=nn_plot,
            errorbar="se",
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
        )
        nn_plot.set_title(
            "Prediction accuracy: Area of RPE\nin test organoids by deep learning",
            fontsize=cfg.TITLE_SIZE,
        )
        nn_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        nn_plot.set_ylim(-0.1, 1.1)
        nn_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        nn_plot.set_xlabel("n experiments used for training", fontsize=cfg.AXIS_LABEL_SIZE)
        nn_plot.legend().remove()

        clf_plot = fig.add_subplot(fig_sgs[1])
        sns.lineplot(
            data=classifier_data,
            x="n_experiments",
            y="f1_score",
            hue="experiment",
            ax=clf_plot,
            errorbar="se",
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
        )

        clf_plot.set_title(
            "Prediction accuracy: Area of RPE\nin test organoids by morphometrics",
            fontsize=cfg.TITLE_SIZE,
        )
        clf_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        clf_plot.set_ylim(-0.1, 1.1)
        clf_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        clf_plot.set_xlabel("n experiments used for training", fontsize=cfg.AXIS_LABEL_SIZE)
        clf_plot.legend(
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.TITLE_SIZE,
            **cfg.TWO_COL_LEGEND,
        )

        return


    def generate_subfigure_d(
        fig: Figure, ax: Axes, gs: SubplotSpec, subfigure_label
    ) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x=-0.4)

        neural_net_data = pd.read_csv(
            "./0_initial_submission_files/figure_data/Lens_classes_classification/Lens_classes_nexp.txt",
            index_col=False,
        )
        neural_net_data = _prep_data(neural_net_data)
        neural_net_data = neural_net_data.sort_values(
            ["ValExpID", "n_experiments"], ascending=[True, True]
        )
        neural_net_data["ValExpID"] = neural_net_data["ValExpID"].map(cfg.EXPERIMENT_MAP)

        classifier_data = pd.read_csv(
            "./0_initial_submission_files/figure_data/classifier_n_experiments.log", index_col=False
        )
        classifier_data = classifier_data[classifier_data["readout"] == "Lens_classes"]
        classifier_data = classifier_data.sort_values(
            ["experiment", "n_experiments"], ascending=[True, True]
        )
        classifier_data["experiment"] = classifier_data["experiment"].map(
            cfg.EXPERIMENT_MAP
        )

        fig_sgs = gs.subgridspec(1, 2)

        nn_plot = fig.add_subplot(fig_sgs[0])
        sns.lineplot(
            data=neural_net_data,
            x="n_experiments",
            y="ValF1",
            hue="ValExpID",
            ax=nn_plot,
            errorbar="se",
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
        )
        nn_plot.set_title(
            "Prediction accuracy: Area of lenses\nin validation organoids by deep learning",
            fontsize=cfg.TITLE_SIZE,
        )
        nn_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        nn_plot.set_ylim(-0.1, 1.1)
        nn_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        nn_plot.set_xlabel("n experiments used for training", fontsize=cfg.AXIS_LABEL_SIZE)
        nn_plot.legend().remove()

        clf_plot = fig.add_subplot(fig_sgs[1])
        sns.lineplot(
            data=classifier_data,
            x="n_experiments",
            y="f1_score",
            hue="experiment",
            ax=clf_plot,
            errorbar="se",
            palette=cfg.EXPERIMENT_LEGEND_CMAP,
        )

        clf_plot.set_title(
            "Prediction accuracy: Area of lenses\nin validation organoids by morphometrics",
            fontsize=cfg.TITLE_SIZE,
        )
        clf_plot.set_ylabel("F1 score", fontsize=cfg.AXIS_LABEL_SIZE)
        clf_plot.set_ylim(-0.1, 1.1)
        clf_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        clf_plot.set_xlabel("n experiments used for training", fontsize=cfg.AXIS_LABEL_SIZE)
        clf_plot.legend(
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fontsize=cfg.TITLE_SIZE,
            **cfg.TWO_COL_LEGEND,
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
    plt.savefig(output_dir, dpi=300, bbox_inches="tight", transparent = True)

    return

def figure_S28_generation(
    figure_output_dir: str,
    **kwargs
) -> None:
    _generate_main_figure(figure_output_dir, "S28_Fig")

    rpe_nn = pd.read_csv(
        "./0_initial_submission_files/figure_data/RPE_classification/RPE_Final_nexp.txt",
        index_col=False,
    )
    lens_nn= pd.read_csv(
        "./0_initial_submission_files/figure_data/Lens_classification/Lens_Final_nexp.txt",
        index_col=False,
    )
    rpe_classes_nn = pd.read_csv(
        "./0_initial_submission_files/figure_data/RPE_classes_classification/RPE_classes_nexp.txt",
        index_col=False,
    )
    lens_classes_nn= pd.read_csv(
        "./0_initial_submission_files/figure_data/Lens_classes_classification/Lens_classes_nexp.txt",
        index_col=False,
    )
    rpe_nn = _prep_data(rpe_nn)
    lens_nn = _prep_data(lens_nn)
    rpe_classes_nn = _prep_data(rpe_classes_nn)
    lens_classes_nn= _prep_data(lens_classes_nn)
    nn_data = pd.concat([rpe_nn, lens_nn, rpe_classes_nn, lens_classes_nn], axis = 0)

    nn_data_output_dir = os.path.join(figure_output_dir, "S87_Data.csv")
    nn_data.to_csv(nn_data_output_dir, index = False)

    classifier_data = pd.read_csv(
        "./0_initial_submission_files/figure_data/classifier_n_experiments.log", index_col=False
    )

    clf_data_output_dir = os.path.join(figure_output_dir, "S88_Data.csv")
    classifier_data.to_csv(clf_data_output_dir, index = False)

    return


