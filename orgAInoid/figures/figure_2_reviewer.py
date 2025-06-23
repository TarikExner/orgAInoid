import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import figure_config as cfg
from . import figure_utils as utils

from .figure_data_generation import (get_morphometrics_frame,
                                     get_data_columns_morphometrics,
                                     calculate_organoid_dimensionality_reduction,
                                     compare_neighbors_by_experiment,
                                     neighbors_per_well_by_experiment,
                                     PC_COLUMNS)

def _generate_main_figure(jaccard_tsne_pca: pd.DataFrame,
                          jaccard_umap_pca: pd.DataFrame,
                          jaccard_tsne_raw: pd.DataFrame,
                          jaccard_umap_raw: pd.DataFrame,
                          well_enrichment_pca,
                          well_enrichment_raw,
                          figure_output_dir: str = "",
                          figure_name: str = ""):

    def generate_subfigure_a(fig: Figure,
                             ax: Axes,
                             gs: GridSpec,
                             subfigure_label) -> None:

        sns.violinplot
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.45)

        fig_sgs = gs.subgridspec(1,1)
        data = jaccard_tsne_pca.copy()
        data["hours"] = data["loop"] / 2
        data["experiment"] = data["experiment"].astype("category")

        fig_sgs = gs.subgridspec(1,1)
        distance_plot = fig.add_subplot(fig_sgs[0])
        
        sns.lineplot(data = data, x = "hours", y = "mean_jaccard", hue = "experiment", ax = distance_plot, palette = "tab20")
        distance_plot.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left", fontsize = cfg.AXIS_LABEL_SIZE, ncols = 2)
        distance_plot.set_xlabel('hours', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.set_ylabel('jaccard score', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        distance_plot.set_title("Jaccard scores of 30 neighbors in TSNE space vs. 30 neighbors in PCA space (euclidean distance)", fontsize = cfg.TITLE_SIZE)
        return

    def generate_subfigure_b(fig: Figure,
                             ax: Axes,
                             gs: GridSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.45)

        fig_sgs = gs.subgridspec(1,1)
        data = jaccard_tsne_raw.copy()
        data["hours"] = data["loop"] / 2
        data["experiment"] = data["experiment"].astype("category")

        fig_sgs = gs.subgridspec(1,1)
        distance_plot = fig.add_subplot(fig_sgs[0])
        
        sns.lineplot(data = data, x = "hours", y = "mean_jaccard", hue = "experiment", ax = distance_plot, palette = "tab20")
        distance_plot.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left", fontsize = cfg.AXIS_LABEL_SIZE, ncols = 2)
        distance_plot.set_xlabel('hours', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.set_ylabel('jaccard score', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        distance_plot.set_title("Jaccard scores of 30 neighbors in TSNE space vs. 30 neighbors in raw data space (euclidean distance)", fontsize = cfg.TITLE_SIZE)
        return

    def generate_subfigure_c(fig: Figure,
                             ax: Axes,
                             gs: GridSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.45)

        fig_sgs = gs.subgridspec(1,1)
        data = jaccard_umap_pca.copy()
        data["hours"] = data["loop"] / 2
        data["experiment"] = data["experiment"].astype("category")

        fig_sgs = gs.subgridspec(1,1)
        distance_plot = fig.add_subplot(fig_sgs[0])
        
        sns.lineplot(data = data, x = "hours", y = "mean_jaccard", hue = "experiment", ax = distance_plot, palette = "tab20")
        distance_plot.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left", fontsize = cfg.AXIS_LABEL_SIZE, ncols = 2)
        distance_plot.set_xlabel('hours', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.set_ylabel('jaccard score', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        distance_plot.set_title("Jaccard scores of 30 neighbors in UMAP space vs. 30 neighbors in PCA space (euclidean distance)", fontsize = cfg.TITLE_SIZE)

        return

    def generate_subfigure_d(fig: Figure,
                             ax: Axes,
                             gs: GridSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.45)

        fig_sgs = gs.subgridspec(1,1)
        data = jaccard_umap_raw.copy()
        data["hours"] = data["loop"] / 2
        data["experiment"] = data["experiment"].astype("category")

        fig_sgs = gs.subgridspec(1,1)
        distance_plot = fig.add_subplot(fig_sgs[0])
        
        sns.lineplot(data = data, x = "hours", y = "mean_jaccard", hue = "experiment", ax = distance_plot, palette = "tab20")
        distance_plot.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left", fontsize = cfg.AXIS_LABEL_SIZE, ncols = 2)
        distance_plot.set_xlabel('hours', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.set_ylabel('jaccard score', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        distance_plot.set_title("Jaccard scores of 30 neighbors in UMAP space vs. 30 neighbors in raw_data space (euclidean distance)", fontsize = cfg.TITLE_SIZE)

        return

    def generate_subfigure_e(fig: Figure,
                             ax: Axes,
                             gs: GridSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.45)

        fig_sgs = gs.subgridspec(1,1)
        data = well_enrichment_pca.copy()
        data["hours"] = data["loop"] / 2
        data["experiment"] = data["experiment"].astype("category")

        fig_sgs = gs.subgridspec(1,1)
        distance_plot = fig.add_subplot(fig_sgs[0])
        
        sns.lineplot(data = data, x = "hours", y = "obs_frac", hue = "experiment", ax = distance_plot, palette = "tab20")
        distance_plot.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left", fontsize = cfg.AXIS_LABEL_SIZE, ncols = 2)
        distance_plot.set_xlabel('hours', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.set_ylabel('fraction', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        distance_plot.set_title("Fraction of 30 nearest neighbors (pca space)\nwithin the same organoid", fontsize = cfg.TITLE_SIZE)

        return    
    
    def generate_subfigure_f(fig: Figure,
                             ax: Axes,
                             gs: GridSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.45)

        fig_sgs = gs.subgridspec(1,1)
        data = well_enrichment_raw.copy()
        data["hours"] = data["loop"] / 2
        data["experiment"] = data["experiment"].astype("category")

        fig_sgs = gs.subgridspec(1,1)
        distance_plot = fig.add_subplot(fig_sgs[0])
        
        sns.lineplot(data = data, x = "hours", y = "obs_frac", hue = "experiment", ax = distance_plot, palette = "tab20")
        distance_plot.legend(bbox_to_anchor = (1.01, 0.5), loc = "center left", fontsize = cfg.AXIS_LABEL_SIZE, ncols = 2)
        distance_plot.set_xlabel('hours', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.set_ylabel('fraction', fontsize = cfg.AXIS_LABEL_SIZE)
        distance_plot.tick_params(**cfg.TICKPARAMS_PARAMS)
        distance_plot.set_title("Fraction of 30 nearest neighbors (raw data space)\nwithin the same organoid", fontsize = cfg.TITLE_SIZE)

        return
    
    fig = plt.figure(layout = "constrained",
                     figsize = (cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_FULL))
    gs = GridSpec(ncols = 6,
                  nrows = 6,
                  figure = fig,
                  height_ratios = [1,1,1,1,1,1])

    a_coords = gs[0,:]
    b_coords = gs[1,:]
    c_coords = gs[2,:]
    d_coords = gs[3,:]
    e_coords = gs[4,:]
    f_coords = gs[5,:]

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
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.png")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

    return

def figure_2_reviewer_generation(morphometrics_dir: str,
                                 figure_data_dir: str,
                                 figure_output_dir: str,
                                 **kwargs):

    morphometrics = get_morphometrics_frame(morphometrics_dir)
    data_columns = get_data_columns_morphometrics(morphometrics)

    # we use n_pcs 20 for everything
    pc_columns = PC_COLUMNS

    dimreds_pca = calculate_organoid_dimensionality_reduction(morphometrics,
                                                              data_columns,
                                                              use_pca = True,
                                                              output_dir = figure_data_dir)
    dimreds = calculate_organoid_dimensionality_reduction(morphometrics,
                                                          data_columns,
                                                          use_pca = False,
                                                          output_dir = figure_data_dir)

    jaccard_pca_umap = compare_neighbors_by_experiment(dimreds_pca,
                                                       dimred = "UMAP",
                                                       user_suffix = "pca",
                                                       data_cols = pc_columns,
                                                       output_dir = figure_data_dir)
    jaccard_pca_tsne = compare_neighbors_by_experiment(dimreds_pca,
                                                       dimred = "TSNE",
                                                       user_suffix = "pca",
                                                       data_cols = pc_columns,
                                                       output_dir = figure_data_dir)
    jaccard_raw_umap = compare_neighbors_by_experiment(dimreds,
                                                       dimred = "UMAP",
                                                       user_suffix = "raw",
                                                       data_cols = data_columns,
                                                       output_dir = figure_data_dir)
    jaccard_raw_tsne = compare_neighbors_by_experiment(dimreds,
                                                       dimred = "TSNE",
                                                       user_suffix = "raw",
                                                       data_cols = data_columns,
                                                       output_dir = figure_data_dir)

    well_frac_pca_umap = neighbors_per_well_by_experiment(dimreds_pca,
                                                          dimred = "UMAP",
                                                          user_suffix = "pca",
                                                          data_cols = pc_columns,
                                                          output_dir = figure_data_dir)
    well_frac_raw_umap = neighbors_per_well_by_experiment(dimreds,
                                                          dimred = "UMAP",
                                                          user_suffix = "raw",
                                                          data_cols = data_columns,
                                                          output_dir = figure_data_dir)

    _generate_main_figure(jaccard_tsne_pca = jaccard_pca_tsne,
                          jaccard_tsne_raw = jaccard_raw_tsne,
                          jaccard_umap_pca = jaccard_pca_umap,
                          jaccard_umap_raw = jaccard_raw_umap,
                          well_enrichment_raw = well_frac_raw_umap,
                          well_enrichment_pca = well_frac_pca_umap,
                          figure_output_dir = figure_output_dir,
                          figure_name = "Reviewer_Figure_2")
    return
