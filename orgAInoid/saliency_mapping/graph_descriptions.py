import os

import pandas as pd
import networkx as nx
import numpy as np
import time

from typing import Optional

from ._graph_utils import (
    get_images_and_masks,
    slic_segment_stack,
    regionprops_stack,
    compute_path_level_coverage,
    forward_paths_from_backward_paths,
    overlap_stats,
    mask_selected_inputs,
    save_to_zarr,
    compute_input_to_last_costs,
    shortest_paths_last_to_first,
    build_backwards_graph,
    build_forwards_graph,
)
from ..classification._dataset import OrganoidDataset
from ..image_handling._image_handler import ImageHandler


def graph_descriptions(
    dataset: OrganoidDataset,
    output_dir: str,
    zarr_file: str,
    segmentator_input_dir: str = "../segmentation/segmentators",
    parameter_grid: Optional[dict] = None,
) -> None:
    """\
    Function for hyperparameter testing of the graph descriptions.

    """
    img_handler = ImageHandler(
        segmentator_input_dir=segmentator_input_dir,
        segmentator_input_size=dataset.image_metadata.segmentator_input_size,
        segmentation_model_name="DEEPLABV3",
    )
    imgs = dataset.X  # shape [n_images, 1, 224, 224]
    metadata: pd.DataFrame = dataset.metadata

    organoid_wells = metadata["well"].unique()
    experiments = metadata["experiment"].unique()
    if len(experiments) > 1:
        raise ValueError("There should only be one experiment")
    experiment = experiments[0]

    if not parameter_grid:
        parameter_grid = {
            "compactness": [0.001, 0.01, 0.1, 1],  # 4
            "pixels_per_superpixel": [900, 300, 100],  # 3
            "n_connections_per_node": [
                1,
                3,
            ],  # 3 should be fine for keeping connections :)
            "alpha": [0.1, 0.5],  # 2
            "beta": [0.1, 0.5],  # 2
        }

    path_level_coverage_file_name = f"{experiment}_path_level_coverage.csv"
    cost_analysis_file_name = f"{experiment}_path_cost_analysis.csv"

    zarr_path = os.path.join(output_dir, zarr_file)

    def _already_analyzed(well: str, output_dir: str):
        try:
            path_coverage_df = pd.read_csv(
                os.path.join(output_dir, path_level_coverage_file_name)
            )
        except FileNotFoundError:
            return False
        if well not in path_coverage_df["well"].unique():
            return False
        try:
            cost_analysis_df = pd.read_csv(
                os.path.join(output_dir, cost_analysis_file_name)
            )
        except FileNotFoundError:
            return False
        if well not in cost_analysis_df["well"].unique():
            return False
        return True

    for well in organoid_wells:
        well_start = time.time()
        if _already_analyzed(well, output_dir):
            print(f"""
                Skipping well {well} as it has been already analyzed!
            """)
            continue

        cost_analysis_result = pd.DataFrame()
        path_level_coverage_result = pd.DataFrame()

        imgs, masks = get_images_and_masks(dataset, well, img_handler)

        for pix_per_suppix in parameter_grid["pixels_per_superpixel"]:
            for compactness in parameter_grid["compactness"]:
                labels = slic_segment_stack(
                    imgs,
                    masks,
                    pixels_per_superpixel=pix_per_suppix,
                    compactness=compactness,
                )
                props = regionprops_stack(imgs, labels)

                for n_connections in parameter_grid["n_connections_per_node"]:
                    for alpha in parameter_grid["alpha"]:
                        for beta in parameter_grid["beta"]:
                            G_fwd = build_forwards_graph(
                                labels,
                                props,
                                n_successors=n_connections,
                                alpha=alpha,
                                beta=beta,
                            )
                            G_bwd = build_backwards_graph(
                                labels,
                                props,
                                n_predecessors=n_connections,
                                alpha=alpha,
                                beta=beta,
                            )

                            parameters = {
                                "experiment": experiment,
                                "well": well,
                                "compactness": compactness,
                                "pixels_per_superpixel": pix_per_suppix,
                                "n_connections_per_node": n_connections,
                                "alpha": alpha,
                                "beta": beta,
                            }

                            analysis_start = time.time()
                            cost_per_node, path_coverage = analyze_graphs(
                                G_fwd, G_bwd, labels, parameters, zarr_path
                            )

                            cost_analysis_result = pd.concat(
                                [cost_analysis_result, cost_per_node], axis=0
                            )
                            path_level_coverage_result = pd.concat(
                                [path_level_coverage_result, path_coverage], axis=0
                            )
                            print(parameters)
                            analysis_time = time.time() - analysis_start
                            print(f"... analyzed in {analysis_time} seconds!")

        # we save the data for every well
        _save_csv(path_level_coverage_result, output_dir, path_level_coverage_file_name)
        _save_csv(cost_analysis_result, output_dir, cost_analysis_file_name)

        well_time = time.time() - well_start
        print(
            f"""\
            \nANALYZED WELL {well} in {well_time} seconds!\n
            """
        )

    return


def _save_csv(file: pd.DataFrame, output_dir: str, file_name: str) -> None:
    file_path = os.path.join(output_dir, file_name)
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, index_col=False)
        file = pd.concat([df, file], axis=0)
    file.to_csv(file_path, index=False)


def _transfer_parameters_to_df(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    for label, val in params.items():
        df[label] = val
    return df


def analyze_graphs(
    G_fwd: nx.DiGraph,
    G_bwd: nx.DiGraph,
    labels: np.ndarray,
    parameters: dict,
    zarr_path: str,
) -> tuple[pd.DataFrame, ...]:
    """\
    This function is supposed to run the analysis for the graph descriptions.
    Organoids are segmented using SLIC and a directed graph is constructed
    in both directions, meaning from loop 144 to 1 and 1 to 143.

    There are several readouts:
        - The number of active nodes per frame (n_paths)
        - The area of active nodes/superpixels per frame (path_area)
    
        - The overlap of the forward graphs corresponding to the nodes
          that were reached from the backwards calculations with the actual
          backwards graphs
        - A statistical evaluation if the few input nodes have a lower cost
          to reach the final output nodes
        - the label masks with the proposed important nodes are saved to a .zarr file

    The function will run iteratively on a couple of hyperparameters, just
    to be sure that we do not produce any artifact.

    """
    overlap_percentage_bwd = overlap_stats(G_bwd)
    overlap_percentage_fwd = overlap_stats(G_fwd)

    backwards_paths = shortest_paths_last_to_first(G_bwd, weighted=True)
    path_coverage: pd.DataFrame = compute_path_level_coverage(G_bwd, backwards_paths)

    inferred_input_nodes = {
        n for path in backwards_paths.values() for n in path if n[0] == 0
    }
    forward_paths = forward_paths_from_backward_paths(
        G_fwd, backwards_paths, weighted=True
    )

    cost_per_node = compute_input_to_last_costs(
        G_fwd, backwards_paths, forward_paths, weighted=True
    )
    cost_per_node["non_zero_weights_fwd_total"] = overlap_percentage_fwd
    cost_per_node["non_zero_weights_bwd_total"] = overlap_percentage_bwd

    path_coverage = _transfer_parameters_to_df(path_coverage, parameters)
    cost_per_node = _transfer_parameters_to_df(cost_per_node, parameters)

    masked_label = mask_selected_inputs(labels, inferred_input_nodes)
    save_to_zarr(masked_label, parameters, zarr_path)

    # path_overlap: pd.DataFrame = compare_forward_backward_paths(forward_paths, backwards_paths)
    # path_overlap = _transfer_parameters_to_df(path_overlap, parameters)

    return cost_per_node, path_coverage
