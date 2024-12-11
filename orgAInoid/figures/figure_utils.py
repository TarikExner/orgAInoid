from matplotlib.axes import Axes
import pandas as pd
import numpy as np

def _figure_label(ax: Axes, label, x = 0, y = 1):
    """labels individual subfigures. Requires subgrid to not use figure axis coordinates."""
    ax.text(x,y,label, fontsize = 12)
    return

def _prep_image_axis(ax: Axes):
    ax.axis("off")
    return

def remove_ticks_and_labels(ax):
    ax.set_xlabel('')
    ax.set_ylabel('')#
    ax.set_xticklabels([])
    ax.set_yticklabels([])#
    ax.tick_params(left=False, right=False, top=False, bottom=False)

def _remove_axis_labels(ax: Axes):
    ax.tick_params(left = False, bottom = False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def _preprocess_two_class_results(results: pd.DataFrame):
    results = results.reset_index()
    results["loop"] = [int(loop.split("LO")[1]) for loop in results["loop"].tolist()]
    results["loop"] = results["loop"].astype(int)

    flattened_data = pd.DataFrame({
        "loop": results["loop"].repeat(4),
        "value": results["percentage_matrix"].apply(lambda x: x.ravel()).explode().astype(float).values,
        "component": np.tile(["tn", "fp", "fn", "tp"], len(results))
    })
    
    plot_data = flattened_data.pivot(index="loop", columns="component", values="value")
    
    plot_data = plot_data[["fp", "fn", "tp", "tn"][::-1]]

    plot_data.index = plot_data.index / 2

    return plot_data

def _preprocess_four_class_results(results: pd.DataFrame,
                                   col_to_process: str):
    results = results.reset_index()
    results["loop"] = [int(loop.split("LO")[1]) for loop in results["loop"].tolist()]
    results["loop"] = results["loop"].astype(int)

    flattened_data = pd.DataFrame({
        "loop": results["loop"].repeat(4),
        "value": results[col_to_process].apply(lambda x: x.ravel()).explode().astype(float).values,
        "component": np.tile(["tn", "fp", "fn", "tp"], len(results))
    })
    
    plot_data = flattened_data.pivot(index="loop", columns="component", values="value")
    
    plot_data = plot_data[["fp", "fn", "tp", "tn"][::-1]]

    plot_data.index = plot_data.index / 2

    return plot_data

def classwise_confusion_matrix(matrix):
    """
    Compute 2x2 numpy arrays for all classes from a 4x4 confusion matrix.

    Args:
        matrix (ndarray): A 4x4 percentage confusion matrix.

    Returns:
        tuple: A tuple of 4 numpy arrays, each representing a 2x2 confusion matrix for a class.
    """
    n_classes = matrix.shape[0]
    results = []

    for focus_class in range(n_classes):
        tp = matrix[focus_class, focus_class]
        fn = np.sum(matrix[focus_class, :]) - tp
        fp = np.sum(matrix[:, focus_class]) - tp
        tn = np.sum(matrix) - tp - fn - fp
        results.append(np.array([[tn, fp], [fn, tp]]))
    return tuple(results)
