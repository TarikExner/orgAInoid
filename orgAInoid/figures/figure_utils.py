from matplotlib.axes import Axes
import pandas as pd
import numpy as np

def _figure_label(ax: Axes, label, x: float = 0.0, y: float = 1.0):
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
