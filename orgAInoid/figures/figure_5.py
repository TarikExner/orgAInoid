import os
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, SubplotSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import cv2

from . import figure_config as cfg
from . import figure_utils as utils


def _generate_main_figure(figure_output_dir: str = "",
                          sketch_dir: str = "",
                          figure_name: str = ""):

    def generate_subfigure_a(fig: Figure,
                             ax: Axes,
                             gs: SubplotSpec,
                             subfigure_label) -> None:
        ax.axis("off")
        utils._figure_label(ax, subfigure_label, x = -0.3)
        fig_sgs = gs.subgridspec(1,1)

        sketch = fig.add_subplot(fig_sgs[0])
        utils._prep_image_axis(sketch)
        img = cv2.imread(os.path.join(sketch_dir, "Figure_5.png"), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sketch.imshow(img)
        return


    fig = plt.figure(layout = "constrained",
                     figsize = (cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_HALF))
    gs = GridSpec(ncols = 6,
                  nrows = 2,
                  figure = fig,
                  height_ratios = [0.8,1])
    a_coords = gs[0,:]

    fig_a = fig.add_subplot(a_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.png")
    plt.savefig(output_dir, dpi = 300, bbox_inches = "tight")

def figure_5_generation(sketch_dir: str,
                        figure_output_dir: str,
                        **kwargs):

    _generate_main_figure(figure_output_dir = figure_output_dir,
                          figure_name = "Figure_5",
                          sketch_dir = sketch_dir)

