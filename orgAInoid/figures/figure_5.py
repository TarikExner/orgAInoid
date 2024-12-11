from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from matplotlib.figure import Figure
from matplotlib.axes import Axes

import cv2

import .figure_config as cfg
import .figure_utils as utils


def generate_subfigure_a(fig: Figure,
                         ax: Axes,
                         gs: GridSpec,
                         subfigure_label) -> None:
    """Will contain the experimental overview sketch"""
    ax.axis("off")
    utils._figure_label(ax, subfigure_label, x = -0.3)
    fig_sgs = gs.subgridspec(1,1)

    sketch = fig.add_subplot(fig_sgs[0])
    utils._prep_image_axis(sketch)
    img = cv2.imread('./sketches/Figure_5.png', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sketch.imshow(img)
    return


if __name__ == "__main__":
    fig = plt.figure(layout = "constrained",
                     figsize = (cfg.FIGURE_WIDTH_FULL, cfg.FIGURE_HEIGHT_HALF))
    gs = GridSpec(ncols = 6,
                  nrows = 2,
                  figure = fig,
                  height_ratios = [0.8,1])
    a_coords = gs[0,:]

    fig_a = fig.add_subplot(a_coords)

    generate_subfigure_a(fig, fig_a, a_coords, "")

    plt.savefig("./prefinal_figures/Figure5.pdf", dpi = 300, bbox_inches = "tight")
    plt.savefig("./prefinal_figures/Figure5.png", dpi = 300, bbox_inches = "tight")
    plt.show()
