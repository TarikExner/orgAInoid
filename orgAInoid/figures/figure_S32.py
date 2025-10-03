import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

METHOD_ORDER = ["IG_NT", "SAL_NT", "DLS", "GC", "GGC", "OCC", "FAB", "KSH"]


def _extract_loop_key(loop: int | str) -> str:
    if isinstance(loop, str) and "LO" in loop:
        return loop
    return f"LO{int(loop):03d}"


def _make_gray(img):
    if img.ndim == 4:
        return np.mean(img[0], axis=0)
    if img.ndim == 3:
        return np.mean(img, axis=0)
    return img.squeeze()


def _zscore_in_mask(arr: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    m = mask2d > 0
    if not np.any(m):
        return arr.astype(np.float32)
    vals = arr[m].astype(np.float32)
    mu = vals.mean()
    sd = vals.std()
    if sd < 1e-8:
        sd = 1.0
    out = (arr.astype(np.float32) - mu) / sd
    return out


def _effective_pixel_size_um(
    original_px: int = 2048, downscaled_px: int = 512, original_nm_per_px: float = 650.0
) -> float:
    """
    Compute the microns-per-pixel after downscaling 2048->512 and then cropping.
    650 nm/px at 2048; after 4x downscale -> 2600 nm/px = 2.6 µm/px.
    """
    scale = original_px / float(downscaled_px)  # 4.0
    nm_per_px = original_nm_per_px * scale  # 2600 nm
    return nm_per_px / 1000.0  # µm/px


def add_scalebar(
    ax,
    img_shape: tuple[int, int],
    pixel_size_um: float,
    bar_um: float = 100.0,
    thickness_px: int = 4,
    margin_px: int = 10,
    label: str | None = None,
):
    """
    Draw a simple scalebar in bottom-right (two-layer bar for contrast).
    """
    H, W = img_shape
    bar_px = int(round(bar_um / pixel_size_um))
    x0 = W - margin_px - bar_px
    y0 = H - margin_px - thickness_px

    ax.add_patch(
        Rectangle(
            (x0 - 1, y0 - 1),
            bar_px + 2,
            thickness_px + 2,
            facecolor="black",
            edgecolor="none",
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (x0, y0),
            bar_px,
            thickness_px,
            facecolor="white",
            edgecolor="none",
            zorder=6,
        )
    )
    if label is None:
        label = f"{int(bar_um)} µm"
    ax.text(
        x0 + bar_px / 2,
        y0 - 4,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        color="white",
        path_effects=[],
        zorder=7,
    )


def visualize_models_at_loops(
    h5_dir: str,
    readout: str,
    experiment: str,
    well: str,
    figure_output_dir: str,
    figure_name: str,
    models: list[str] = ("DenseNet121", "ResNet50", "MobileNetV3_Large"),
    loops: list[str] = ("LO001", "LO030", "LO144"),
    loop_row_labels: dict[str, str] | None = None,
    methods: list[str] | None = None,
    cmap: str = "coolwarm",
    show_mask_outline: bool = True,
    clip_sigma: float = 3.0,
    include_input: bool = True,
    figsize_per_cell: float = 2.2,
    fontsize: int = 11,
    draw_separators: bool = True,
    sep_kwargs: dict | None = None,
):
    if sep_kwargs is None:
        sep_kwargs = {"color": "black", "linewidth": 3.0, "alpha": 0.9}

    fname = f"{experiment}_{well}_{readout}.h5"
    h5_path = os.path.join(h5_dir, fname)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"File not found: {h5_path}")

    if loop_row_labels is None:
        loop_row_labels = {"LO001": "0 h", "LO030": "15 h", "LO144": "72 h"}

    with h5py.File(h5_path, "r") as h5f:
        if well not in h5f:
            raise KeyError(f"Well '{well}' not found in {fname}")
        grp_well = h5f[well]

        loops_present = []
        for lp in loops:
            k = _extract_loop_key(lp)
            if k in grp_well:
                loops_present.append(k)
        if not loops_present:
            raise KeyError("None of the requested loops exist in the H5 file.")

        if methods is None:
            common = None
            for mdl in models:
                probe = None
                for k in loops_present:
                    if mdl in grp_well[k]:
                        probe = set(grp_well[k][mdl].keys())
                        break
                if probe is None:
                    continue
                common = probe if common is None else (common & probe)
            if not common:
                raise ValueError(
                    "Could not find a common set of methods across the given models."
                )
            found_methods = sorted(list(common))
            ordered = [m for m in METHOD_ORDER if m in found_methods]
            extras = [m for m in found_methods if m not in ordered]
            methods = ordered + sorted(extras)

        rows = []
        for mdl in models:
            for k in loops_present:
                grp_loop = grp_well[k]
                img2d = None
                mask = None
                if "input_image" in grp_loop and "mask" in grp_loop:
                    img = grp_loop["input_image"][...]
                    mask = grp_loop["mask"][...].squeeze().astype(np.float32)
                    img2d = _make_gray(img)

                maps = {}
                if mdl in grp_loop:
                    grp_model = grp_loop[mdl]
                    for m in methods:
                        if m in grp_model and "trained" in grp_model[m]:
                            arr = (
                                grp_model[m]["trained"][...]
                                .squeeze()
                                .astype(np.float32)
                            )
                            if mask is not None:
                                arr = _zscore_in_mask(arr, mask)
                            arr = np.clip(arr, -clip_sigma, clip_sigma)
                            maps[m] = arr

                rows.append(
                    {
                        "model": mdl,
                        "loop": k,
                        "img2d": img2d,
                        "mask": mask,
                        "maps": maps,
                    }
                )

    n_rows = len(rows)
    n_cols = (1 if include_input else 0) + len(methods)
    fig_w = max(8.0, figsize_per_cell * n_cols)
    fig_h = max(6.0, figsize_per_cell * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    vlim = clip_sigma
    pixel_size_um = _effective_pixel_size_um()

    im_last = None
    block_len = len(loops_present)
    model_block_starts = [i * block_len for i in range(len(models))]
    model_block_ends = [(i + 1) * block_len - 1 for i in range(len(models))]

    for r, row in enumerate(rows):
        model = row["model"]
        loop_key = row["loop"]
        img2d = row["img2d"]
        mask = row["mask"]
        maps = row["maps"]

        col = 0
        if include_input:
            ax0 = axes[r, col]
            if img2d is not None:
                ax0.imshow(img2d, cmap="gray")
                if show_mask_outline and mask is not None and np.any(mask > 0):
                    ax0.contour(mask > 0, levels=[0.5], colors=["lime"], linewidths=0.6)
                add_scalebar(ax0, img2d.shape, pixel_size_um, 100.0)

            tp_label = loop_row_labels.get(loop_key, loop_key)
            ax0.text(
                -0.06,
                0.5,
                tp_label,
                transform=ax0.transAxes,
                ha="right",
                va="center",
                fontsize=fontsize,
            )

            if r in model_block_starts:
                ax0.set_title(model, fontsize=fontsize + 2, pad=6)

            ax0.axis("off")
            col += 1

        for m in methods:
            ax = axes[r, col]
            if m in maps and maps[m] is not None:
                im_last = ax.imshow(maps[m], cmap=cmap, vmin=-vlim, vmax=vlim)
                if show_mask_outline and mask is not None and np.any(mask > 0):
                    ax.contour(mask > 0, levels=[0.5], colors=["k"], linewidths=0.5)
            if r == 0:
                ax.set_title(m, fontsize=fontsize)
            ax.axis("off")
            col += 1

    if draw_separators and n_rows > 0:
        left, right = 0.01, 0.90
        y = 0.33
        line = Line2D([left, right], [y, y], transform=fig.transFigure, **sep_kwargs)
        fig.add_artist(line)

        y = 0.64
        line = Line2D([left, right], [y, y], transform=fig.transFigure, **sep_kwargs)
        fig.add_artist(line)
    if im_last is not None:
        cax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
        cb = plt.colorbar(im_last, cax=cax)
        cb.set_label("z-scored saliency", fontsize=fontsize)

    fig.suptitle(f"{experiment} | {well} | {readout}", y=0.995, fontsize=fontsize + 1)
    plt.tight_layout(rect=[0.01, 0.01, 0.90, 0.97])

    output_dir = os.path.join(figure_output_dir, f"{figure_name}.pdf")
    plt.savefig(output_dir, dpi=300, bbox_inches="tight")


def figure_S32_generation(figure_output_dir, **kwargs):
    visualize_models_at_loops(
        h5_dir="../classification/saliencies/results/",
        readout="RPE_classes",
        experiment="E001",
        well="B003",
        models=["DenseNet121", "ResNet50", "MobileNetV3_Large"],
        loops=["LO001", "LO030", "LO144"],
        loop_row_labels={"LO001": "0 h", "LO030": "15 h", "LO144": "72 h"},
        methods=None,
        cmap="coolwarm",
        clip_sigma=3.0,
        include_input=True,
        figure_output_dir=figure_output_dir,
        figure_name="Supplementary_Figure_S32",
    )
