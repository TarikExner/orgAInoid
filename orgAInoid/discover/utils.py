"""Utility helpers for DISCOVER experiments
========================================
Common tasks:
  • Building a DataLoader from an image *index* slice or list.
  • Listing checkpoints.
  • Plotting training curves from TensorBoard event files.
  • Displaying the reconstruction grids logged each epoch.

These helpers keep notebooks short and avoid boilerplate.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tensorboard.backend.event_processing import event_accumulator

from .training import build_dataloader, build_dataloader_from_dir

__all__ = [
    "build_loader_from_index",
    "get_checkpoint_paths",
    "plot_loss_curves",
    "show_recon_grid",
]

# ─────────────────────────── data helper ──────────────────────────

def build_loader_from_index(
    data: Sequence[torch.Tensor | np.ndarray],
    idx: Sequence[int] | slice,
    batch_size: int = 32,
):
    """Create DataLoader from a subset (index X) of an in‑memory dataset."""
    subset = [data[i] for i in range(len(data))[idx]] if isinstance(idx, slice) else [data[i] for i in idx]
    return build_dataloader(subset, batch_size=batch_size)


# ───────────────────────── checkpoint utils ────────────────────────

def get_checkpoint_paths(out_dir: str | Path, pattern: str = "discover_epoch*.pth") -> list[str]:
    """Return sorted list of checkpoint paths in *out_dir*."""
    return sorted(glob.glob(str(Path(out_dir) / pattern)))


# ───────────────────────── plot loss curves ───────────────────────

def _extract_scalars(log_dir: str | Path, tag: str):
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals = [e.value for e in events]
    return steps, vals


def plot_loss_curves(log_dir: str | Path, tags: list[str] | None = None):
    """Plot specified scalar tags from TensorBoard logs."""
    if tags is None:
        tags = ["loss/g_total", "loss/d_total"]
    plt.figure(figsize=(6, 4))
    for tag in tags:
        steps, vals = _extract_scalars(log_dir, tag)
        if steps:
            plt.plot(steps, vals, label=tag)
    plt.xlabel("global step")
    plt.ylabel("loss value")
    plt.legend()
    plt.title("Training curves")
    plt.tight_layout()
    plt.show()


# ───────────────────── display recon grids ─────────────────────────

def show_recon_grid(log_dir: str | Path, epoch: int):
    """Display the reconstruction grid saved at *epoch* in TensorBoard."""
    tag = "sample/recon"
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    images = ea.Images(tag)
    for im in images:
        if im.step == epoch:
            fig = plt.figure(figsize=(8, 4))
            img = torchvision.io.decode_image(torch.tensor(im.encoded_image_string))
            img = img.permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis("off")
            plt.show()
            return
    print(f"Epoch {epoch} not found in {tag}.")
