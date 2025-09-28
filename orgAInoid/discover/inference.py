"""Inference & explanation utilities for DISCOVER
================================================
Implements the saliency-map procedure exactly as described in the DISCOVER
paper (Methods, section “Latent perturbation and visualisation”).

* `explain_image` – core function; perturbs latent units ±σ, decodes,
  computes patch-wise (1 − SSIM) maps, smoothes them, ranks units by the
  change in frozen-classifier score, and returns the top-k maps.
* `load_model_from_ckpt` – convenience loader that restores encoder &
  decoder from a training checkpoint.

External deps: scikit-image (for SSIM). Both skimage and scipy are
pure-python wheels so `pip install scikit-image` is fine.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from .models import build_models


def load_model_from_ckpt(
    ckpt_path: str | Path,
    *,
    latent_dim: int = 350,
    img_size: int = 224,
    device: Union[str, torch.device] = "cpu",
) -> tuple[torch.nn.Module, ...]:
    """Restore encoder & decoder from a checkpoint produced by train_loop."""
    ckpt = torch.load(ckpt_path, map_location=device)
    enc, dec, _ = build_models(latent_dim=latent_dim, img_size=img_size)
    enc.load_state_dict(ckpt["enc"])
    dec.load_state_dict(ckpt["dec"])
    enc.eval().to(device)
    dec.eval().to(device)
    return enc, dec


def _to_numpy(img: torch.Tensor) -> np.ndarray:
    """Tensor CHW in [-1,1] or [0,1] → float32 HWC in [0,1] for skimage."""
    img = img.detach().cpu()
    if img.min() < 0:
        img = (img + 1.0) / 2.0  # map to 0-1
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img.astype(np.float32)


def _patchwise_ssim(
    a: torch.Tensor, b: torch.Tensor, win_size: int = 11
) -> torch.Tensor:
    """Compute per-pixel (1 − SSIM) map via skimage, then return torch H×W."""
    a_np, b_np = _to_numpy(a), _to_numpy(b)
    _, ssim_map = ssim(
        a_np,
        b_np,
        channel_axis=-1,
        data_range=1.0,
        win_size=win_size,
        gaussian_weights=True,
        full=True,
    )
    return torch.from_numpy(1.0 - ssim_map.astype(np.float32))  # H×W


def _gaussian_blur(
    map_: torch.Tensor, kernel_size: int = 7, sigma: float = 2.0
) -> torch.Tensor:
    map_ = map_.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return F.gaussian_blur(map_, (kernel_size, kernel_size), (sigma, sigma)).squeeze()


def explain_image(
    img: torch.Tensor,
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    classifier: torch.nn.Module,
    latent_std: Optional[torch.Tensor] = None,
    sigma: float = 3.0,
    top_k: int = 3,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[float], List[int]]:
    """Generate saliency maps for *img*.

    Parameters
    ----------
    img : torch.Tensor (C×H×W), values in [-1,1] or [0,1]
    encoder, decoder : trained modules
    classifier : frozen model we explain (outputs scalar score 0-1)
    latent_std : per-dim std-dev of latent codes from training set. If
                 None, `sigma` is used as an absolute ± value.
    sigma : multiplier for std (default 3σ as per paper)
    top_k : return maps for K most influential latent units.

    Returns
    -------
    recon : reconstructed image (torch tensor C×H×W)
    maps  : list[torch H×W] saliency maps (0-1) length K
    deltas: list[float] absolute classifier-score deltas length K
    idxs  : list[int] latent indices corresponding to each map
    """
    device = next(encoder.parameters()).device
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        z = encoder(img)  # [1, D]
        recon = decoder(z)[0]  # C H W

    D = z.shape[1]
    std_vec = (
        latent_std.to(device)
        if latent_std is not None
        else torch.ones(D, device=device)
    )
    delta_scores = torch.zeros(D, device=device)
    maps = []

    for j in range(D):
        perturb = sigma * std_vec[j]
        z_plus = z.clone()
        z_plus[0, j] += perturb
        z_minus = z.clone()
        z_minus[0, j] -= perturb
        with torch.no_grad():
            x_plus = decoder(z_plus)[0]
            x_minus = decoder(z_minus)[0]
            # classifier score difference
            y_plus = classifier(x_plus.unsqueeze(0)).item()
            y_minus = classifier(x_minus.unsqueeze(0)).item()
            delta_scores[j] = abs(y_plus - y_minus)
            # saliency map
            diff_map = _patchwise_ssim(x_plus, x_minus)
            diff_map = _gaussian_blur(diff_map)
            diff_map = (diff_map - diff_map.min()) / (
                diff_map.max() - diff_map.min() + 1e-8
            )
            maps.append(diff_map)

    # select top-k influential units
    topk_vals, topk_idx = torch.topk(delta_scores, k=top_k)
    top_maps = [maps[i] for i in topk_idx.tolist()]

    return recon.cpu(), top_maps, topk_vals.cpu().tolist(), topk_idx.cpu().tolist()
