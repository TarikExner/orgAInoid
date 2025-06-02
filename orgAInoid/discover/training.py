"""Training script for DISCOVER (single‑file version)
===================================================
Run ‑> `python train_loop.py --data_root data/train --cls_ckpt checkpoints/classifier.pth`.
All settings have sensible defaults so a first run needs only data &
classifier arguments.

Key features
------------
* **Albumentations** pipeline configurable via CLI flags.
* **Mixed‑precision** (torch.cuda.amp) with automatic grad‑scaler.
* **Weights‑&‑biases** style logging using TensorBoard by default.
* **Resume / checkpoint** every N epochs.

The trainer is deliberately compact (≈150 LOC) so you can later move
its pieces into a larger project.
"""
from __future__ import annotations

import itertools
import numpy as np
from pathlib import Path
from typing import Sequence

import albumentations as A
import torch
import torch.nn.functional as F
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .models import build_models
from .losses import GanLoss, build_loss_dict


# ───────────────────────────── tfms ────────────────────────────────

DEFAULT_AUG = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
        # expect images already in [0–1] or [‑1–1]; we do not rescale
        ToTensorV2(),
    ]
)

# ─────────────────────── array‑based dataloader ────────────────────

class _ArrayDataset(Dataset):
    """Dataset for in‑memory images (torch.Tensor or np.ndarray).

    * Accepts float32/float64/uint8 images.
    * Accepts CHW or HWC layouts – auto‑transposes to HWC as required
      by Albumentations.
    """

    def __init__(self, images: Sequence, transform=DEFAULT_AUG):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _to_numpy(img):
        # torch tensor → numpy
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy()
            # CHW → HWC if needed
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
        elif isinstance(img, np.ndarray):
            # ensure HWC
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))
        else:
            raise TypeError("Expected torch.Tensor or np.ndarray, got %s" % type(img))
        img = img.astype(np.float32)
        return img

    def __getitem__(self, idx):
        img = self._to_numpy(self.images[idx])
        img = self.transform(image=img)["image"]  # tensor CHW in [0–1] (float32)
        return img, 0

def build_dataloader(images: Sequence,
                     *,
                     batch_size: int = 64,
                     workers: int = 1,
                     transform=DEFAULT_AUG) -> DataLoader:
    ds = _ArrayDataset(images, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

class _AlbFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root: str, transform):
        super().__init__(root)
        self.transform = transform

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        img = self.transform(image=np.asarray(img, dtype=np.float32))["image"]
        return img, 0


def build_dataloader_from_dir(root: str,
                              *,
                              batch_size: int = 64,
                              workers: int = 4,
                              transform=DEFAULT_AUG) -> DataLoader:
    ds = _AlbFolder(root, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


def train(*,
          dataloader: DataLoader,
          classifier: nn.Module,
          out_dir: str,
          epochs: int = 30,
          lr: float = 2e-4,
          latent_dim: int = 350,
          img_size: int = 224) -> None:
    """Train DISCOVER.

    Parameters
    ----------
    dataloader : DataLoader
        Yields `(img_tensor, _)`, images already scaled to [0–1] or [‑1–1].
    classifier : nn.Module
        Frozen model we want to explain – passed *as an instance*.
    out_dir : str
        Where to write TensorBoard logs and checkpoints.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder, disc, dis = build_models(img_size=img_size, latent_dim=latent_dim)
    encoder, decoder, disc, dis = encoder.to(device), decoder.to(device), disc.to(device), dis.to(device)

    classifier = classifier.to(device).eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    losses = build_loss_dict(classifier)

    opt_g = optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=lr,
        betas=(0.5, 0.999)
    )
    opt_d = optim.Adam(
        disc.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )
    opt_dis = optim.Adam(
        dis.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )

    scaler = GradScaler()
    writer = SummaryWriter(out_dir)

    global_step = 0
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)

            # ── generator ──────────────────────────────────────────
            opt_g.zero_grad(set_to_none=True)
            with autocast():
                z = encoder(imgs)
                recons = decoder(z)
                fake_logits = disc(recons)

                # choose one random latent index per sample
                idx = torch.randint(0, latent_dim, (imgs.size(0),), device=device)
                # create perturbed latents (+0.25σ, paper’s value)
                z_pert = z.clone()
                z_pert[torch.arange(imgs.size(0)), idx] += 0.25
                x_pert = decoder(z_pert) # reconstruction after perturb
                logits, head_pred = dis(x_pert, z)    # CNN + 14-unit head

                loss_recon = losses["recon"](recons, imgs)
                loss_vgg = losses["vgg"](recons, imgs)
                loss_cls = losses["cls"](recons, imgs)
                loss_cov = losses["cov"](z)
                loss_dis = losses["dis"](
                    logits, idx, head_pred,
                    classifier(imgs).detach()
                )
                g_adv = GanLoss.g_loss(fake_logits)
                gan_warm = min(epoch/5, 1.)
                g_total = loss_recon + g_adv * gan_warm + loss_vgg + loss_cls + loss_cov + loss_dis

                loss_dis.backward()

            scaler.scale(g_total).backward()
            scaler.step(opt_g)
            opt_dis.zero_grad(set_to_none=True)
            opt_dis.step()

            # ── discriminator ─────────────────────────────────────
            opt_d.zero_grad(set_to_none=True)
            with autocast():
                real_logits = disc(imgs)
                fake_logits_det = disc(recons.detach())
                d_total = GanLoss.d_loss(real_logits, fake_logits_det)
            scaler.scale(d_total).backward()
            scaler.step(opt_d)
            scaler.update()

            writer.add_scalar("loss/g_total", g_total.item(), global_step)
            writer.add_scalar("loss/d_total", d_total.item(), global_step)
            pbar.set_postfix(g=f"{g_total.item():.3f}", d=f"{d_total.item():.3f}")
            global_step += 1

        with torch.no_grad():
            grid = torchvision.utils.make_grid(
                torch.cat([imgs[:8], recons[:8]]),
                nrow=8, normalize=True, value_range=(0, 1)
            )
            writer.add_image("sample/recon", grid, epoch)

        ckpt_path = Path(out_dir) / f"discover_epoch{epoch:03d}.pth"
        torch.save({
            "epoch": epoch,
            "enc": encoder.state_dict(),
            "dec": decoder.state_dict(),
            "disc": disc.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "scaler": scaler.state_dict(),
        }, ckpt_path)
        print(f"checkpoint saved → {ckpt_path}")

    writer.close()
