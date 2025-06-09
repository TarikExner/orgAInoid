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
from .losses import GanLoss, build_loss_dict, sample_z_noise


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
          img_size: int = 224,
          ) -> None:
    """
    Full training loop for DISCOVER (all six losses), with TensorBoard logging
    and periodic checkpoints.  Assumes:
      - `dataloader` yields batches of images x (shape (B, 3, 224, 224)) already
        normalized to ImageNet/DenseNet statistics and with all required augmentations.
      - `classifier` is a pretrained IVF-CLF network; we freeze it and only use it
        for feature‐extraction in ClassificationPerceptualLoss and for true‐score extraction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    # ── Build models: encoder, decoder, discriminator (latent D), disentangler ──
    encoder, decoder, disc, disent = build_models(
        img_size=img_size,
        latent_dim=latent_dim
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    disc = disc.to(device)
    disent = disent.to(device)

    # ── Freeze classifier (we only use it for perceptual losses / true scores) ──
    classifier = classifier.to(device).eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    # ── Build all loss modules, moved to device ───────────────────────────────
    losses = build_loss_dict(classifier, disent, device=device)
    vgg_loss_fn = losses["vgg"] # L_ImageNet-CLF
    gan_loss_fn = losses["gan"] # GanLoss instance
    cls_loss_fn = losses["cls"] # ClassificationPerceptualLoss
    cov_loss_fn = losses["cov"] # WhiteningLoss
    dis_loss_fn = losses["dis"] # DisentangleLoss
    csl_loss_fn = losses["csl"] # ClassificationSubsetLoss

    # ── Set up optimizers ─────────────────────────────────────────────────────
    opt_g   = optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    opt_d   = optim.Adam(
        disc.parameters(),
        lr=lr, betas=(0.5, 0.999)
    )
    # We will use opt_dis to update BOTH disentangler AND encoder+decoder for disent loss
    opt_dis = optim.Adam(
        itertools.chain(
            disent.parameters(),
            encoder.parameters(),
            decoder.parameters()
        ),
        lr=lr, betas=(0.5, 0.999)
    )

    scaler = GradScaler()
    writer = SummaryWriter(out_dir)
    global_step = 0

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for batch, _ in pbar:
            # ── 1) Load batch ───────────────────────────────────────────────────
            # Assume dataloader returns just the image tensor (B,3,224,224)
            x = batch.to(device)                           # real images
            B = x.size(0)

            # ── 2) Train DISCRIMINATOR (Latent GAN loss #2) ─────────────────
            # a) Compute z = E(x) and detach so encoder is not updated here
            with torch.no_grad():
                z_enc = encoder(x)                         # (B, latent_dim)
            z_enc_det = z_enc.detach()

            # b) Sample z_noise ∼ N(0,I)
            z_noise = sample_z_noise(B, latent_dim, device = device)

            # c) Evaluate D on real noise and on fake (encoder) latents
            real_scores = disc(z_noise)                    # (B,1)
            fake_scores = disc(z_enc_det)                  # (B,1)

            # d) Compute discriminator hinge loss and update
            loss_d = gan_loss_fn.d_loss(real_scores, fake_scores)
            opt_d.zero_grad()
            scaler.scale(loss_d).backward()
            scaler.step(opt_d)
            scaler.update()

            # ── 3) Train GENERATOR (Encoder + Decoder) ───────────────────────
            # a) Forward through encoder+decoder under mixed precision
            with autocast():
                z = encoder(x)
                recons = decoder(z)

                # 3a) L_ImageNet-CLF (#1): VGG perceptual between recons & x
                loss_vgg = vgg_loss_fn(recons, x)

                # 3b) L_IVF-CLF (#3): classification perceptual between recons & x
                loss_cls = cls_loss_fn(recons, x)

                # 3c) Adversarial “generator” loss L_adv (#2): encourage D(z) → +1
                fake_scores_g = disc(z)                    # (B,1)
                loss_gan = gan_loss_fn.g_loss(fake_scores_g)  # negative hinge

                # 3d) L_COV (#4): whitening on z
                loss_cov = cov_loss_fn(z)

                # 3e) L_classification_subset (#6): need IVF-CLF(x) “true score”
                #    Obtain true scores from `classifier(x)`.  We assume classifier(x)
                #    returns either a single‐logit (B,1) or two logits (B,2).  We extract
                #    the probability of the “positive” class in [0,1].
                clf_logits = classifier(x)
                if clf_logits.dim() == 2 and clf_logits.size(1) == 2:
                    # Two‐class output: take P(class=1)
                    true_probs = F.softmax(clf_logits, dim=1)[:, 1]  # (B,)
                else:
                    # Single‐logit output: assume sigmoid‐needed
                    true_probs = torch.sigmoid(clf_logits.view(B, -1).squeeze(1))  # (B,)

                loss_csl = csl_loss_fn(z, true_probs)        # BCE on first 14 dims

                # 3f) L_disentangle (#5): build diff_images and compute dis loss
                #    i) Pick a random latent index for each sample
                true_indices = torch.randint(
                    0, latent_dim, size=(B,), device=device
                )  # (B,)

                #   ii) Standard deviation of z over batch
                std_z = z.std(dim=0, unbiased=False)       # (latent_dim,)

                #  iii) Random ±1 sign for each sample
                signs = torch.randint(0, 2, (B,), device=device) * 2 - 1  # (B,)

                #  iv) Perturbation = ±1.5 * std_z[k] for each b
                perturb_mag = 1.5 * std_z[true_indices]     # (B,)
                delta = signs * perturb_mag                 # (B,)

                #  v) Construct z_perturbed
                z_perturbed = z.clone()
                z_perturbed[torch.arange(B), true_indices] += delta

                #  vi) Decode both
                x_rec = recons                         # (B,3,224,224)
                x_rec_pert = decoder(z_perturbed)            # (B,3,224,224)

                #  vii) diff_images = x_rec_pert − x_rec
                diff_images = x_rec_pert - x_rec              # (B,3,224,224)

                loss_disent = dis_loss_fn(diff_images, true_indices)

                # 3g) Total generator‐side loss = weighted sum of #1, #2(gen), #3, #4, #5, #6
                total_gen_loss = (
                      loss_vgg
                    + loss_gan
                    + loss_cls
                    + loss_cov
                    + loss_disent
                    + loss_csl
                )

            # b) Backpropagate generator loss
            opt_g.zero_grad()
            scaler.scale(total_gen_loss).backward()
            scaler.unscale_(opt_g)

            # 3) Clip generator gradients, exploding losses for COV
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(encoder.parameters(), decoder.parameters()),
                max_norm=1.0
            )
            scaler.step(opt_g)
            scaler.update()

            # ── 4) Train DISENTANGLER (only its own weights) ───────────────────
            # We re‐compute diff_images inside no‐grad for x and z
            with torch.no_grad():
                z_for_disent = encoder(x)
                true_indices2 = torch.randint(
                    0, latent_dim, size=(B,), device=device
                )
                std_z2 = z_for_disent.std(dim=0, unbiased=False)
                signs2 = torch.randint(0, 2, (B,), device=device) * 2 - 1
                perturb2 = signs2 * (1.5 * std_z2[true_indices2])
                z_pert2 = z_for_disent.clone()
                z_pert2[torch.arange(B), true_indices2] += perturb2
                x_rec2 = decoder(z_for_disent)
                x_rec_pert2  = decoder(z_pert2)
                diff_images2 = x_rec_pert2 - x_rec2

            # Forward and backward for disentangler alone
            loss_disent_only = dis_loss_fn(diff_images2, true_indices2)
            opt_dis.zero_grad()
            scaler.scale(loss_disent_only).backward()
            scaler.step(opt_dis)
            scaler.update()

            # ── 5) Log all losses to TensorBoard ──────────────────────────────
            writer.add_scalar("loss/d_loss", loss_d.item(), global_step)
            writer.add_scalar("loss/vgg_loss", loss_vgg.item(), global_step)
            writer.add_scalar("loss/gan_g_loss", loss_gan.item(), global_step)
            writer.add_scalar("loss/cls_loss", loss_cls.item(), global_step)
            writer.add_scalar("loss/cov_loss", loss_cov.item(), global_step)
            writer.add_scalar("loss/disent_loss", loss_disent.item(), global_step)
            writer.add_scalar("loss/csl_loss", loss_csl.item(), global_step)
            writer.add_scalar("loss/disent_only", loss_disent_only.item(), global_step)

            global_step += 1

            pbar.set_postfix(g=f"{loss_gan.item():.2f}",
                 d=f"{loss_d.item():.2f}",
                 dis=f"{loss_disent.item():.2f}")

        # ── 6) At epoch end: write a sample reconstruction grid ─────────────
        with torch.no_grad():
            sample_imgs = x[:8]       # these are still normalized
            sample_recons = recons[:8]  # these are still normalized

            # Unnormalize both
            imgs_to_show = sample_imgs * imagenet_std + imagenet_mean
            recons_to_show = sample_recons * imagenet_std + imagenet_mean

            # Clamp to [0,1] so make_grid doesn’t try to rescale weirdly
            imgs_to_show = imgs_to_show.clamp(0.0, 1.0)
            recons_to_show = recons_to_show.clamp(0.0, 1.0)

            # Now create a grid of 16 images (8 real + 8 recon)
            grid = torchvision.utils.make_grid(
                torch.cat([imgs_to_show, recons_to_show], dim=0),
                nrow=8,    # eight images per row → first row is real, second row is recon
                normalize=False  # we already put them in [0,1]
            )
            writer.add_image("sample/recon", grid, epoch)

        # ── 7) Save checkpoint for this epoch ───────────────────────────────
        ck = {
            "epoch": epoch,
            "enc": encoder.state_dict(),
            "dec": decoder.state_dict(),
            "disc": disc.state_dict(),
            "disent": disent.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "opt_dis": opt_dis.state_dict(),
            "scaler": scaler.state_dict(),
        }
        ckpt_path = Path(out_dir) / f"discover_epoch{epoch:03d}.pth"
        torch.save(ck, ckpt_path)
        print(f"checkpoint saved → {ckpt_path}")

    writer.close()
