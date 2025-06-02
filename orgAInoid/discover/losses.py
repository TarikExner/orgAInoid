"""Modular loss functions for DISCOVER
------------------------------------------------
Six independent loss terms are wrapped as tiny `nn.Module`s so you can
mix‑and‑match weights from a config dict.  All take a `weight` argument
so their contribution can be scaled at runtime.

Available losses
::::::::::::::::
* **ReconLoss** ‒ L1 or L2 image reconstruction.
* **GanLoss** ‒ hinge‑style generator / discriminator terms.
* **VggPerceptual** ‒ perceptual distance on VGG‑19 reluX_1 features.
* **ClassifierPerceptual** ‒ same idea but uses a *frozen* user model.
* **CovWhitening** ‒ penalises off‑diagonal covariance in the latent batch.
* **DisentangleLoss** ‒ categorical CE for the “which‑feature” CNN **plus**
  BCE on the 14‑unit head that matches the frozen classifier score.

Typical usage
:::::::::::::
>>> crit = ReconLoss(mode='l1', weight=5.0)
>>> loss = crit(recon, target)  # already scaled by weight

>>> g_loss, d_loss = gan(fake_logits, real_logits)

>>> total = sum(loss_dict.values())

Weights
:::::::
The paper sets λ = [5, 1, 5, 1, 1, 1] in that order; you can of course
pass any floats when you instantiate each loss.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class ReconLoss(nn.Module):
    """Pixel reconstruction loss (L1 or L2)."""

    def __init__(self, mode: str = "l1", weight: float = 1.0):
        super().__init__()
        self.weight = weight
        if mode == "l1":
            self.fn = F.l1_loss
        elif mode == "l2":
            self.fn = F.mse_loss
        else:
            raise ValueError("mode must be 'l1' or 'l2'")

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.weight * self.fn(recon, target)

class GanLoss:
    """Stateless hinge GAN loss.

    Use like:
        g_loss = GanLoss.g_loss(fake_logits)
        d_loss = GanLoss.d_loss(real_logits, fake_logits)
    """

    @staticmethod
    def d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        loss_real = F.relu(1.0 - real_logits).mean()
        loss_fake = F.relu(1.0 + fake_logits).mean()
        return weight * 0.5 * (loss_real + loss_fake)

    @staticmethod
    def g_loss(fake_logits: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
        return weight * (-fake_logits.mean())

class VggPerceptual(nn.Module):
    """Perceptual distance in VGG‑19 reluX_1 features."""

    def __init__(self, weight: float = 1.0, layer: str = "relu2_1", resize: bool = False):
        super().__init__()
        self.weight = weight
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.sub = nn.Sequential()
        for name, module in vgg._modules.items():
            self.sub.add_module(name, module)
            if name == {
                "relu1_1": "1",
                "relu2_1": "6",
                "relu3_1": "11",
                "relu4_1": "20",
                "relu5_1": "29",
            }[layer]:
                break
        self.resize = resize

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.resize and recon.shape[-1] != 224:
            recon = F.interpolate(recon, size=224, mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=224, mode="bilinear", align_corners=False)
        recon_f = self.sub(self._norm(recon))
        target_f = self.sub(self._norm(target))
        return self.weight * F.l1_loss(recon_f, target_f)

class ClassifierPerceptual(nn.Module):
    """Keeps the frozen classifier score close between x and x̂."""

    def __init__(self, classifier: nn.Module, weight: float = 1.0):
        super().__init__()
        self.cls = classifier.eval()
        for p in self.cls.parameters():
            p.requires_grad_(False)
        self.weight = weight

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y_target = self.cls(target).detach()
        y_recon = self.cls(recon)
        return self.weight * F.l1_loss(y_recon, y_target)

class CovWhitening(nn.Module):
    """Penalise off‑diagonal covariance to encourage disentanglement."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D]
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.t() @ z) / (z.size(0) - 1)  # [D, D]
        off_diag = cov - torch.diag(torch.diag(cov))
        return self.weight * (off_diag.pow(2).mean())


# ───────────────────── disentanglement + BCE head ──────────────────


class DisentangleLoss(nn.Module):
    """Combines CE for which‑feature task and BCE on 14‑unit head."""

    def __init__(self, weight_ce: float = 1.0, weight_bce: float = 1.0):
        super().__init__()
        self.w_ce = weight_ce
        self.w_bce = weight_bce
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()

    def forward(
        self,
        logits: torch.Tensor,  # [B, D] from which‑feature CNN
        targets: torch.Tensor,  # [B] integer idx of feature perturbed
        head_pred: torch.Tensor,  # [B, 1] from 14‑unit head
        cls_score: torch.Tensor,  # [B, 1] frozen classifier score of original
    ) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_bce = self.bce(head_pred, cls_score)
        return self.w_ce * loss_ce + self.w_bce * loss_bce

def build_loss_dict(
    classifier: nn.Module,
    lambdas: Dict[str, float] | None = None,
) -> Dict[str, nn.Module]:
    """Factory that returns the six losses in a dict keyed by name."""

    if lambdas is None:
        # paper defaults
        lambdas = {"recon": 5, "gan": 1, "vgg": 5, "cls": 1, "cov": 1, "dis": 1}

    loss_dict: Dict[str, nn.Module] = {
        "recon": ReconLoss(weight=lambdas["recon"]),
        # gan handled separately because needs both real & fake logits
        "vgg": VggPerceptual(weight=lambdas["vgg"]),
        "cls": ClassifierPerceptual(classifier, weight=lambdas["cls"]),
        "cov": CovWhitening(weight=lambdas["cov"]),
        # DisentangleLoss splits its two parts inside
        "dis": DisentangleLoss(weight_ce=lambdas["dis"], weight_bce=lambdas["dis"]),
    }
    return loss_dict
