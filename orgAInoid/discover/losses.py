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

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class ReconLoss(nn.Module):
    def __init__(self, mode: str = "l1", weight: float = 1.0):
        super().__init__()
        self.weight = weight
        if mode == "l1":
            self.fn = F.l1_loss
        elif mode == "l2":
            self.fn = F.mse_loss
        else:
            raise ValueError("mode must be 'l1' or 'l2'")

    def forward(self, recon: torch.Tensor, target: torch.Tensor):
        return self.weight * self.fn(recon, target)

class GanLoss:
    @staticmethod
    def d_loss(real: torch.Tensor, fake: torch.Tensor, weight: float = 1.0):
        return weight * 0.5 * (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())

    @staticmethod
    def g_loss(fake: torch.Tensor, weight: float = 1.0):
        return weight * (-fake.mean())

class VggPerceptual(nn.Module):
    def __init__(self, weight: float = 1.0, layer: str = "relu2_1", resize: bool = False):
        super().__init__()
        self.weight = weight
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        # slice until desired layer id
        cut = {
            "relu1_1": 2,
            "relu2_1": 7,
            "relu3_1": 12,
            "relu4_1": 21,
            "relu5_1": 30,
        }[layer]
        self.sub = nn.Sequential(*list(vgg.children())[: cut + 1])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def _norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor):
        if self.resize and x_hat.shape[-1] != 224:
            x_hat = F.interpolate(x_hat, size=224, mode="bilinear", align_corners=False)
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        fx = self.sub(self._norm(x))
        fhat = self.sub(self._norm(x_hat))
        return self.weight * F.l1_loss(fhat, fx)

class ClassifierPerceptual(nn.Module):
    def __init__(self, classifier: nn.Module, weight: float = 1.0):
        super().__init__()
        self.cls = classifier.eval()
        for p in self.cls.parameters():
            p.requires_grad_(False)
        self.weight = weight

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor):
        with torch.no_grad():
            y = self.cls(x).detach()
        y_hat = self.cls(x_hat)
        return self.weight * F.l1_loss(y_hat, y)

class CovWhitening(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, z: torch.Tensor):
        z = z - z.mean(0, keepdim=True)
        cov = (z.t() @ z) / (z.size(0) - 1)
        off = cov - torch.diag(torch.diag(cov))
        return self.weight * off.pow(2).mean()

class DisentangleLoss(nn.Module):
    def __init__(self, weight_ce: float = 1.0, weight_bce: float = 1.0):
        super().__init__()
        self.w_ce, self.w_bce = weight_ce, weight_bce
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self,
                logits: torch.Tensor,
                idx: torch.Tensor,
                head: torch.Tensor,
                cls_score: torch.Tensor):
        return self.w_ce * self.ce(logits, idx) + self.w_bce * self.bce(head, cls_score)

def build_loss_dict(classifier: nn.Module,
                    *,
                    lambdas: Optional[Dict[str, float]] = None,
                    device: Optional[Union[torch.device, str]] = None) -> Dict[str, nn.Module]:
    """Return loss dict, all moved to *device* (default → classifier.device)."""
    if lambdas is None:
        # corresponds to settings in the paper
        lambdas = {"recon": 5, "gan": 1, "vgg": 5, "cls": 1, "cov": 1, "dis": 1}

    if device is None:
        device = next(classifier.parameters()).device
    device = torch.device(device)

    losses: Dict[str, nn.Module] = {
        "recon": ReconLoss(weight=lambdas["recon"]),
        "vgg": VggPerceptual(weight=lambdas["vgg"]),
        "cls": ClassifierPerceptual(classifier, weight=lambdas["cls"]),
        "cov": CovWhitening(weight=lambdas["cov"]),
        "dis": DisentangleLoss(weight_ce=lambdas["dis"], weight_bce=lambdas["dis"]),
    }

    # move param‑bearing losses to the target device
    for m in losses.values():
        if isinstance(m, nn.Module):
            m.to(device)
    return losses
