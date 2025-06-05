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

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

def sample_z_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    """
    Draws z_noise ∼ N(0, I).
    Returns: Tensor of shape (batch_size, latent_dim) on the given device.
    """
    return torch.randn(batch_size, latent_dim, device=device)

class GanLoss(nn.Module):
    """
    Adversarial hinge‐style GAN losses for latent‐space discrimination.

    Usage:
        # Instantiate with a single weight λ
        gan_loss = GanLoss(weight=1.0)

        # During discriminator update:
        loss_d = gan_loss.d_loss(real_scores, fake_scores)

        # During generator (encoder) update:
        loss_g = gan_loss.g_loss(fake_scores)
    """
    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight: A scalar factor λ to multiply both the discriminator and generator losses.
        """
        super().__init__()
        self.weight = weight

    def d_loss(self,
               real_scores: torch.Tensor,
               fake_scores: torch.Tensor
               ) -> torch.Tensor:
        """
        Discriminator loss L_disc = −E[ log(real_scores) ] − E[ log(1 − fake_scores) ].

        real_scores = D(z_noise)   ∈ (0,1),  should be driven toward 1.
        fake_scores = D(z_real)    ∈ (0,1),  should be driven toward 0.

        Returns:
            weight * (−mean(log(real_scores)) − mean(log(1−fake_scores))).
        """
        eps = 1e-7
        real_clamped = real_scores.clamp(min=eps, max=1.0 - eps)
        fake_clamped = fake_scores.clamp(min=eps, max=1.0 - eps)

        loss_real = -torch.log(real_clamped).mean()         # −E[log D(z_noise)]
        loss_fake = -torch.log(1.0 - fake_clamped).mean()   # −E[log(1 − D(z_real))]

        return self.weight * (loss_real + loss_fake)

    def g_loss(
        self,
        fake_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Generator/Encoder adversarial loss L_adv = −E[ log(fake_scores) ].

        fake_scores = D(z_real) ∈ (0,1),  we want D(z_real) → 1.

        Returns:
            weight * (−mean(log(fake_scores))).
        """
        eps = 1e-7
        fake_clamped = fake_scores.clamp(min=eps, max=1.0 - eps)
        return self.weight * (-torch.log(fake_clamped).mean())

class VggPerceptual(nn.Module):
    """
    Computes sum of L1 losses between feature maps of x and x_rec
    at these VGG-19 layers (indices in torchvision vgg19.features):
      [block3_conv1 (10), block3_conv2 (12), block3_conv3 (14),
       block4_conv1 (19), block4_conv2 (21), block4_conv3 (23), block4_conv4 (25),
       block5_conv1 (28), block5_conv2 (30), block5_conv3 (32), block5_conv4 (34)]
    Assumes x and x_rec are already normalized with ImageNet mean/std.
    """
    def __init__(self, weight: float = 1.0, resize: bool = False):
        super().__init__()
        self.weight = weight
        self.resize = resize

        # Load pretrained VGG-19 features, freeze weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg_features = vgg

        # Indices of target conv layers
        self.target_indices = [10, 12, 14, 19, 21, 23, 25, 28, 30, 32, 34]

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        out = x
        for idx, layer in enumerate(self.vgg_features):
            out = layer(out)
            if idx in self.target_indices:
                feats.append(out)
        return feats

    def forward(self, x_rec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rec: reconstructed image, shape (B,3,H,W), already ImageNet-normalized
            x:    original image,       shape (B,3,H,W), already ImageNet-normalized
        Returns:
            weight * sum_i L1(Φ_i(x_rec), Φ_i(x))
        """
        if self.resize and x_rec.shape[-1] != 224:
            x_rec = F.interpolate(x_rec, size=224, mode="bilinear", align_corners=False)
            x     = F.interpolate(x, size=224, mode="bilinear", align_corners=False)

        feats_x     = self._extract_features(x)
        feats_x_rec = self._extract_features(x_rec)

        loss = 0.0
        for f_orig, f_rec in zip(feats_x, feats_x_rec):
            loss += F.l1_loss(f_rec, f_orig, reduction="mean")

        return self.weight * loss

class FeatureHook:
    def __init__(self, model: nn.Module, layer_names: list[str]):
        """
        Registers forward hooks on `model` at every layer whose “dotted” name
        matches one of `layer_names`.  Captured outputs are stored in self.features.
        
        Example of a “dotted” name: "layer3.0.conv1" in ResNet50, or "features.denseblock3"
        in DenseNet121, or "features.15" in MobileNetV3_Large.  
        """
        self.model = model
        self.layer_names = set(layer_names)
        self.features = {}  # maps layer_name → output tensor
        
        # Walk all submodules and their names; register hook if name matches
        for name, module in model.named_modules():
            if name in self.layer_names:
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook_fn(module, inp, out):
            # Store a copy to avoid accidental in-place modifications
            self.features[name] = out.detach()
        return hook_fn

    def clear(self):
        """Clear captured feature maps before a new forward pass."""
        self.features.clear()

"""\
This extracts the bottom 10 conv layers for better runtime
"""
MOBILENETV3_LAYER_NAMES_TOP10 = [
    "mobilenet_v3_large.features.13.conv.0",
    "mobilenet_v3_large.features.13.conv.3",
    "mobilenet_v3_large.features.13.conv.6",

    "mobilenet_v3_large.features.14.conv.0",
    "mobilenet_v3_large.features.14.conv.3",
    "mobilenet_v3_large.features.14.conv.6",

    "mobilenet_v3_large.features.15.conv.0",
    "mobilenet_v3_large.features.15.conv.3",
    "mobilenet_v3_large.features.15.conv.6",

    "mobilenet_v3_large.features.16.conv",

    "mobilenet_v3_large.avgpool",
    "mobilenet_v3_large.classifier.3"
]

MOBILENETV3_LAYER_NAMES_FULL = [
    "mobilenet_v3_large.features.15.conv.0",
    "mobilenet_v3_large.features.15.conv.3",
    "mobilenet_v3_large.features.15.conv.6",

    "mobilenet_v3_large.features.16.conv",

    "mobilenet_v3_large.avgpool",
    "mobilenet_v3_large.classifier.3"
]

DENSENET121_LAYER_NAMES_TOP10 = [
    "densenet121.features.denseblock4.denselayer11.conv1",
    "densenet121.features.denseblock4.denselayer11.conv2",
    "densenet121.features.denseblock4.denselayer12.conv1",
    "densenet121.features.denseblock4.denselayer12.conv2",
    "densenet121.features.denseblock4.denselayer13.conv1",
    "densenet121.features.denseblock4.denselayer13.conv2",
    "densenet121.features.denseblock4.denselayer14.conv1",
    "densenet121.features.denseblock4.denselayer14.conv2",
    "densenet121.features.denseblock4.denselayer15.conv1",
    "densenet121.features.denseblock4.denselayer15.conv2",
    "densenet121.features.norm5",
    "densenet121.classifier.1"
]

DENSENET121_LAYER_NAMES_FULL = [
    *[f"densenet121.features.denseblock3.denselayer{i}.conv1" for i in range(24)],
    *[f"densenet121.features.denseblock3.denselayer{i}.conv2" for i in range(24)],

    *[f"densenet121.features.denseblock4.denselayer{i}.conv1" for i in range(16)],
    *[f"densenet121.features.denseblock4.denselayer{i}.conv2" for i in range(16)],

    "densenet121.features.norm5",
    "densenet121.classifier.1"
]

RESNET50_LAYER_NAMES_TOP10 = [
    "resnet50.layer4.2.conv3",
    "resnet50.layer4.2.conv2",
    "resnet50.layer4.2.conv1",
    "resnet50.layer4.1.conv3",
    "resnet50.layer4.1.conv2",
    "resnet50.layer4.1.conv1",
    "resnet50.layer4.0.conv3",
    "resnet50.layer4.0.conv2",
    "resnet50.layer4.0.conv1",

    "resnet50.layer3.5.conv3",
    "resnet50.avgpool",
    "resnet50.fc.1"
]

RESNET50_LAYER_NAMES_FULL = [
    "resnet50.layer3.0.conv1", "resnet50.layer3.0.conv2", "resnet50.layer3.0.conv3",
    "resnet50.layer3.1.conv1", "resnet50.layer3.1.conv2", "resnet50.layer3.1.conv3",
    "resnet50.layer3.2.conv1", "resnet50.layer3.2.conv2", "resnet50.layer3.2.conv3",
    "resnet50.layer3.3.conv1", "resnet50.layer3.3.conv2", "resnet50.layer3.3.conv3",
    "resnet50.layer3.4.conv1", "resnet50.layer3.4.conv2", "resnet50.layer3.4.conv3",
    "resnet50.layer3.5.conv1", "resnet50.layer3.5.conv2", "resnet50.layer3.5.conv3",

    "resnet50.layer4.1.conv1", "resnet50.layer4.1.conv2", "resnet50.layer4.1.conv3",
    "resnet50.layer4.2.conv1", "resnet50.layer4.2.conv2", "resnet50.layer4.2.conv3",

    "resnet50.avgpool",
    "resnet50.fc.1"
]


class ClassificationPerceptualLoss(nn.Module):
    """
    Computes the sum of L1 (MAE) errors between feature maps of x and x_rec
    at selected layers (convolutional, flatten, and dense) of a given
    classification network.  The user must supply the exact list of module‐names
    (as they appear in model.named_modules()) to “hook” and compare.

    Example usage:
        # 1) Instantiate your classifier (e.g. ResNet50, DenseNet121, or MobileNetV3_Large)
        clf = ResNet50(num_classes=2, pretrained=True)
        #    (or DenseNet121(num_classes=2), or mobilenet_v3_large(pretrained=True), etc.)

        # 2) Build a list of layer‐names that exactly matches model.named_modules().
        #    For instance, for ResNet50 you might choose:
        #       layer_names = [
        #           "resnet50.layer3.0.conv1",
        #           "resnet50.layer3.0.conv2",
        #           "resnet50.layer3.0.conv3",
        #           "resnet50.layer4.0.conv1",
        #           "resnet50.layer4.0.conv2",
        #           "resnet50.layer4.0.conv3",
        #           "resnet50.layer4.1.conv1",
        #           "resnet50.avgpool",
        #           "resnet50.fc.1"   # The final Linear in the classifier
        #       ]

        # 3) Create the loss module:
        #       loss_cls = ClassificationPerceptualLoss(
        #           clf_model=clf,
        #           layer_names=layer_names,
        #           weight=1.0,
        #           resize=False
        #       )

        # 4) During training, pass your already‐normalized (ImageNet mean/std)
        #    images `x` and their reconstructions `x_rec` into:
        #       loss_value = loss_cls(x_rec, x)

    """

    def __init__(self,
                 clf_model: nn.Module,
                 lightweight: bool = True,
                 weight: float = 1.0,
                 resize: bool = False,
                 target_size: int = 224):
        """
        Args:
            clf_model:   A pretrained (or partially frozen) classification network.
            lightweight: If True, chooses the bottom 10 conv layers, else all conv
                         layers from the last two blocks. Adds the Dense and Flatten
                         layers as well, independent of the variable
            weight:      A float that multiplies the final sum of L1 losses.
            resize:      If True, x and x_rec will be bilinearly resized to (target_size×target_size)
                         before feature extraction.
            target_size: The spatial size to resize to (only used if resize=True).
        """
        super().__init__()
        self.model = clf_model.eval()   # Freeze the classifier—no grads through it.
        self.weight = weight
        self.resize = resize
        self.target_size = target_size
        self.lightweight = lightweight
        """
        layer_names: A list of strings, each one being the exact “dotted” module‐name
                     as it appears in clf_model.named_modules().  At each such name,
                     we will register a forward hook and capture that layer’s output.
        """
        model_name = self.model.__class__.__name__
        self.layer_names = self._choose_layers(model_name)

        # This dict will temporarily hold activations at each hooked layer:
        self._features: Dict[str, torch.Tensor] = {}

        # Register a forward hook on each requested layer name
        # We walk through model.named_modules() and compare the name string.
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                module.register_forward_hook(self._make_hook(name))

    def _choose_layers(self,
                       model_name: str):
        if model_name == "DenseNet121":
            return DENSENET121_LAYER_NAMES_TOP10 if self.lightweight else DENSENET121_LAYER_NAMES_FULL
        elif model_name == "ResNet50":
            return RESNET50_LAYER_NAMES_TOP10 if self.lightweight else RESNET50_LAYER_NAMES_FULL
        elif model_name == "MobileNetV3_Large":
            return MOBILENETV3_LAYER_NAMES_TOP10 if self.lightweight else MOBILENETV3_LAYER_NAMES_FULL
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _make_hook(self, layer_name: str):
        """
        Returns a hook function that saves the output of the submodule
        into self._features[layer_name].
        """
        def hook(module, input, output):
            # We detach so that no gradients flow back through the classification net.
            self._features[layer_name] = output.detach()
        return hook

    def forward(self, x_rec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_rec: Reconstructed image batch, shape (B, 3, H, W), already normalized
                   for the classification network (e.g. ImageNet mean/std).
            x:     Original image batch,       shape (B, 3, H, W), also already normalized.

        Returns:
            A scalar tensor = weight * sum_i [ L1( feat_i(x_rec), feat_i(x) ) ],
            where feat_i is the activation at layer_names[i].
        """
        # 1) If requested, resize both to (target_size × target_size)
        if self.resize and (x.shape[-1] != self.target_size or x_rec.shape[-1] != self.target_size):
            x_rec = F.interpolate(
                x_rec,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False
            )
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False
            )

        # 2) Forward‐pass on the original image x, capturing its features
        self._features = {}
        _ = self.model(x)
        feats_orig = {name: self._features[name] for name in self.layer_names}

        # 3) Forward‐pass on the reconstructed image x_rec, capturing its features
        self._features = {}
        _ = self.model(x_rec)
        feats_rec = {name: self._features[name] for name in self.layer_names}

        # 4) Compute the sum of L1 losses across all hooked layers
        total_loss = 0.0
        for name in self.layer_names:
            f_o = feats_orig[name]
            f_r = feats_rec[name]
            total_loss = total_loss + F.l1_loss(f_r, f_o, reduction="mean")

        return self.weight * total_loss

class WhiteningLoss(nn.Module):
    """
    Implements L_COV = ½ ∑_i (cov(z)_{ii} − 1)^2  +  ½ ∑_{i≠j} (cov(z)_{ij})^2,
    where z has shape (B, latent_dim).  The covariance is computed over the batch dimension.
    A scalar weight multiplies the final result.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: FloatTensor of shape (B, latent_dim).  Each row is a latent vector for one sample.
        Returns:
            A scalar tensor equal to weight * whitening loss from Eq. (5).
        """
        B, D = z.shape  # Batch size, latent dimension

        # 1) Center each latent dimension
        mean_z = z.mean(dim=0, keepdim=True)      # (1, D)
        z_centered = z - mean_z                   # (B, D)

        # 2) Covariance matrix: (D × D)
        cov = (z_centered.transpose(0, 1) @ z_centered) / (B - 1)  # (D, D)

        # 3) Extract diagonal (cov_{ii}) and off-diagonal terms
        diag_cov = torch.diagonal(cov, offset=0)  # (D,)

        cov_off = cov.clone()
        cov_off.fill_diagonal_(0.0)               # zeros on diagonal

        # 4) Compute squared differences
        diag_diff_squared = (diag_cov - 1.0).pow(2).sum()  # sum_i (cov_{ii}−1)^2
        offdiag_squared = cov_off.pow(2).sum()            # sum_{i≠j} (cov_{ij})^2

        # 5) Combine with ½ factors and apply weight
        loss = 0.5 * diag_diff_squared + 0.5 * offdiag_squared
        return self.weight * loss


class DisentangleLoss(nn.Module):
    """
    Implements L_disentangle = −(1 / latent_dim) ∑_{b=1..B} log p_pred(k_b | diff_images[b]),
    where p_pred is the Softmax output of a Disentangler network.  A scalar weight multiplies
    the final result.
    """

    def __init__(self, disentangler_model: nn.Module, weight: float = 1.0):
        """
        Args:
            disentangler_model: An instance of the Disentangler (or identical network), which,
                                when called as disentangler_model(x, z_dummy), returns
                                (softmaxed_logits, head_out).  softmaxed_logits has shape (B, latent_dim)
                                and is already passed through Softmax.
            weight:             A scalar factor to multiply the computed loss.
        """
        super().__init__()
        self.disentangler = disentangler_model
        self.weight = weight

    def forward(
        self,
        diff_images: torch.Tensor,
        true_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            diff_images:  FloatTensor of shape (B, 3, 224, 224), each entry is
                          the “difference image” generated by subtracting the altered
                          reconstruction from the original reconstruction.
                          Must be normalized as the Disentangler expects.
            true_indices: LongTensor of shape (B,) with values in [0..latent_dim−1].
                          Each entry is the index of the latent feature that was altered.

        Returns:
            A scalar tensor = weight * [−(1/latent_dim) ∑_{b=1..B} log p_pred(k_b)].
        """
        B = diff_images.size(0)
        device = diff_images.device

        # 1) Determine latent_dim from the Disentangler’s 350‐way Softmax head:
        latent_dim = self.disentangler.fc_logits.out_features

        # 2) Create a dummy z of shape (B, latent_dim). Values do not matter for logits_350.
        dummy_z = torch.zeros(B, latent_dim, device=device, dtype=diff_images.dtype)

        # 3) Forward‐pass through Disentangler to get Softmaxed logits
        logits_350, _ = self.disentangler(diff_images, dummy_z)
        # logits_350: (B, latent_dim), already Softmax’ed

        # 4) Clamp to avoid log(0)
        eps = 1e-7
        preds = logits_350.clamp(min=eps, max=1.0 - eps)  # (B, latent_dim)

        # 5) Gather probability at the true index for each batch element
        p_true = preds[torch.arange(B, device=device), true_indices]  # (B,)

        # 6) Compute −(1/latent_dim) * sum_b log p_true[b]
        log_p_true = torch.log(p_true)            # (B,)
        loss = - (1.0 / latent_dim) * log_p_true.sum()

        return self.weight * loss


class ClassificationSubsetLoss(nn.Module):
    """
    Implements L_classification_subset = BCE( logit(subset_of_z),  IVF-CLF(x) ),
    where the first `subset_size` dimensions of z are linearly combined into a single
    logit, and we compare that (via a sigmoid internally) to the true IVF-CLF score.
    We use BCEWithLogitsLoss to ensure proper behavior under AMP.

    In the paper, they wrote “Dense(14→1) with Sigmoid, then binary cross‐entropy against IVF-CLF(x).”
    Using BCEWithLogitsLoss here is mathematically identical but safe under torch.cuda.amp.
    """
    def __init__(self, subset_size: int = 14, weight: float = 1.0):
        """
        Args:
            subset_size: Number of latent dims (starting at index 0) to use for this head.
            weight:      Scalar multiplier λ₆ for this loss.
        """
        super().__init__()
        self.subset_size = subset_size
        self.weight = weight

        # Single‐neuron “logit‐head”: Linear( subset_size → 1 ), no bias suppression
        self.fc = nn.Linear(subset_size, 1)

        # BCEWithLogitsLoss combines sigmoid + BCE, safe for autocast
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, z: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:           FloatTensor of shape (B, latent_dim), latent vectors from encoder.
            true_scores: FloatTensor of shape (B,) or (B,1), containing the IVF-CLF(x) “score”
                         in [0,1] for each sample.

        Returns:
            Scalar tensor = weight * BCEWithLogits( logit, true_score ).
        """
        B, D = z.shape
        assert D >= self.subset_size, (
            f"Latent dimension {D} is smaller than subset_size {self.subset_size}"
        )

        # 1) Extract the first `subset_size` features from z
        z_subset = z[:, : self.subset_size]    # shape (B, subset_size)

        # 2) Compute the raw logit (no sigmoid here)
        logit = self.fc(z_subset)              # shape (B, 1)

        # 3) Ensure true_scores is shape (B, 1)
        if true_scores.dim() == 1:
            true_scores = true_scores.view(B, 1)
        else:
            true_scores = true_scores.view(B, 1)

        # 4) Compute BCEWithLogitsLoss between raw logit and true [0,1] score
        loss = self.bce_logits(logit, true_scores)  # mean over batch

        return self.weight * loss

def build_loss_dict(
    classifier: nn.Module,
    disentangler: nn.Module,
    *,
    lambdas: Optional[Dict[str, float]] = None,
    device: Optional[Union[torch.device, str]] = None
) -> Dict[str, nn.Module]:
    """
    Return a dictionary of loss modules, each moved to `device`.  The keys correspond to:
      - "vgg":   VggPerceptual (loss #1)
      - "gan":   GanLoss          (loss #2)
      - "cls":   ClassificationPerceptualLoss (loss #3, IVF-CLF)
      - "cov":   WhiteningLoss    (loss #4)
      - "dis":   DisentangleLoss  (loss #5)
      - "csl":   ClassificationSubsetLoss (loss #6)

    The `lambdas` dict gives per‐loss weights; if None, defaults to the paper’s settings:
      λ1=5 for vgg, λ2=1 for gan, λ3=5 for cls, λ4=1 for cov, λ5=1 for dis, λ6=1 for csl.

    Args:
        classifier:      nn.Module that will be wrapped by ClassificationPerceptualLoss.
        disentangler:    nn.Module that will be wrapped by DisentangleLoss.
        lambdas:         Optional mapping from loss‐key to weight.  Expected keys:
                         "vgg", "gan", "cls", "cov", "dis", "csl".
        device:          torch.device or device‐string; if None, uses classifier’s device.

    Returns:
        A dict mapping each string key ("vgg", "gan", "cls", "cov", "dis", "csl") to its
        corresponding loss module, all moved to `device` (when applicable).
    """
    if lambdas is None:
        # Paper defaults: vgg=5, gan=1, cls=5, cov=1, dis=1, csl=1
        lambdas = {
            "vgg": 5.0,
            "gan": 1.0,
            "cls": 5.0,
            "cov": 1.0,
            "dis": 1.0,
            "csl": 1.0
        }

    if device is None:
        # Use classifier’s device by default
        device = next(classifier.parameters()).device
    device = torch.device(device)

    losses: Dict[str, nn.Module] = {
        # Loss #1: ImageNet‐CLF perceptual loss (VGG)
        "vgg": VggPerceptual(weight=lambdas["vgg"]),

        # Loss #2: Adversarial GAN loss
        # (GanLoss is a class with static methods; store the class itself or an instance.
        #  Here we store an instance, even though no parameters are updated inside GanLoss.)
        "gan": GanLoss(weight=lambdas["gan"]),

        # Loss #3: IVF‐CLF perceptual loss (ClassificationPerceptualLoss wraps `classifier`)
        "cls": ClassificationPerceptualLoss(classifier, weight=lambdas["cls"]),

        # Loss #4: Covariance whitening loss
        "cov": WhiteningLoss(weight=lambdas["cov"]),

        # Loss #5: Disentanglement‐index classification loss (wraps `disentangler`)
        "dis": DisentangleLoss(disentangler, weight=lambdas["dis"]),

        # Loss #6: Classification‐driving subset BCE loss
        "csl": ClassificationSubsetLoss(subset_size=14, weight=lambdas["csl"])
    }

    for loss_module in losses.values():
        if isinstance(loss_module, nn.Module):
            loss_module.to(device)

    return losses
