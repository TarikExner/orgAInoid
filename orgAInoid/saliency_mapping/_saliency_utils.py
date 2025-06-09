import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy, kurtosis

from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.fx as fx

import torch.nn as nn

from typing import Optional, Literal, Union, Any

from captum.attr import (IntegratedGradients,
                         NoiseTunnel,
                         Saliency,
                         DeepLiftShap,
                         GradientShap,
                         LayerGradCam,
                         GuidedGradCam,
                         Occlusion,
                         FeatureAblation,
                         KernelShap)

from ..classification._dataset import OrganoidDataset
from ..classification._utils import create_dataloader
from ..classification.models import (DenseNet121,
                                     ResNet50,
                                     MobileNetV3_Large)


def initialize_model(model: torch.nn.Module,
                     state_dict_path: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    model.cuda()
    return model

def instantiate_model(model_name: str) -> torch.nn.Module:
    if model_name == "DenseNet121":
        return DenseNet121()
    elif model_name == "ResNet50":
        return ResNet50()
    elif model_name == "MobileNetV3_Large":
        return MobileNetV3_Large()
    else:
        raise ValueError(f"Model name not known: {model_name}")

def disable_inplace_relu(model: nn.Module):
    def fn(m):
        if isinstance(m, nn.ReLU):
            m.inplace = False
    model.apply(fn)

def remove_relu_modules(model: nn.Module) -> nn.Module:
    gm: fx.GraphModule = fx.symbolic_trace(model)

    for node in list(gm.graph.nodes):
        if node.op == "call_module":
            submod = dict(gm.named_modules())[node.target]
            if isinstance(submod, nn.ReLU):
                # insert a functional F.relu node in its place
                with gm.graph.inserting_after(node):
                    new_node = gm.graph.call_function(
                        F.relu, args=node.args, kwargs={"inplace": False}
                    )
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm

def initialize_models(models: list[str],
                      experiment: str,
                      readout: str,
                      model_directory: str,
                      baseline_directory: str) -> dict[str, torch.nn.Module]:
    model_dict = {}
    for model_name in models:
        state_dict_path = os.path.join(
            model_directory,
            f"{model_name}_val_f1_{experiment}_{readout}_base_model.pth"
        )
        baseline_state_dict_path = os.path.join(
            baseline_directory,
            f"{model_name}_val_f1_{experiment}_{readout}_base_model.pth"
        )
        raw_model = instantiate_model(model_name)
        if model_name == "ResNet50":
            raw_model = remove_relu_modules(raw_model)
            # disable_inplace_relu(raw_model)
        model_dict[model_name] = initialize_model(raw_model, state_dict_path)

        raw_model = instantiate_model(model_name)
        if model_name == "ResNet50":
            raw_model = remove_relu_modules(raw_model)
            # disable_inplace_relu(raw_model)
        model_dict[f"{model_name}_baseline"] = initialize_model(raw_model, baseline_state_dict_path)

    return model_dict

Readouts = Literal["RPE_Final", "Lens_Final", "RPE_classes", "Lens_classes"]
def get_dataloader(dataset: OrganoidDataset,
                   well: str,
                   readout: Readouts) -> DataLoader:
    metadata = dataset.metadata[
        (dataset.metadata["well"] == well) &
        (dataset.metadata["slice"].isin(dataset.dataset_metadata.slices))
    ].copy()
    metadata = metadata.sort_values("loop", ascending = True)
    well_idxs = metadata["IMAGE_ARRAY_INDEX"].to_numpy()
    data_images = dataset.X[well_idxs]
    data_classes = dataset.y[readout][well_idxs]
    return create_dataloader(data_images, data_classes,
                             batch_size = 1,
                             shuffle = False,
                             train = False)

def masks_to_tensor(mask_stack: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask_stack).float()  # (n_images,1,224,224)


def create_baseline_image(img: Union[torch.Tensor, np.ndarray],
                          mask: Union[torch.Tensor, np.ndarray],
                          method: Literal["gaussian", "mean", "zero"] = "mean") -> torch.Tensor:
    """
    Given a three‐channel image and its single‐channel mask (shapes (1,3,H,W) and (1,1,H,W)),
    produce a baseline of shape (1,3,H,W) according to:
      - "gaussian": Apply Gaussian blur (σ=5) to each channel, then multiply by mask.
      - "mean":     For each channel, fill foreground (mask=1) with that channel's mean over foreground.
      - "zero":     Return a zero tensor of shape (1,3,H,W).

    Args:
        img:    torch.Tensor or numpy array, shape (1, 3, H, W), dtype float32.
        mask:   torch.Tensor or numpy array, shape (1, 1, H, W), values {0,1}.
        method: One of "gaussian", "mean", "zero".

    Returns:
        baseline_t: torch.FloatTensor of shape (1, 3, H, W) on CPU.
    """
    # Convert to torch.Tensor on CPU if needed
    if isinstance(img, np.ndarray):
        img_t = torch.from_numpy(img).float()
    else:
        img_t = img.detach().cpu().float()

    if isinstance(mask, np.ndarray):
        mask_t = torch.from_numpy(mask).float()
    else:
        mask_t = mask.detach().cpu().float()

    # Validate shapes
    if img_t.ndim != 4 or img_t.shape[0] != 1 or img_t.shape[1] != 3:
        raise ValueError("`img` must have shape (1,3,H,W).")
    if mask_t.ndim != 4 or mask_t.shape[0] != 1 or mask_t.shape[1] != 1:
        raise ValueError("`mask` must have shape (1,1,H,W).")
    if img_t.shape[2:] != mask_t.shape[2:]:
        raise ValueError("Spatial dimensions of `img` and `mask` must match.")

    _, _, H, W = img_t.shape

    if method == "zero":
        return torch.zeros_like(img_t)

    # Prepare numpy arrays for processing
    img_np = img_t.numpy()[0]        # shape (3, H, W)
    mask_np = mask_t.numpy()[0, 0]   # shape (H, W)

    baseline_np = np.zeros((3, H, W), dtype=np.float32)

    if method == "gaussian":
        # Blur each channel and apply mask
        for c in range(3):
            channel = img_np[c]                   # (H, W)
            blurred = gaussian_filter(channel, sigma=5)
            baseline_np[c] = blurred * mask_np

    elif method == "mean":
        # For each channel, compute mean over foreground and fill
        for c in range(3):
            channel = img_np[c]               # (H, W)
            fg_vals = channel[mask_np == 1]
            if fg_vals.size == 0:
                mean_val = 0.0
            else:
                mean_val = float(fg_vals.mean())
            filled = np.full((H, W), mean_val, dtype=np.float32)
            baseline_np[c] = filled * mask_np

    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert back to torch with shape (1,3,H,W)
    baseline_t = torch.from_numpy(baseline_np).unsqueeze(0).float()
    return baseline_t

def compute_integrated_gradients(**kwargs: Any) -> torch.Tensor:
    """
    Integrated Gradients wrapped in a NoiseTunnel (SmoothGrad-IG).
    Expects in kwargs:
      - model
      - image  (1,3,224,224) on device
      - baseline (1,3,224,224) on device
      - target (optional)
      - nt_samples (optional, default=50)
      - stdevs (optional, default=0.1)

    Returns:
      attributions: (1,3,224,224) float32 tensor on device.
    """
    model = kwargs["model"]
    image = kwargs["image"]
    baseline = kwargs["baseline"]
    
    # Determine target
    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()
    
    nt_samples = kwargs.get("nt_samples", 50)
    stdevs     = kwargs.get("stdevs", 0.1)
    
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    
    attributions = nt.attribute(
        inputs=image,
        baselines=baseline,
        target=target,
        nt_type="smoothgrad",
        nt_samples=nt_samples,
        n_steps=10,
        stdevs=stdevs
    )
    return attributions

def compute_saliency(**kwargs: Any) -> torch.Tensor:
    """
    Vanilla gradients (Saliency) wrapped in a NoiseTunnel (SmoothGrad-Saliency).
    Expects in kwargs:
      - model
      - image  (1,3,224,224) on device
      - baseline (ignored, but must be present)
      - target (optional)
      - nt_samples (optional, default=50)
      - stdevs (optional, default=0.1)

    Returns:
      attributions: (1,3,224,224) float32 tensor on device.
    """
    model = kwargs["model"]
    image = kwargs["image"]
    # baseline is accepted but not used by Saliency
    
    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()
    
    nt_samples = kwargs.get("nt_samples", 50)
    stdevs = kwargs.get("stdevs", 0.1)
    
    sal = Saliency(model)
    nt  = NoiseTunnel(sal)
    
    attributions = nt.attribute(
        inputs=image,
        target=target,
        nt_type="smoothgrad",
        nt_samples=nt_samples,
        stdevs=stdevs
    )
    return attributions

def compute_deeplift_shap_equal_baseline(**kwargs: Any) -> torch.Tensor:
    """
    DeepLiftShap attributions.
    Expects in kwargs:
      - model
      - image    (1,3,224,224) on device
      - baseline (1,3,224,224) on device   → will be repeated n_baselines times
      - target (optional)
      - nt_samples (optional, used as n_baselines, default=10)
      - stdevs (ignored)

    Returns:
      attributions: (1,3,224,224) float32 tensor on device.
    """
    model = kwargs["model"]
    image = kwargs["image"]
    baseline = kwargs["baseline"]
    
    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()
    
    n_baselines = kwargs.get("nt_samples", 10)
    # Repeat the single baseline n_baselines times
    b = baseline.repeat(n_baselines, 1, 1, 1)  # (n_baselines,3,224,224)
    
    dls = DeepLiftShap(model)
    attributions = dls.attribute(inputs=image, baselines=b, target=target)
    return attributions

def compute_gradient_shap_equal_baseline(**kwargs: Any) -> torch.Tensor:
    """
    GradientShap attributions.
    Expects in kwargs:
      - model
      - image   (1,3,224,224) on device
      - baseline (1,3,224,224) on device → will be repeated n_baselines times
      - target (optional)
      - nt_samples (optional, used as n_baselines, default=10)
      - stdevs (ignored)

    Returns:
      attributions: (1,3,224,224) float32 tensor on device.
    """
    model = kwargs["model"]
    image = kwargs["image"]
    baseline = kwargs["baseline"]
    
    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()
    
    n_baselines = kwargs.get("nt_samples", 10)
    b = baseline.repeat(n_baselines, 1, 1, 1)
    
    gs = GradientShap(model)
    attributions = gs.attribute(inputs=image, baselines=b, target=target)
    return attributions

def compute_gradient_shap(**kwargs: Any) -> torch.Tensor:
    """
    GradientShap attributions, but with 
    Expects in kwargs:
      - model
      - image   (1,3,224,224) on device
      - baseline (1,3,224,224) on device → will be repeated n_baselines times
      - target (optional)
      - nt_samples (optional, used as n_baselines, default=10)
      - stdevs (ignored)

    Returns:
      attributions: (1,3,224,224) float32 tensor on device.
    """
    model = kwargs["model"]
    image = kwargs["image"]
    baseline = kwargs["baseline"]
    
    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()
    
    n_baselines = kwargs.get("nt_samples", 10)
    eps = 1
    noisy_refs = []
    for _ in range(n_baselines):
        noise = torch.randn_like(baseline) * eps
        noisy_refs.append(baseline + noise)
    b = torch.clamp(torch.cat(noisy_refs, dim=0), 0.0, 1.0)
    
    gs = GradientShap(model)
    attributions = gs.attribute(inputs=image, baselines=b, target=target)
    return attributions

def compute_deeplift_shap(**kwargs: Any) -> torch.Tensor:
    """
    DeepLiftShap attributions.
    Expects in kwargs:
      - model
      - image    (1,3,224,224) on device
      - baseline (1,3,224,224) on device   → will be repeated n_baselines times
      - target (optional)
      - nt_samples (optional, used as n_baselines, default=10)
      - stdevs (ignored)

    Returns:
      attributions: (1,3,224,224) float32 tensor on device.
    """
    model = kwargs["model"]
    image = kwargs["image"]
    baseline = kwargs["baseline"]
    
    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()
    
    n_baselines = kwargs.get("nt_samples", 10)
    eps = 1
    noisy_refs = []
    for _ in range(n_baselines):
        noise = torch.randn_like(baseline) * eps
        noisy_refs.append(baseline + noise)
    b = torch.clamp(torch.cat(noisy_refs, dim=0), 0.0, 1.0)
    
    dls = DeepLiftShap(model)
    attributions = dls.attribute(inputs=image, baselines=b, target=target)
    return attributions

def compute_grad_cam(**kwargs: Any) -> torch.Tensor:
    """
    Grad-CAM heatmap.
    Expects in kwargs:
      - model
      - image        (1,3,224,224) on device
      - baseline     (ignored, but must be present)
      - target_layer (nn.Module): the conv layer to hook
      - target (optional)
      - nt_samples, stdevs (ignored)

    Returns:
      attributions: (1,3,224,224) float32 heatmap on device.
    """
    model = kwargs["model"]
    image = kwargs["image"]
    target_layer = kwargs["target_layer"]
    
    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()
    
    layer_gc = LayerGradCam(model, target_layer)
    cam = layer_gc.attribute(inputs=image, target=target)  # (1, C', h', w')
    
    # If C' > 1, average over channel dimension
    if cam.shape[1] > 1:
        spatial = cam.mean(dim=1, keepdim=True)  # (1,1,h',w')
    else:
        spatial = cam  # already (1,1,h',w')
    
    cam_upsampled = F.interpolate(spatial, size=image.shape[-2:], mode='bilinear', align_corners=False)
    cam_relu = torch.relu(cam_upsampled)       # (1,1,224,224)
    heatmap_3ch = cam_relu.repeat(1, 3, 1, 1)      # (1,3,224,224)
    return heatmap_3ch

def compute_guided_grad_cam(**kwargs: Any) -> torch.Tensor:
    """
    Guided Grad-CAM attributions.
    Expects in kwargs:
      - model
      - image        (1,3,224,224) on device
      - baseline     (ignored, but must be present)
      - target_layer (nn.Module): the conv layer to hook
      - target (optional)
      - nt_samples, stdevs (ignored)

    Returns:
      attributions: (1,3,224,224) float32 tensor on device.
    """
    model = kwargs["model"]
    image = kwargs["image"]
    target_layer = kwargs["target_layer"]
    
    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()
    
    guided_gc = GuidedGradCam(model, target_layer)
    attributions = guided_gc.attribute(inputs=image, target=target)  # (1,3,224,224)
    return attributions

def compute_smooth_occlusion(**kwargs: Any) -> torch.Tensor:
    """
    Occlusion wrapped in a NoiseTunnel (“Smooth Occlusion”).
    Expects kwargs:
      - model, image, baseline            (same as others)
      - target (optional)
      - nt_samples (optional, default=25)   ← # noisy runs
      - stdevs      (ignored)              Occlusion is already a perturbation
      - patch_size  (optional, default=15) patch (H,W)
      - stride      (optional, default=8)  stride (H,W)

    Returns: (1,3,H,W) tensor
    """
    model    = kwargs["model"]
    image    = kwargs["image"]            # (1,3,H,W)
    baseline = kwargs["baseline"]

    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()

    nt_samples = kwargs.get("nt_samples", 25)
    patch = kwargs.get("patch_size", 15)
    stride_val = kwargs.get("stride", 8)

    # patch covers all 3 channels; stride likewise
    patch_shape  = (3, patch, patch)
    stride_shape = (3, stride_val, stride_val)

    occ = Occlusion(model)
    nt  = NoiseTunnel(occ)

    attr = nt.attribute(
        inputs=image,
        baselines=baseline,
        sliding_window_shapes=patch_shape,
        strides=stride_shape,
        nt_type="smoothgrad_sq",    # SmoothGrad-like averaging
        nt_samples=nt_samples,
        target=target
    )
    return attr

def compute_feature_ablation(**kwargs: Any) -> torch.Tensor:
    """
    Feature Ablation using a pre-computed super-pixel mask.
    Expects in kwargs:
      - model, image, baseline
      - feature_mask : (1,1,H,W) tensor of integer labels (SLIC super‐pixels)
      - target (optional)

    Returns: (1,3,H,W) tensor
    """
    model = kwargs["model"]
    image = kwargs["image"]
    baseline = kwargs["baseline"]
    feature_mask = kwargs["feature_mask"]    # (1,1,H,W)

    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()

    fa = FeatureAblation(model)
    # Captum expects mask without batch/channel dims
    mask_2d = feature_mask.squeeze(0).squeeze(0)  # (H,W)

    # attribute returns a list per tensor-input; we supply one input → get tensor
    attr = fa.attribute(
        inputs=image,
        baselines=baseline,
        additional_forward_args=None,
        feature_mask=mask_2d.long(),
        target=target
    )
    return attr

def compute_kernel_shap(**kwargs: Any) -> torch.Tensor:
    """
    Kernel SHAP (LIME-based) attribution.
    Expects in kwargs:
      - model, image
      - baseline       (single scalar or (1,3,H,W) tensor)  ← passed as 'baselines'
      - feature_mask   (optional) same semantics as FeatureAblation
      - n_samples      (optional, default=50)
      - target         (optional)
    Returns: (1,3,H,W) tensor (matches input shape).
    """
    model = kwargs["model"]
    image = kwargs["image"]
    baseline = kwargs.get("baseline", 0.0)      # KernelShap can take scalar
    fmask = kwargs.get("feature_mask", None)

    if "target" in kwargs and kwargs["target"] is not None:
        target = kwargs["target"]
    else:
        with torch.no_grad():
            target = torch.argmax(model(image), dim=1).item()

    n_samples = kwargs.get("n_samples", 50)

    ks = KernelShap(model)
    attr = ks.attribute(
        inputs=image,
        baselines=baseline,
        target=target,
        feature_mask=fmask.squeeze(0).squeeze(0).long() if fmask is not None else None,
        n_samples=n_samples,
        return_input_shape=True
    )
    return attr

def preprocess_attribution(att: torch.Tensor,
                           take_abs: bool = True,
                           eps: float = 1e-8) -> np.ndarray:
    """
    Average the 3 channels of an attribution map → 2‑D
    then z‑score normalise (optional abs).

    Args
    ----
    att : torch.Tensor  shape (1, 3, H, W)
    take_abs : bool     if True take |att| before averaging
    eps : float         small number to avoid div/0

    Returns
    -------
    heat : np.ndarray   shape (H, W), float32, zero‑mean / unit‑var
    """
    if att.ndim != 4 or att.shape[1] != 3:
        raise ValueError("attribution must have shape (1,3,H,W)")
    hm = att.detach().cpu().squeeze(0)
    if take_abs:
        hm = hm.abs()
    hm = hm.mean(dim=0)
    hm = (hm - hm.mean()) / (hm.std() + eps)
    return hm.numpy().astype(np.float32)

def pearson_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Pixel‑wise Pearson r between two normalised maps."""
    return float(np.corrcoef(A.flatten(), B.flatten())[0, 1])


def ssim_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Structural Similarity Index (SSIM) over the whole image."""
    return float(ssim(A, B, gaussian_weights=True, data_range=B.max() - B.min()))


def jaccard_topk_similarity(A: np.ndarray, B: np.ndarray,
                            topk_ratio: float = 0.01) -> float:
    """
    Convert each map to a binary mask that keeps the
    top k% pixels then compute Jaccard |A∩B| / |A∪B|.
    """
    k = max(1, int(topk_ratio * A.size))
    thresh_A = np.partition(A.flatten(), -k)[-k]
    thresh_B = np.partition(B.flatten(), -k)[-k]
    bin_A = A >= thresh_A
    bin_B = B >= thresh_B
    inter = (bin_A & bin_B).sum()
    union = (bin_A | bin_B).sum()
    return float(inter) / (union + 1e-8)


_METRIC_FUNS = {
    "pearson": pearson_similarity,
    "ssim": ssim_similarity,
    "jaccard": jaccard_topk_similarity,
}

def compute_similarity_matrix(attributions: dict[str, torch.Tensor],
                              metrics: tuple[str, ...] = ("pearson", "ssim", "jaccard")
                              ) -> dict[str, np.ndarray]:
    """
    Build a |M|×|M| similarity matrix per metric for one image.

    Parameters
    ----------
    attributions : dict  {method_name: (1,3,H,W) tensor}
    metrics      : tuple which metrics to compute

    Returns
    -------
    sims : dict  {metric_name: np.ndarray of shape (M,M)}
    """
    names = list(attributions.keys())
    processed = {n: preprocess_attribution(attributions[n]) for n in names}

    sims = {m: np.zeros((len(names), len(names)), dtype=np.float32) for m in metrics}

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j < i:   # matrix is symmetric, copy later
                continue
            Ai = processed[ni]
            Aj = processed[nj]
            for m in metrics:
                s = _METRIC_FUNS[m](Ai, Aj)
                sims[m][i, j] = sims[m][j, i] = s
    return sims

def average_similarity_matrices(list_of_mats: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """
    Average a list of per‑image similarity dictionaries.

    Returns
    -------
    avg : dict  {metric_name: averaged matrix}
    """
    if not list_of_mats:
        raise ValueError("Need at least one similarity matrix")
    metrics = list(list_of_mats[0].keys())
    avg = {m: np.mean([d[m] for d in list_of_mats], axis=0) for m in metrics}
    return avg

def compute_snr(real_atts: dict[str, torch.Tensor],
                baseline_atts: dict[str, torch.Tensor],
                metric: str = "pearson") -> dict[str, float]:
    """
    For each method, compare its attribution map on the
    real model to its map from a shuffled‑label baseline.

    SNR_m = similarity(real_m, real_m) / similarity(real_m, shuffled_m)

    The numerator is always 1 (self‑similarity), so effectively
    SNR_m = 1 / sim(real, shuffled).  Higher ⇒ better.

    Returns
    -------
    snr : dict {method_name: float}
    """
    if metric not in _METRIC_FUNS:
        raise ValueError(f"metric must be one of {list(_METRIC_FUNS)}")

    sim_fun = _METRIC_FUNS[metric]
    snr = {}
    for name in real_atts:
        A = preprocess_attribution(real_atts[name])
        B = preprocess_attribution(baseline_atts[name])
        noise_sim = sim_fun(A, B)
        snr[name] = float(1.0 / (noise_sim + 1e-8))
    return snr

def _flatten_probs(att, mask=None, eps=1e-12):
    """Return p (1-D np.array) with positive entries summing to 1."""
    if torch.is_tensor(att):
        a = att.detach().cpu().numpy()
    else:
        a = np.asarray(att)
    if mask is not None:
        if torch.is_tensor(mask):
            m = mask.detach().cpu().numpy()
        else:
            m = np.asarray(mask)
        a = a * m
    p = np.abs(a).ravel()
    p = p / (p.sum() + eps)
    return p

def struct_entropy(att, mask=None) -> float:
    p = _flatten_probs(att, mask)
    H = entropy(p)                             # natural log base e
    return 1.0 - H / np.log(p.size) # 0 (uniform) … 1 (single pixel)

def struct_gini(att, mask=None) -> float:
    p = _flatten_probs(att, mask)
    sorted_p = np.sort(p)
    n = p.size
    cum = np.cumsum(sorted_p)
    gini = 1.0 - 2.0 * np.sum(cum) / (n - 1)
    return gini # 0 = uniform, 1 = spike

def struct_topk(att, mask=None, k_ratio=0.01) -> float:
    p = _flatten_probs(att, mask)
    k = max(1, int(k_ratio * p.size))
    thresh = np.partition(p, -k)[-k]
    mass = p[p >= thresh].sum()
    return mass # ∈ (0,1], larger = denser

def struct_kurtosis(att, mask=None, normalise=True, C=10.0) -> float:
    p = _flatten_probs(att, mask)
    k = kurtosis(p, fisher=False) # Pearson form (3 = Gaussian)
    if not normalise:
        return float(k)
    # Rescale via k / (k + C) so that k→∞ → 1  and k=0 → 0
    return float(k / (k + C))

def area_fraction_for_mass(att, mask=None, mass_thr=0.9) -> float:
    """
    Return minimal pixel fraction needed to accumulate `mass_thr`
    (e.g. 0.9 for 90 %) of |attribution| mass.
    """
    p = _flatten_probs(att, mask)
    idx = np.argsort(p)[::-1]
    cum = np.cumsum(p[idx])
    n_needed = np.searchsorted(cum, mass_thr) + 1
    return n_needed / p.size

def combined_struct_score(att,
                          mask=None,
                          weights=None,
                          k_ratio=0.01,
                          mass_thr=0.9) -> float:
    """
    Combine four monotone-increasing metrics into one score in [0,1].

    Steps:
      • Compute:
          E  = entropy-struct      (0-1)
          G  = gini-struct         (0-1)
          T  = top-k mass ratio    (0-1)
          K  = rescaled kurtosis   (0-1)
          A  = 1 − area_fraction_for_mass     (0-1, larger = better)
      • Optionally give each a weight; default = equal.
      • Return weighted average.
    """
    E = struct_entropy(att, mask)
    G = struct_gini(att, mask)
    T = struct_topk(att, mask, k_ratio=k_ratio)
    K = struct_kurtosis(att, mask)
    A = 1.0 - area_fraction_for_mass(att, mask, mass_thr=mass_thr)

    metrics = np.array([E, G, T, K, A], dtype=np.float32)
    if weights is None:
        weights = np.ones_like(metrics)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.sum()

    return float((metrics * weights).sum())

def compute_structuredness_for_dicts(real_atts: dict[str, torch.Tensor],
                                     baseline_atts: dict[str, torch.Tensor],
                                     mask: Optional[torch.Tensor] = None,
                                     k_ratio: float = 0.01,
                                     mass_thr: float = 0.90
                                    ) -> dict[str, dict[str, float]]:
    """
    Parameters
    ----------
    real_atts      : {method_name : (1,3,H,W) tensor}
    baseline_atts  : same keys, shuffled-model attributions
    mask           : optional (1,1,H,W) tensor selecting organoid pixels
    k_ratio        : for Top-k mass metric (default top 1 %)
    mass_thr       : area-for-mass threshold (default 90 %)

    Returns
    -------
    result : dict
        {
          method :
            {
              'entropy'          : float,
              'entropy_base'     : float,
              'gini'             : float,
              'gini_base'        : float,
              'topk'             : float,
              'topk_base'        : float,
              'kurtosis'         : float,
              'kurtosis_base'    : float,
              'area_inv'         : float,
              'area_inv_base'    : float,
              'combined'         : float,
              'combined_base'    : float
            },
          …
        }
    """
    out: dict[str, dict[str, float]] = {}
    for name in real_atts.keys():
        A_real = real_atts[name]
        A_base = baseline_atts[name]

        if mask is not None:
            m = mask
        else:
            m = None

        e  = struct_entropy(A_real, m)
        eb = struct_entropy(A_base, m)

        g  = struct_gini(A_real, m)
        gb = struct_gini(A_base, m)

        t  = struct_topk(A_real, m, k_ratio=k_ratio)
        tb = struct_topk(A_base, m, k_ratio=k_ratio)

        k  = struct_kurtosis(A_real, m)
        kb = struct_kurtosis(A_base, m)

        a  = 1.0 - area_fraction_for_mass(A_real, m, mass_thr=mass_thr)
        ab = 1.0 - area_fraction_for_mass(A_base, m, mass_thr=mass_thr)

        c  = combined_struct_score(A_real, m, k_ratio=k_ratio, mass_thr=mass_thr)
        cb = combined_struct_score(A_base, m, k_ratio=k_ratio, mass_thr=mass_thr)

        out[name] = dict(
            entropy=e, entropy_base=eb,
            gini=g, gini_base=gb,
            topk=t, topk_base=tb,
            kurtosis=k, kurtosis_base=kb,
            area_inv=a, area_inv_base=ab,
            combined=c, combined_base=cb
        )

    return out
