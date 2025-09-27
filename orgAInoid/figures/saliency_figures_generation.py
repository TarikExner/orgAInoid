import os
import glob
import h5py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from skimage.measure import regionprops
from scipy.ndimage import zoom, center_of_mass

from .figure_data_utils import Readouts

from typing import Dict, Tuple

METHOD_FAMILIES = {
    "grad_based": {"IG_NT", "SAL_NT", "DLS"},
    "cam_based": {"GC", "GGC"},
    "occ_based": {"OCC", "FAB", "KSH"},
}


TOPK_PCTS = [1, 5, 10] # %
MULTISCALES = [1, 0.5, 0.25]

def zscore_in_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score arr within mask (mask>0). Keeps sign. Returns same shape float32."""
    m = mask > 0
    vals = arr[m]
    mu = vals.mean() if vals.size else 0.0
    sd = vals.std() if vals.size else 1.0
    sd = sd if sd > 1e-8 else 1.0
    out = (arr - mu) / sd
    return out.astype(np.float32)


def downsample(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1:
        return img
    # zoom expects (H, W) for 2D
    return zoom(img, (scale, scale), order=1)


def topk_mask_from_abs(arr: np.ndarray, pct: float, mask: np.ndarray) -> np.ndarray:
    """Binary mask of top pct% absolute |arr| inside mask."""
    m = mask > 0
    vals = np.abs(arr[m])
    if vals.size == 0:
        return np.zeros_like(arr, dtype=bool)
    k = max(1, int(np.ceil(vals.size * (pct / 100.0))))
    thresh = np.partition(vals, -k)[-k]
    out = (np.abs(arr) >= thresh) & m
    return out


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    size = a.sum() + b.sum()
    return 2 * inter / size if size > 0 else 0.0


def kendall_tau_like(rank_a: np.ndarray, rank_b: np.ndarray) -> float:
    """Fast Spearman rank corr as a proxy (since exact Kendall tau is heavy for full images)."""
    # Spearman rho
    a = rank_a.ravel().astype(np.float64)
    b = rank_b.ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom > 0 else 0.0


def to_rank(abs_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return dense ranks (1 = strongest) over |map| within mask; elsewhere 0."""
    m = mask > 0
    out = np.zeros_like(abs_map, dtype=np.int32)
    vals = np.abs(abs_map[m]).astype(np.float64)
    if vals.size == 0:
        return out
    # ranks: highest gets rank 1
    order = np.argsort(-vals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, vals.size + 1)
    out[m] = ranks.astype(np.int32)
    return out


def rank_consensus(maps: Dict[str, np.ndarray], mask: np.ndarray, weights: Dict[str, float] | None = None) -> np.ndarray:
    """Rank aggregation over |z|; lower summed rank = stronger. Returns float32 score (normalized 0..1 in mask)."""
    m = mask > 0
    H, W = mask.shape
    agg = np.zeros((H, W), dtype=np.float64)
    total_w = 0.0
    for name, amap in maps.items():
        r = to_rank(amap, mask).astype(np.float64)
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        # Zero ranks outside mask; inside mask, higher weight reduces summed rank
        agg += w * r
        total_w += w
    # Normalize: invert ranks so higher = stronger, map to 0..1 inside mask
    # Inside mask, agg ranges from ~total_w to ~total_w * Npix; invert
    out = np.zeros((H, W), dtype=np.float32)
    if m.sum() > 0 and total_w > 0:
        vals = agg[m]
        inv = (vals.max() - vals)  # higher is better
        inv -= inv.min()
        rng = inv.max() - inv.min()
        inv = inv / rng if rng > 1e-12 else np.zeros_like(inv)
        out[m] = inv.astype(np.float32)
    return out


def slic_regions(image_like: np.ndarray, mask: np.ndarray, n_segments: int = 500, compactness: float = 0.1) -> np.ndarray:
    """Create SLIC labels restricted to mask. Works with 2D image-like array."""
    img = image_like
    if img.ndim == 2:
        img3 = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 3:
        img3 = img
    else:
        # fallback to uniform texture
        img3 = np.repeat(mask[..., None], 3, axis=-1).astype(np.float32)
    labels = slic(img3, n_segments=n_segments, compactness=compactness, start_label=1, mask=mask.astype(bool))
    return labels.astype(np.int32)


def region_vote(abs_maps: Dict[str, np.ndarray], labels: np.ndarray, top_quantile: float = 0.9) -> pd.DataFrame:
    """Per region vote: a method votes if region mean(|map|) is in top_quantile for that image.
    Returns a tidy DataFrame with columns: region, method, voted (0/1), score.
    """
    rows = []
    for mname, amap in abs_maps.items():
        props = regionprops(labels, intensity_image=np.abs(amap))
        scores = np.array([p.mean_intensity for p in props], dtype=np.float32)
        if scores.size == 0:
            continue
        thr = np.quantile(scores, top_quantile)
        for ridx, sc in enumerate(scores, start=1):
            rows.append({"region": ridx, "method": mname, "voted": int(sc >= thr), "score": float(sc)})
    return pd.DataFrame(rows)

@dataclass
class samplekey:
    experiment: str
    well: str
    loop: int

def extract_loop_int(loop: str):
    return int(loop.split("LO")[1])

def iter_h5_samples(h5_path: str):
    """Yield (well, loop, sample_dict) where sample_dict mirrors `save_saliency_h5` content."""
    with h5py.File(h5_path, "r") as h5f:
        for well in h5f.keys():
            grp_well = h5f[well]
            for loop in grp_well.keys():
                grp_loop = grp_well[loop]
                sample = {}
                sample["image"] = grp_loop["input_image"][...]
                sample["mask"] = grp_loop["mask"][...]
                for model in grp_loop.keys():
                    if model in ("input_image", "mask"):
                        continue
                    grp_model = grp_loop[model]
                    sample[model] = {}
                    for fn in grp_model.keys():
                        grp_fn = grp_model[fn]
                        sample[model][fn] = {
                            "trained": grp_fn["trained"][...],
                            "baseline": grp_fn["baseline"][...],
                        }
                yield well, extract_loop_int(loop), sample


def collect_maps(sample: dict, use_trained: bool = True) -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray, np.ndarray]:
    """Return (maps_by_model, image2d, mask2d). Each map is 2D (H, W) float32.
    Your stored maps are already combined across channels.
    """
    img = sample["image"]
    # "image" is shape (1,3,H,W) from your saver; make a 2D magnitude to drive SLIC if needed
    if img.ndim == 4:
        img2d = np.mean(img[0], axis=0)
    elif img.ndim == 3:
        img2d = np.mean(img, axis=0)
    else:
        img2d = img.squeeze()
    msk = sample["mask"]
    mask2d = msk.squeeze().astype(np.float32)

    key = "trained" if use_trained else "baseline"
    maps_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    for model, algs in sample.items():
        if model in ("image", "mask"):
            continue
        maps_by_model[model] = {}
        for fn, dct in algs.items():
            arr = dct[key]
            # ensure 2D
            arr2 = arr.squeeze().astype(np.float32)
            maps_by_model[model][fn] = arr2
    return maps_by_model, img2d, mask2d


def normalize_maps(maps_by_model: Dict[str, Dict[str, np.ndarray]], mask2d: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for model, md in maps_by_model.items():
        out[model] = {fn: zscore_in_mask(arr, mask2d) for fn, arr in md.items()}
    return out


def method_agreement_for_sample(norm_maps_by_model: Dict[str, Dict[str, np.ndarray]], mask2d: np.ndarray) -> pd.DataFrame:
    """Method×method Dice across all models, averaged over models and k in TOPK_PCTS."""
    # stack across models by averaging maps per method (keeps method signal while reducing model noise)
    methods = sorted({fn for md in norm_maps_by_model.values() for fn in md.keys()})
    merged: Dict[str, np.ndarray] = {}
    for fn in methods:
        stack = []
        for model, md in norm_maps_by_model.items():
            if fn in md:
                stack.append(md[fn])
        if stack:
            merged[fn] = np.mean(np.stack(stack, axis=0), axis=0)
    # compute pairwise Dice on top‑k
    rows = []
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if j <= i:
                continue
            score_list = []
            for k in TOPK_PCTS:
                Ma = topk_mask_from_abs(merged[a], k, mask2d)
                Mb = topk_mask_from_abs(merged[b], k, mask2d)
                score_list.append(dice(Ma, Mb))
            rows.append({"method_a": a, "method_b": b, "dice_avg": float(np.mean(score_list))})
    return pd.DataFrame(rows)


def cross_model_consistency(norm_maps_by_model: Dict[str, Dict[str, np.ndarray]], mask2d: np.ndarray) -> pd.DataFrame:
    """For each method, correlation between models (Spearman-like on ranks)."""
    models = sorted(norm_maps_by_model.keys())
    methods = sorted({fn for md in norm_maps_by_model.values() for fn in md.keys()})
    rows = []
    # precompute ranks
    ranks = {model: {fn: to_rank(md.get(fn, np.zeros_like(mask2d)), mask2d) for fn, _ in norm_maps_by_model[model].items()} for model, md in norm_maps_by_model.items()}
    for fn in methods:
        # compare across all model pairs
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                if fn not in norm_maps_by_model[m1] or fn not in norm_maps_by_model[m2]:
                    continue
                r1 = ranks[m1][fn]
                r2 = ranks[m2][fn]
                corr = kendall_tau_like(r1, r2)
                rows.append({"method": fn, "model_a": m1, "model_b": m2, "rank_corr": float(corr)})
    return pd.DataFrame(rows)

def dice_to_peak_timeseries(consensus_by_time: Dict[int, np.ndarray], mask2d: np.ndarray, top_pct: float = 5.0) -> pd.DataFrame:
    """Return DataFrame with columns: time, dice_to_peak."""
    if not consensus_by_time:
        return pd.DataFrame(columns=["time", "dice_to_peak"])
    # pick peak map as pseudo-ROI: the time key with max time or provided upstream
    # Upstream we will pass the loop of max F1.
    items = list(consensus_by_time.items())
    times = [t for t, _ in items]
    # find peak is handled by caller
    peak_time = max(times)
    peak_map = consensus_by_time[peak_time]
    peak_mask = topk_mask_from_abs(peak_map, top_pct, mask2d)
    rows = []
    for t, cmap in sorted(consensus_by_time.items()):
        Mt = topk_mask_from_abs(cmap, top_pct, mask2d)
        rows.append({"time": t, "dice_to_peak": dice(Mt, peak_mask)})
    return pd.DataFrame(rows)


def entropy_and_drift(consensus_by_time: Dict[int, np.ndarray], mask2d: np.ndarray) -> pd.DataFrame:
    rows = []
    com_prev = None
    for t in sorted(consensus_by_time.keys()):
        cmap = np.abs(consensus_by_time[t])
        m = mask2d > 0
        vals = cmap[m].astype(np.float64)
        if vals.size == 0:
            ent = 0.0
            com = (np.nan, np.nan)
        else:
            p = vals / (vals.sum() + 1e-12)
            ent = float(-(p * (np.log(p + 1e-12))).sum())
            # center of mass in (row, col)
            grid = np.zeros_like(cmap, dtype=np.float64)
            grid[m] = vals
            com = center_of_mass(grid)
        drift = float(np.linalg.norm(np.array(com) - np.array(com_prev))) if (com_prev is not None and not np.any(np.isnan(com))) else 0.0
        rows.append({"time": t, "entropy": ent, "drift": drift})
        com_prev = com
    return pd.DataFrame(rows)

def build_consensus(norm_maps: Dict[str, np.ndarray], mask2d: np.ndarray, weights: Dict[str, float] | None = None) -> np.ndarray:
    return rank_consensus(norm_maps, mask2d, weights)


def dice_to_peak_timeseries(consensus_by_time: Dict[int, np.ndarray], mask2d: np.ndarray, top_pct: float = 5.0) -> pd.DataFrame:
    """Return DataFrame with columns: time, dice_to_peak."""
    if not consensus_by_time:
        return pd.DataFrame(columns=["time", "dice_to_peak"])
    # pick peak map as pseudo-ROI: the time key with max time or provided upstream
    # Upstream we will pass the loop of max F1.
    items = list(consensus_by_time.items())
    times = [t for t, _ in items]
    # find peak is handled by caller
    peak_time = max(times)
    peak_map = consensus_by_time[peak_time]
    peak_mask = topk_mask_from_abs(peak_map, top_pct, mask2d)
    rows = []
    for t, cmap in sorted(consensus_by_time.items()):
        Mt = topk_mask_from_abs(cmap, top_pct, mask2d)
        rows.append({"time": t, "dice_to_peak": dice(Mt, peak_mask)})
    return pd.DataFrame(rows)


def run_saliency_analysis(h5_glob: str,
                          readout: Readouts,
                          f1_csv: pd.DataFrame,
                          output_dir: str,
                          timepoints: list[int] = [0, 12, 24, 36, 48, 60, 72],
                          n_segments: int = 500,
                          compactness: float = 0.1,
                          topk_pct: float = 5.0):

    h5_files = sorted(glob.glob(h5_glob))
    h5_files = [file for file in h5_files if readout in file]

    h5_files = ["../classification/saliencies/results/E001_A001_RPE_Final.h5"]

    # Collect per-sample outputs
    all_method_pairs = []
    all_cross_model = []
    all_region_votes = []
    dice_to_peak_rows = []
    ent_drift_rows = []

    f1 = f1_csv

    # Get global peak loop across all wells (since F1 is aggregated)
    idxmax = f1["f1"].idxmax()
    series_peak_loop = int(f1.loc[idxmax, "loop"]) if pd.notna(idxmax) else None
    for h5_path in h5_files:
        # parse experiment, well from filename for joins
        base = os.path.basename(h5_path)
        file_name = base.split(".h5")[0]
        exp, well, readout1, readout2 = file_name.split("_")
        _readout = "_".join([readout1, readout2])

        # Per-loop aggregation for this file
        for well_key, loop, sample in iter_h5_samples(h5_path):
            maps_by_model, img2d, mask2d = collect_maps(sample, use_trained=True)
            norm_by_model = normalize_maps(maps_by_model, mask2d)

            # (1) Method agreement
            df_pairs = method_agreement_for_sample(norm_by_model, mask2d)
            df_pairs["experiment"] = exp
            df_pairs["well"] = well_key
            df_pairs["loop"] = loop
            all_method_pairs.append(df_pairs)

            # (2) Cross-model consistency
            df_cm = cross_model_consistency(norm_by_model, mask2d)
            df_cm["experiment"] = exp
            df_cm["well"] = well_key
            df_cm["loop"] = loop
            all_cross_model.append(df_cm)

            # (3) Region votes (one set of labels per loop)
            labels = slic_regions(img2d, (mask2d > 0).astype(np.uint8), n_segments=args.n_segments, compactness=args.compactness)
            # use absolute z-maps averaged per method across models for stability
            merged = {fn: np.mean(np.stack([md[fn] for md in norm_by_model.values() if fn in md], axis=0), axis=0) for fn in sorted({fn for md in norm_by_model.values() for fn in md.keys()})}
            votes_df = region_vote(merged, labels, top_quantile=0.9)
            votes_df["experiment"] = exp
            votes_df["well"] = well_key
            votes_df["loop"] = loop
            all_region_votes.append(votes_df)

        # After iterating loops of this file: temporal metrics need consensus per loop
        # Re-open to build consensus maps per loop (across methods, averaged across models)
        consensus_by_loop: Dict[int, np.ndarray] = {}
        with h5py.File(h5_path, "r") as h5f:
            for well_key in h5f.keys():
                grp_well = h5f[well_key]
                for loop_str in grp_well.keys():
                    loop = extract_loop_int(loop_str)
                    grp_loop = grp_well[loop_str]
                    img = grp_loop["input_image"][...]
                    img2d = np.mean(img[0], axis=0) if img.ndim == 4 else np.mean(img, axis=0)
                    mask2d = grp_loop["mask"][...].squeeze().astype(np.float32)

                    # gather maps per method across models
                    maps_per_method: Dict[str, List[np.ndarray]] = {}
                    for model in grp_loop.keys():
                        if model in ("input_image", "mask"):
                            continue
                        for fn in grp_loop[model].keys():
                            arr = grp_loop[model][fn]["trained"][...].squeeze().astype(np.float32)
                            arr = zscore_in_mask(arr, mask2d)
                            maps_per_method.setdefault(fn, []).append(arr)
                    # average per method across models
                    method_maps = {fn: np.mean(np.stack(lst, axis=0), axis=0) for fn, lst in maps_per_method.items()}
                    # consensus across methods (equal weights or supply custom weights here)
                    consensus = build_consensus(method_maps, mask2d, weights=None)
                    consensus_by_loop[loop] = consensus

        if series_peak_loop is not None and len(consensus_by_loop) > 0:
            # Align to desired timepoints if given
            # For plotting, we will just use available loops present in H5; user can filter upstream.
            # Compute Dice vs peak using peak loop as pseudo-ROI
            peak_map = consensus_by_loop.get(series_peak_loop, None)
            if peak_map is not None:
                peak_mask = topk_mask_from_abs(peak_map, topk_pct, mask2d)
                for t, cmap in sorted(consensus_by_loop.items()):
                    Mt = topk_mask_from_abs(cmap, topk_pct, mask2d)
                    dice_val = dice(Mt, peak_mask)
                    dice_to_peak_rows.append({"experiment": exp, "well": well, "time": t, "dice_to_peak": dice_val})
                # entropy and drift
                ed = entropy_and_drift(consensus_by_loop, mask2d)
                ed["experiment"] = exp
                ed["well"] = well
                ent_drift_rows.append(ed)

    # Concatenate and save metrics
    if all_method_pairs:
        df_pairs_all = pd.concat(all_method_pairs, ignore_index=True)
        df_pairs_all.to_csv(os.path.join(output_dir, "metrics", "agreement_method_pairwise.csv"), index=False)

    if all_cross_model:
        df_cm_all = pd.concat(all_cross_model, ignore_index=True)
        df_cm_all.to_csv(os.path.join(output_dir, "metrics", "cross_model_correlation.csv"), index=False)

    if all_region_votes:
        votes_all = pd.concat(all_region_votes, ignore_index=True)
        votes_all.to_csv(os.path.join(output_dir, "metrics", "region_votes_summary.csv"), index=False)

    if dice_to_peak_rows:
        d2p = pd.DataFrame(dice_to_peak_rows)
        d2p.to_csv(os.path.join(output_dir, "metrics", "dice_to_peak_timeseries.csv"), index=False)

    if ent_drift_rows:
        ed_all = pd.concat(ent_drift_rows, ignore_index=True)
        ed_all.to_csv(os.path.join(output_dir, "metrics", "entropy_drift_timeseries.csv"), index=False)













