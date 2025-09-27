from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import os
import glob
import h5py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import slic
from scipy.ndimage import zoom, center_of_mass

from .figure_data_utils import Readouts
from .figure_data_generation import get_classification_f1_data

from typing import Dict, Tuple, Optional

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

def iter_h5_samples(
    h5_path: str,
    loops: list[int] | None = None,
    models: list[str] | None = None,
    methods: list[str] | None = None,
    load_trained: bool = True,
    load_baseline: bool = True,
):
    """
    Yield (well, loop_int, sample_dict) where sample_dict mirrors `save_saliency_h5`,
    but only loads requested subsets to speed things up.

    Parameters
    ----------
    h5_path : str
        Path to a single {experiment}_{well}_{readout}.h5 file.
    loops : list[int] | None
        If given, only these loop IDs are read; else read all.
    models : list[str] | None
        If given, only these model groups are read.
    methods : list[str] | None
        If given, only these attribution function groups are read.
    load_trained / load_baseline : bool
        Toggle reading the corresponding datasets.
    """
    loops_set = None if loops is None else {int(x) for x in loops}

    with h5py.File(h5_path, "r") as h5f:
        for well in h5f.keys():
            grp_well = h5f[well]

            # Pick loop keys (fast, no scanning beyond the needed ones)
            loop_keys = list(grp_well.keys())
            if loops_set is not None:
                loop_keys = [lk for lk in loop_keys if extract_loop_int(lk) in loops_set]
            loop_keys.sort(key=extract_loop_int)

            for loop_key in loop_keys:
                grp_loop = grp_well[loop_key]
                sample = {}

                # Always read image + mask
                sample["image"] = grp_loop["input_image"][...]
                sample["mask"]  = grp_loop["mask"][...]

                # Restrict model groups
                model_keys = [k for k in grp_loop.keys() if k not in ("input_image", "mask")]
                if models is not None:
                    allowed_models = set(models)
                    model_keys = [m for m in model_keys if m in allowed_models]

                for model in model_keys:
                    grp_model = grp_loop[model]
                    sample[model] = {}

                    # Restrict attribution methods
                    fn_keys = list(grp_model.keys())
                    if methods is not None:
                        allowed_methods = set(methods)
                        fn_keys = [fn for fn in fn_keys if fn in allowed_methods]

                    for fn in fn_keys:
                        grp_fn = grp_model[fn]
                        entry = {}
                        if load_trained and "trained" in grp_fn:
                            entry["trained"] = grp_fn["trained"][...]
                        if load_baseline and "baseline" in grp_fn:
                            entry["baseline"] = grp_fn["baseline"][...]
                        if entry:
                            sample[model][fn] = entry

                yield well, extract_loop_int(loop_key), sample


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




def _worker_process_file(h5_path: str,
                         loops: list[int],
                         n_segments: int,
                         compactness: float,
                         topk_pct: float,
                         series_peak_loop: int | None):
    """Runs all metrics for one H5 file. Returns plain Python data (lists/dfs)."""
    # Containers for this file
    method_pairs_rows = []
    cross_model_rows = []
    region_votes_frames = []
    dice_to_peak_rows = []
    ent_drift_frames = []

    base = os.path.basename(h5_path)
    try:
        # adapt to your file naming
        file_stem = base.split(".h5")[0]
        parts = file_stem.split("_")
        if len(parts) == 4:
            exp, well_from_name, r1, r2 = parts
            readout_in_path = f"{r1}_{r2}"
        elif len(parts) == 3:
            exp, well_from_name, readout_in_path = parts
        else:
            exp, well_from_name, readout_in_path = "unknown", "unknown", file_stem
    except Exception:
        exp, well_from_name, readout_in_path = "unknown", "unknown", base

    with h5py.File(h5_path, "r", swmr=True) as h5f:
        for well_key in h5f.keys():
            grp_well = h5f[well_key]

            # choose loops
            loop_keys = sorted(grp_well.keys(), key=extract_loop_int)
            if loops is not None:
                want = set(int(x) for x in loops)
                loop_keys = [lk for lk in loop_keys if extract_loop_int(lk) in want]
            if not loop_keys:
                continue

            consensus_by_loop: dict[int, np.ndarray] = {}
            mask_for_series = None

            for loop_str in loop_keys:
                loop = extract_loop_int(loop_str)
                grp_loop = grp_well[loop_str]

                img = grp_loop["input_image"][...]
                img2d = np.mean(img[0], axis=0) if img.ndim == 4 else np.mean(img, axis=0)
                mask2d = grp_loop["mask"][...].squeeze().astype(np.float32)
                mask_for_series = mask2d

                # collect maps (trained only) and normalize in-mask
                maps_by_model = {}
                maps_per_method = {}
                for model in grp_loop.keys():
                    if model in ("input_image", "mask"):
                        continue
                    gmodel = grp_loop[model]
                    model_maps = {}
                    for fn in gmodel.keys():
                        atr = gmodel[fn]["trained"][...].squeeze().astype(np.float32)
                        atr = zscore_in_mask(atr, mask2d)
                        model_maps[fn] = atr
                        maps_per_method.setdefault(fn, []).append(atr)
                    maps_by_model[model] = model_maps

                methods = sorted(maps_per_method.keys())
                merged_per_method = {fn: np.mean(np.stack(lst, axis=0), axis=0)
                                     for fn, lst in maps_per_method.items()}

                # (1) method agreement (pairwise Dice over top-k)
                for i, a in enumerate(methods):
                    for j in range(i + 1, len(methods)):
                        b = methods[j]
                        scores = []
                        for k in (1, 5, 10):
                            Ma = topk_mask_from_abs(merged_per_method[a], k, mask2d)
                            Mb = topk_mask_from_abs(merged_per_method[b], k, mask2d)
                            scores.append(dice(Ma, Mb))
                        method_pairs_rows.append({
                            "experiment": exp, "well": well_key, "loop": loop,
                            "method_a": a, "method_b": b, "dice_avg": float(np.mean(scores))
                        })

                # (2) cross-model consistency (rank corr per method)
                rank_maps = {m: {fn: to_rank(amap, mask2d) for fn, amap in md.items()}
                             for m, md in maps_by_model.items()}
                model_list = sorted(rank_maps.keys())
                for fn in methods:
                    for i in range(len(model_list)):
                        for j in range(i + 1, len(model_list)):
                            m1, m2 = model_list[i], model_list[j]
                            if fn not in rank_maps[m1] or fn not in rank_maps[m2]:
                                continue
                            r1 = rank_maps[m1][fn]
                            r2 = rank_maps[m2][fn]
                            corr = kendall_tau_like(r1, r2)
                            cross_model_rows.append({
                                "experiment": exp, "well": well_key, "loop": loop,
                                "method": fn, "model_a": m1, "model_b": m2, "rank_corr": float(corr)
                            })

                # (3) region votes (SLIC once per loop)
                labels = slic_regions(img2d, (mask2d > 0).astype(np.uint8),
                                      n_segments=n_segments, compactness=compactness)
                votes_df = region_vote(merged_per_method, labels, top_quantile=0.9)
                if not votes_df.empty:
                    votes_df["experiment"] = exp
                    votes_df["well"] = well_key
                    votes_df["loop"] = loop
                    region_votes_frames.append(votes_df)

                # consensus for temporal metrics
                consensus = build_consensus(merged_per_method, mask2d, weights=None)
                consensus_by_loop[loop] = consensus

            # temporal metrics vs global peak
            if series_peak_loop is not None and consensus_by_loop:
                peak_map = consensus_by_loop.get(series_peak_loop)
                if peak_map is not None and mask_for_series is not None:
                    peak_mask = topk_mask_from_abs(peak_map, topk_pct, mask_for_series)
                    for t, cmap in sorted(consensus_by_loop.items()):
                        Mt = topk_mask_from_abs(cmap, topk_pct, mask_for_series)
                        dice_val = dice(Mt, peak_mask)
                        dice_to_peak_rows.append({
                            "experiment": exp, "well": well_key, "loop": t, "dice_to_peak": dice_val
                        })
                    ed = entropy_and_drift(consensus_by_loop, mask_for_series)
                    ed["experiment"] = exp
                    ed["well"] = well_key
                    ent_drift_frames.append(ed)

    return {
        "pairs": method_pairs_rows,
        "cross": cross_model_rows,
        "votes": (pd.concat(region_votes_frames, ignore_index=True)
                  if region_votes_frames else pd.DataFrame()),
        "d2p": dice_to_peak_rows,
        "ed": (pd.concat(ent_drift_frames, ignore_index=True)
               if ent_drift_frames else pd.DataFrame()),
    }


def run_saliency_analysis_parallel(saliency_input_dir: str,
                                   readout: Readouts,
                                   raw_data_dir: str,
                                   morphometrics_dir: str,
                                   hyperparameter_dir: str,
                                   rpe_classification_dir: str,
                                   lens_classification_dir: str,
                                   rpe_classes_classification_dir: str,
                                   lens_classes_classification_dir: str,
                                   figure_data_dir: str,
                                   evaluator_results_dir: str,
                                   loops: list[int] | None = None,
                                   n_segments: int = 100,
                                   compactness: float = 0.01,
                                   topk_pct: float = 5.0,
                                   max_workers: int = 6,
                                   **kwargs) -> None:
    """Parallel version: one process per H5 file."""

    # keep BLAS threads in check inside workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

    path_map = {
        "RPE_Final": rpe_classification_dir,
        "Lens_Final": lens_classification_dir,
        "RPE_classes": rpe_classes_classification_dir,
        "Lens_classes": lens_classes_classification_dir,
    }

    f1_scores = get_classification_f1_data(
        readout=readout,
        output_dir=figure_data_dir,
        proj="",
        hyperparameter_dir=hyperparameter_dir,
        classification_dir=path_map[readout],
        baseline_dir=None,
        morphometrics_dir=morphometrics_dir,
        raw_data_dir=raw_data_dir,
        evaluator_results_dir=evaluator_results_dir,
    )
    f1_scores: pd.DataFrame = (f1_scores[f1_scores["classifier"] == "Ensemble_val"]
                               .copy().reset_index(drop=True))

    # global peak loop from aggregated F1
    idxmax = f1_scores["F1"].idxmax()
    series_peak_loop = int(f1_scores.loc[idxmax, "loop"]) if pd.notna(idxmax) else None

    # files to process
    h5_files = sorted(glob.glob(os.path.join(saliency_input_dir, "*")))
    # keep only files for this readout
    h5_files = [p for p in h5_files if str(readout) in os.path.basename(p)]

    if loops is None:
        loops = list(np.arange(0, 145, 6))

    # parallel map over files
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                _worker_process_file,
                h5_path=p,
                loops=loops,
                n_segments=n_segments,
                compactness=compactness,
                topk_pct=topk_pct,
                series_peak_loop=series_peak_loop,
            )
            for p in h5_files
        ]
        for fut in as_completed(futs):
            results.append(fut.result())

    # merge outputs
    all_method_pairs = list(itertools.chain.from_iterable(r["pairs"] for r in results))
    all_cross_model = list(itertools.chain.from_iterable(r["cross"] for r in results))
    all_d2p = list(itertools.chain.from_iterable(r["d2p"] for r in results))

    votes_list = [r["votes"] for r in results if isinstance(r["votes"], pd.DataFrame) and not r["votes"].empty]
    ed_list = [r["ed"] for r in results if isinstance(r["ed"], pd.DataFrame) and not r["ed"].empty]
    votes_df = pd.concat(votes_list, ignore_index=True) if votes_list else pd.DataFrame()
    ed_df = pd.concat(ed_list, ignore_index=True) if ed_list else pd.DataFrame()

    # save
    os.makedirs(os.path.join(figure_data_dir, "metrics"), exist_ok=True)

    if all_method_pairs:
        pd.DataFrame(all_method_pairs).to_csv(
            os.path.join(figure_data_dir, "metrics", f"agreement_method_pairwise_{readout}.csv"),
            index=False,
        )
    if all_cross_model:
        pd.DataFrame(all_cross_model).to_csv(
            os.path.join(figure_data_dir, "metrics", f"cross_model_correlation_{readout}.csv"),
            index=False,
        )
    if not votes_df.empty:
        votes_df.to_csv(
            os.path.join(figure_data_dir, "metrics", f"region_votes_summary_{readout}.csv"),
            index=False,
        )
    if all_d2p:
        pd.DataFrame(all_d2p).to_csv(
            os.path.join(figure_data_dir, "metrics", f"dice_to_peak_timeseries_{readout}.csv"),
            index=False,
        )
    if not ed_df.empty:
        ed_df.to_csv(
            os.path.join(figure_data_dir, "metrics", f"entropy_drift_timeseries_{readout}.csv"),
            index=False,
        )

    return










