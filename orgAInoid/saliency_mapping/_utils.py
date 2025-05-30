import numpy as np
import pandas as pd

from skimage.segmentation import slic
from skimage.measure import regionprops_table

import xarray as xr
import shutil

from pathlib import Path

import networkx as nx

from tqdm import tqdm

from collections import defaultdict

from typing import Optional, Union, Sequence

from ..classification._dataset import OrganoidDataset
from ..image_handling import OrganoidImage
from ..image_handling._image_handler import ImageHandler

Node = tuple[int, int]

def get_images_and_masks(dataset: OrganoidDataset,
                         well: str,
                         img_handler: ImageHandler) -> tuple[np.ndarray, ...]:
    metadata = dataset.metadata[
        (dataset.metadata["well"] == well) &
        (dataset.metadata["slice"].isin(dataset.dataset_metadata.slices))
    ].copy()
    metadata = metadata.sort_values("loop", ascending = True)
    file_paths = metadata["image_path"].tolist()
    imgs = []
    masks = []

    for path in tqdm(file_paths, desc = f"Image generation for well {well}"):
        array_index = metadata.loc[
            metadata["image_path"] == path,
            "IMAGE_ARRAY_INDEX"
        ].iloc[0]
        img = dataset.X[array_index,:,:,:]

        img_to_mask = OrganoidImage(path)
        _, mask = img_handler.get_mask_and_image(
            img = img_to_mask,
            image_target_dimension = dataset.image_metadata.dimension,
            mask_threshold = dataset.image_metadata.mask_threshold,
            clean_mask = dataset.image_metadata.cleaned_mask,
            crop_bounding_box = dataset.image_metadata.crop_bounding_box,
            rescale_cropped_image = dataset.image_metadata.rescale_cropped_image,
            crop_bounding_box_dimension = dataset.image_metadata.crop_bounding_box_dimension
        )
        if mask is None:
            print("ERROR! MASK CREATION FAILED! SKIPPING LOOP!")
            continue
        mask = mask[np.newaxis, ...]
        imgs.append(img)
        masks.append(mask)
    
    imgs = np.array(imgs)
    masks = np.array(masks)

    # imgs.shape = [n_images, 1, 224, 224]
    # masks.shape = [n_images, 1, 224, 224]
    return imgs, masks

def slic_segment_stack(imgs: np.ndarray,
                       masks: np.ndarray,
                       pixels_per_superpixel: int = 1000,
                       compactness: float = 0.06) -> np.ndarray:
    if imgs.ndim == 4 and imgs.shape[1] == 1:
        imgs_np = imgs[:, 0]
    else:
        imgs_np = imgs
    masks = masks.astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks_np = masks[:, 0]
    else:
        masks_np = masks

    T, H, W = imgs_np.shape
    labels_stack = np.empty((T, H, W), dtype=np.int32)

    for t in tqdm(range(T), desc="SLIC"):
        mask = masks_np[t]
        masked_px = int(mask.sum())
        n_segments = max(1, round(masked_px / pixels_per_superpixel))

        lab = slic(imgs_np[t], mask=mask,
                   n_segments=n_segments,
                   compactness=compactness,
                   start_label=1,
                   channel_axis=None)
        lab[~mask] = -1
        labels_stack[t] = lab.astype(np.int32)

    return labels_stack

def regionprops_stack(imgs: np.ndarray,
                      labels_stack: np.ndarray,
                      props: Optional[list[str]] = None,
                      hist_bins: int = 8) -> pd.DataFrame:
    """
    Compute regionprops for a labelled stack.

    * Uses intensity-weighted centroids.
    * Reads `intensity_mean` / `intensity_std` (or their legacy names).
    * Adds an `hist_bins`-bin normalised histogram per region.
    * Ignores background label -1.
    """
    if props is None:
        props = [
            "label", "area", "perimeter",
            "weighted_centroid",
            "weighted_moments_central",
            "weighted_moments_hu",
            "intensity_mean",
            "intensity_std"
        ]

    imgs_np = imgs[:, 0] if imgs.ndim == 4 and imgs.shape[1] == 1 else imgs

    rows = []
    for t in tqdm(range(labels_stack.shape[0]), desc="regionprops"):
        lab = labels_stack[t].copy()
        assert 0 not in lab
        lab[lab == -1] = 0  # background to 0 for sklearn ignorings

        tbl = regionprops_table(lab,
                                intensity_image=imgs_np[t],
                                properties=props)

        df = pd.DataFrame(tbl)

        if "weighted_centroid-0" in df.columns:
            df.rename(columns={"weighted_centroid-0": "wcentroid_row",
                               "weighted_centroid-1": "wcentroid_col"},
                      inplace=True)

        hist_cols = [f"hist_{i}" for i in range(hist_bins)]
        hist_dict = {c: [] for c in hist_cols}

        for lbl in df["label"]:
            pix = imgs_np[t][lab == lbl]
            h, _ = np.histogram(pix,
                                bins=hist_bins,
                                range=(0, 1),
                                density=True)
            for i, v in enumerate(h):
                hist_dict[f"hist_{i}"].append(float(v))

        for c in hist_cols:
            df[c] = hist_dict[c]

        df["image_idx"] = t
        rows.append(df)

    region_df = pd.concat(rows, ignore_index=True)

    # drop background (label 0 after shift)
    region_df = region_df[region_df["label"] > 0].reset_index(drop=True)
    return region_df

def _add_cost_attr(G: nx.DiGraph, eps: float = 1e-6) -> None:
    """Adds a cost attribute = 1/(weight+eps) for Dijkstra."""
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0.0)
        d["cost"] = 1.0/(w+eps) if w>0 else 1.0/eps


def build_backwards_graph(labels_stack: np.ndarray,
                          region_df: pd.DataFrame,
                          n_predecessors: int = 1,
                          m_candidates: int = 10,
                          alpha: float = 1.0,
                          beta: float = 0.5) -> nx.DiGraph:
    """
    Build a backward graph with *normalized* edge weights:

      weight = overlap_ratio / (1 + α·d_c_norm + β·d_p_norm)

    • overlap_ratio = IoU = overlap_px / (area_cur + area_prv - overlap_px)
    • d_c_norm      = centroid_dist / image_diagonal
    • d_p_norm      = prop_dist / n_props
    """
    # image diag for normalizing centroid distances
    H, W = labels_stack.shape[1], labels_stack.shape[2]
    max_dist = np.hypot(H, W).astype(np.float32)

    # which columns go into the prop vector?
    prop_cols = [
        c for c in region_df.columns
        if c.startswith(
            ("intensity_", "weighted_moments_hu-", "hist_")
        )
    ]
    n_props = float(len(prop_cols))

    # build standardized feature dict and weighted centroid dict
    prop_mat  = region_df[prop_cols].to_numpy(np.float32, copy=False)
    sigma_vec = prop_mat.std(axis=0, ddof=0)
    sigma_vec[sigma_vec == 0] = 1.0

    feat_dict: dict[tuple[int,int], np.ndarray] = {}
    wcent_dict: dict[tuple[int,int], tuple[float,float]] = {}
    area_dict: dict[dict[int,int], float] = {}

    keep_cols = [c for c in region_df.columns
                 if c not in {"image_idx", "label"}]

    G = nx.DiGraph()
    for _, r in region_df.iterrows():
        key = (int(r.image_idx), int(r.label))
        vec = r[prop_cols].to_numpy(np.float32, copy=False) / sigma_vec
        feat_dict[key] = vec
        wcent_dict[key] = (float(r.wcentroid_row), float(r.wcentroid_col))
        area_dict[key] = float(r.area)
        G.add_node(key, **r[keep_cols].to_dict())

    T = labels_stack.shape[0]

    def _cdist(u, v) -> float:
        ra, ca = wcent_dict.get(u, (np.nan, np.nan))
        rb, cb = wcent_dict.get(v, (np.nan, np.nan))
        return float(np.hypot(ra - rb, ca - cb))

    # build edges
    for t in tqdm(range(T - 1, 0, -1), desc="edges (backward)"):
        cur_lab = labels_stack[t]
        prv_lab = labels_stack[t - 1]

        la = cur_lab.astype(np.int32) + 1
        lb = prv_lab.astype(np.int32) + 1
        mask = (la > 0) | (lb > 0)
        la, lb = la[mask], lb[mask]
        if la.size == 0:
            continue

        max_prev = lb.max()
        key = la.astype(np.int64) * (max_prev + 1) + lb
        cnt = np.bincount(key)

        overlap_map: dict[int, dict[int,int]] = {}
        for pid, px in enumerate(cnt):
            if px == 0:
                continue
            cur = pid // (max_prev + 1) - 1
            prv = pid %  (max_prev + 1) - 1
            if cur >= 0 and prv >= 0:
                overlap_map.setdefault(cur, {})[prv] = int(px)

        cur_labels = np.unique(cur_lab)
        prv_labels = np.unique(prv_lab)

        for cur in cur_labels:
            if cur == -1:
                continue
            u = (t, cur)
            v_cur = feat_dict[u]
            area_u = area_dict[u]

            # gather candidate predecessors
            cand: list[tuple[float,int,int,float,float]] = []  # (-score, prv, px, d_c, d_p)

            # (a) real overlap patches
            for prv, px in overlap_map.get(cur, {}).items():
                v = (t-1, prv)
                d_c = _cdist(u, v)
                score = px / (1 + d_c)           # quick filter
                cand.append((-score, prv, px, d_c, 0.0))

            # (b) add nearest by centroid if needed
            if len(cand) < m_candidates:
                ra, ca = wcent_dict[u]
                if not np.isnan(ra):
                    dist_list = []
                    for prv in prv_labels:
                        if prv in (-1, *[x[1] for x in cand]):
                            continue
                        v = (t-1, prv)
                        rb, cb = wcent_dict.get(v, (np.nan, np.nan))
                        if np.isnan(rb):
                            continue
                        d_c = np.hypot(ra - rb, ca - cb)
                        dist_list.append((d_c, prv))
                    dist_list.sort()
                    for d_c, prv in dist_list[: m_candidates - len(cand)]:
                        cand.append((-1/(1+d_c), prv, 0, d_c, 0.0))

            # refine final weights on this trimmed set
            refined = []
            for _, prv, px, d_c, _ in cand:
                v = (t-1, prv)
                d_p = float(np.abs(v_cur - feat_dict[v]).sum())
                area_v = area_dict[v]
                union  = area_u + area_v - px
                overlap_ratio = px / (union + 1e-8)

                d_c_norm = d_c / max_dist
                d_p_norm = d_p / n_props

                w = (overlap_ratio /
                     (1.0 + alpha * d_c_norm + beta * d_p_norm)) if px>0 else 0.0
                refined.append((-w, prv, px, d_c_norm, d_p_norm))

            # pick top n_predecessors
            refined.sort()
            for _, prv, px, d_cn, d_pn in refined[:n_predecessors]:
                v = (t-1, prv)
                overlap_ratio = (px / (area_u + area_dict[v] - px + 1e-8))
                w = (overlap_ratio /
                     (1.0 + alpha * d_cn + beta * d_pn)) if px>0 else 0.0

                G.add_edge(u, v,
                           overlap_px=px,
                           overlap_ratio=overlap_ratio,
                           d_c_norm=d_cn,
                           d_p_norm=d_pn,
                           weight=w)

    _add_cost_attr(G)

    return G

def build_forwards_graph(labels_stack: np.ndarray,
                         region_df: pd.DataFrame,
                         n_successors: int = 1,
                         m_candidates: int = 10,
                         alpha: float = 1.0,
                         beta: float = 0.5) -> nx.DiGraph:
    """
    Link each super-pixel at frame t → up to n_successors at t+1,
    using the same normalized weight formula as the backwards graph:

      weight = overlap_ratio / (1 + α·d_c_norm + β·d_p_norm)

    • overlap_ratio = IoU = overlap_px / (area_t + area_{t+1} - overlap_px)
    • d_c_norm      = centroid_dist / image_diagonal
    • d_p_norm      = prop_dist   / n_props
    """
    # image diagonal for centroid normalization
    H, W = labels_stack.shape[1], labels_stack.shape[2]
    max_dist = np.hypot(H, W).astype(np.float32)

    # which properties to include
    prop_cols = [c for c in region_df.columns
                 if c.startswith(("intensity_", "weighted_moments_hu-", "hist_"))]
    n_props = float(len(prop_cols))

    # prepare feature, centroid & area caches
    prop_mat  = region_df[prop_cols].to_numpy(np.float32, copy=False)
    sigma_vec = prop_mat.std(axis=0, ddof=0)
    sigma_vec[sigma_vec == 0] = 1.0

    feat_dict: dict[tuple[int,int], np.ndarray] = {}
    wcent_dict: dict[tuple[int,int], tuple[float,float]] = {}
    area_dict: dict[tuple[int,int], float] = {}

    keep_cols = [c for c in region_df.columns if c not in {"image_idx","label"}]
    G = nx.DiGraph()

    # add nodes
    for _, r in region_df.iterrows():
        key = (int(r.image_idx), int(r.label))
        feat_dict[key]  = r[prop_cols].to_numpy(np.float32, copy=False) / sigma_vec
        wcent_dict[key] = (float(r.wcentroid_row), float(r.wcentroid_col))
        area_dict[key]  = float(r.area)
        G.add_node(key, **r[keep_cols].to_dict())

    T = labels_stack.shape[0]

    # helper to compute Euclidean centroid distance
    def _cdist(u, v) -> float:
        ra, ca = wcent_dict.get(u, (np.nan, np.nan))
        rb, cb = wcent_dict.get(v, (np.nan, np.nan))
        return float(np.hypot(ra - rb, ca - cb))

    # build forward edges
    for t in tqdm(range(T - 1), desc="edges (forward)"):
        cur_lab = labels_stack[t]
        nxt_lab = labels_stack[t + 1]

        # shift labels to non-negative for bincount
        la = cur_lab.astype(np.int32) + 1
        lb = nxt_lab.astype(np.int32) + 1
        mask = (la > 0) | (lb > 0)
        la, lb = la[mask], lb[mask]
        if la.size == 0:
            continue

        max_next = lb.max()
        key = la.astype(np.int64) * (max_next + 1) + lb
        cnt = np.bincount(key)

        # map curr_label → {next_label: overlap_px}
        overlap_map: dict[int, dict[int,int]] = {}
        for pid, px in enumerate(cnt):
            if px == 0:
                continue
            i = pid // (max_next + 1) - 1
            j = pid %  (max_next + 1) - 1
            if i >= 0 and j >= 0:
                overlap_map.setdefault(i, {})[j] = int(px)

        cur_labels = np.unique(cur_lab)
        nxt_labels = np.unique(nxt_lab)

        for cur in cur_labels:
            if cur == -1:
                continue
            u = (t, int(cur))
            v_cur = feat_dict[u]
            area_u = area_dict[u]

            # 1) gather overlap‐based candidates
            cand: list[tuple[float,int,int,float,float]] = []
            for nxt, px in overlap_map.get(cur, {}).items():
                v = (t + 1, nxt)
                d_c = _cdist(u, v)
                score = px / (1 + d_c)           # quick pre‐filter
                cand.append((-score, nxt, px, d_c, 0.0))

            # 2) if too few, add nearest by centroid
            if len(cand) < m_candidates:
                ra, ca = wcent_dict[u]
                if not np.isnan(ra):
                    dist_list = []
                    for nxt in nxt_labels:
                        if nxt in (-1, *[c[1] for c in cand]):
                            continue
                        v = (t + 1, nxt)
                        rb, cb = wcent_dict.get(v, (np.nan, np.nan))
                        if np.isnan(rb):
                            continue
                        d_c = float(np.hypot(ra - rb, ca - cb))
                        dist_list.append((d_c, nxt))
                    dist_list.sort()
                    for d_c, nxt in dist_list[: m_candidates - len(cand)]:
                        cand.append((-1/(1+d_c), nxt, 0, d_c, 0.0))

            # 3) refine weights on this trimmed set
            refined = []
            for _, nxt, px, d_c, _ in cand:
                v = (t + 1, nxt)
                d_p = float(np.abs(v_cur - feat_dict[v]).sum())
                area_v = area_dict[v]
                union  = area_u + area_v - px
                overlap_ratio = px / (union + 1e-8)

                d_c_norm = d_c / max_dist
                d_p_norm = d_p / n_props

                w = (overlap_ratio /
                     (1.0 + alpha * d_c_norm + beta * d_p_norm)) if px > 0 else 0.0
                refined.append((-w, nxt, px, d_c_norm, d_p_norm))

            # 4) pick top n_successors and add edges
            refined.sort()
            for _, nxt, px, d_cn, d_pn in refined[:n_successors]:
                v = (t + 1, nxt)
                overlap_ratio = px / (area_u + area_dict[v] - px + 1e-8)
                w = (overlap_ratio /
                     (1.0 + alpha * d_cn + beta * d_pn)) if px > 0 else 0.0

                G.add_edge(u, v,
                           overlap_px=px,
                           overlap_ratio=overlap_ratio,
                           d_c_norm=d_cn,
                           d_p_norm=d_pn,
                           weight=w)

    _add_cost_attr(G)

    return G

def overlap_stats(G: nx.DiGraph) -> float:
    overlaps = np.array([d["overlap_px"] for _,_,d in G.edges(data=True)])
    nonzero = np.count_nonzero(overlaps)
    total = len(overlaps)

    return (nonzero/total) * 100

def shortest_paths_last_to_first(G: nx.DiGraph,
                                 weighted: bool = False) -> dict[Node, list[Node]]:
    """
    For every node in the *last frame*, compute one shortest path
    back to *any* node in frame 0. Uses Dijkstra if weighted else topological graph.

    """
    first_f = min(f for f, _ in G.nodes)
    last_f  = max(f for f, _ in G.nodes)
    first_nodes = {n for n in G.nodes if n[0] == first_f}

    paths: dict[Node, list[Node]] = {}

    # choose which NetworkX routine and weight key
    if weighted:
        path_func = nx.single_source_dijkstra_path
        length_func = nx.single_source_dijkstra_path_length
        w_key = "cost"            # lower cost ⇒ stronger edge
    else:
        path_func = nx.single_source_shortest_path
        length_func = None              # lengths by len(path)
        w_key = None

    for src in (n for n in G.nodes if n[0] == last_f):
        try:
            p_dict = path_func(G, src, weight=w_key)
        except nx.NetworkXNoPath:
            continue

        reachable = first_nodes & p_dict.keys()
        if not reachable:
            continue

        # choose the best first-frame target
        if weighted:
            l_dict = length_func(G, src, weight=w_key)
            tgt = min(reachable, key=l_dict.get)
        else:
            tgt = min(reachable, key=lambda n: len(p_dict[n]))

        paths[src] = p_dict[tgt]        # already in backward order

    return paths

def shortest_paths_first_to_last(G: nx.DiGraph,
                                 inputs: list[Node],
                                 weighted: bool = False) -> dict[Node, list[Node]]:
    """
    For each `start` in `inputs` (frame 0), find ONE shortest path
    to any node in the last frame of G.  
    Returns {start_node → path list}, or omits if unreachable.

    weighted=False → fewest hops  
    weighted=True  → minimal total 'cost' (edge-attr 'cost')
    """
    # identify last‐frame nodes
    last_f = max(f for f,_ in G.nodes)
    last_nodes = {n for n in G.nodes if n[0] == last_f}

    # pick algorithms
    if weighted:
        path_fn   = nx.single_source_dijkstra_path
        length_fn = nx.single_source_dijkstra_path_length
        w_key     = "cost"
    else:
        path_fn   = nx.single_source_shortest_path
        length_fn = None
        w_key     = None

    out: dict[Node, list[Node]] = {}
    for src in inputs:
        try:
            p_dict = path_fn(G, src, weight=w_key)
        except nx.NetworkXNoPath:
            continue
        reach = last_nodes & p_dict.keys()
        if not reach:
            continue

        # choose the best end‐node
        if weighted:
            l_dict = length_fn(G, src, weight=w_key)
            tgt = min(reach, key=l_dict.get)
        else:
            tgt = min(reach, key=lambda n: len(p_dict[n]))

        out[src] = p_dict[tgt]

    return out

def forward_paths_from_backward(G_fwd: nx.DiGraph,
                                back_paths: dict[Node, list[Node]],
                                weighted: bool = False) -> dict[Node, list[Node]]:
    """
    For each output_node in back_paths, take its backward path,
    pick the final element (the input_node at frame 0), then compute
    the shortest forward path from that input_node to output_node in G_fwd.

    weighted=False → fewest hops (unweighted BFS)
    weighted=True  → minimal total 'cost' (Dijkstra on edge-attr 'cost')
    """
    out: Dict[Node, List[Node]] = {}

    for output_node, bpath in back_paths.items():
        if not bpath:
            continue
        input_node = bpath[-1]   # the seed at frame 0

        try:
            if weighted:
                # explicit Dijkstra on 'cost'
                fwd_path = nx.dijkstra_path(G_fwd,
                                            source=input_node,
                                            target=output_node,
                                            weight="cost")
            else:
                # simple shortest‐path (BFS)
                fwd_path = nx.shortest_path(G_fwd,
                                            source=input_node,
                                            target=output_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        out[output_node] = fwd_path

    return out


def compare_forward_backward_paths(fwd: dict[Node, list[Node]],
                                   bwd: dict[Node, list[Node]]) -> pd.DataFrame:
    """
    Compare forward‐inferred tracks (start→…→end) with backward‐inferred tracks (end→…→start0),
    and return a DataFrame summarising their overlap for each start node.

    Columns:
      - start_node     : (frame 0, label)
      - end_node       : (last frame, label)
      - fwd_length     : number of nodes in forward path
      - bwd_length     : number of nodes in backward path
      - intersection   : |intersection of node sets|
      - union          : |union of node sets|
      - jaccard        : intersection / union
    """
    records = []
    for start, path_f in fwd.items():
        end = path_f[-1]
        path_b = bwd.get(end)
        if path_b is None:
            continue

        path_b_rev = list(reversed(path_b))

        set_f = set(path_f)
        set_b = set(path_b_rev)
        i = len(set_f & set_b)
        u = len(set_f | set_b)
        j = i / u if u else 0.0

        records.append({
            "start_node": start,
            "end_node": end,
            "fwd_length": len(path_f),
            "bwd_length": len(path_b_rev),
            "intersection": i,
            "union": u,
            "jaccard": j
        })

    return pd.DataFrame(records)

def compute_path_level_coverage(G: nx.DiGraph,
                                paths: dict[Node, list[Node]]) -> pd.DataFrame:
    """
    Summarise how many nodes per frame belong to at least one shortest path.

    Returns a DataFrame with columns:
        frame, in_paths, total_nodes, percent
    """
    # gather nodes that lie on at least one path
    nodes_in_paths = defaultdict(set)        # frame → {nodes}
    for path in paths.values():
        for n in path:
            nodes_in_paths[n[0]].add(n)

    records = []
    for frame in sorted({f for f, _ in G.nodes}):
        in_paths = len(nodes_in_paths.get(frame, ()))
        total    = sum(1 for f, _ in G.nodes if f == frame)
        pct      = (in_paths / total * 100) if total else 0.0
        records.append(dict(frame=frame,
                            in_paths=in_paths,
                            total_nodes=total,
                            percent=pct))

    return pd.DataFrame(records)

def compute_input_to_last_costs(G: nx.DiGraph,
                                weighted: bool = True) -> pd.DataFrame:
    """
    For each input node (frame 0) in the forward graph G, compute the minimal
    path cost to any output node (last frame).

    Returns a DataFrame with columns:
      - input_node: tuple (frame, label)
      - label: the super-pixel label (int)
      - min_cost: minimal cost (float), np.inf if unreachable
      - reachable: bool, True if any path exists
    """
    # identify input (frame 0) and output (last frame) nodes
    first_f = min(f for f, _ in G.nodes())
    last_f  = max(f for f, _ in G.nodes())
    inputs  = [n for n in G.nodes() if n[0] == first_f]
    outputs = {n for n in G.nodes() if n[0] == last_f}

    # select the appropriate shortest-path length function
    if weighted:
        length_fn = nx.single_source_dijkstra_path_length
        weight_key = "cost"
    else:
        length_fn = nx.single_source_shortest_path_length
        weight_key = None

    records = []
    for inp in inputs:
        # compute all distances/costs from this input
        try:
            lengths = length_fn(G, inp, weight=weight_key)
        except Exception:
            lengths = {}

        # pick minimal cost among reachable outputs
        out_costs = [lengths[out] for out in outputs if out in lengths]
        if out_costs:
            min_cost = float(min(out_costs))
            reachable = True
        else:
            min_cost = float(np.inf)
            reachable = False

        records.append({
            "input_node": inp,
            "label": inp[1],
            "min_cost": min_cost,
            "reachable": reachable
        })

    return pd.DataFrame(records)

def mask_selected_inputs(labels_stack: np.ndarray,
                         input_nodes: Sequence[Node]) -> np.ndarray:
    """
    From labels_stack (T×H×W) and a list of input_nodes [(0,label),…],
    return a 2D mask for frame 0 where pixels = 1 if their label is in
    input_nodes, else 0.
    """
    selected_labels = {lab for (t, lab) in input_nodes if t == 0}
    lab0 = labels_stack[0]

    # build mask: 1 where label is selected, else 0
    mask0 = np.isin(lab0, list(selected_labels)).astype(np.uint8)
    return mask0


def save_to_zarr(mask: np.ndarray,
                 parameters: dict[str, Union[int, float, str]],
                 zarr_path: Union[str, Path]) -> None:
    """
    Store or append one 2D mask (0/1) and its metadata to a Zarr store
    along the 'run' dimension. If the store doesn't exist, it's created.
    If it does exist, the new run is appended.

    Parameters
    ----------
    mask       : 2D array (H, W) of 0/1
    parameters : dict of scalar metadata, e.g. {'well':'A001', 'alpha':1.0}
    zarr_path  : path to Zarr store directory

    Returns
    -------
    None
    """
    zarr_path = Path(zarr_path)

    if zarr_path.exists():
        # Append to existing store
        ds_old = xr.open_zarr(zarr_path)
        run_index = ds_old.dims['run']

        # New DataArray for this run
        da_new = xr.DataArray(
            mask[np.newaxis, ...],
            dims=('run', 'y', 'x'),
            coords={'run': [run_index]}
        )
        ds_new = da_new.to_dataset(name='mask')
        # Attach metadata
        for k, v in parameters.items():
            ds_new.coords[k] = ('run', [v])
        # Append
        ds_new.to_zarr(zarr_path, mode='a', append_dim='run')

    else:
        # Create fresh store (remove any partial)
        if zarr_path.exists():
            shutil.rmtree(zarr_path)

        arr = mask[np.newaxis, ...]  # shape (1, H, W)
        coords = {'run': [0]}
        coords.update({k: ('run', [v]) for k, v in parameters.items()})
        ds = xr.Dataset(
            {'mask': (('run', 'y', 'x'), arr)},
            coords=coords
        )
        ds.to_zarr(zarr_path, mode='w')
