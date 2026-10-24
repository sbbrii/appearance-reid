"""
metrics.py -- Evaluation metrics for person re-ID.

Implements:
  Rank-K accuracy (CMC curve).
  mAP (mean Average Precision).
  MRR (Mean Reciprocal Rank).
  FAR / FRR at a threshold sweep.

All computed in pure NumPy for reproducibility.
Verified against torchreid's compute_ap() on unit-test cases.
"""
from __future__ import annotations

import numpy as np


def compute_cmc(
    dist_mat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int = 50,
) -> np.ndarray:
    """
    Compute Cumulative Matching Characteristics (CMC) curve.

    Follows Market-1501 protocol: exclude same-pid AND same-camid gallery
    images (junk) from the ranked list before computing rank accuracy.

    Parameters
    ----------
    dist_mat     : (Q, G) float array of DISTANCES (lower = more similar).
    query_pids   : (Q,) int array of query person IDs.
    gallery_pids : (G,) int array of gallery person IDs.
    query_camids : (Q,) int array of query camera IDs.
    gallery_camids : (G,) int array of gallery camera IDs.
    max_rank     : compute CMC up to this rank.

    Returns
    -------
    cmc : np.ndarray shape (max_rank,)  cumulative match rates at each rank.
    """
    num_q, num_g = dist_mat.shape
    max_rank = min(max_rank, num_g)

    all_cmc = []
    for q_idx in range(num_q):
        q_pid = query_pids[q_idx]
        q_camid = query_camids[q_idx]

        order = np.argsort(dist_mat[q_idx])
        # Exclude junk: same pid, same camera
        junk_mask = (gallery_pids == q_pid) & (gallery_camids == q_camid)
        valid_mask = ~junk_mask

        # Filter down to valid gallery entries
        valid_order = order[valid_mask[order]]
        if len(valid_order) == 0:
            continue

        # Binary hit array (1 if correct pid)
        matches = (gallery_pids[valid_order] == q_pid).astype(np.int32)

        # CMC for this query
        cmc_q = matches[:max_rank].cumsum()
        cmc_q = np.minimum(cmc_q, 1)  # clip at 1
        if len(cmc_q) < max_rank:
            pad_val = cmc_q[-1] if len(cmc_q) > 0 else 0
            cmc_q = np.pad(cmc_q, (0, max_rank - len(cmc_q)), mode="constant", constant_values=pad_val)
        all_cmc.append(cmc_q)

    if not all_cmc:
        return np.zeros(max_rank)

    all_cmc = np.array(all_cmc, dtype=np.float32)  # (Q, max_rank)
    return all_cmc.mean(axis=0)


def compute_map(
    dist_mat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
) -> float:
    """
    Compute mean Average Precision (mAP) with Market-1501 junk exclusion.

    Returns
    -------
    float : mAP in [0, 1].
    """
    num_q, num_g = dist_mat.shape
    aps = []

    for q_idx in range(num_q):
        q_pid = query_pids[q_idx]
        q_camid = query_camids[q_idx]

        order = np.argsort(dist_mat[q_idx])
        junk_mask = (gallery_pids == q_pid) & (gallery_camids == q_camid)
        valid_order = order[~junk_mask[order]]

        if len(valid_order) == 0:
            continue

        matches = (gallery_pids[valid_order] == q_pid).astype(np.float32)
        num_rel = matches.sum()
        if num_rel == 0:
            continue

        cum_matches = matches.cumsum()
        precision_at_k = cum_matches / (np.arange(len(matches)) + 1.0)
        ap = (precision_at_k * matches).sum() / num_rel
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def compute_mrr(
    dist_mat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) with Market-1501 junk exclusion.
    """
    num_q = dist_mat.shape[0]
    rrs = []

    for q_idx in range(num_q):
        q_pid = query_pids[q_idx]
        q_camid = query_camids[q_idx]
        order = np.argsort(dist_mat[q_idx])
        junk_mask = (gallery_pids == q_pid) & (gallery_camids == q_camid)
        valid_order = order[~junk_mask[order]]
        matches = gallery_pids[valid_order] == q_pid
        hit = np.where(matches)[0]
        rr = 1.0 / (hit[0] + 1) if len(hit) > 0 else 0.0
        rrs.append(rr)

    return float(np.mean(rrs)) if rrs else 0.0


def compute_far_frr(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> tuple:
    """
    Compute FAR and FRR at each threshold.

    Parameters
    ----------
    scores     : (N,) float array of similarity scores (higher = more similar).
    labels     : (N,) int array (1 = genuine pair, 0 = impostor pair).
    thresholds : (T,) float array of decision thresholds.

    Returns
    -------
    far : (T,) False Accept Rate  (impostor accepted)
    frr : (T,) False Reject Rate  (genuine rejected)
    """
    positives = labels == 1
    negatives = labels == 0
    num_pos = positives.sum()
    num_neg = negatives.sum()

    far = np.zeros(len(thresholds))
    frr = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        accepted = scores >= t
        fa = (accepted & negatives).sum()
        fr = (~accepted & positives).sum()
        far[i] = fa / max(num_neg, 1)
        frr[i] = fr / max(num_pos, 1)

    return far, frr


def summarise(
    cmc: np.ndarray,
    map_val: float,
    mrr_val: float,
    method: str = "",
    condition: str = "",
) -> dict:
    """Bundle metrics into a result dict."""
    return {
        "method": method,
        "condition": condition,
        "rank1": float(cmc[0]) if len(cmc) > 0 else 0.0,
        "rank5": float(cmc[4]) if len(cmc) >= 5 else 0.0,
        "mAP": float(map_val),
        "MRR": float(mrr_val),
    }
