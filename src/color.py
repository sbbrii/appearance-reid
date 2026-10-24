"""
color.py -- Color label extraction from OpenVINO sampling points.

The OpenVINO person-attributes model outputs two (x, y) coordinates in [0,1]
(upper-body and lower-body sampling points).  We sample a small patch in
CIELAB space, then map to a fixed label set via nearest-centroid.

Skin / background / shadow risk is flagged by low confidence.
If confidence < threshold, 'unknown' is returned instead of a label.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Fallback sampling points (normalized coordinates in [0, 1]).
# Used when model-predicted coordinates are missing or invalid.
FALLBACK_UPPER_XY: Tuple[float, float] = (0.50, 0.33)
FALLBACK_LOWER_XY: Tuple[float, float] = (0.50, 0.70)

# Fixed label set and their Lab centroids.
# Centroids measured from representative patches; adjust per deployment.
_COLOR_LABELS = [
    "black", "white", "gray", "red", "orange", "yellow",
    "green", "blue", "purple", "brown", "pink", "beige",
]

# Lab (L*, a*, b*) centroids.
# L in [0,100]; a,b approximately in [-128, 127].
_LAB_CENTROIDS = np.array([
    [10,   0,   0],   # black
    [95,   0,   0],   # white
    [55,   0,   0],   # gray
    [40,  50,  35],   # red
    [60,  25,  45],   # orange
    [85, -10,  60],   # yellow
    [45, -35,  25],   # green
    [40,  10, -45],   # blue
    [30,  25, -30],   # purple
    [35,  15,  20],   # brown
    [75,  30,   5],   # pink
    [80,   5,  20],   # beige
], dtype=np.float32)


def _is_valid_coord(pt: object) -> bool:
    """Check if pt is a 2-element tuple/list of finite numbers."""
    if pt is None or not isinstance(pt, (tuple, list)) or len(pt) < 2:
        return False
    try:
        x, y = float(pt[0]), float(pt[1])
        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            return False
        return True
    except (TypeError, ValueError):
        return False


def _sample_lab_patch(
    crop_bgr: np.ndarray,
    x_norm: float,
    y_norm: float,
    patch_size: int,
) -> np.ndarray:
    """
    Sample a patch around normalized (x_norm, y_norm) and return median Lab colour.

    Scales from normalized coords to crop pixels and clips to valid pixel bounds.
    Returns float32 array of shape (3,) in standard Lab space.
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.array([50.0, 0.0, 0.0], dtype=np.float32)

    # Clip normalized coords to [0.0, 1.0]
    x_c = float(np.clip(x_norm, 0.0, 1.0))
    y_c = float(np.clip(y_norm, 0.0, 1.0))

    # Scale to crop pixel coordinates, clipped to valid pixel indices
    cx = int(np.clip(round(x_c * (w - 1)), 0, max(0, w - 1)))
    cy = int(np.clip(round(y_c * (h - 1)), 0, max(0, h - 1)))
    half = patch_size // 2

    x1 = max(0, cx - half); x2 = min(w, cx + half + 1)
    y1 = max(0, cy - half); y2 = min(h, cy + half + 1)

    patch = crop_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([50.0, 0.0, 0.0], dtype=np.float32)

    # OpenCV Lab: L in [0,255], a/b in [0,255] with 128 bias.
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = lab[:, :, 0] * (100.0 / 255.0)
    lab[:, :, 1] = lab[:, :, 1] - 128.0
    lab[:, :, 2] = lab[:, :, 2] - 128.0
    return np.median(lab.reshape(-1, 3), axis=0).astype(np.float32)


def _nearest_centroid(lab: np.ndarray) -> tuple:
    """
    Find nearest centroid in Lab space.

    Returns (label, nearest_dist, margin) where margin = second - nearest.
    """
    dists = np.linalg.norm(_LAB_CENTROIDS - lab, axis=1)
    idx = np.argsort(dists)
    nearest_dist = float(dists[idx[0]])
    second_dist = float(dists[idx[1]]) if len(idx) > 1 else nearest_dist + 1
    return _COLOR_LABELS[idx[0]], nearest_dist, second_dist - nearest_dist


def classify_color(
    crop_bgr: np.ndarray,
    x_norm: float,
    y_norm: float,
    cfg: dict,
) -> tuple:
    """
    Map one sampling point to a color label with confidence.

    Returns
    -------
    label : str  ('unknown' if confidence < threshold)
    confidence : float in [0, 1]
    """
    patch_size = int(cfg.get("patch_size", 7))
    conf_thresh = float(cfg.get("confidence_threshold", 0.25))

    lab = _sample_lab_patch(crop_bgr, x_norm, y_norm, patch_size)
    label, dist, margin = _nearest_centroid(lab)

    eps = 1e-6
    confidence = float(np.clip(margin / (dist + margin + eps), 0.0, 1.0))

    if confidence < conf_thresh:
        label = "unknown"

    return label, confidence


def extract_colors(
    crop_bgr: np.ndarray,
    upper_xy: tuple | list | None = None,
    lower_xy: tuple | list | None = None,
    cfg: dict | None = None,
) -> dict:
    """
    Extract upper and lower body color labels.

    Parameters
    ----------
    crop_bgr  : person crop (BGR).
    upper_xy  : (x_norm, y_norm) for upper body (or None for fallback).
    lower_xy  : (x_norm, y_norm) for lower body (or None for fallback).
    cfg       : ``color`` section of default.yaml (or None for defaults).

    Returns
    -------
    dict: {upper, upper_conf, lower, lower_conf, sampling_path}
    """
    if cfg is None:
        cfg = {"patch_size": 7, "confidence_threshold": 0.25}

    if _is_valid_coord(upper_xy) and _is_valid_coord(lower_xy):
        sampling_path = "model points"
        u_pt = (float(np.clip(upper_xy[0], 0.0, 1.0)), float(np.clip(upper_xy[1], 0.0, 1.0)))
        l_pt = (float(np.clip(lower_xy[0], 0.0, 1.0)), float(np.clip(lower_xy[1], 0.0, 1.0)))
    else:
        sampling_path = "fallback"
        u_pt = FALLBACK_UPPER_XY
        l_pt = FALLBACK_LOWER_XY

    if logger.isEnabledFor(logging.DEBUG):
        msg = f"[DEBUG] Color sampling path: {sampling_path} (upper={u_pt}, lower={l_pt})"
        logger.debug(msg)
        print(msg)

    u_label, u_conf = classify_color(crop_bgr, *u_pt, cfg)
    l_label, l_conf = classify_color(crop_bgr, *l_pt, cfg)
    return {
        "upper": u_label, "upper_conf": u_conf,
        "lower": l_label, "lower_conf": l_conf,
        "sampling_path": sampling_path,
    }
