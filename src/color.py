"""
color.py -- Color label extraction from OpenVINO sampling points.

The OpenVINO person-attributes model outputs two (x, y) coordinates in [0,1]
(upper-body and lower-body sampling points).  We sample a small patch in
CIELAB space, then map to a fixed label set via nearest-centroid.

Skin / background / shadow risk is flagged by low confidence.
If confidence < threshold, 'unknown' is returned instead of a label.
"""
from __future__ import annotations

import cv2
import numpy as np

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


def _sample_lab_patch(
    crop_bgr: np.ndarray,
    x_norm: float,
    y_norm: float,
    patch_size: int,
) -> np.ndarray:
    """
    Sample a patch around (x_norm, y_norm) and return median Lab colour.

    Returns float32 array of shape (3,) in standard Lab space.
    """
    h, w = crop_bgr.shape[:2]
    cx = int(round(x_norm * w))
    cy = int(round(y_norm * h))
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
    patch_size = int(cfg["patch_size"])
    conf_thresh = float(cfg["confidence_threshold"])

    lab = _sample_lab_patch(crop_bgr, x_norm, y_norm, patch_size)
    label, dist, margin = _nearest_centroid(lab)

    eps = 1e-6
    confidence = float(np.clip(margin / (dist + margin + eps), 0.0, 1.0))

    if confidence < conf_thresh:
        label = "unknown"

    return label, confidence


def extract_colors(
    crop_bgr: np.ndarray,
    upper_xy: tuple,
    lower_xy: tuple,
    cfg: dict,
) -> dict:
    """
    Extract upper and lower body color labels.

    Parameters
    ----------
    crop_bgr  : person crop (BGR).
    upper_xy  : (x_norm, y_norm) for upper body.
    lower_xy  : (x_norm, y_norm) for lower body.
    cfg       : ``color`` section of default.yaml.

    Returns
    -------
    dict: {upper, upper_conf, lower, lower_conf}
    """
    u_label, u_conf = classify_color(crop_bgr, *upper_xy, cfg)
    l_label, l_conf = classify_color(crop_bgr, *lower_xy, cfg)
    return {
        "upper": u_label, "upper_conf": u_conf,
        "lower": l_label, "lower_conf": l_conf,
    }
