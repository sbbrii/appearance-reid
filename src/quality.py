"""
quality.py -- Body quality estimator.

Computes four quality scores from a BGR person crop and its bounding box:
  blur        Laplacian variance, calibrated to [0,1].
  visibility  Proxy for occlusion/truncation (aspect ratio + edge density).
  size        Bbox height relative to reference, clipped to [0,1].
  overall     Documented weighted combination (weights from config).

LIMITATIONS (documented honestly):
- blur:       Laplacian variance correlates with sharpness but can be high for
              textured clothing.  Percentile calibration is dataset-dependent.
- visibility: Uses aspect ratio and lower-vs-upper edge density as proxy for
              body completeness.  Does NOT detect crops at image borders or
              detect partial limb visibility.  A pose estimator would be needed
              for true part-visibility (out of scope for M3).
- size:       Calibrate size_ref_height_px per deployment and camera height.
"""
from __future__ import annotations

import cv2
import numpy as np


def _laplacian_variance(gray: np.ndarray) -> float:
    """Return variance of the Laplacian of a grayscale image."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _calibrate_blur(raw: float, p5: float, p95: float) -> float:
    """
    Map Laplacian variance to [0, 1] using percentile calibration.

    p5  -> 0.0  (very blurry)
    p95 -> 1.0  (sharp)
    Clipped to [0, 1].
    """
    if p95 <= p5:
        return 0.5
    score = (raw - p5) / (p95 - p5)
    return float(np.clip(score, 0.0, 1.0))


def _visibility_score(
    crop_bgr: np.ndarray,
    min_aspect: float,
    edge_weight: float,
) -> float:
    """
    Estimate body visibility as a proxy score in [0, 1].

    Method (documented):
    1. Aspect ratio term: score_ar = clip((h/w - 1) / (min_aspect - 1), 0, 1)
    2. Edge density term: Canny lower half / upper half density ratio.
       Truncated bodies typically have fewer edges at the bottom.
    3. Final = (1 - edge_weight) * score_ar + edge_weight * score_ed
    """
    h, w = crop_bgr.shape[:2]
    if w == 0:
        return 0.0

    # Aspect ratio component
    aspect = h / w
    score_ar = float(np.clip((aspect - 1.0) / max(min_aspect - 1.0, 0.5), 0.0, 1.0))

    # Edge density component
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mid = h // 2
    upper_e = cv2.Canny(gray[:mid], 50, 150).astype(np.float32).mean()
    lower_e = cv2.Canny(gray[mid:], 50, 150).astype(np.float32).mean()
    eps = 1e-6
    score_ed = float(np.clip(lower_e / (upper_e + eps), 0.0, 1.0))

    return float((1.0 - edge_weight) * score_ar + edge_weight * score_ed)


def _size_score(bbox_height: int, min_height: int, ref_height: int) -> float:
    """Map bbox height to [0, 1] linearly, clipped."""
    if ref_height <= min_height:
        return 1.0
    score = (bbox_height - min_height) / (ref_height - min_height)
    return float(np.clip(score, 0.0, 1.0))


def estimate_quality(crop_bgr: np.ndarray, bbox: list, cfg: dict) -> dict:
    """
    Compute quality scores for one person crop.

    Parameters
    ----------
    crop_bgr : np.ndarray   H x W x 3 uint8 BGR person crop.
    bbox     : list         [x1, y1, x2, y2] pixel coordinates.
    cfg      : dict         ``quality`` section of default.yaml.

    Returns
    -------
    dict with keys: blur, visibility, size, overall  (floats in [0,1])
    """
    blur_cal = cfg["blur_calibration"]
    vis_min_asp = float(cfg["visibility_min_aspect"])
    vis_edge_w = float(cfg["visibility_edge_weight"])
    size_ref = int(cfg["size_ref_height_px"])
    size_min = int(cfg["size_min_height_px"])
    w_blur = float(cfg["weights"]["blur"])
    w_vis = float(cfg["weights"]["visibility"])
    w_size = float(cfg["weights"]["size"])

    # Blur
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    raw_var = _laplacian_variance(gray)
    blur_score = _calibrate_blur(
        raw_var,
        p5=float(blur_cal["p5"]),
        p95=float(blur_cal["p95"]),
    )

    # Visibility
    vis_score = _visibility_score(crop_bgr, vis_min_asp, vis_edge_w)

    # Size
    bbox_h = int(bbox[3]) - int(bbox[1])
    size_sc = _size_score(bbox_h, size_min, size_ref)

    # Overall (documented weighted sum, normalised weights should sum to ~1)
    overall = w_blur * blur_score + w_vis * vis_score + w_size * size_sc
    overall = float(np.clip(overall, 0.0, 1.0))

    return {
        "blur": blur_score,
        "visibility": vis_score,
        "size": size_sc,
        "overall": overall,
    }
