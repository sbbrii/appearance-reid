"""
degrade.py -- Synthetic degradation transforms for evaluation.

Applied to QUERY images only.  Gallery images are kept clean.
Each transform is a pure function: (bgr_crop, **params) -> bgr_crop.
"""
from __future__ import annotations

import cv2
import numpy as np


def gaussian_blur(crop: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur.  kernel_size=0 means no-op."""
    if kernel_size <= 0:
        return crop
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(crop, (k, k), 0)


def motion_blur(crop: np.ndarray, kernel_size: int) -> np.ndarray:
    """Simulate horizontal motion blur."""
    if kernel_size <= 0:
        return crop
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(crop, -1, kernel)


def random_occlusion(
    crop: np.ndarray,
    frac: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Occlude a random rectangular region covering `frac` of the crop area.

    Occluded pixels are set to mean colour (less aggressive than black).
    """
    if frac <= 0:
        return crop
    h, w = crop.shape[:2]
    area = h * w * frac
    rng = np.random.RandomState(seed)
    occ_h = int(np.sqrt(area * (h / w)))
    occ_w = int(area / max(occ_h, 1))
    occ_h = min(occ_h, h)
    occ_w = min(occ_w, w)
    y0 = rng.randint(0, max(1, h - occ_h))
    x0 = rng.randint(0, max(1, w - occ_w))
    out = crop.copy()
    mean_col = crop.mean(axis=(0, 1)).astype(np.uint8)
    out[y0:y0 + occ_h, x0:x0 + occ_w] = mean_col
    return out


def low_resolution(crop: np.ndarray, scale_factor: float) -> np.ndarray:
    """Downscale then upscale back to original size."""
    if scale_factor >= 1.0:
        return crop
    h, w = crop.shape[:2]
    small_h = max(1, int(h * scale_factor))
    small_w = max(1, int(w * scale_factor))
    small = cv2.resize(crop, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def brightness_shift(crop: np.ndarray, delta: int) -> np.ndarray:
    """Add a constant to all pixels, clipped to [0, 255]."""
    if delta == 0:
        return crop
    out = crop.astype(np.int16) + delta
    return np.clip(out, 0, 255).astype(np.uint8)


# Registry for convenience
DEGRADATIONS = {
    "none": lambda c, **kw: c,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "occlusion": random_occlusion,
    "low_resolution": low_resolution,
    "brightness": brightness_shift,
}


def apply_all_conditions(crop: np.ndarray, cfg_eval: dict) -> list:
    """
    Generate all (condition_name, degraded_crop) pairs from config.

    Returns
    -------
    list of (condition_label: str, degraded: np.ndarray)
    """
    conditions = [("none_0", crop)]
    dg = cfg_eval["degradation"]

    for k in dg.get("blur_kernels", []):
        if k > 0:
            conditions.append((f"gaussian_blur_{k}", gaussian_blur(crop, k)))

    for k in dg.get("motion_blur_kernels", []):
        if k > 0:
            conditions.append((f"motion_blur_{k}", motion_blur(crop, k)))

    for frac in dg.get("occlusion_fracs", []):
        if frac > 0:
            conditions.append((f"occlusion_{int(frac*100)}pct",
                                random_occlusion(crop, frac)))

    for sf in dg.get("scale_factors", []):
        if sf < 1.0:
            conditions.append((f"lowres_{int(sf*100)}pct", low_resolution(crop, sf)))

    for delta in dg.get("brightness_shifts", []):
        if delta != 0:
            conditions.append((f"brightness_{delta:+d}", brightness_shift(crop, delta)))

    return conditions
