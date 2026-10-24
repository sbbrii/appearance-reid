"""
degrade.py -- Synthetic degradation transforms for evaluation.

Applied to QUERY images only.  Gallery images are kept clean.
Each transform is a pure function: (bgr_crop, **params) -> bgr_crop.
"""
from __future__ import annotations
from pathlib import Path

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


_DONOR_POOL_CACHE = None


def get_donor_pool() -> list:
    global _DONOR_POOL_CACHE
    if _DONOR_POOL_CACHE is None:
        cands = [
            Path("data/market1501/Market-1501-v15.09.15/bounding_box_train"),
            Path("data/market1501/bounding_box_train"),
            Path("data/market1501/Market-1501-v15.09.15/bounding_box_test"),
        ]
        for c in cands:
            if c.exists():
                _DONOR_POOL_CACHE = [str(p) for p in c.glob("*.jpg")]
                break
        if not _DONOR_POOL_CACHE:
            _DONOR_POOL_CACHE = []
    return _DONOR_POOL_CACHE


def generate_occluder_patch(
    crop: np.ndarray,
    occ_h: int,
    occ_w: int,
    occluder_type: str,
    donor_pool: list | None,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate occluder patch for (a) donor_crop, (b) solid_gray, or (c) smooth_block."""
    if occluder_type == "donor_crop":
        pool = donor_pool or get_donor_pool()
        if pool:
            donor_path = pool[rng.randint(len(pool))]
            donor_img = cv2.imread(donor_path)
            if donor_img is not None and donor_img.size > 0:
                dh, dw = donor_img.shape[:2]
                if dh >= occ_h and dw >= occ_w:
                    dy = rng.randint(0, dh - occ_h + 1)
                    dx = rng.randint(0, dw - occ_w + 1)
                    return donor_img[dy : dy + occ_h, dx : dx + occ_w].copy()
                else:
                    return cv2.resize(donor_img, (occ_w, occ_h), interpolation=cv2.INTER_LINEAR)
        # Fallback to solid gray if donor pool unavailable
        occluder_type = "solid_gray"

    if occluder_type == "solid_gray":
        gray_val = rng.choice([128, 100, 70, int(crop.mean())])
        return np.full((occ_h, occ_w, 3), gray_val, dtype=np.uint8)

    elif occluder_type == "smooth_block":
        c1 = rng.randint(40, 220, size=(3,), dtype=np.int16)
        c2 = rng.randint(40, 220, size=(3,), dtype=np.int16)
        if rng.rand() > 0.5:
            alpha = np.linspace(0, 1, occ_w)[None, :, None]
        else:
            alpha = np.linspace(0, 1, occ_h)[:, None, None]
        patch = (c1 * (1.0 - alpha) + c2 * alpha).clip(0, 255).astype(np.uint8)
        k = max(3, (min(occ_h, occ_w) // 4) * 2 + 1)
        return cv2.GaussianBlur(patch, (k, k), 0)
    else:
        raise ValueError(f"Unknown occluder type: {occluder_type}")


def random_occlusion(
    crop: np.ndarray,
    frac: float,
    occluder_type: str = "mix",
    donor_pool: list | None = None,
    seed: int | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple:
    """
    Occlude a random rectangular region covering `frac` of the crop area.

    Occluder types:
      - 'donor_crop': patch pasted from a different image (simulates another pedestrian)
      - 'solid_gray': solid mean-gray or dark-gray rectangle
      - 'smooth_block': smooth low-frequency color block
      - 'mix': equal probability mix of the three types above

    Guarantees measured area is within 2 percentage points of target frac.
    """
    if frac <= 0:
        if return_meta:
            return crop.copy(), 0.0, "none"
        return crop.copy()

    h, w = crop.shape[:2]
    total_area = h * w
    target_area = int(round(total_area * frac))

    if seed is None:
        h_idx = [0, h // 4, h // 2, 3 * h // 4, h - 1]
        w_idx = [0, w // 4, w // 2, 3 * w // 4, w - 1]
        sample_bytes = crop[np.ix_(h_idx, w_idx)].tobytes()
        seed = (hash(sample_bytes) + int(frac * 10000)) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)

    # Constrain width and height so neither spans full dimension (<= 0.88 * w, <= 0.88 * h)
    max_w_allowed = min(w - 2, max(2, int(0.88 * w)))
    max_h_allowed = min(h - 2, max(2, int(0.88 * h)))

    min_w = max(2, int(round(target_area / max_h_allowed)))
    max_w = min(max_w_allowed, int(round(target_area / 2)))

    if min_w > max_w:
        min_w = max(1, int(round(np.sqrt(target_area * (w / h)))))
        max_w = min(w - 1, int(round(np.sqrt(target_area * (w / h)))))
        min_w = min(min_w, max_w)

    occ_w = rng.randint(min_w, max_w + 1) if max_w >= min_w else min_w
    occ_h = int(round(target_area / max(occ_w, 1)))

    occ_h = min(max(1, occ_h), h - 1)
    occ_w = min(max(1, occ_w), w - 1)

    # Measured area check
    measured_area = (occ_h * occ_w) / total_area
    assert abs(measured_area - frac) <= 0.02, (
        f"Measured area {measured_area:.4f} exceeds 2 points from target {frac:.4f}"
    )

    y0 = rng.randint(0, max(1, h - occ_h + 1))
    x0 = rng.randint(0, max(1, w - occ_w + 1))

    if occluder_type == "mix":
        chosen_type = rng.choice(["donor_crop", "solid_gray", "smooth_block"])
    else:
        chosen_type = occluder_type

    patch = generate_occluder_patch(crop, occ_h, occ_w, chosen_type, donor_pool, rng)

    out = crop.copy()
    out[y0 : y0 + occ_h, x0 : x0 + occ_w] = patch

    if return_meta:
        return out, measured_area, chosen_type
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
