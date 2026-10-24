"""
generate_rich_samples_grid.py

Regenerates results/degrade_samples.png showing:
- Every condition and occluder type
- 3 random crops each
- Measured area printed on each tile
"""
import logging
from pathlib import Path
import random
import sys

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from eval.datasets import load_market1501
from eval.degrade import (
    brightness_shift,
    gaussian_blur,
    get_donor_pool,
    low_resolution,
    motion_blur,
    random_occlusion,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def apply_cond_with_meta(crop: np.ndarray, cond: str, seed: int, donor_pool: list):
    h, w = crop.shape[:2]
    if cond == "clean":
        return crop.copy(), 0.0, "clean"
    elif cond.startswith("gaussian_blur_"):
        k = int(cond.split("_")[-1])
        return gaussian_blur(crop, k), 0.0, f"k={k}"
    elif cond.startswith("motion_blur_"):
        k = int(cond.split("_")[-1])
        return motion_blur(crop, k), 0.0, f"k={k}"
    elif cond.startswith("lowres_"):
        pct = int(cond.replace("lowres_", "").replace("pct", ""))
        return low_resolution(crop, pct / 100.0), 0.0, f"{pct}%"
    elif cond.startswith("brightness_"):
        delta = int(cond.replace("brightness_", ""))
        return brightness_shift(crop, delta), 0.0, f"delta={delta:+d}"
    elif cond.startswith("occ_"):
        # Format: occ_<pct>_<type>
        parts = cond.split("_")
        pct = int(parts[1].replace("pct", ""))
        occ_type = "_".join(parts[2:])
        frac = pct / 100.0
        out, area, chosen = random_occlusion(
            crop, frac, occluder_type=occ_type, donor_pool=donor_pool, seed=seed, return_meta=True
        )
        return out, area, chosen
    else:
        raise ValueError(f"Unknown condition: {cond}")


def main():
    Path("results").mkdir(exist_ok=True, parents=True)
    cfg = yaml.safe_load(open("configs/default.yaml", "r"))["eval"]
    ds = load_market1501(cfg)
    donor_pool = get_donor_pool()

    # Pick 3 distinct query sample crops
    sample_indices = [5, 42, 108]
    sample_crops = [cv2.imread(ds["query"][idx][0]) for idx in sample_indices]

    # Conditions list:
    # 1. Clean & Blur
    # 2. Motion blur & Low-res & Brightness
    # 3. Occlusion levels (10%, 20%, 30%, 40%, 50%, 70%)
    # 4. Occluder types at 30% (donor_crop, solid_gray, smooth_block, mix)
    conditions_to_show = [
        ("clean", "Clean (Reference)"),
        ("gaussian_blur_3", "Gaussian Blur k=3"),
        ("gaussian_blur_7", "Gaussian Blur k=7"),
        ("gaussian_blur_15", "Gaussian Blur k=15"),
        ("motion_blur_9", "Motion Blur k=9"),
        ("motion_blur_15", "Motion Blur k=15"),
        ("lowres_50pct", "Low-Res 50%"),
        ("lowres_25pct", "Low-Res 25%"),
        ("brightness_-50", "Brightness -50"),
        ("brightness_+50", "Brightness +50"),
        ("occ_10pct_mix", "Occlusion 10% (Mix)"),
        ("occ_20pct_mix", "Occlusion 20% (Mix)"),
        ("occ_30pct_mix", "Occlusion 30% (Mix)"),
        ("occ_40pct_mix", "Occlusion 40% (Mix)"),
        ("occ_30pct_donor_crop", "Occ 30% (a: Donor Crop)"),
        ("occ_30pct_solid_gray", "Occ 30% (b: Solid Gray)"),
        ("occ_30pct_smooth_block", "Occ 30% (c: Smooth Block)"),
        ("occ_50pct_mix", "Occ 50% (Appendix)"),
        ("occ_70pct_mix", "Occ 70% (Appendix)"),
    ]

    n_rows = len(conditions_to_show)
    n_cols = 3  # 3 random crops each

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, n_rows * 2.3))

    for r_idx, (cond_id, label) in enumerate(conditions_to_show):
        for c_idx in range(n_cols):
            base_crop = sample_crops[c_idx]
            seed = 1000 * (r_idx + 1) + c_idx * 77 + 42
            deg, area, info = apply_cond_with_meta(base_crop, cond_id, seed=seed, donor_pool=donor_pool)

            deg_rgb = cv2.cvtColor(deg, cv2.COLOR_BGR2RGB)
            ax = axes[r_idx, c_idx]
            ax.imshow(deg_rgb)
            ax.axis("off")

            # Title / Annotation
            if "occ" in cond_id:
                txt = f"{label}\nArea: {area*100:.1f}%\nType: {info}"
            else:
                txt = f"{label}\nArea: 0.0%"
            ax.set_title(txt, fontsize=8, pad=3)

    plt.suptitle("Degradation Conditions & Occluder Types (3 Random Crops Each)", fontsize=13, y=0.995)
    plt.tight_layout()
    out_path = Path("results/degrade_samples.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    logger.info("Saved regenerated rich sample grid to %s", out_path)


if __name__ == "__main__":
    main()
