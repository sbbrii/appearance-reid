"""
generate_samples_and_split.py

1. Generates results/degrade_samples.png showing a sample crop under all 13 conditions.
2. Creates results/split.json: 3-way identity split of Market-1501 TEST identities (seed 42):
   - fit_pids (250 IDs): to train interaction LR on degraded-query pairs from identities OSNet never saw
   - tune_pids (250 IDs): to tune tau, aggregation, penalty, C
   - report_pids (250 IDs): final numbers only, untouched during development
"""
import json
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
    low_resolution,
    motion_blur,
    random_occlusion,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def apply_condition(crop: np.ndarray, cond: str) -> np.ndarray:
    if cond == "clean":
        return crop.copy()
    elif cond.startswith("gaussian_blur_"):
        k = int(cond.split("_")[-1])
        return gaussian_blur(crop, k)
    elif cond.startswith("motion_blur_"):
        k = int(cond.split("_")[-1])
        return motion_blur(crop, k)
    elif cond == "occlusion_30pct":
        return random_occlusion(crop, 0.30)
    elif cond == "occlusion_50pct":
        return random_occlusion(crop, 0.50)
    elif cond == "occlusion_70pct":
        return random_occlusion(crop, 0.70)
    elif cond == "lowres_50pct":
        return low_resolution(crop, 0.50)
    elif cond == "lowres_25pct":
        return low_resolution(crop, 0.25)
    elif cond == "brightness_-50":
        return brightness_shift(crop, -50)
    elif cond == "brightness_+50":
        return brightness_shift(crop, 50)
    else:
        raise ValueError(f"Unknown condition: {cond}")


def main():
    Path("results").mkdir(exist_ok=True, parents=True)
    cfg = yaml.safe_load(open("configs/default.yaml", "r"))["eval"]
    ds = load_market1501(cfg)

    # 1. Inspect and save sample grid of all 13 conditions
    # Pick a clear query sample image (e.g. index 0 or index 10)
    sample_item = ds["query"][10]
    sample_img_path = sample_item[0]
    logger.info("Using sample image: %s (PID %s, Cam %s)", sample_img_path, sample_item[1], sample_item[2])
    sample_bgr = cv2.imread(sample_img_path)

    conditions = [
        "clean",
        "gaussian_blur_3",
        "gaussian_blur_7",
        "gaussian_blur_15",
        "motion_blur_9",
        "motion_blur_15",
        "occlusion_30pct",
        "occlusion_50pct",
        "occlusion_70pct",
        "lowres_50pct",
        "lowres_25pct",
        "brightness_-50",
        "brightness_+50",
    ]

    # Create figure: 2 rows of 7 (or 2 x 7 grid, 13 images)
    fig, axes = plt.subplots(2, 7, figsize=(18, 7))
    axes = axes.flatten()

    for i, cond in enumerate(conditions):
        deg = apply_condition(sample_bgr, cond)
        deg_rgb = cv2.cvtColor(deg, cv2.COLOR_BGR2RGB)
        axes[i].imshow(deg_rgb)
        axes[i].set_title(cond, fontsize=10, pad=4)
        axes[i].axis("off")

    # Hide 14th empty subplot
    axes[13].axis("off")

    plt.suptitle("Degradation Conditions Sample Grid (Market-1501)", fontsize=14, y=0.98)
    plt.tight_layout()
    out_grid_path = Path("results/degrade_samples.png")
    plt.savefig(out_grid_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved sample grid to %s", out_grid_path)

    # 2. 3-way Identity Split of Market-1501 TEST identities
    query_pids_unique = sorted(list({pid for _, pid, _ in ds["query"]}))
    logger.info("Total unique TEST query identities: %d", len(query_pids_unique))

    rng = random.Random(42)
    shuffled_pids = list(query_pids_unique)
    rng.shuffle(shuffled_pids)

    n_ids = len(shuffled_pids)
    n_third = n_ids // 3  # 250
    fit_pids = sorted(shuffled_pids[:n_third])
    tune_pids = sorted(shuffled_pids[n_third:2 * n_third])
    report_pids = sorted(shuffled_pids[2 * n_third:])

    fit_set = set(fit_pids)
    tune_set = set(tune_pids)
    report_set = set(report_pids)

    # Counts
    fit_queries = [it for it in ds["query"] if it[1] in fit_set]
    tune_queries = [it for it in ds["query"] if it[1] in tune_set]
    report_queries = [it for it in ds["query"] if it[1] in report_set]

    # For full protocol, each query searches against the entire 15,913 gallery.
    # We can also track gallery items belonging to these pids
    gallery_pids = {pid for _, pid, _ in ds["gallery"]}
    logger.info("Gallery unique identities: %d", len(gallery_pids))

    split_info = {
        "seed": 42,
        "description": "3-way split of Market-1501 TEST identities (OSNet never saw any of these 750 identities)",
        "fit": {
            "n_ids": len(fit_pids),
            "n_queries": len(fit_queries),
            "pids": fit_pids,
        },
        "tune": {
            "n_ids": len(tune_pids),
            "n_queries": len(tune_queries),
            "pids": tune_pids,
        },
        "report": {
            "n_ids": len(report_pids),
            "n_queries": len(report_queries),
            "pids": report_pids,
        },
    }

    out_split_path = Path("results/split.json")
    with open(out_split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info("Saved 3-way split info to %s", out_split_path)

    print("\n" + "=" * 60)
    print("NEW 3-WAY SPLIT SUMMARY (MARKET-1501 TEST SET, SEED 42)")
    print("=" * 60)
    print(f"Total Test IDs : {n_ids}")
    print(f"Fit IDs        : {len(fit_pids):4d}  |  Query Images: {len(fit_queries):4d}")
    print(f"Tune IDs       : {len(tune_pids):4d}  |  Query Images: {len(tune_queries):4d}")
    print(f"Report IDs     : {len(report_pids):4d}  |  Query Images: {len(report_queries):4d}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
