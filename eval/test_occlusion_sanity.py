"""
test_occlusion_sanity.py
Diagnostic sanity check on OSNet-only and occluder types (a, b, c) at 30% occlusion.
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
import numpy as np
from torchreid.reid.metrics.rank import eval_market1501
import yaml

from eval.datasets import load_market1501
from src.embed import EmbeddingExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_occluder_patch(
    crop: np.ndarray,
    occ_h: int,
    occ_w: int,
    occluder_type: str,
    donor_pool: list,
    rng: np.random.RandomState,
) -> np.ndarray:
    if occluder_type == "donor_crop":
        # a) Crop pasted from a different-identity Market image
        donor_path = donor_pool[rng.randint(len(donor_pool))]
        donor_img = cv2.imread(donor_path)
        dh, dw = donor_img.shape[:2]
        # Crop a patch of size occ_h, occ_w or resize from donor
        if dh >= occ_h and dw >= occ_w:
            dy = rng.randint(0, dh - occ_h + 1)
            dx = rng.randint(0, dw - occ_w + 1)
            patch = donor_img[dy : dy + occ_h, dx : dx + occ_w].copy()
        else:
            patch = cv2.resize(donor_img, (occ_w, occ_h), interpolation=cv2.INTER_LINEAR)
        return patch

    elif occluder_type == "solid_gray":
        # b) Solid mean-gray or dark-gray rectangle
        gray_val = rng.choice([128, 100, 64, int(crop.mean())])
        patch = np.full((occ_h, occ_w, 3), gray_val, dtype=np.uint8)
        return patch

    elif occluder_type == "smooth_block":
        # c) Smooth low-frequency color block
        c1 = rng.randint(40, 220, size=(3,), dtype=np.int16)
        c2 = rng.randint(40, 220, size=(3,), dtype=np.int16)
        # 1D linear ramp along width or height
        if rng.rand() > 0.5:
            alpha = np.linspace(0, 1, occ_w)[None, :, None]
        else:
            alpha = np.linspace(0, 1, occ_h)[:, None, None]
        patch = (c1 * (1 - alpha) + c2 * alpha).clip(0, 255).astype(np.uint8)
        k = max(3, (min(occ_h, occ_w) // 4) * 2 + 1)
        patch = cv2.GaussianBlur(patch, (k, k), 0)
        return patch
    else:
        raise ValueError(f"Unknown occluder type: {occluder_type}")


def apply_realistic_occlusion(
    crop: np.ndarray,
    frac: float,
    occluder_type: str = "mix",
    donor_pool: list = None,
    seed: int = None,
) -> tuple[np.ndarray, float, str]:
    """
    Returns (occluded_crop, measured_area_ratio, chosen_type).
    """
    if frac <= 0:
        return crop.copy(), 0.0, "none"
    h, w = crop.shape[:2]
    area = int(round(h * w * frac))

    if seed is None:
        h_idx = [0, h // 4, h // 2, 3 * h // 4, h - 1]
        w_idx = [0, w // 4, w // 2, 3 * w // 4, w - 1]
        sample_bytes = crop[np.ix_(h_idx, w_idx)].tobytes()
        seed = (hash(sample_bytes) + int(frac * 10000)) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)

    # Random aspect ratio, avoiding full-width or full-height bands (<= 0.88 * w, <= 0.88 * h)
    max_w_allowed = min(w - 2, max(2, int(0.88 * w)))
    max_h_allowed = min(h - 2, max(2, int(0.88 * h)))

    min_w = max(2, int(round(area / max_h_allowed)))
    max_w = min(max_w_allowed, int(round(area / 2)))

    if min_w > max_w:
        min_w = max(1, int(round(np.sqrt(area * (w / h)))))
        max_w = min(w - 1, int(round(np.sqrt(area * (w / h)))))
        min_w = min(min_w, max_w)

    occ_w = rng.randint(min_w, max_w + 1) if max_w >= min_w else min_w
    occ_h = int(round(area / max(occ_w, 1)))

    occ_h = min(max(1, occ_h), h - 1)
    occ_w = min(max(1, occ_w), w - 1)

    y0 = rng.randint(0, max(1, h - occ_h + 1))
    x0 = rng.randint(0, max(1, w - occ_w + 1))

    if occluder_type == "mix":
        chosen_type = rng.choice(["donor_crop", "solid_gray", "smooth_block"])
    else:
        chosen_type = occluder_type

    patch = generate_occluder_patch(crop, occ_h, occ_w, chosen_type, donor_pool, rng)

    out = crop.copy()
    out[y0 : y0 + occ_h, x0 : x0 + occ_w] = patch
    measured_area = (occ_h * occ_w) / (h * w)
    return out, measured_area, chosen_type


def main():
    cfg = yaml.safe_load(open("configs/default.yaml", "r"))
    ds = load_market1501(cfg["eval"])

    with open("results/split.json", "r") as f:
        split_info = json.load(f)

    tune_pids = set(split_info["tune"]["pids"])
    tune_queries = [it for it in ds["query"] if it[1] in tune_pids]

    # Donor pool: images from FIT split gallery (identities different from tune)
    fit_pids = set(split_info["fit"]["pids"])
    donor_pool = [it[0] for it in ds["gallery"] if it[1] in fit_pids]
    logger.info("Donor pool size: %d images from FIT split", len(donor_pool))

    # Gallery features
    g_cache = np.load("cache/gallery_features_full.npz")
    g_emb = g_cache["embeddings"]
    g_pids = g_cache["pids"]
    g_camids = g_cache["camids"]

    extractor = EmbeddingExtractor(cfg["model"])

    # Test each occluder type at 30% occlusion on 300 tune queries
    sub_queries = tune_queries[:300]
    types_to_test = ["donor_crop", "solid_gray", "smooth_block", "mix"]

    print("\n" + "=" * 70)
    print("SANITY CHECK: OSNet-Only Performance at 30% Occlusion by Occluder Type")
    print("=" * 70)

    for occ_type in types_to_test:
        q_embs = []
        pids = []
        cams = []
        measured_areas = []

        for i, (q_path, q_pid, q_cam) in enumerate(sub_queries):
            img = cv2.imread(q_path)
            deg, m_area, _ = apply_realistic_occlusion(img, 0.30, occluder_type=occ_type, donor_pool=donor_pool, seed=42 + i)
            measured_areas.append(m_area)
            emb = extractor.extract(deg)
            q_embs.append(emb)
            pids.append(q_pid)
            cams.append(q_cam)

        q_embs = np.array(q_embs)
        distmat = 1.0 - np.dot(q_embs, g_emb.T)
        cmc, mAP = eval_market1501(distmat, np.array(pids), g_pids, np.array(cams), g_camids, max_rank=10)

        mean_area = np.mean(measured_areas) * 100
        print(f"Type: {occ_type:<14} | Mean Area: {mean_area:.2f}% | Rank-1: {cmc[0]:.4f} | Rank-5: {cmc[4]:.4f} | mAP: {mAP:.4f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
