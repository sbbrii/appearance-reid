"""
eval_new_occlusion.py

Evaluates:
1. OSNet-only sanity check per occluder type (a: donor_crop, b: solid_gray, c: smooth_block) at 30% occlusion on the TUNE split.
2. Control (OSNet emb_sim only) vs Mean-Centered Interaction Model on new occlusion levels:
   - occlusion_10pct
   - occlusion_20pct
   - occlusion_30pct
   - occlusion_40pct
   - occlusion_50pct (appendix)
   - occlusion_70pct (appendix)
3. Verifies measured area across all samples and computes mean measured area.
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
from eval.degrade import get_donor_pool, random_occlusion
from src.attributes import AttributeInference
from src.color import extract_colors, _COLOR_LABELS
from src.embed import EmbeddingExtractor
from src.quality import estimate_quality

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

COLOR_CENTROIDS = list(_COLOR_LABELS) + ["unknown"]
COLOR_MAP = {name: i for i, name in enumerate(COLOR_CENTROIDS)}


def extract_features_batch(
    crops_bgr: list,
    extractor: EmbeddingExtractor,
    attr_model: AttributeInference,
    cfg_quality: dict,
    cfg_color: dict,
) -> dict:
    embs = extractor.extract_batch(crops_bgr)

    qualities = []
    attr_probs = []
    c_upper = []
    c_lower = []
    c_upper_conf = []
    c_lower_conf = []

    for crop in crops_bgr:
        h, w = crop.shape[:2]
        q_dict = estimate_quality(crop, [0, 0, w, h], cfg_quality)
        qualities.append(float(q_dict["overall"]))

        pred = attr_model.predict(crop)
        probs = pred["probs"]
        attr_vec = [
            probs.get("is_male", 0.5),
            probs.get("has_bag", 0.0),
            probs.get("has_backpack", 0.0),
            probs.get("has_hat", 0.0),
            probs.get("has_longsleeves", 0.5),
            probs.get("has_longpants", 0.5),
            probs.get("has_longhair", 0.0),
            probs.get("has_coat_jacket", 0.0),
        ]
        attr_probs.append(attr_vec)

        top_pt, bot_pt = pred["coords"]
        color_dict = extract_colors(crop, top_pt, bot_pt, cfg_color)
        c_upper.append(COLOR_MAP.get(color_dict["upper"], COLOR_MAP["unknown"]))
        c_lower.append(COLOR_MAP.get(color_dict["lower"], COLOR_MAP["unknown"]))
        c_upper_conf.append(float(color_dict["upper_conf"]))
        c_lower_conf.append(float(color_dict["lower_conf"]))

    return {
        "embeddings": embs,
        "quality": np.array(qualities, dtype=np.float32),
        "attr_probs": np.array(attr_probs, dtype=np.float32),
        "color_upper": np.array(c_upper, dtype=np.int64),
        "color_lower": np.array(c_lower, dtype=np.int64),
        "color_upper_conf": np.array(c_upper_conf, dtype=np.float32),
        "color_lower_conf": np.array(c_lower_conf, dtype=np.float32),
    }


def main():
    cfg = yaml.safe_load(open("configs/default.yaml", "r"))
    ds = load_market1501(cfg["eval"])

    with open("results/split.json", "r") as f:
        split_info = json.load(f)

    tune_pids = set(split_info["tune"]["pids"])
    tune_queries = [it for it in ds["query"] if it[1] in tune_pids]
    q_pids = np.array([it[1] for it in tune_queries])
    q_cams = np.array([it[2] for it in tune_queries])

    donor_pool = get_donor_pool()

    # Load gallery features
    g_cache = np.load("cache/gallery_features_full.npz")
    g_emb = g_cache["embeddings"]
    g_pids = g_cache["pids"]
    g_camids = g_cache["camids"]
    g_q = g_cache["quality"]
    g_attr = g_cache["attr_probs"]
    g_up_idx = g_cache["color_upper"]
    g_lo_idx = g_cache["color_lower"]
    g_up_conf = g_cache["color_upper_conf"]
    g_lo_conf = g_cache["color_lower_conf"]

    extractor = EmbeddingExtractor(cfg["model"])
    attr_model = AttributeInference(cfg["attributes"])
    attr_model.load()

    # Load fitted verification model info (trained on FIT split with mean centering)
    with open("results/step1_to_3_verification.json", "r") as f:
        v_info = json.load(f)
    means = v_info["means"]
    coefs = v_info["coefs"]
    best_C = v_info["best_C"]

    # 1. Sanity check: OSNet-only across occluder types at 30% on the entire TUNE split (1,118 queries)
    logger.info("Running Sanity Check on entire TUNE split (1,118 queries) at 30% occlusion...")
    sanity_results = {}
    sanity_types = ["donor_crop", "solid_gray", "smooth_block", "mix"]

    for occ_type in sanity_types:
        crops = []
        measured_areas = []
        for i, (path, _, _) in enumerate(tune_queries):
            img = cv2.imread(path)
            deg, m_area, _ = random_occlusion(
                img, 0.30, occluder_type=occ_type, donor_pool=donor_pool, seed=42 + i, return_meta=True
            )
            crops.append(deg)
            measured_areas.append(m_area)

        embs = extractor.extract_batch(crops)
        distmat = 1.0 - np.dot(embs, g_emb.T)
        cmc, mAP = eval_market1501(distmat, q_pids, g_pids, q_cams, g_camids, max_rank=10)
        mean_area = float(np.mean(measured_areas))
        sanity_results[occ_type] = {
            "mean_area": mean_area,
            "rank1": float(cmc[0]),
            "rank5": float(cmc[4]),
            "map": float(mAP),
        }

    # 2. Evaluate all occlusion conditions (10%, 20%, 30%, 40%, 50%, 70%) with occluder_type="mix"
    occlusion_conditions = [
        ("occlusion_10pct", 0.10),
        ("occlusion_20pct", 0.20),
        ("occlusion_30pct", 0.30),
        ("occlusion_40pct", 0.40),
        ("occlusion_50pct", 0.50),
        ("occlusion_70pct", 0.70),
    ]

    all_occ_results = {}
    N_q = len(tune_queries)

    for cond_name, frac in occlusion_conditions:
        logger.info("Evaluating condition: %s (frac=%.2f)...", cond_name, frac)
        crops = []
        measured_areas = []
        for i, (path, _, _) in enumerate(tune_queries):
            img = cv2.imread(path)
            deg, m_area, _ = random_occlusion(
                img, frac, occluder_type="mix", donor_pool=donor_pool, seed=42 + i, return_meta=True
            )
            crops.append(deg)
            measured_areas.append(m_area)

        # Batch extraction of features
        q_feats = extract_features_batch(crops, extractor, attr_model, cfg["quality"], cfg["color"])
        q_emb = q_feats["embeddings"]
        q_q = q_feats["quality"]
        q_attr = q_feats["attr_probs"]
        q_up_idx = q_feats["color_upper"]
        q_lo_idx = q_feats["color_lower"]
        q_up_conf = q_feats["color_upper_conf"]
        q_lo_conf = q_feats["color_lower_conf"]

        # Control distmat
        distmat_ctrl = 1.0 - np.dot(q_emb, g_emb.T)
        cmc_ctrl, map_ctrl = eval_market1501(distmat_ctrl, q_pids, g_pids, q_cams, g_camids, max_rank=10)

        # Interaction model (re-ranking top-100)
        distmat_int = distmat_ctrl.copy()

        # Center using fit-split means
        mu_emb = means["mu_emb"]
        mu_col = means["mu_col"]
        mu_attr = means["mu_attr"]
        mu_q = means["mu_q"]
        mu_inv_q = means["mu_inv_q"]

        # Predict interaction scores for top 100
        for i in range(N_q):
            top100_idx = np.argsort(distmat_ctrl[i])[:100]
            emb_sims = np.dot(g_emb[top100_idx], q_emb[i])
            q_min_vals = np.minimum(q_q[i], g_q[top100_idx])
            attr_diffs = np.mean(np.abs(g_attr[top100_idx] - q_attr[i]), axis=1)
            attr_sims = 1.0 - attr_diffs

            up_match = (g_up_idx[top100_idx] == q_up_idx[i]).astype(float)
            lo_match = (g_lo_idx[top100_idx] == q_lo_idx[i]).astype(float)
            up_min_c = np.minimum(q_up_conf[i], g_up_conf[top100_idx])
            lo_min_c = np.minimum(q_lo_conf[i], g_lo_conf[top100_idx])
            col_sims = 0.5 * up_match * up_min_c + 0.5 * lo_match * lo_min_c

            c_emb_i = emb_sims - mu_emb
            c_col_i = col_sims - mu_col
            c_attr_i = attr_sims - mu_attr
            c_q_i = q_min_vals - mu_q
            c_inv_q_i = (1.0 - q_min_vals) - mu_inv_q

            # Features: emb_sim, color_sim, attr_sim, q_min, emb_sim_x_q, attr_sim_x_inv_q, color_sim_x_inv_q
            score_i = (
                coefs["emb_sim"] * c_emb_i +
                coefs["color_sim"] * c_col_i +
                coefs["attr_sim"] * c_attr_i +
                coefs["q_min"] * c_q_i +
                coefs["emb_sim_x_q"] * (c_emb_i * c_q_i) +
                coefs["attr_sim_x_inv_q"] * (c_attr_i * c_inv_q_i) +
                coefs["color_sim_x_inv_q"] * (c_col_i * c_inv_q_i)
            )
            # Higher score means more similar -> lower distance
            distmat_int[i, top100_idx] = -score_i

        cmc_int, map_int = eval_market1501(distmat_int, q_pids, g_pids, q_cams, g_camids, max_rank=10)
        mean_measured = float(np.mean(measured_areas))
        max_deviation = float(np.max(np.abs(np.array(measured_areas) - frac)))

        all_occ_results[cond_name] = {
            "target_frac": frac,
            "mean_measured_area": mean_measured,
            "max_deviation_points": max_deviation * 100.0,
            "control": {
                "rank1": float(cmc_ctrl[0]),
                "rank5": float(cmc_ctrl[4]),
                "map": float(map_ctrl),
            },
            "interaction": {
                "rank1": float(cmc_int[0]),
                "rank5": float(cmc_int[4]),
                "map": float(map_int),
            },
        }

    # Save output to json
    results_summary = {
        "sanity_check_30pct": sanity_results,
        "occlusion_conditions": all_occ_results,
    }
    with open("results/occlusion_corrected_metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Print clean formatted tables
    print("\n" + "=" * 80)
    print("SANITY CHECK: OSNet-Only Performance at 30% Occlusion by Occluder Type (TUNE SPLIT)")
    print("=" * 80)
    print(f"{'Occluder Type':<20} | {'Mean Measured Area':<18} | {'Rank-1':<10} | {'Rank-5':<10} | {'mAP':<10}")
    print("-" * 80)
    for t in sanity_types:
        r = sanity_results[t]
        print(f"{t:<20} | {r['mean_area']*100:<17.2f}% | {r['rank1']:<10.4f} | {r['rank5']:<10.4f} | {r['map']:<10.4f}")
    print("=" * 80)

    print("\n" + "=" * 85)
    print("CORRECTED OCCLUSION VALIDATION METRICS (TUNE SPLIT, 1,118 QUERIES vs 15,913 GALLERY)")
    print("=" * 85)
    print(f"{'Condition':<18} | {'Mean Area':<10} | {'Max Dev':<9} | {'Control (OSNet only)':<20} | {'Interaction Model':<20}")
    print(f"{'':<18} | {'':<10} | {'':<9} | {'R-1    R-5    mAP':<20} | {'R-1    R-5    mAP':<20}")
    print("-" * 85)
    for cond_name, _ in occlusion_conditions:
        r = all_occ_results[cond_name]
        ctrl = r["control"]
        inter = r["interaction"]
        str_ctrl = f"{ctrl['rank1']:.3f}  {ctrl['rank5']:.3f}  {ctrl['map']:.3f}"
        str_inter = f"{inter['rank1']:.3f}  {inter['rank5']:.3f}  {inter['map']:.3f}"
        print(f"{cond_name:<18} | {r['mean_measured_area']*100:>5.2f}%    | {r['max_deviation_points']:>4.2f} pt   | {str_ctrl:<20} | {str_inter:<20}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
