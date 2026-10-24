"""
refit_and_validate.py -- Refit learned fusion with standardisation, interaction terms,
and rigorous validation re-ranking evaluation.

Key Requirements Addressed:
1. Control model: LR on emb_sim only, same fit/val pairs.
2. Standardisation: StandardScaler inside Pipeline, report standardized coefficients.
3. Interaction features: emb_sim * q_min, emb_sim * q_prod, color_sim * (1 - q_min),
   attr_sim * (1 - q_min). Collinearity / VIF analysis to drop redundant standalone terms.
4. Validation re-ranking: Rank-1 and mAP on degraded val queries vs clean val gallery
   (validation identities ONLY).
5. Effective embedding weight vs quality curve: sweep q_min in [0, 1] and verify dW/dq.
6. Weight leakage ablation: compare Market-trained vs ImageNet-pretrained OSNet.
7. Pure quantitative reporting (no post-hoc qualitative narrative).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pathlib
import pickle
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yaml

from eval.datasets import load_market1501
from eval.degrade import (
    brightness_shift,
    gaussian_blur,
    low_resolution,
    motion_blur,
    random_occlusion,
)
from eval.metrics import compute_cmc, compute_map
from src.attributes import AttributeInference
from src.color import extract_colors
from src.embed import EmbeddingExtractor
from src.fusion import _attribute_similarity, _color_similarity
from src.quality import estimate_quality
from src.record import AttributeInfo, ColorInfo, FeatureRecord, QualityInfo

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_degrade_crop(crop_bgr: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, str]:
    choice = rng.choice(["clean", "gaussian_blur", "motion_blur", "occlusion", "lowres", "brightness"])
    if choice == "clean":
        return crop_bgr.copy(), "clean"
    elif choice == "gaussian_blur":
        k = int(rng.choice([3, 5, 7, 11, 15]))
        return gaussian_blur(crop_bgr, k), f"gaussian_blur_{k}"
    elif choice == "motion_blur":
        k = int(rng.choice([5, 9, 13, 15, 19]))
        return motion_blur(crop_bgr, k), f"motion_blur_{k}"
    elif choice == "occlusion":
        frac = float(rng.uniform(0.20, 0.70))
        return random_occlusion(crop_bgr, frac, seed=int(rng.randint(0, 100000))), f"occlusion_{int(frac*100)}pct"
    elif choice == "lowres":
        sf = float(rng.choice([0.5, 0.35, 0.25]))
        return low_resolution(crop_bgr, sf), f"lowres_{int(sf*100)}pct"
    elif choice == "brightness":
        delta = int(rng.choice([-60, -40, -20, 20, 40, 60]))
        return brightness_shift(crop_bgr, delta), f"brightness_{delta:+d}"
    return crop_bgr.copy(), "clean"


def extract_record(
    crop_bgr: np.ndarray,
    extractor: EmbeddingExtractor,
    attr_model: AttributeInference,
    cfg_quality: dict,
    cfg_color: dict,
    pid: int = -1,
    camid: int = -1,
) -> FeatureRecord:
    h, w = crop_bgr.shape[:2]
    bbox = [0, 0, w, h]

    q_dict = estimate_quality(crop_bgr, bbox, cfg_quality)
    q_info = QualityInfo(
        blur=q_dict["blur"],
        visibility=q_dict["visibility"],
        size=q_dict["size"],
        overall=q_dict["overall"],
    )

    pred = attr_model.predict(crop_bgr)
    probs = pred["probs"]
    attr_info = AttributeInfo(
        is_male=probs.get("is_male", 0.5),
        has_bag=probs.get("has_bag", 0.0),
        has_backpack=probs.get("has_backpack", 0.0),
        has_hat=probs.get("has_hat", 0.0),
        has_longsleeves=probs.get("has_longsleeves", 0.5),
        has_longpants=probs.get("has_longpants", 0.5),
        has_longhair=probs.get("has_longhair", 0.0),
        has_coat_jacket=probs.get("has_coat_jacket", 0.0),
    )

    top_pt, bot_pt = pred["coords"]
    color_dict = extract_colors(crop_bgr, top_pt, bot_pt, cfg_color)
    color_info = ColorInfo(
        upper=color_dict["upper"],
        lower=color_dict["lower"],
        upper_conf=color_dict["upper_conf"],
        lower_conf=color_dict["lower_conf"],
    )

    emb = extractor.extract(crop_bgr)

    return FeatureRecord(
        track_id=pid,
        camera_id=f"c{camid}",
        bbox=bbox,
        embedding=emb,
        colors=color_info,
        attributes=attr_info,
        quality=q_info,
    )


def extract_full_feature_dict(q_rec: FeatureRecord, g_rec: FeatureRecord) -> Dict[str, float]:
    emb_sim = float(np.clip(float(np.dot(q_rec.embedding, g_rec.embedding)), 0.0, 1.0))
    color_sim = _color_similarity(q_rec.colors, g_rec.colors)
    attr_sim = _attribute_similarity(q_rec.attributes, g_rec.attributes)

    q_q = float(q_rec.quality.overall)
    q_g = float(g_rec.quality.overall)
    q_min = min(q_q, q_g)
    q_prod = math.sqrt(max(q_q * q_g, 0.0))
    inv_q = 1.0 - q_min

    return {
        "emb_sim": emb_sim,
        "q_quality_overall": q_q,
        "g_quality_overall": q_g,
        "q_blur": float(q_rec.quality.blur),
        "g_blur": float(g_rec.quality.blur),
        "q_visibility": float(q_rec.quality.visibility),
        "g_visibility": float(g_rec.quality.visibility),
        "color_sim": color_sim,
        "attr_sim": attr_sim,
        "q_min": q_min,
        "q_prod": q_prod,
        "emb_sim_x_q_min": emb_sim * q_min,
        "emb_sim_x_q_prod": emb_sim * q_prod,
        "color_sim_x_inv_q": color_sim * inv_q,
        "attr_sim_x_inv_q": attr_sim * inv_q,
    }


def compute_vif(X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Compute VIF from correlation matrix diagonal inversion."""
    R = np.corrcoef(X, rowvar=False)
    # Ridge regularization for stable numerical inversion if near-singular
    inv_R = np.linalg.pinv(R + 1e-6 * np.eye(len(feature_names)))
    vifs = np.diag(inv_R)
    return {name: float(round(vifs[i], 2)) for i, name in enumerate(feature_names)}


def run_experiment(extractor_weights_path: str = "models/osnet_x1_0_market1501.pth", tag: str = "market1501") -> dict:
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["eval"]["seed"])
    _set_seeds(seed)
    rng = np.random.RandomState(seed)

    cfg_model = dict(cfg["model"])
    cfg_model["weights_path"] = extractor_weights_path
    if extractor_weights_path == "imagenet":
        cfg_model["osnet_pretrained"] = True

    logger.info("Initializing EmbeddingExtractor (%s)...", tag)
    extractor = EmbeddingExtractor(cfg_model)
    attr_model = AttributeInference(cfg["attributes"])
    attr_model.load()

    ds = load_market1501(cfg["eval"])
    train_items = ds["train"]

    pid_to_items: Dict[int, list] = {}
    for img_path, pid, camid in train_items:
        if pid in (-1, 0):
            continue
        pid_to_items.setdefault(pid, []).append((img_path, pid, camid))

    all_pids = sorted(list(pid_to_items.keys()))
    rng.shuffle(all_pids)
    split_idx = int(0.80 * len(all_pids))
    fit_pids = set(all_pids[:split_idx])
    val_pids = set(all_pids[split_idx:])

    clean_cache: Dict[str, FeatureRecord] = {}

    def get_clean(item: tuple) -> FeatureRecord:
        path, p, c = item
        if path not in clean_cache:
            crop = cv2.imread(path)
            clean_cache[path] = extract_record(
                crop, extractor, attr_model, cfg["quality"], cfg["color"], pid=p, camid=c
            )
        return clean_cache[path]

    def build_pairs(target_pids: set, n_pos: int, n_neg: int, desc: str) -> Tuple[List[dict], List[int]]:
        pids_list = [p for p in target_pids if len(pid_to_items[p]) >= 2]
        records, labels = [], []

        # Positives
        for i in range(n_pos):
            pid = rng.choice(pids_list)
            items = pid_to_items[pid]
            idx_a, idx_b = rng.choice(len(items), size=2, replace=False)
            item_a, item_b = items[idx_a], items[idx_b]

            crop_a = cv2.imread(item_a[0])
            crop_a_deg, _ = random_degrade_crop(crop_a, rng)
            q_rec = extract_record(crop_a_deg, extractor, attr_model, cfg["quality"], cfg["color"], item_a[1], item_a[2])
            g_rec = get_clean(item_b)

            records.append(extract_full_feature_dict(q_rec, g_rec))
            labels.append(1)
            if (i + 1) % 500 == 0:
                logger.info("  [%s] positives: %d / %d", desc, i + 1, n_pos)

        # Negatives
        for i in range(n_neg):
            pid_a = rng.choice(pids_list)
            list_a = pid_to_items[pid_a]
            item_a = list_a[rng.randint(0, len(list_a))]
            crop_a = cv2.imread(item_a[0])
            crop_a_deg, _ = random_degrade_crop(crop_a, rng)
            q_rec = extract_record(crop_a_deg, extractor, attr_model, cfg["quality"], cfg["color"], item_a[1], item_a[2])

            if i < (n_neg // 2):
                pid_b = rng.choice(pids_list)
                while pid_b == pid_a:
                    pid_b = rng.choice(pids_list)
                list_b = pid_to_items[pid_b]
                item_b = list_b[rng.randint(0, len(list_b))]
                g_rec = get_clean(item_b)
            else:
                best_sim = -1.0
                best_rec = None
                for _ in range(8):
                    pid_cand = rng.choice(pids_list)
                    while pid_cand == pid_a:
                        pid_cand = rng.choice(pids_list)
                    cand_list = pid_to_items[pid_cand]
                    cand_item = cand_list[rng.randint(0, len(cand_list))]
                    cand_rec = get_clean(cand_item)
                    sim = float(np.dot(q_rec.embedding, cand_rec.embedding))
                    if sim > best_sim:
                        best_sim = sim
                        best_rec = cand_rec
                g_rec = best_rec

            records.append(extract_full_feature_dict(q_rec, g_rec))
            labels.append(0)
            if (i + 1) % 500 == 0:
                logger.info("  [%s] negatives: %d / %d (hard negatives included)", desc, i + 1, n_neg)

        return records, labels

    logger.info("[%s] Generating 1500 pos / 1500 neg fit pairs...", tag)
    fit_dicts, y_fit = build_pairs(fit_pids, 1500, 1500, desc="fit")
    logger.info("[%s] Generating 500 pos / 500 neg val pairs...", tag)
    val_dicts, y_val = build_pairs(val_pids, 500, 500, desc="val")

    # Define Feature Sets
    feats_emb_only = ["emb_sim"]
    feats_additive = [
        "emb_sim", "q_quality_overall", "g_quality_overall",
        "q_blur", "g_blur", "q_visibility", "g_visibility",
        "color_sim", "attr_sim",
    ]
    # Pruned interaction model: drop redundant gallery quality terms (g_quality, g_blur, g_visibility)
    feats_interaction = [
        "emb_sim", "q_min", "q_prod", "q_blur", "q_visibility",
        "color_sim", "attr_sim",
        "emb_sim_x_q_min", "emb_sim_x_q_prod",
        "color_sim_x_inv_q", "attr_sim_x_inv_q",
    ]

    def to_matrix(dict_list: List[dict], feat_names: List[str]) -> np.ndarray:
        return np.array([[d[f] for f in feat_names] for d in dict_list], dtype=np.float32)

    X_fit_emb = to_matrix(fit_dicts, feats_emb_only)
    X_val_emb = to_matrix(val_dicts, feats_emb_only)

    X_fit_add = to_matrix(fit_dicts, feats_additive)
    X_val_add = to_matrix(val_dicts, feats_additive)

    X_fit_int = to_matrix(fit_dicts, feats_interaction)
    X_val_int = to_matrix(val_dicts, feats_interaction)

    y_fit_arr = np.array(y_fit, dtype=np.int32)
    y_val_arr = np.array(y_val, dtype=np.int32)

    # VIF Analysis on Additive and Interaction matrices
    vif_additive = compute_vif(X_fit_add, feats_additive)
    vif_interaction = compute_vif(X_fit_int, feats_interaction)

    # Train Pipelines
    pipe_emb = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs")),
    ])
    pipe_emb.fit(X_fit_emb, y_fit_arr)

    pipe_add = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs")),
    ])
    pipe_add.fit(X_fit_add, y_fit_arr)

    pipe_int = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs")),
    ])
    pipe_int.fit(X_fit_int, y_fit_arr)

    # Validation Metrics (AUC & PR-AUC)
    val_p_emb = pipe_emb.predict_proba(X_val_emb)[:, 1]
    val_p_add = pipe_add.predict_proba(X_val_add)[:, 1]
    val_p_int = pipe_int.predict_proba(X_val_int)[:, 1]

    auc_emb = float(roc_auc_score(y_val_arr, val_p_emb))
    prauc_emb = float(average_precision_score(y_val_arr, val_p_emb))

    auc_add = float(roc_auc_score(y_val_arr, val_p_add))
    prauc_add = float(average_precision_score(y_val_arr, val_p_add))

    auc_int = float(roc_auc_score(y_val_arr, val_p_int))
    prauc_int = float(average_precision_score(y_val_arr, val_p_int))

    # Standardized Coefficients
    coef_emb = {f: float(c) for f, c in zip(feats_emb_only, pipe_emb.named_steps["lr"].coef_[0])}
    coef_add = {f: float(c) for f, c in zip(feats_additive, pipe_add.named_steps["lr"].coef_[0])}
    coef_int = {f: float(c) for f, c in zip(feats_interaction, pipe_int.named_steps["lr"].coef_[0])}

    # Validation Set Re-Ranking Evaluation (Validation Identities ONLY)
    logger.info("[%s] Running validation re-ranking evaluation on 151 val IDs...", tag)
    val_pids_list = [p for p in val_pids if len(pid_to_items[p]) >= 2]
    val_query_recs = []
    val_gallery_recs = []

    for p in val_pids_list:
        items = pid_to_items[p]
        # First crop as degraded query (gaussian blur 7)
        q_item = items[0]
        crop_q = cv2.imread(q_item[0])
        crop_q_deg = gaussian_blur(crop_q, 7)
        q_rec = extract_record(crop_q_deg, extractor, attr_model, cfg["quality"], cfg["color"], p, q_item[2])
        val_query_recs.append(q_rec)

        # 2 distinct clean crops from other views as gallery
        for g_item in items[1:3]:
            g_rec = get_clean(g_item)
            val_gallery_recs.append(g_rec)

    q_pids = np.array([r.track_id for r in val_query_recs], dtype=np.int32)
    q_cams = np.array([int(r.camera_id.replace("c", "")) for r in val_query_recs], dtype=np.int32)
    g_pids = np.array([r.track_id for r in val_gallery_recs], dtype=np.int32)
    g_cams = np.array([int(r.camera_id.replace("c", "")) for r in val_gallery_recs], dtype=np.int32)

    N_q = len(val_query_recs)
    N_g = len(val_gallery_recs)

    # Vectorized score computation for all validation query-gallery pairs
    fdicts = []
    for qr in val_query_recs:
        for gr in val_gallery_recs:
            fdicts.append(extract_full_feature_dict(qr, gr))

    sims = np.array([fd["emb_sim"] for fd in fdicts], dtype=np.float32).reshape(N_q, N_g)
    dist_emb = 1.0 - sims

    X_val_eval_add = np.array([[fd[f] for f in feats_additive] for fd in fdicts], dtype=np.float32)
    scores_add = pipe_add.predict_proba(X_val_eval_add)[:, 1].reshape(N_q, N_g)
    dist_add = 1.0 - scores_add

    X_val_eval_int = np.array([[fd[f] for f in feats_interaction] for fd in fdicts], dtype=np.float32)
    scores_int = pipe_int.predict_proba(X_val_eval_int)[:, 1].reshape(N_q, N_g)
    dist_int = 1.0 - scores_int

    cmc_emb = compute_cmc(dist_emb, q_pids, g_pids, q_cams, g_cams)
    map_emb = compute_map(dist_emb, q_pids, g_pids, q_cams, g_cams)

    cmc_add = compute_cmc(dist_add, q_pids, g_pids, q_cams, g_cams)
    map_add = compute_map(dist_add, q_pids, g_pids, q_cams, g_cams)

    cmc_int = compute_cmc(dist_int, q_pids, g_pids, q_cams, g_cams)
    map_int = compute_map(dist_int, q_pids, g_pids, q_cams, g_cams)

    # Effective embedding weight vs quality curve
    # d(logit)/d(emb_sim) as a function of q in [0, 1]
    scaler = pipe_int.named_steps["scaler"]
    lr_int = pipe_int.named_steps["lr"]
    means = scaler.mean_
    scales = scaler.scale_
    feat_to_idx = {f: idx for idx, f in enumerate(feats_interaction)}

    idx_emb = feat_to_idx["emb_sim"]
    idx_min = feat_to_idx["emb_sim_x_q_min"]
    idx_prd = feat_to_idx["emb_sim_x_q_prod"]

    w_emb_std = lr_int.coef_[0][idx_emb] / scales[idx_emb]
    w_min_std = lr_int.coef_[0][idx_min] / scales[idx_min]
    w_prd_std = lr_int.coef_[0][idx_prd] / scales[idx_prd]

    q_grid = np.linspace(0.0, 1.0, 21)
    marginal_weights = [float(w_emb_std + (w_min_std + w_prd_std) * q) for q in q_grid]

    # Save plot
    Path("results").mkdir(parents=True, exist_ok=True)
    plot_path = Path("results") / f"effective_embedding_weight_vs_quality_{tag}.png"
    plt.figure(figsize=(7, 4.5), dpi=150)
    plt.plot(q_grid, marginal_weights, marker="o", color="#1f77b4", linewidth=2.0)
    plt.xlabel(r"Quality Metric $q = \min(Q_q, Q_g)$", fontsize=11)
    plt.ylabel(r"Effective Embedding Logit Weight $\frac{\partial\,\mathrm{logit}}{\partial\,S_{\mathrm{emb}}}$", fontsize=11)
    plt.title(f"SAFEPASS: Marginal Embedding Weight vs Quality ({tag})", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info("Saved marginal weight curve: %s", plot_path)

    # Save Pipeline to models/fusion_lr.pkl if market tag
    if tag == "market1501":
        with open("models/fusion_lr.pkl", "wb") as f:
            pickle.dump(pipe_int, f)
        logger.info("Saved standardized interaction pipeline to models/fusion_lr.pkl")

    return {
        "tag": tag,
        "auc_emb": auc_emb, "prauc_emb": prauc_emb,
        "auc_add": auc_add, "prauc_add": prauc_add,
        "auc_int": auc_int, "prauc_int": prauc_int,
        "vif_additive": vif_additive,
        "vif_interaction": vif_interaction,
        "coef_emb": coef_emb,
        "coef_add": coef_add,
        "coef_int": coef_int,
        "rank1_emb": float(cmc_emb[0]), "map_emb": float(map_emb),
        "rank1_add": float(cmc_add[0]), "map_add": float(map_add),
        "rank1_int": float(cmc_int[0]), "map_int": float(map_int),
        "q_grid": q_grid.tolist(),
        "marginal_weights": marginal_weights,
        "plot_path": str(plot_path),
    }


def main() -> None:
    print("=" * 70)
    print("SAFEPASS: FUSION REFIT, INTERACTION MODELING & RIGOROUS VALIDATION")
    print("=" * 70)

    # Run 1: Market-1501 trained OSNet
    res_market = run_experiment("models/osnet_x1_0_market1501.pth", tag="market1501")

    # Run 2: ImageNet pretrained OSNet (Leakage Control)
    res_imagenet = run_experiment("imagenet", tag="imagenet")

    out_json = Path("results/fusion_refit_summary.json")
    with open(out_json, "w") as f:
        json.dump({"market1501": res_market, "imagenet": res_imagenet}, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY -- QUANTITATIVE METRICS ONLY")
    print("=" * 70)
    for res in [res_market, res_imagenet]:
        t = res["tag"]
        print(f"\n--- OSNet Backbone: {t.upper()} ---")
        print("Validation Pair Classification (Held-out 151 Identities):")
        print(f"  Control (emb_sim only)      : ROC-AUC = {res['auc_emb']:.4f} | PR-AUC = {res['prauc_emb']:.4f}")
        print(f"  Additive Model (Standardized): ROC-AUC = {res['auc_add']:.4f} | PR-AUC = {res['prauc_add']:.4f}")
        print(f"  Interaction Model (Standardized): ROC-AUC = {res['auc_int']:.4f} | PR-AUC = {res['prauc_int']:.4f}")
        print("\nValidation Re-Ranking Performance (Degraded Queries, Held-out 151 Identities):")
        print(f"  Control (emb_sim only)      : Rank-1 = {res['rank1_emb']:.4f} | mAP = {res['map_emb']:.4f}")
        print(f"  Additive Model (Standardized): Rank-1 = {res['rank1_add']:.4f} | mAP = {res['map_add']:.4f}")
        print(f"  Interaction Model (Standardized): Rank-1 = {res['rank1_int']:.4f} | mAP = {res['map_int']:.4f}")
        print("\nStandardized Coefficients (Interaction Model):")
        for f_name, c_val in res["coef_int"].items():
            print(f"  {f_name:<20}: {c_val:+.4f}")
        print(f"\nEffective Embedding Weight sweep: q=0.0 -> {res['marginal_weights'][0]:.4f}, q=1.0 -> {res['marginal_weights'][-1]:.4f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
