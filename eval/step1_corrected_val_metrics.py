"""
step1_corrected_val_metrics.py -- Compute corrected validation re-ranking metrics
using Torchreid's official eval_market1501 on the full 15,913 gallery protocol.

Implements:
1. Simplified interaction model:
   Features: emb_sim, color_sim, attr_sim, q_min, emb_sim*q_min, attr_sim*(1-q_min), color_sim*(1-q_min)
   Drop q_prod and standalone gallery quality terms.
   Tune L2 regularization strength via 5-fold cross-validation.
   Compute and report VIF.
2. Clean tuning split:
   Split Market-1501 TEST queries by identity (seed 42) into tune half (375 IDs, 1690 queries).
   Evaluate against FULL Market-1501 gallery (15,913 images) using torchreid.reid.metrics.rank.eval_market1501.
3. Full evaluation across all 13 degradation conditions:
   Report Rank-1, Rank-5, and mAP together for:
   - Control (emb_sim only / osnet_only)
   - Additive Model (emb_sim, color_sim, attr_sim, q_min)
   - Simplified Interaction Model
"""
from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchreid.reid.metrics.rank import eval_market1501
import yaml

from eval.datasets import load_market1501
from eval.degrade import (
    brightness_shift,
    gaussian_blur,
    low_resolution,
    motion_blur,
    random_occlusion,
)
from src.attributes import AttributeInference
from src.color import extract_colors, _COLOR_LABELS
from src.embed import EmbeddingExtractor
from src.quality import estimate_quality
from src.record import AttributeInfo, ColorInfo, FeatureRecord, QualityInfo

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

COLOR_CENTROIDS = list(_COLOR_LABELS) + ["unknown"]
COLOR_MAP = {name: i for i, name in enumerate(COLOR_CENTROIDS)}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_vif(X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Compute VIF from correlation matrix diagonal inversion."""
    R = np.corrcoef(X, rowvar=False)
    inv_R = np.linalg.pinv(R + 1e-6 * np.eye(len(feature_names)))
    vifs = np.diag(inv_R)
    return {name: float(round(vifs[i], 2)) for i, name in enumerate(feature_names)}


def extract_features_single(
    crop_bgr: np.ndarray,
    extractor: EmbeddingExtractor,
    attr_model: AttributeInference,
    cfg_quality: dict,
    cfg_color: dict,
) -> dict:
    h, w = crop_bgr.shape[:2]
    q_dict = estimate_quality(crop_bgr, [0, 0, w, h], cfg_quality)
    q_overall = float(q_dict["overall"])

    pred = attr_model.predict(crop_bgr)
    probs = pred["probs"]
    attr_vec = np.array([
        probs.get("is_male", 0.5),
        probs.get("has_bag", 0.0),
        probs.get("has_backpack", 0.0),
        probs.get("has_hat", 0.0),
        probs.get("has_longsleeves", 0.5),
        probs.get("has_longpants", 0.5),
        probs.get("has_longhair", 0.0),
        probs.get("has_coat_jacket", 0.0),
    ], dtype=np.float32)

    top_pt, bot_pt = pred["coords"]
    color_dict = extract_colors(crop_bgr, top_pt, bot_pt, cfg_color)
    upper_idx = COLOR_MAP.get(color_dict["upper"], COLOR_MAP["unknown"])
    lower_idx = COLOR_MAP.get(color_dict["lower"], COLOR_MAP["unknown"])

    emb = extractor.extract(crop_bgr)

    return {
        "embedding": emb,
        "quality": q_overall,
        "attr_probs": attr_vec,
        "color_upper": upper_idx,
        "color_lower": lower_idx,
        "color_upper_conf": float(color_dict["upper_conf"]),
        "color_lower_conf": float(color_dict["lower_conf"]),
    }


def compute_color_sim_vectorized(
    q_upper: np.ndarray, q_upper_conf: np.ndarray,
    q_lower: np.ndarray, q_lower_conf: np.ndarray,
    g_upper: np.ndarray, g_upper_conf: np.ndarray,
    g_lower: np.ndarray, g_lower_conf: np.ndarray,
) -> np.ndarray:
    """
    Vectorized computation of _color_similarity between Q queries and G gallery items.
    Returns (Q, G) float32 matrix.
    """
    unk_idx = COLOR_MAP["unknown"]

    # Upper region
    # Q: (N_q, 1), G: (1, N_g)
    qu = q_upper[:, None]
    gu = g_upper[None, :]
    qu_c = q_upper_conf[:, None]
    gu_c = g_upper_conf[None, :]
    min_conf_u = np.minimum(qu_c, gu_c)

    is_unk_u = (qu == unk_idx) | (gu == unk_idx)
    same_u = (qu == gu)

    sim_u = np.where(is_unk_u, 0.5, np.where(same_u, 0.5 + 0.5 * min_conf_u, 0.5 - 0.5 * min_conf_u))

    # Lower region
    ql = q_lower[:, None]
    gl = g_lower[None, :]
    ql_c = q_lower_conf[:, None]
    gl_c = g_lower_conf[None, :]
    min_conf_l = np.minimum(ql_c, gl_c)

    is_unk_l = (ql == unk_idx) | (gl == unk_idx)
    same_l = (ql == gl)

    sim_l = np.where(is_unk_l, 0.5, np.where(same_l, 0.5 + 0.5 * min_conf_l, 0.5 - 0.5 * min_conf_l))

    return (sim_u + sim_l) / 2.0


def compute_attr_sim_vectorized(q_attr: np.ndarray, g_attr: np.ndarray, chunk_size: int = 500) -> np.ndarray:
    """
    Mean 1 - |p_q - p_g| across 8 attributes, chunked along queries for low memory footprint.
    """
    N_q = len(q_attr)
    sims = []
    for start in range(0, N_q, chunk_size):
        chunk = q_attr[start : start + chunk_size]
        diff = np.abs(chunk[:, None, :] - g_attr[None, :, :]).mean(axis=-1)
        sims.append(1.0 - diff)
    return np.concatenate(sims, axis=0)


def build_or_load_gallery_cache(
    gallery_items: list,
    extractor: EmbeddingExtractor,
    attr_model: AttributeInference,
    cfg: dict,
    cache_path: Path,
) -> dict:
    if cache_path.exists():
        logger.info("Loading cached gallery features: %s", cache_path)
        data = np.load(str(cache_path))
        return {k: data[k] for k in data.files}

    logger.info("Extracting gallery features for %d items...", len(gallery_items))

    # Check for existing gallery embeddings cache
    existing_emb_path = Path("cache/embeddings_gallery_036e2e9f.npz")
    if existing_emb_path.exists():
        logger.info("Loading existing gallery embeddings from %s", existing_emb_path)
        d = np.load(str(existing_emb_path))
        embeddings = d["embeddings"]
    else:
        embeddings = None

    pids = []
    camids = []
    qualities = []
    attr_probs = []
    c_upper = []
    c_lower = []
    c_upper_conf = []
    c_lower_conf = []
    extracted_embs = []

    batch_size = 500
    total = len(gallery_items)
    for start in range(0, total, batch_size):
        chunk_items = gallery_items[start : start + batch_size]
        chunk_crops = []
        for path, pid, camid in chunk_items:
            img = cv2.imread(path)
            chunk_crops.append(img)
            pids.append(pid)
            camids.append(camid)

            h, w = img.shape[:2]
            q_dict = estimate_quality(img, [0, 0, w, h], cfg["quality"])
            pred = attr_model.predict(img)
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
            top_pt, bot_pt = pred["coords"]
            color_dict = extract_colors(img, top_pt, bot_pt, cfg["color"])

            qualities.append(float(q_dict["overall"]))
            attr_probs.append(attr_vec)
            c_upper.append(COLOR_MAP.get(color_dict["upper"], COLOR_MAP["unknown"]))
            c_lower.append(COLOR_MAP.get(color_dict["lower"], COLOR_MAP["unknown"]))
            c_upper_conf.append(float(color_dict["upper_conf"]))
            c_lower_conf.append(float(color_dict["lower_conf"]))

        if embeddings is None:
            extracted_embs.append(extractor.extract_batch(chunk_crops))

        logger.info("  Processed %d / %d gallery images...", min(start + batch_size, total), total)

    if embeddings is None:
        embeddings = np.concatenate(extracted_embs, axis=0)

    out_dict = {
        "embeddings": np.array(embeddings, dtype=np.float32),
        "pids": np.array(pids, dtype=np.int32),
        "camids": np.array(camids, dtype=np.int32),
        "quality": np.array(qualities, dtype=np.float32),
        "attr_probs": np.array(attr_probs, dtype=np.float32),
        "color_upper": np.array(c_upper, dtype=np.int32),
        "color_lower": np.array(c_lower, dtype=np.int32),
        "color_upper_conf": np.array(c_upper_conf, dtype=np.float32),
        "color_lower_conf": np.array(c_lower_conf, dtype=np.float32),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), **out_dict)
    logger.info("Saved gallery cache to %s", cache_path)
    return out_dict


def build_or_load_query_cache(
    query_items: list,
    cond_name: str,
    extractor: EmbeddingExtractor,
    attr_model: AttributeInference,
    cfg: dict,
    cache_path: Path,
) -> dict:
    if cache_path.exists():
        logger.info("Loading cached query features [%s]: %s", cond_name, cache_path)
        data = np.load(str(cache_path))
        return {k: data[k] for k in data.files}

    logger.info("Extracting query features for %d items [%s]...", len(query_items), cond_name)
    pids = []
    camids = []
    crops = []
    for pth, pid, cam in query_items:
        img = cv2.imread(pth)
        if img is None:
            continue

        # Apply degradation condition
        if cond_name == "clean" or cond_name == "none_0":
            deg = img
        elif cond_name.startswith("gaussian_blur_"):
            k = int(cond_name.split("_")[-1])
            deg = gaussian_blur(img, k)
        elif cond_name.startswith("motion_blur_"):
            k = int(cond_name.split("_")[-1])
            deg = motion_blur(img, k)
        elif cond_name.startswith("occlusion_"):
            pct = int(cond_name.split("_")[1].replace("pct", ""))
            deg = random_occlusion(img, pct / 100.0, seed=42)
        elif cond_name.startswith("lowres_"):
            pct = int(cond_name.split("_")[1].replace("pct", ""))
            deg = low_resolution(img, pct / 100.0)
        elif cond_name.startswith("brightness_"):
            delta = int(cond_name.split("_")[1])
            deg = brightness_shift(img, delta)
        else:
            deg = img

        crops.append(deg)
        pids.append(pid)
        camids.append(cam)

    embeddings = extractor.extract_batch(crops)

    qualities = []
    attr_probs = []
    c_upper = []
    c_lower = []
    c_upper_conf = []
    c_lower_conf = []

    for i, img in enumerate(crops):
        h, w = img.shape[:2]
        q_dict = estimate_quality(img, [0, 0, w, h], cfg["quality"])
        pred = attr_model.predict(img)
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
        top_pt, bot_pt = pred["coords"]
        color_dict = extract_colors(img, top_pt, bot_pt, cfg["color"])

        qualities.append(float(q_dict["overall"]))
        attr_probs.append(attr_vec)
        c_upper.append(COLOR_MAP.get(color_dict["upper"], COLOR_MAP["unknown"]))
        c_lower.append(COLOR_MAP.get(color_dict["lower"], COLOR_MAP["unknown"]))
        c_upper_conf.append(float(color_dict["upper_conf"]))
        c_lower_conf.append(float(color_dict["lower_conf"]))

    out_dict = {
        "embeddings": np.array(embeddings, dtype=np.float32),
        "pids": np.array(pids, dtype=np.int32),
        "camids": np.array(camids, dtype=np.int32),
        "quality": np.array(qualities, dtype=np.float32),
        "attr_probs": np.array(attr_probs, dtype=np.float32),
        "color_upper": np.array(c_upper, dtype=np.int32),
        "color_lower": np.array(c_lower, dtype=np.int32),
        "color_upper_conf": np.array(c_upper_conf, dtype=np.float32),
        "color_lower_conf": np.array(c_lower_conf, dtype=np.float32),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), **out_dict)
    logger.info("Saved query cache [%s] to %s", cond_name, cache_path)
    return out_dict


def fit_simplified_interaction_model(
    train_items: list,
    extractor: EmbeddingExtractor,
    attr_model: AttributeInference,
    cfg: dict,
    seed: int = 42,
) -> Tuple[Pipeline, Pipeline, Pipeline, dict]:
    """Fit Control, Additive, and Simplified Interaction models with L2 CV tuning."""
    logger.info("Fitting simplified interaction model on Market-1501 train split...")
    rng = np.random.RandomState(seed)

    pid_to_items = {}
    for pth, pid, cam in train_items:
        if pid not in (-1, 0):
            pid_to_items.setdefault(pid, []).append((pth, pid, cam))

    all_pids = sorted(list(pid_to_items.keys()))
    rng.shuffle(all_pids)
    split_idx = int(0.80 * len(all_pids))
    fit_pids = set(all_pids[:split_idx])
    pids_list = [p for p in fit_pids if len(pid_to_items[p]) >= 2]

    # Pre-extract clean items for fit identities to avoid redundant inference
    clean_dict = {}
    for p in pids_list:
        for it in pid_to_items[p]:
            img = cv2.imread(it[0])
            clean_dict[it[0]] = extract_features_single(img, extractor, attr_model, cfg["quality"], cfg["color"])

    def random_degrade(img: np.ndarray) -> np.ndarray:
        choice = rng.choice(["clean", "gaussian_blur", "motion_blur", "occlusion", "lowres", "brightness"])
        if choice == "gaussian_blur":
            return gaussian_blur(img, int(rng.choice([3, 5, 7, 11, 15])))
        elif choice == "motion_blur":
            return motion_blur(img, int(rng.choice([5, 9, 13, 15, 19])))
        elif choice == "occlusion":
            frac = float(rng.uniform(0.20, 0.70))
            return random_occlusion(img, frac, seed=int(rng.randint(0, 100000)))
        elif choice == "lowres":
            sf = float(rng.choice([0.5, 0.35, 0.25]))
            return low_resolution(img, sf)
        elif choice == "brightness":
            delta = int(rng.choice([-60, -40, -20, 20, 40, 60]))
            return brightness_shift(img, delta)
        return img.copy()

    records = []
    labels = []
    n_pos = 1500
    n_neg = 1500

    # Positives
    for _ in range(n_pos):
        pid = rng.choice(pids_list)
        items = pid_to_items[pid]
        idx_a, idx_b = rng.choice(len(items), size=2, replace=False)
        item_a, item_b = items[idx_a], items[idx_b]

        img_a = cv2.imread(item_a[0])
        img_a_deg = random_degrade(img_a)
        fa = extract_features_single(img_a_deg, extractor, attr_model, cfg["quality"], cfg["color"])
        fb = clean_dict[item_b[0]]

        emb_sim = float(fa["embedding"] @ fb["embedding"])
        attr_sim = 1.0 - float(np.mean(np.abs(fa["attr_probs"] - fb["attr_probs"])))

        # Color sim
        su = compute_color_sim_vectorized(
            np.array([fa["color_upper"]]), np.array([fa["color_upper_conf"]]),
            np.array([fa["color_lower"]]), np.array([fa["color_lower_conf"]]),
            np.array([fb["color_upper"]]), np.array([fb["color_upper_conf"]]),
            np.array([fb["color_lower"]]), np.array([fb["color_lower_conf"]]),
        )[0, 0]

        q_min = min(fa["quality"], fb["quality"])
        inv_q = 1.0 - q_min

        records.append({
            "emb_sim": emb_sim,
            "color_sim": float(su),
            "attr_sim": float(attr_sim),
            "q_min": q_min,
            "emb_sim_x_q_min": emb_sim * q_min,
            "attr_sim_x_inv_q": attr_sim * inv_q,
            "color_sim_x_inv_q": float(su) * inv_q,
        })
        labels.append(1)

    # Negatives
    for i in range(n_neg):
        pid_a = rng.choice(pids_list)
        items_a = pid_to_items[pid_a]
        item_a = items_a[rng.randint(0, len(items_a))]
        img_a = cv2.imread(item_a[0])
        img_a_deg = random_degrade(img_a)
        fa = extract_features_single(img_a_deg, extractor, attr_model, cfg["quality"], cfg["color"])

        pid_b = rng.choice(pids_list)
        while pid_b == pid_a:
            pid_b = rng.choice(pids_list)
        items_b = pid_to_items[pid_b]
        item_b = items_b[rng.randint(0, len(items_b))]
        fb = clean_dict[item_b[0]]

        emb_sim = float(fa["embedding"] @ fb["embedding"])
        attr_sim = 1.0 - float(np.mean(np.abs(fa["attr_probs"] - fb["attr_probs"])))
        su = compute_color_sim_vectorized(
            np.array([fa["color_upper"]]), np.array([fa["color_upper_conf"]]),
            np.array([fa["color_lower"]]), np.array([fa["color_lower_conf"]]),
            np.array([fb["color_upper"]]), np.array([fb["color_upper_conf"]]),
            np.array([fb["color_lower"]]), np.array([fb["color_lower_conf"]]),
        )[0, 0]

        q_min = min(fa["quality"], fb["quality"])
        inv_q = 1.0 - q_min

        records.append({
            "emb_sim": emb_sim,
            "color_sim": float(su),
            "attr_sim": float(attr_sim),
            "q_min": q_min,
            "emb_sim_x_q_min": emb_sim * q_min,
            "attr_sim_x_inv_q": attr_sim * inv_q,
            "color_sim_x_inv_q": float(su) * inv_q,
        })
        labels.append(0)

    y = np.array(labels, dtype=np.int32)

    # Simplified Interaction features
    feats_int = [
        "emb_sim", "color_sim", "attr_sim", "q_min",
        "emb_sim_x_q_min", "attr_sim_x_inv_q", "color_sim_x_inv_q"
    ]
    X_int = np.array([[r[f] for f in feats_int] for r in records], dtype=np.float32)

    # Additive features
    feats_add = ["emb_sim", "color_sim", "attr_sim", "q_min"]
    X_add = np.array([[r[f] for f in feats_add] for r in records], dtype=np.float32)

    # Control
    X_emb = np.array([[r["emb_sim"]] for r in records], dtype=np.float32)

    # Compute VIF
    vif_int = compute_vif(X_int, feats_int)
    vif_add = compute_vif(X_add, feats_add)

    # Fit models with L2 CV tuning (Cs=10, cv=5)
    scaler_emb = StandardScaler()
    lr_emb = LogisticRegression(penalty="l2", C=1.0, random_state=seed)
    pipe_emb = Pipeline([("scaler", scaler_emb), ("lr", lr_emb)]).fit(X_emb, y)

    scaler_add = StandardScaler()
    lrcv_add = LogisticRegressionCV(Cs=10, cv=5, penalty="l2", random_state=seed, max_iter=1000)
    pipe_add = Pipeline([("scaler", scaler_add), ("lr", lrcv_add)]).fit(X_add, y)

    scaler_int = StandardScaler()
    lrcv_int = LogisticRegressionCV(Cs=10, cv=5, penalty="l2", random_state=seed, max_iter=1000)
    pipe_int = Pipeline([("scaler", scaler_int), ("lr", lrcv_int)]).fit(X_int, y)

    best_C_add = float(lrcv_add.C_[0])
    best_C_int = float(lrcv_int.C_[0])

    coef_int = {f: float(c) for f, c in zip(feats_int, pipe_int.named_steps["lr"].coef_[0])}
    coef_add = {f: float(c) for f, c in zip(feats_add, pipe_add.named_steps["lr"].coef_[0])}

    info = {
        "feats_int": feats_int,
        "feats_add": feats_add,
        "vif_int": vif_int,
        "vif_add": vif_add,
        "best_C_int": best_C_int,
        "best_C_add": best_C_add,
        "coef_int": coef_int,
        "coef_add": coef_add,
    }

    # Save fitted model
    with open("models/fusion_lr.pkl", "wb") as f:
        pickle.dump(pipe_int, f)
    logger.info("Saved simplified interaction model to models/fusion_lr.pkl")

    return pipe_emb, pipe_add, pipe_int, info


def evaluate_condition_full_gallery(
    q_dict: dict,
    g_dict: dict,
    pipe_emb: Pipeline,
    pipe_add: Pipeline,
    pipe_int: Pipeline,
    feats_int: list,
    feats_add: list,
) -> dict:
    """Evaluate emb-only, additive, and interaction models on full gallery protocol."""
    q_emb = q_dict["embeddings"]
    g_emb = g_dict["embeddings"]
    q_pids = q_dict["pids"]
    g_pids = g_dict["pids"]
    q_cams = q_dict["camids"]
    g_cams = g_dict["camids"]

    # Vectorized similarity matrices
    emb_sim = q_emb @ g_emb.T  # (N_q, N_g)
    dist_emb = 1.0 - emb_sim

    # Torchreid eval for emb-only control
    cmc_emb, map_emb = eval_market1501(dist_emb, q_pids, g_pids, q_cams, g_cams, max_rank=50)

    # Color & Attribute similarity matrices
    color_sim = compute_color_sim_vectorized(
        q_dict["color_upper"], q_dict["color_upper_conf"],
        q_dict["color_lower"], q_dict["color_lower_conf"],
        g_dict["color_upper"], g_dict["color_upper_conf"],
        g_dict["color_lower"], g_dict["color_lower_conf"],
    )
    attr_sim = compute_attr_sim_vectorized(q_dict["attr_probs"], g_dict["attr_probs"])

    # Quality matrix
    q_q = q_dict["quality"][:, None]
    q_g = g_dict["quality"][None, :]
    q_min = np.minimum(q_q, q_g)
    inv_q = 1.0 - q_min

    # Additive model prediction via standardized pipeline:
    # logit = sum_j (w_j / scale_j) * (x_j - mean_j) + intercept
    s_add = pipe_add.named_steps["scaler"]
    lr_add = pipe_add.named_steps["lr"]
    w_add = lr_add.coef_[0] / s_add.scale_
    b_add = lr_add.intercept_[0] - np.sum(lr_add.coef_[0] * s_add.mean_ / s_add.scale_)

    logit_add = (w_add[0] * emb_sim +
                 w_add[1] * color_sim +
                 w_add[2] * attr_sim +
                 w_add[3] * q_min + b_add)
    scores_add = 1.0 / (1.0 + np.exp(-logit_add))
    dist_add = 1.0 - scores_add

    cmc_add, map_add = eval_market1501(dist_add, q_pids, g_pids, q_cams, g_cams, max_rank=50)

    # Interaction model prediction via standardized pipeline:
    s_int = pipe_int.named_steps["scaler"]
    lr_int = pipe_int.named_steps["lr"]
    w_int = lr_int.coef_[0] / s_int.scale_
    b_int = lr_int.intercept_[0] - np.sum(lr_int.coef_[0] * s_int.mean_ / s_int.scale_)

    logit_int = (w_int[0] * emb_sim +
                 w_int[1] * color_sim +
                 w_int[2] * attr_sim +
                 w_int[3] * q_min +
                 w_int[4] * (emb_sim * q_min) +
                 w_int[5] * (attr_sim * inv_q) +
                 w_int[6] * (color_sim * inv_q) + b_int)
    scores_int = 1.0 / (1.0 + np.exp(-logit_int))
    dist_int = 1.0 - scores_int

    cmc_int, map_int = eval_market1501(dist_int, q_pids, g_pids, q_cams, g_cams, max_rank=50)

    return {
        "emb_only": {"rank1": float(cmc_emb[0]), "rank5": float(cmc_emb[4]), "map": float(map_emb)},
        "additive": {"rank1": float(cmc_add[0]), "rank5": float(cmc_add[4]), "map": float(map_add)},
        "interaction": {"rank1": float(cmc_int[0]), "rank5": float(cmc_int[4]), "map": float(map_int)},
    }


def main() -> None:
    set_seed(42)
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    logger.info("Initializing models...")
    extractor = EmbeddingExtractor(cfg["model"])
    attr_model = AttributeInference(cfg["attributes"])
    attr_model.load()

    ds = load_market1501(cfg["eval"])

    # 1. Fit Simplified Interaction Model & Check VIF
    pipe_emb, pipe_add, pipe_int, fit_info = fit_simplified_interaction_model(
        ds["train"], extractor, attr_model, cfg, seed=42
    )

    # 2. Build or Load Gallery Cache (Full 15,913 images)
    gallery_cache_path = Path("cache/gallery_features_full.npz")
    g_dict = build_or_load_gallery_cache(ds["gallery"], extractor, attr_model, cfg, gallery_cache_path)

    # 3. Split Test Queries by Identity (Seed 42)
    query_pids_unique = sorted(list(set([it[1] for it in ds["query"]])))
    rng = np.random.RandomState(42)
    shuffled_pids = list(query_pids_unique)
    rng.shuffle(shuffled_pids)
    tune_pids = set(shuffled_pids[:len(shuffled_pids)//2])
    report_pids = set(shuffled_pids[len(shuffled_pids)//2:])

    tune_queries = [it for it in ds["query"] if it[1] in tune_pids]
    logger.info("Tune queries: %d images across %d IDs", len(tune_queries), len(tune_pids))

    # All 13 conditions
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

    all_results = {}
    for cond in conditions:
        q_cache_path = Path(f"cache/query_tune_{cond}.npz")
        q_dict = build_or_load_query_cache(tune_queries, cond, extractor, attr_model, cfg, q_cache_path)
        logger.info("Evaluating condition on full gallery protocol: %s", cond)
        res = evaluate_condition_full_gallery(
            q_dict, g_dict, pipe_emb, pipe_add, pipe_int,
            fit_info["feats_int"], fit_info["feats_add"]
        )
        all_results[cond] = res
        logger.info("  [%s] Control: Rank-1=%.4f mAP=%.4f | Interaction: Rank-1=%.4f mAP=%.4f",
                    cond, res["emb_only"]["rank1"], res["emb_only"]["map"],
                    res["interaction"]["rank1"], res["interaction"]["map"])

    # Output full summary json
    out_summary = {
        "fit_info": fit_info,
        "results": all_results,
    }
    with open("results/step1_corrected_val_metrics.json", "w") as f:
        json.dump(out_summary, f, indent=2)

    # Print pure numbers table
    print("\n" + "=" * 90)
    print("STEP 1: CORRECTED VALIDATION METRICS (FULL 15,913 GALLERY PROTOCOL, TUNE HALF)")
    print("=" * 90)
    print(f"L2 CV Tuned C: Additive={fit_info['best_C_add']}, Interaction={fit_info['best_C_int']}\n")

    print("Variance Inflation Factor (VIF) - Simplified Interaction Model:")
    for f, v in fit_info["vif_int"].items():
        print(f"  {f:<22}: VIF = {v:.2f}")

    print("\nStandardized Coefficients - Simplified Interaction Model:")
    for f, c in fit_info["coef_int"].items():
        print(f"  {f:<22}: {c:+.4f}")

    print("\n" + "-" * 90)
    print(f"{'Condition':<18} | {'Control (emb_sim only)':<22} | {'Additive Model':<22} | {'Interaction Model':<22}")
    print(f"{'':<18} | {'R-1    R-5    mAP':<22} | {'R-1    R-5    mAP':<22} | {'R-1    R-5    mAP':<22}")
    print("-" * 90)
    for cond in conditions:
        r_emb = all_results[cond]["emb_only"]
        r_add = all_results[cond]["additive"]
        r_int = all_results[cond]["interaction"]
        str_emb = f"{r_emb['rank1']:.3f}  {r_emb['rank5']:.3f}  {r_emb['map']:.3f}"
        str_add = f"{r_add['rank1']:.3f}  {r_add['rank5']:.3f}  {r_add['map']:.3f}"
        str_int = f"{r_int['rank1']:.3f}  {r_int['rank5']:.3f}  {r_int['map']:.3f}"
        print(f"{cond:<18} | {str_emb:<22} | {str_add:<22} | {str_int:<22}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
