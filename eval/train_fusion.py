"""
train_fusion.py -- Train and validate the quality-aware learned fusion model.

Trains a LogisticRegression model on degraded query and clean gallery pairs:
  Features: [q_quality_overall, g_quality_overall, q_blur, g_blur,
             q_visibility, g_visibility, emb_sim, color_sim, attr_sim]
  Target: 1 (same PID) vs 0 (different PID).

Training methodology:
1. Identity-disjoint split on Market-1501 train set: 80% fit PIDs, 20% val PIDs (seeded).
2. Degraded training queries: random degradation (blur, motion blur, occlusion, lowres, brightness).
3. Positives + Mined Hard Negatives (high embedding similarity, different PID).
4. Save trained estimator to models/fusion_lr.pkl.
5. Report validation ROC-AUC, PR-AUC, accuracy, precision, recall, F1, and feature coefficients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
from src.color import extract_colors
from src.embed import EmbeddingExtractor
from src.fusion import _attribute_similarity, _build_learned_features, _color_similarity
from src.quality import estimate_quality
from src.record import AttributeInfo, ColorInfo, FeatureRecord, QualityInfo

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def random_degrade_crop(crop_bgr: np.ndarray, rng: np.random.RandomState) -> Tuple[np.ndarray, str]:
    """Apply a randomly sampled surveillance degradation to a query crop."""
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


def extract_single_record(
    crop_bgr: np.ndarray,
    extractor: EmbeddingExtractor,
    attr_model: AttributeInference,
    cfg_quality: dict,
    cfg_color: dict,
    pid: int = -1,
    camid: int = -1,
) -> FeatureRecord:
    """Extract full FeatureRecord (embedding, quality, colors, attributes) for a crop."""
    h, w = crop_bgr.shape[:2]
    bbox = [0, 0, w, h]

    # Quality
    q_dict = estimate_quality(crop_bgr, bbox, cfg_quality)
    q_info = QualityInfo(
        blur=q_dict["blur"],
        visibility=q_dict["visibility"],
        size=q_dict["size"],
        overall=q_dict["overall"],
    )

    # Attributes & Color sampling coords
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

    # Colors
    top_pt, bot_pt = pred["coords"]
    color_dict = extract_colors(crop_bgr, top_pt, bot_pt, cfg_color)
    color_info = ColorInfo(
        upper=color_dict["upper"],
        lower=color_dict["lower"],
        upper_conf=color_dict["upper_conf"],
        lower_conf=color_dict["lower_conf"],
    )

    # Embedding
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


def build_pair_features(
    q_rec: FeatureRecord,
    g_rec: FeatureRecord,
    cfg_fusion: dict,
) -> List[float]:
    """Compute 9-D feature vector for a pair of FeatureRecords."""
    emb_sim = float(np.clip(float(np.dot(q_rec.embedding, g_rec.embedding)), 0.0, 1.0))
    color_sim = _color_similarity(q_rec.colors, g_rec.colors)
    attr_sim = _attribute_similarity(q_rec.attributes, g_rec.attributes)
    return _build_learned_features(q_rec, g_rec, emb_sim, color_sim, attr_sim, cfg_fusion)


def train_learned_fusion(
    cfg: dict,
    n_fit_pos: int = 2500,
    n_fit_neg: int = 2500,
    n_val_pos: int = 800,
    n_val_neg: int = 800,
    output_path: str = "models/fusion_lr.pkl",
) -> dict:
    """
    Main training and validation routine for quality-aware fusion LR model.
    """
    seed = int(cfg["eval"]["seed"])
    _set_seeds(seed)
    rng = np.random.RandomState(seed)

    logger.info("Initializing models for feature extraction...")
    extractor = EmbeddingExtractor(cfg["model"])
    attr_model = AttributeInference(cfg["attributes"])
    attr_model.load()

    # Load dataset
    ds = load_market1501(cfg["eval"])
    train_items = ds["train"]
    logger.info("Total Market-1501 train items: %d", len(train_items))

    # Group by PID
    pid_to_items: Dict[int, list] = {}
    for img_path, pid, camid in train_items:
        if pid == -1 or pid == 0:
            continue
        pid_to_items.setdefault(pid, []).append((img_path, pid, camid))

    all_pids = sorted(list(pid_to_items.keys()))
    rng.shuffle(all_pids)

    # Disjoint 80/20 train/val split by identity
    split_idx = int(0.80 * len(all_pids))
    fit_pids = set(all_pids[:split_idx])
    val_pids = set(all_pids[split_idx:])
    logger.info("Identity Split: %d fit PIDs, %d val PIDs", len(fit_pids), len(val_pids))

    def generate_pairs(
        target_pids: set,
        n_pos: int,
        n_neg: int,
        desc: str = "fit",
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Generating %d positive and %d negative pairs for %s split...", n_pos, n_neg, desc)
        pids_list = [p for p in target_pids if len(pid_to_items[p]) >= 2]

        X, y = [], []

        clean_cache: Dict[str, FeatureRecord] = {}

        def get_clean_record(item_tuple: tuple) -> FeatureRecord:
            path, pid, camid = item_tuple
            if path not in clean_cache:
                crop = cv2.imread(path)
                clean_cache[path] = extract_single_record(
                    crop, extractor, attr_model, cfg["quality"], cfg["color"],
                    pid=pid, camid=camid
                )
            return clean_cache[path]

        # 1. Positives
        count_pos = 0
        attempts = 0
        while count_pos < n_pos and attempts < n_pos * 10:
            attempts += 1
            pid = rng.choice(pids_list)
            items = pid_to_items[pid]
            idx_a, idx_b = rng.choice(len(items), size=2, replace=False)
            item_a, item_b = items[idx_a], items[idx_b]

            # Query: crop A with random degradation
            crop_a = cv2.imread(item_a[0])
            if crop_a is None:
                continue
            crop_a_deg, _ = random_degrade_crop(crop_a, rng)
            q_rec = extract_single_record(
                crop_a_deg, extractor, attr_model, cfg["quality"], cfg["color"],
                pid=item_a[1], camid=item_a[2]
            )

            # Gallery: crop B clean (cached)
            g_rec = get_clean_record(item_b)

            feat = build_pair_features(q_rec, g_rec, cfg["fusion"])
            X.append(feat)
            y.append(1)
            count_pos += 1
            if count_pos % 500 == 0:
                logger.info("  %s positives: %d / %d", desc, count_pos, n_pos)

        # 2. Negatives (50% random, 50% hard negatives with high emb similarity)
        count_neg = 0
        while count_neg < n_neg:
            pid_a = rng.choice(pids_list)
            list_a = pid_to_items[pid_a]
            item_a = list_a[rng.randint(0, len(list_a))]

            crop_a = cv2.imread(item_a[0])
            if crop_a is None:
                continue

            crop_a_deg, _ = random_degrade_crop(crop_a, rng)
            q_rec = extract_single_record(
                crop_a_deg, extractor, attr_model, cfg["quality"], cfg["color"],
                pid=item_a[1], camid=item_a[2]
            )

            if count_neg < (n_neg // 2):
                # Standard random negative
                pid_b = rng.choice(pids_list)
                while pid_b == pid_a:
                    pid_b = rng.choice(pids_list)
                list_b = pid_to_items[pid_b]
                item_b = list_b[rng.randint(0, len(list_b))]
                g_rec = get_clean_record(item_b)
            else:
                # Mined hard negative: candidate with highest embedding similarity from 8 candidates
                best_sim = -1.0
                best_rec = None
                for _ in range(8):
                    pid_cand = rng.choice(pids_list)
                    while pid_cand == pid_a:
                        pid_cand = rng.choice(pids_list)
                    cand_list = pid_to_items[pid_cand]
                    cand_item = cand_list[rng.randint(0, len(cand_list))]
                    cand_rec = get_clean_record(cand_item)
                    sim = float(np.dot(q_rec.embedding, cand_rec.embedding))
                    if sim > best_sim:
                        best_sim = sim
                        best_rec = cand_rec
                g_rec = best_rec

            feat = build_pair_features(q_rec, g_rec, cfg["fusion"])
            X.append(feat)
            y.append(0)
            count_neg += 1
            if count_neg % 500 == 0:
                logger.info("  %s negatives: %d / %d (hard negatives included)", desc, count_neg, n_neg)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    # Generate Fit Pairs
    X_fit, y_fit = generate_pairs(fit_pids, n_fit_pos, n_fit_neg, desc="fit")
    # Generate Val Pairs
    X_val, y_val = generate_pairs(val_pids, n_val_pos, n_val_neg, desc="val")

    logger.info("Fitting Logistic Regression on %d pairs...", len(y_fit))
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs")
    lr.fit(X_fit, y_fit)

    # Evaluate on Validation Split
    val_probs = lr.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    val_auc = float(roc_auc_score(y_val, val_probs))
    val_ap = float(average_precision_score(y_val, val_probs))
    val_acc = float(accuracy_score(y_val, val_preds))
    val_prec = float(precision_score(y_val, val_preds, zero_division=0))
    val_rec = float(recall_score(y_val, val_preds, zero_division=0))
    val_f1 = float(f1_score(y_val, val_preds, zero_division=0))

    feature_names = cfg["fusion"]["learned"]["features"]
    coefs = lr.coef_[0].tolist()
    intercept = float(lr.intercept_[0])

    coef_dict = {name: round(c, 4) for name, c in zip(feature_names, coefs)}

    # Save model
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(lr, f)
    logger.info("Model saved to: %s", out_file)

    report = {
        "val_auc": val_auc,
        "val_ap": val_ap,
        "val_accuracy": val_acc,
        "val_precision": val_prec,
        "val_recall": val_rec,
        "val_f1": val_f1,
        "intercept": intercept,
        "coefficients": coef_dict,
        "n_fit_samples": len(y_fit),
        "n_val_samples": len(y_val),
        "fit_pids_count": len(fit_pids),
        "val_pids_count": len(val_pids),
        "git_commit": _git_commit(),
        "seed": seed,
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAFEPASS Quality-Aware Learned Fusion")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="models/fusion_lr.pkl")
    parser.add_argument("--fit-pos", type=int, default=2000, help="Number of positive fit pairs")
    parser.add_argument("--fit-neg", type=int, default=2000, help="Number of negative fit pairs")
    parser.add_argument("--val-pos", type=int, default=600, help="Number of positive val pairs")
    parser.add_argument("--val-neg", type=int, default=600, help="Number of negative val pairs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    report = train_learned_fusion(
        cfg=cfg,
        n_fit_pos=args.fit_pos,
        n_fit_neg=args.fit_neg,
        n_val_pos=args.val_pos,
        n_val_neg=args.val_neg,
        output_path=args.output,
    )

    print("\n" + "=" * 60)
    print("SAFEPASS QUALITY-AWARE LEARNED FUSION -- VALIDATION REPORT")
    print("=" * 60)
    print(f"Validation ROC-AUC : {report['val_auc']:.4f}")
    print(f"Validation PR-AUC  : {report['val_ap']:.4f}")
    print(f"Validation Accuracy: {report['val_accuracy']:.4f}")
    print(f"Validation F1-Score: {report['val_f1']:.4f}")
    print(f"Validation P / R   : {report['val_precision']:.4f} / {report['val_recall']:.4f}")
    print(f"Intercept          : {report['intercept']:.4f}")
    print("-" * 60)
    print("Learned Coefficients:")
    for feat, coef in report["coefficients"].items():
        print(f"  {feat:<20}: {coef:+.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
