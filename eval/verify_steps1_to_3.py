"""
verify_steps1_to_3.py

1. Fits interaction model on degraded-query pairs from the FIT split (test IDs OSNet never saw).
2. Computes and reports VIF with mean-centered main effects before forming interaction terms.
3. Re-evaluates occlusion_30pct, occlusion_50pct, occlusion_70pct on the new realistic textured occlusion.
4. Reports split sizes, VIF comparison (uncentered vs centered), and re-run occlusion metrics.
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
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchreid.reid.metrics.rank import eval_market1501
import yaml

from eval.datasets import load_market1501
from eval.degrade import random_occlusion
from src.attributes import AttributeInference
from src.color import extract_colors, _COLOR_LABELS
from src.embed import EmbeddingExtractor
from src.quality import estimate_quality
from src.record import AttributeInfo, ColorInfo, FeatureRecord, QualityInfo

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

COLOR_CENTROIDS = list(_COLOR_LABELS) + ["unknown"]
COLOR_MAP = {name: i for i, name in enumerate(COLOR_CENTROIDS)}


def compute_vif(X: np.ndarray, feature_names: list) -> dict:
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
    up_conf = float(color_dict["upper_conf"])
    lo_conf = float(color_dict["lower_conf"])

    emb = extractor.extract(crop_bgr)

    return {
        "emb": emb.astype(np.float32),
        "q": q_overall,
        "attr": attr_vec,
        "up_idx": upper_idx,
        "lo_idx": lower_idx,
        "up_conf": up_conf,
        "lo_conf": lo_conf,
    }


def compute_pair_features_raw(f1: dict, f2: dict) -> dict:
    emb_sim = float(np.dot(f1["emb"], f2["emb"]))
    attr_sim = float(1.0 - np.mean(np.abs(f1["attr"] - f2["attr"])))
    color_sim = float(
        0.5 * (1.0 if f1["up_idx"] == f2["up_idx"] else 0.0) * min(f1["up_conf"], f2["up_conf"]) +
        0.5 * (1.0 if f1["lo_idx"] == f2["lo_idx"] else 0.0) * min(f1["lo_conf"], f2["lo_conf"])
    )
    q_min = float(min(f1["q"], f2["q"]))
    return {
        "emb_sim": emb_sim,
        "color_sim": color_sim,
        "attr_sim": attr_sim,
        "q_min": q_min,
    }


def main():
    cfg = yaml.safe_load(open("configs/default.yaml", "r"))
    ds = load_market1501(cfg["eval"])

    with open("results/split.json", "r") as f:
        split_info = json.load(f)

    fit_pids = set(split_info["fit"]["pids"])
    tune_pids = set(split_info["tune"]["pids"])

    extractor = EmbeddingExtractor(cfg["model"])
    attr_model = AttributeInference(cfg["attributes"])
    attr_model.load()

    # Load gallery cache
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

    logger.info("Building training pairs from FIT split (250 IDs OSNet never saw)...")
    fit_gallery = [it for it in ds["gallery"] if it[1] in fit_pids]
    fit_queries = [it for it in ds["query"] if it[1] in fit_pids]

    gal_by_pid = {}
    for path, pid, cam in fit_gallery:
        gal_by_pid.setdefault(pid, []).append((path, pid, cam))

    rng = random.Random(42)
    pairs = []
    
    # Sample 1500 positive pairs and 1500 negative pairs
    pids_list = list(fit_pids)
    for _ in range(1500):
        # Positive
        pid = rng.choice(pids_list)
        if pid in gal_by_pid and len(gal_by_pid[pid]) > 0:
            q_candidates = [it for it in fit_queries if it[1] == pid]
            if q_candidates:
                q_item = rng.choice(q_candidates)
                g_item = rng.choice(gal_by_pid[pid])
                pairs.append((q_item, g_item, 1))

        # Negative
        pid1, pid2 = rng.sample(pids_list, 2)
        q_cand = [it for it in fit_queries if it[1] == pid1]
        g_cand = gal_by_pid.get(pid2, [])
        if q_cand and g_cand:
            pairs.append((rng.choice(q_cand), rng.choice(g_cand), 0))

    logger.info("Extracting features for %d FIT pairs...", len(pairs))
    raw_rows = []
    y_list = []

    # Cache single crops to speed up pair generation
    crop_cache = {}
    def get_feat(path, is_query=False):
        if path not in crop_cache:
            img = cv2.imread(path)
            if is_query and rng.random() > 0.4:
                # Random degradation during fit pair generation
                frac = rng.choice([0.3, 0.5])
                img = random_occlusion(img, frac, seed=rng.randint(0, 100000))
            crop_cache[path] = extract_features_single(img, extractor, attr_model, cfg["quality"], cfg["color"])
        return crop_cache[path]

    for (q_path, q_pid, _), (g_path, g_pid, _), label in pairs:
        f1 = get_feat(q_path, is_query=True)
        f2 = get_feat(g_path, is_query=False)
        r = compute_pair_features_raw(f1, f2)
        raw_rows.append(r)
        y_list.append(label)

    # Convert to array
    raw_emb = np.array([r["emb_sim"] for r in raw_rows])
    raw_col = np.array([r["color_sim"] for r in raw_rows])
    raw_attr = np.array([r["attr_sim"] for r in raw_rows])
    raw_q = np.array([r["q_min"] for r in raw_rows])
    y_arr = np.array(y_list)

    # 1. Uncentered VIF (for baseline comparison)
    uncentered_matrix = np.column_stack([
        raw_emb,
        raw_col,
        raw_attr,
        raw_q,
        raw_emb * raw_q,
        raw_attr * (1.0 - raw_q),
        raw_col * (1.0 - raw_q),
    ])
    feat_names = [
        "emb_sim",
        "color_sim",
        "attr_sim",
        "q_min",
        "emb_sim_x_q",
        "attr_sim_x_inv_q",
        "color_sim_x_inv_q",
    ]
    vif_uncentered = compute_vif(uncentered_matrix, feat_names)

    # 2. Mean-centered features & interactions
    mu_emb = float(np.mean(raw_emb))
    mu_col = float(np.mean(raw_col))
    mu_attr = float(np.mean(raw_attr))
    mu_q = float(np.mean(raw_q))
    mu_inv_q = float(np.mean(1.0 - raw_q))

    c_emb = raw_emb - mu_emb
    c_col = raw_col - mu_col
    c_attr = raw_attr - mu_attr
    c_q = raw_q - mu_q
    c_inv_q = (1.0 - raw_q) - mu_inv_q

    int_emb_q = c_emb * c_q
    int_attr_inv_q = c_attr * c_inv_q
    int_col_inv_q = c_col * c_inv_q

    centered_matrix = np.column_stack([
        c_emb,
        c_col,
        c_attr,
        c_q,
        int_emb_q,
        int_attr_inv_q,
        int_col_inv_q,
    ])
    vif_centered = compute_vif(centered_matrix, feat_names)

    # Fit LogisticRegressionCV with mean-centering
    pipe_centered = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(Cs=10, cv=5, penalty="l2", scoring="roc_auc", random_state=42, max_iter=1000)),
    ])
    pipe_centered.fit(centered_matrix, y_arr)
    clf = pipe_centered.named_steps["clf"]
    best_C = float(clf.C_[0])
    coefs = {feat_names[i]: float(clf.coef_[0][i]) for i in range(len(feat_names))}

    # Re-evaluate occlusion conditions on TUNE queries with new realistic occlusion
    tune_queries = [it for it in ds["query"] if it[1] in tune_pids]
    logger.info("Evaluating fixed occlusion conditions on %d TUNE queries...", len(tune_queries))

    occlusion_conditions = ["occlusion_30pct", "occlusion_50pct", "occlusion_70pct"]
    occ_results = {}

    for cond in occlusion_conditions:
        frac = float(cond.split("_")[1].replace("pct", "")) / 100.0
        q_emb_list = []
        q_pids_list = []
        q_camids_list = []
        q_quality_list = []
        q_attr_list = []
        q_up_idx_list = []
        q_lo_idx_list = []
        q_up_conf_list = []
        q_lo_conf_list = []

        for q_path, q_pid, q_cam in tune_queries:
            img = cv2.imread(q_path)
            deg = random_occlusion(img, frac, seed=None)
            f = extract_features_single(deg, extractor, attr_model, cfg["quality"], cfg["color"])
            q_emb_list.append(f["emb"])
            q_pids_list.append(q_pid)
            q_camids_list.append(q_cam)
            q_quality_list.append(f["q"])
            q_attr_list.append(f["attr"])
            q_up_idx_list.append(f["up_idx"])
            q_lo_idx_list.append(f["lo_idx"])
            q_up_conf_list.append(f["up_conf"])
            q_lo_conf_list.append(f["lo_conf"])

        q_emb_arr = np.array(q_emb_list)
        q_pids_arr = np.array(q_pids_list)
        q_camids_arr = np.array(q_camids_list)
        q_q_arr = np.array(q_quality_list)
        q_attr_arr = np.array(q_attr_list)
        q_up_idx_arr = np.array(q_up_idx_list)
        q_lo_idx_arr = np.array(q_lo_idx_list)
        q_up_conf_arr = np.array(q_up_conf_list)
        q_lo_conf_arr = np.array(q_lo_conf_list)

        # Baseline control (OSNet embedding distance matrix)
        distmat_control = 1.0 - np.dot(q_emb_arr, g_emb.T)
        cmc_ctrl, map_ctrl = eval_market1501(distmat_control, q_pids_arr, g_pids, q_camids_arr, g_camids, max_rank=10)

        # Interaction model re-ranking (top-100)
        N_q = len(q_emb_arr)
        N_g = len(g_emb)
        distmat_int = distmat_control.copy()

        scaler = pipe_centered.named_steps["scaler"]
        clf_model = pipe_centered.named_steps["clf"]

        for i in range(N_q):
            top100_idx = np.argsort(distmat_control[i])[:100]
            emb_sims = np.dot(g_emb[top100_idx], q_emb_arr[i])
            q_min_vals = np.minimum(q_q_arr[i], g_q[top100_idx])
            attr_diffs = np.mean(np.abs(g_attr[top100_idx] - q_attr_arr[i]), axis=1)
            attr_sims = 1.0 - attr_diffs

            up_match = (g_up_idx[top100_idx] == q_up_idx_arr[i]).astype(float)
            lo_match = (g_lo_idx[top100_idx] == q_lo_idx_arr[i]).astype(float)
            up_min_c = np.minimum(q_up_conf_arr[i], g_up_conf[top100_idx])
            lo_min_c = np.minimum(q_lo_conf_arr[i], g_lo_conf[top100_idx])
            col_sims = 0.5 * up_match * up_min_c + 0.5 * lo_match * lo_min_c

            # Center with stored means
            c_emb_i = emb_sims - mu_emb
            c_col_i = col_sims - mu_col
            c_attr_i = attr_sims - mu_attr
            c_q_i = q_min_vals - mu_q
            c_inv_q_i = (1.0 - q_min_vals) - mu_inv_q

            X_cand = np.column_stack([
                c_emb_i,
                c_col_i,
                c_attr_i,
                c_q_i,
                c_emb_i * c_q_i,
                c_attr_i * c_inv_q_i,
                c_col_i * c_inv_q_i,
            ])
            X_cand_scaled = scaler.transform(X_cand)
            scores = clf_model.predict_proba(X_cand_scaled)[:, 1]
            distmat_int[i, top100_idx] = 1.0 - scores

        cmc_int, map_int = eval_market1501(distmat_int, q_pids_arr, g_pids, q_camids_arr, g_camids, max_rank=10)
        occ_results[cond] = {
            "control": {"rank1": float(cmc_ctrl[0]), "rank5": float(cmc_ctrl[4]), "map": float(map_ctrl)},
            "interaction": {"rank1": float(cmc_int[0]), "rank5": float(cmc_int[4]), "map": float(map_int)},
        }

    # Save verification output
    out = {
        "vif_uncentered": vif_uncentered,
        "vif_centered": vif_centered,
        "best_C": best_C,
        "coefs": coefs,
        "means": {
            "mu_emb": mu_emb,
            "mu_col": mu_col,
            "mu_attr": mu_attr,
            "mu_q": mu_q,
            "mu_inv_q": mu_inv_q,
        },
        "occlusion_results": occ_results,
    }
    with open("results/step1_to_3_verification.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 2: VIF REPORT (UNCENTERED VS MEAN-CENTERED INTERACTION MODEL)")
    print("=" * 80)
    print(f"{'Feature':<22} | {'Uncentered VIF':<16} | {'Mean-Centered VIF':<18} | Status")
    print("-" * 80)
    for f in feat_names:
        u = vif_uncentered[f]
        c = vif_centered[f]
        status = "PASSED (< 10)" if c < 10 else "HIGH (>= 10)"
        print(f"{f:<22} | {u:<16.2f} | {c:<18.2f} | {status}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("STEP 1: RE-RUN OCCLUSION CONDITIONS WITH REALISTIC TEXTURED OCCLUDER")
    print("=" * 80)
    print(f"{'Condition':<18} | {'Control (emb_sim only)':<24} | {'Interaction Model':<24}")
    print(f"{'':<18} | {'R-1      R-5      mAP':<24} | {'R-1      R-5      mAP':<24}")
    print("-" * 80)
    for cond in occlusion_conditions:
        ctrl = occ_results[cond]["control"]
        inter = occ_results[cond]["interaction"]
        str_ctrl = f"{ctrl['rank1']:.4f}   {ctrl['rank5']:.4f}   {ctrl['map']:.4f}"
        str_inter = f"{inter['rank1']:.4f}   {inter['rank5']:.4f}   {inter['map']:.4f}"
        print(f"{cond:<18} | {str_ctrl:<24} | {str_inter:<24}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
