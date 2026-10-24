"""
fusion.py -- Quality-Aware Fusion (RESEARCH CORE).

Four strategies behind one interface:
  A. osnet_only       -- cosine similarity only.
  B. fixed_weight     -- config-specified fixed weights.
  C. adaptive_rule    -- quality-adaptive softmax gating.
  D. adaptive_learned -- logistic regression / MLP trained on quality features.

All methods return per-modality sub-scores and the weights used so the
dashboard can display them later.

Hard mismatch rule: configurable penalty for high-confidence disagreements on
both sides.  Clothing change is possible in long temporal gaps, so this penalty
is ablation-controlled via config.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


# ── color similarity ─────────────────────────────────────────────────────────

def _color_similarity(c_q: Any, c_g: Any) -> float:
    """
    Similarity between two ColorInfo objects.

    Unknown regions are treated as neutral (0.5, not penalised).
    Matching labels -> [0.5, 1.0]; mismatching -> [0.0, 0.5].
    """
    def _region(lq, cq, lg, cg):
        if lq == "unknown" or lg == "unknown":
            return 0.5
        if lq == lg:
            return 0.5 + 0.5 * min(cq, cg)
        return 0.5 - 0.5 * min(cq, cg)

    upper = _region(c_q.upper, c_q.upper_conf, c_g.upper, c_g.upper_conf)
    lower = _region(c_q.lower, c_q.lower_conf, c_g.lower, c_g.lower_conf)
    return float((upper + lower) / 2.0)


def _attribute_similarity(a_q: Any, a_g: Any) -> float:
    """
    Probability agreement per attribute, averaged.

    score = 1 - |p_q - p_g| per attribute.
    """
    fields = list(a_q.__dataclass_fields__.keys())
    sims = [1.0 - abs(getattr(a_q, f) - getattr(a_g, f)) for f in fields]
    return float(np.mean(sims))


def _mismatch_penalty(c_q: Any, c_g: Any, cfg_adp: dict) -> float:
    """
    Hard penalty for high-confidence color mismatches on both sides.

    Configurable so ablations can disable it entirely.
    """
    penalty_per = float(cfg_adp["mismatch_penalty"])
    both_high = bool(cfg_adp["mismatch_requires_both_high"])
    conf_thresh = float(cfg_adp["mismatch_confidence_threshold"])

    if penalty_per <= 0:
        return 0.0

    total = 0.0
    for (lq, cq), (lg, cg) in [
        ((c_q.upper, c_q.upper_conf), (c_g.upper, c_g.upper_conf)),
        ((c_q.lower, c_q.lower_conf), (c_g.lower, c_g.lower_conf)),
    ]:
        if lq == "unknown" or lg == "unknown":
            continue
        if lq != lg:
            if both_high:
                if cq >= conf_thresh and cg >= conf_thresh:
                    total += penalty_per
            else:
                total += penalty_per * min(cq, cg)
    return float(total)


# ── quality gating ───────────────────────────────────────────────────────────

def _quality_gate(q_score: float, temperature: float) -> float:
    """
    Logistic gating: high quality -> weight near 1 (trust embedding).

    gate = sigmoid(temperature * (q - 0.5))
    """
    x = temperature * (q_score - 0.5)
    return float(1.0 / (1.0 + math.exp(-x)))


def _aggregate_quality(q_q: float, q_g: float, method: str) -> float:
    """Aggregate query and gallery quality scores (min or product)."""
    if method == "min":
        return min(q_q, q_g)
    elif method == "product":
        return math.sqrt(max(q_q * q_g, 0.0))
    raise ValueError(f"Unknown quality_aggregation: {method!r}")


# ── learned feature vector ───────────────────────────────────────────────────

def _build_learned_features(q, g, emb_sim, color_sim, attr_sim, cfg) -> list:
    """Build feature vector for the learned fusion model."""
    q_q = float(q.quality.overall)
    q_g = float(g.quality.overall)
    q_min = min(q_q, q_g)
    q_prod = math.sqrt(max(q_q * q_g, 0.0))
    inv_q = 1.0 - q_min

    feature_map = {
        "q_quality_overall": q_q,
        "g_quality_overall": q_g,
        "q_blur": float(q.quality.blur),
        "g_blur": float(g.quality.blur),
        "q_visibility": float(q.quality.visibility),
        "g_visibility": float(g.quality.visibility),
        "emb_sim": float(emb_sim),
        "color_sim": float(color_sim),
        "attr_sim": float(attr_sim),
        "q_min": q_min,
        "q_prod": q_prod,
        "emb_sim_x_q_min": float(emb_sim * q_min),
        "emb_sim_x_q_prod": float(emb_sim * q_prod),
        "color_sim_x_inv_q": float(color_sim * inv_q),
        "attr_sim_x_inv_q": float(attr_sim * inv_q),
    }
    return [feature_map[f] for f in cfg["learned"]["features"]]


# ── public interface ──────────────────────────────────────────────────────────

def fuse(
    query_record: Any,
    gallery_record: Any,
    emb_score: float,
    cfg: dict,
    method: Optional[str] = None,
    learned_model: Optional[Any] = None,
    mismatch_penalty_override: Optional[float] = None,
) -> dict:
    """
    Compute a fusion score between a query and a gallery FeatureRecord.

    Parameters
    ----------
    query_record, gallery_record : FeatureRecord
    emb_score       : float  cosine similarity from FAISS.
    cfg             : ``fusion`` section of default.yaml.
    method          : override cfg['method'] for this call.
    learned_model   : fitted sklearn estimator for 'adaptive_learned'.
    mismatch_penalty_override : override penalty magnitude (ablation).

    Returns
    -------
    dict:
      score      : float in [0,1]
      weights    : dict  per-modality weights
      sub_scores : dict  {emb, color, attr}
      method     : str
      penalty    : float  mismatch penalty applied
    """
    method = method or cfg["method"]
    emb_sim = float(np.clip(emb_score, 0.0, 1.0))
    color_sim = _color_similarity(query_record.colors, gallery_record.colors)
    attr_sim = _attribute_similarity(query_record.attributes, gallery_record.attributes)
    sub_scores = {"emb": emb_sim, "color": color_sim, "attr": attr_sim}

    adp = dict(cfg["adaptive"])
    if mismatch_penalty_override is not None:
        adp = dict(adp)
        adp["mismatch_penalty"] = mismatch_penalty_override

    # -- dispatch ----------------------------------------------------------------
    if method == "osnet_only":
        weights = {"embedding": 1.0, "color": 0.0, "attributes": 0.0}
        raw = emb_sim
        penalty = 0.0

    elif method == "fixed_weight":
        fw = cfg["fixed_weights"]
        w_emb = float(fw["embedding"])
        w_col = float(fw["color"])
        w_att = float(fw["attributes"])
        weights = {"embedding": w_emb, "color": w_col, "attributes": w_att}
        raw = w_emb * emb_sim + w_col * color_sim + w_att * attr_sim
        penalty = _mismatch_penalty(query_record.colors, gallery_record.colors, adp)

    elif method == "adaptive_rule":
        agg_method = str(adp["quality_aggregation"])
        temp = float(adp["temperature"])
        q_q = float(query_record.quality.overall)
        q_g = float(gallery_record.quality.overall)
        q_agg = _aggregate_quality(q_q, q_g, agg_method)
        gate = _quality_gate(q_agg, temp)
        # High quality -> trust embedding; low quality -> trust color + attrs.
        w_emb = gate
        residual = 1.0 - gate
        w_col = 0.5 * residual
        w_att = 0.5 * residual
        weights = {"embedding": w_emb, "color": w_col, "attributes": w_att}
        raw = w_emb * emb_sim + w_col * color_sim + w_att * attr_sim
        penalty = _mismatch_penalty(query_record.colors, gallery_record.colors, adp)

    elif method == "adaptive_learned":
        if learned_model is None:
            logger.warning("No learned model; falling back to adaptive_rule.")
            return fuse(
                query_record, gallery_record, emb_score, cfg,
                method="adaptive_rule", mismatch_penalty_override=mismatch_penalty_override,
            )
        feat = _build_learned_features(
            query_record, gallery_record, emb_sim, color_sim, attr_sim, cfg,
        )
        raw = float(learned_model.predict_proba([feat])[0][1])
        weights = {"embedding": None, "color": None, "attributes": None}
        penalty = 0.0

    else:
        raise ValueError(f"Unknown fusion method: {method!r}")

    final_score = float(np.clip(raw - penalty, 0.0, 1.0))
    return {
        "score": final_score,
        "weights": weights,
        "sub_scores": sub_scores,
        "method": method,
        "penalty": penalty,
    }


def rerank(
    candidates: list,
    query_record: Any,
    cfg: dict,
    method: Optional[str] = None,
    learned_model: Optional[Any] = None,
    mismatch_penalty_override: Optional[float] = None,
) -> list:
    """
    Re-rank a FAISS candidate list using fusion scoring.

    Parameters
    ----------
    candidates   : list of dicts from GalleryIndex.search().
    query_record : FeatureRecord.
    cfg          : fusion section of default.yaml.

    Returns
    -------
    Same list, augmented with 'fusion_result' and 'score', sorted desc.
    """
    for cand in candidates:
        result = fuse(
            query_record=query_record,
            gallery_record=cand["record"],
            emb_score=cand["emb_score"],
            cfg=cfg,
            method=method,
            learned_model=learned_model,
            mismatch_penalty_override=mismatch_penalty_override,
        )
        cand["fusion_result"] = result
        cand["score"] = result["score"]
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
