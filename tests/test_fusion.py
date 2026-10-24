"""
test_fusion.py -- Unit tests for fusion.py.

Tests the fuse() interface contract:
- Score in [0, 1] for all methods.
- Keys present in return dict.
- Adaptive weight sum approximately 1.
- Higher quality -> more weight on embedding (adaptive_rule).
- Hard mismatch penalty is applied and configurable.
"""
import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.record import FeatureRecord, ColorInfo, AttributeInfo, QualityInfo
from src.fusion import fuse, rerank, _quality_gate, _aggregate_quality


# Minimal fusion config
_CFG = {
    "method": "adaptive_rule",
    "fixed_weights": {"embedding": 0.6, "color": 0.2, "attributes": 0.2},
    "adaptive": {
        "quality_aggregation": "min",
        "temperature": 2.0,
        "mismatch_penalty": 0.15,
        "mismatch_requires_both_high": True,
        "mismatch_confidence_threshold": 0.6,
    },
    "learned": {
        "model_path": "models/fusion_lr.pkl",
        "features": [
            "q_quality_overall", "g_quality_overall",
            "q_blur", "g_blur", "q_visibility", "g_visibility",
            "emb_sim", "color_sim", "attr_sim",
        ],
    },
}


def _make_record(
    quality_overall: float = 0.8,
    upper: str = "blue",
    lower: str = "black",
    upper_conf: float = 0.8,
    lower_conf: float = 0.8,
) -> FeatureRecord:
    emb = np.random.RandomState(0).randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return FeatureRecord(
        track_id=1, camera_id="cam0", timestamp=0.0, bbox=[0, 0, 64, 128],
        embedding=emb,
        colors=ColorInfo(upper=upper, lower=lower, upper_conf=upper_conf, lower_conf=lower_conf),
        attributes=AttributeInfo(),
        quality=QualityInfo(blur=quality_overall, visibility=1.0, size=1.0, overall=quality_overall),
    )


class TestFuseInterface:
    @pytest.mark.parametrize("method", ["osnet_only", "fixed_weight", "adaptive_rule"])
    def test_score_in_bounds(self, method):
        q = _make_record()
        g = _make_record()
        result = fuse(q, g, emb_score=0.7, cfg=_CFG, method=method)
        assert 0.0 <= result["score"] <= 1.0, f"[{method}] score={result['score']}"

    @pytest.mark.parametrize("method", ["osnet_only", "fixed_weight", "adaptive_rule"])
    def test_required_keys(self, method):
        q = _make_record()
        g = _make_record()
        result = fuse(q, g, emb_score=0.5, cfg=_CFG, method=method)
        for key in ("score", "weights", "sub_scores", "method", "penalty"):
            assert key in result, f"Missing key {key!r} in result"

    def test_sub_scores_keys(self):
        q = _make_record()
        g = _make_record()
        result = fuse(q, g, emb_score=0.5, cfg=_CFG, method="fixed_weight")
        assert set(result["sub_scores"].keys()) == {"emb", "color", "attr"}

    def test_osnet_only_uses_only_embedding(self):
        q = _make_record()
        g = _make_record()
        result = fuse(q, g, emb_score=0.75, cfg=_CFG, method="osnet_only")
        assert result["score"] == pytest.approx(0.75, abs=0.01)

    def test_fixed_weight_sum(self):
        fw = _CFG["fixed_weights"]
        total = sum(fw.values())
        assert abs(total - 1.0) < 0.01


class TestAdaptiveRule:
    def test_high_quality_gives_more_weight_to_embedding(self):
        q_hq = _make_record(quality_overall=0.95)
        g_hq = _make_record(quality_overall=0.95)
        r_hq = fuse(q_hq, g_hq, 0.7, _CFG, method="adaptive_rule")

        q_lq = _make_record(quality_overall=0.1)
        g_lq = _make_record(quality_overall=0.1)
        r_lq = fuse(q_lq, g_lq, 0.7, _CFG, method="adaptive_rule")

        w_hq = r_hq["weights"]["embedding"]
        w_lq = r_lq["weights"]["embedding"]
        assert w_hq > w_lq, f"High quality emb weight ({w_hq:.3f}) should exceed low ({w_lq:.3f})"

    def test_adaptive_weights_sum_to_one(self):
        q = _make_record(quality_overall=0.6)
        g = _make_record(quality_overall=0.6)
        result = fuse(q, g, 0.5, _CFG, method="adaptive_rule")
        w = result["weights"]
        total = sum(v for v in w.values() if v is not None)
        assert abs(total - 1.0) < 0.01


class TestMismatchPenalty:
    def test_penalty_applied_on_mismatch(self):
        q = _make_record(upper="blue", upper_conf=0.9, lower="black", lower_conf=0.9)
        g = _make_record(upper="red", upper_conf=0.9, lower="white", lower_conf=0.9)
        r_with = fuse(q, g, 0.5, _CFG, method="fixed_weight")
        r_without = fuse(q, g, 0.5, _CFG, method="fixed_weight",
                         mismatch_penalty_override=0.0)
        assert r_with["score"] <= r_without["score"], "Penalty should reduce score"

    def test_no_penalty_when_disabled(self):
        q = _make_record(upper="blue", upper_conf=0.9)
        g = _make_record(upper="red", upper_conf=0.9)
        result = fuse(q, g, 0.5, _CFG, method="fixed_weight",
                      mismatch_penalty_override=0.0)
        assert result["penalty"] == pytest.approx(0.0)

    def test_no_penalty_on_unknown_color(self):
        q = _make_record(upper="unknown", upper_conf=0.0)
        g = _make_record(upper="red", upper_conf=0.9)
        result = fuse(q, g, 0.5, _CFG, method="fixed_weight")
        assert result["penalty"] == pytest.approx(0.0)


class TestRerank:
    def test_sorted_by_score_desc(self):
        q = _make_record()
        candidates = [
            {"record": _make_record(), "faiss_id": i, "emb_score": float(i) * 0.1}
            for i in range(5)
        ]
        ranked = rerank(candidates, q, _CFG, method="osnet_only")
        scores = [c["score"] for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_adds_fusion_result(self):
        q = _make_record()
        cands = [{"record": _make_record(), "faiss_id": 0, "emb_score": 0.7}]
        ranked = rerank(cands, q, _CFG)
        assert "fusion_result" in ranked[0]
        assert "score" in ranked[0]


class TestQualityGate:
    def test_high_quality_gives_high_gate(self):
        gate = _quality_gate(0.95, temperature=2.0)
        assert gate > 0.7

    def test_low_quality_gives_low_gate(self):
        gate = _quality_gate(0.05, temperature=2.0)
        assert gate < 0.3

    def test_midpoint_near_half(self):
        gate = _quality_gate(0.5, temperature=2.0)
        assert abs(gate - 0.5) < 0.01
