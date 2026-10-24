"""
test_quality.py -- Unit tests for the quality estimator.

Tests monotonic behaviour (more blur -> lower blur score),
score bounds [0,1], and weight normalisation.
"""
import numpy as np
import pytest
import cv2

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.quality import estimate_quality, _calibrate_blur, _size_score


# Minimal config mirroring default.yaml
_CFG = {
    "blur_calibration": {"p5": 20.0, "p95": 400.0},
    "visibility_min_aspect": 1.5,
    "visibility_edge_weight": 0.5,
    "size_ref_height_px": 128,
    "size_min_height_px": 32,
    "weights": {"blur": 0.40, "visibility": 0.35, "size": 0.25},
}


def _make_crop(h: int = 128, w: int = 64, blur_k: int = 0) -> np.ndarray:
    """Create a synthetic BGR crop, optionally blurred."""
    rng = np.random.RandomState(0)
    crop = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    if blur_k > 0:
        k = blur_k if blur_k % 2 == 1 else blur_k + 1
        crop = cv2.GaussianBlur(crop, (k, k), 0)
    return crop


class TestCalibrateBlur:
    def test_below_p5_gives_zero(self):
        assert _calibrate_blur(0.0, p5=20.0, p95=400.0) == pytest.approx(0.0)

    def test_at_p95_gives_one(self):
        assert _calibrate_blur(400.0, p5=20.0, p95=400.0) == pytest.approx(1.0)

    def test_midpoint(self):
        mid = (20.0 + 400.0) / 2
        score = _calibrate_blur(mid, p5=20.0, p95=400.0)
        assert 0.45 < score < 0.55

    def test_clip_above_one(self):
        assert _calibrate_blur(1000.0, p5=20.0, p95=400.0) == pytest.approx(1.0)


class TestSizeScore:
    def test_at_ref_height(self):
        assert _size_score(128, min_height=32, ref_height=128) == pytest.approx(1.0)

    def test_at_min_height(self):
        assert _size_score(32, min_height=32, ref_height=128) == pytest.approx(0.0)

    def test_below_min_clips_to_zero(self):
        assert _size_score(10, min_height=32, ref_height=128) == pytest.approx(0.0)

    def test_above_ref_clips_to_one(self):
        assert _size_score(200, min_height=32, ref_height=128) == pytest.approx(1.0)


class TestEstimateQuality:
    def test_output_keys(self):
        crop = _make_crop()
        out = estimate_quality(crop, [0, 0, 64, 128], _CFG)
        assert set(out.keys()) == {"blur", "visibility", "size", "overall"}

    def test_all_scores_in_bounds(self):
        crop = _make_crop()
        out = estimate_quality(crop, [0, 0, 64, 128], _CFG)
        for k, v in out.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_blur_monotonic(self):
        """More blur -> lower blur score."""
        sharp = _make_crop(blur_k=0)
        blurry = _make_crop(blur_k=15)
        score_sharp = estimate_quality(sharp, [0, 0, 64, 128], _CFG)["blur"]
        score_blurry = estimate_quality(blurry, [0, 0, 64, 128], _CFG)["blur"]
        assert score_sharp >= score_blurry, (
            f"Expected sharp ({score_sharp:.3f}) >= blurry ({score_blurry:.3f})"
        )

    def test_size_larger_crop(self):
        """Larger bbox height -> higher size score."""
        crop_small = _make_crop(h=40, w=25)
        crop_large = _make_crop(h=128, w=64)
        s_small = estimate_quality(crop_small, [0, 0, 25, 40], _CFG)["size"]
        s_large = estimate_quality(crop_large, [0, 0, 64, 128], _CFG)["size"]
        assert s_large >= s_small

    def test_weights_sum_approximately_one(self):
        """Config weights should be ≈ 1.0 (not enforced, but a sanity check)."""
        w = _CFG["weights"]
        total = sum(w.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"
