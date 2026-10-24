"""
test_color.py -- Unit tests for color label extraction.

Tests that solid-colour patches map to correct labels, that confidence is
in [0,1], and that low-confidence patches return 'unknown'.
"""
import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.color import classify_color, extract_colors, _COLOR_LABELS

_CFG = {
    "patch_size": 7,
    "confidence_threshold": 0.25,
    "color_space": "lab",
}


def _solid_crop(rgb: tuple, h: int = 64, w: int = 32) -> np.ndarray:
    """Create a solid-colour BGR crop."""
    import cv2
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    crop[:, :] = [rgb[2], rgb[1], rgb[0]]  # RGB -> BGR
    return crop


class TestClassifyColor:
    def test_returns_label_and_float(self):
        crop = _solid_crop((0, 0, 0))
        label, conf = classify_color(crop, 0.5, 0.5, _CFG)
        assert isinstance(label, str)
        assert isinstance(conf, float)

    def test_black_crop_returns_black_or_unknown(self):
        crop = _solid_crop((5, 5, 5))
        label, conf = classify_color(crop, 0.5, 0.5, _CFG)
        # Should be black or unknown if confidence is too low
        assert label in _COLOR_LABELS + ["unknown"]

    def test_white_crop(self):
        crop = _solid_crop((250, 250, 250))
        label, conf = classify_color(crop, 0.5, 0.5, _CFG)
        assert label in ["white", "unknown"]

    def test_confidence_in_bounds(self):
        crop = _solid_crop((200, 50, 50))
        _, conf = classify_color(crop, 0.5, 0.5, _CFG)
        assert 0.0 <= conf <= 1.0

    def test_low_conf_returns_unknown(self):
        """A config with very high threshold should always return unknown."""
        strict_cfg = dict(_CFG, confidence_threshold=0.9999)
        crop = _solid_crop((100, 100, 100))
        label, _ = classify_color(crop, 0.5, 0.5, strict_cfg)
        assert label == "unknown"

    def test_label_in_valid_set_or_unknown(self):
        for rgb in [(0, 0, 0), (255, 255, 255), (200, 0, 0), (0, 0, 200)]:
            crop = _solid_crop(rgb)
            label, _ = classify_color(crop, 0.5, 0.5, _CFG)
            assert label in _COLOR_LABELS + ["unknown"]


class TestExtractColors:
    def test_output_keys(self):
        crop = _solid_crop((100, 150, 100))
        result = extract_colors(crop, (0.5, 0.25), (0.5, 0.75), _CFG)
        assert {"upper", "upper_conf", "lower", "lower_conf", "sampling_path"}.issubset(result.keys())

    def test_conf_in_bounds(self):
        crop = _solid_crop((80, 80, 200))
        result = extract_colors(crop, (0.5, 0.25), (0.5, 0.75), _CFG)
        assert 0.0 <= result["upper_conf"] <= 1.0
        assert 0.0 <= result["lower_conf"] <= 1.0

    def test_synthetic_red_top_blue_bottom(self):
        """
        Synthetic image (red top, blue bottom).
        Tests both model points and fallback paths.
        """
        # 160 height x 80 width, BGR image
        h, w = 160, 80
        crop = np.zeros((h, w, 3), dtype=np.uint8)
        # Top half: Red in BGR
        crop[:h // 2, :] = [40, 48, 174]
        # Bottom half: Blue in BGR
        crop[h // 2:, :] = [168, 93, 45]

        # 1. Model points path
        res_model = extract_colors(crop, upper_xy=(0.5, 0.25), lower_xy=(0.5, 0.75), cfg=_CFG)
        assert res_model["sampling_path"] == "model points"
        assert res_model["upper"] == "red"
        assert res_model["lower"] == "blue"
        assert res_model["upper_conf"] > 0.5
        assert res_model["lower_conf"] > 0.5

        # 2. Fallback path (None coords)
        res_fallback = extract_colors(crop, upper_xy=None, lower_xy=None, cfg=_CFG)
        assert res_fallback["sampling_path"] == "fallback"
        assert res_fallback["upper"] == "red"
        assert res_fallback["lower"] == "blue"
        assert res_fallback["upper_conf"] > 0.5
        assert res_fallback["lower_conf"] > 0.5

    def test_out_of_bounds_clipping(self):
        """Coordinates outside [0, 1] must be clipped without error."""
        crop = np.full((100, 50, 3), [168, 93, 45], dtype=np.uint8)  # Blue
        # Pass coordinates out of bounds (-0.2, 1.4)
        res = extract_colors(crop, upper_xy=(-0.2, 0.5), lower_xy=(1.4, 0.5), cfg=_CFG)
        assert res["sampling_path"] == "model points"
        assert res["upper"] == "blue"
        assert res["lower"] == "blue"
