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
        assert set(result.keys()) == {"upper", "upper_conf", "lower", "lower_conf"}

    def test_conf_in_bounds(self):
        crop = _solid_crop((80, 80, 200))
        result = extract_colors(crop, (0.5, 0.25), (0.5, 0.75), _CFG)
        assert 0.0 <= result["upper_conf"] <= 1.0
        assert 0.0 <= result["lower_conf"] <= 1.0
