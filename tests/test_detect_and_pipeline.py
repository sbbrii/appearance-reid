"""
test_detect_and_pipeline.py -- Integration tests for detect_track and pipeline modules.
"""
import numpy as np
import pytest
import yaml

from src.detect_track import DetectionTracker
from src.pipeline import Pipeline


@pytest.fixture
def cfg():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestDetectionTracker:
    def test_tracker_init_and_empty_frame(self, cfg):
        tracker = DetectionTracker(cfg["detection"], camera_id="test_cam")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = tracker.process_frame(frame)
        assert isinstance(dets, list)
        assert len(dets) == 0


class TestPipeline:
    def test_pipeline_lazy_init_and_empty_frame(self, cfg):
        pipeline = Pipeline(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pipeline.process_single_frame(frame, camera_id="test_cam")
        assert isinstance(results, list)
        assert len(results) == 0
        assert pipeline._attr_loaded is True
