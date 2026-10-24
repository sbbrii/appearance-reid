"""
detect_track.py -- YOLOv8 + ByteTrack person detection and tracking.

Yields per-frame detections: (crop_bgr, track_id, bbox, timestamp, camera_id).
Multi-person detection rewritten from understanding of reference repo main.py
(loop structure; no licence; not copied verbatim).

Key differences from reference:
- Uses YOLOv8 + ByteTrack instead of RetinaNet + Kalman filter.
- Handles multiple people per frame.
- Filters by person class and minimum bbox height.
"""
from __future__ import annotations

import logging
import time
from typing import Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DetectionTracker:
    """
    YOLOv8 + ByteTrack multi-person detector-tracker.

    Parameters
    ----------
    cfg : dict
        ``detection`` section of default.yaml.
    camera_id : str
        Identifier for the camera source (embedded in records).
    """

    def __init__(self, cfg: dict, camera_id: str = "cam0") -> None:
        self.conf_thresh: float = float(cfg["conf_threshold"])
        self.person_class: int = int(cfg["person_class_id"])
        self.min_bbox_h: int = int(cfg["min_bbox_height"])
        self.tracker_cfg: str = str(cfg["tracker"])
        self.camera_id = camera_id

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed.  Run: pip install ultralytics"
            ) from exc

        model_name = cfg["yolo_model"]
        logger.info("DetectionTracker: loading %s  tracker=%s", model_name, self.tracker_cfg)
        self._yolo = YOLO(model_name)
        # ByteTrack requires persist=True across frames.
        self._persist = True

    def process_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> list:
        """
        Run detection + tracking on one BGR frame.

        Returns
        -------
        list of dicts, each with:
          'crop_bgr'   : np.ndarray  person crop
          'track_id'   : int
          'bbox'       : [x1, y1, x2, y2]
          'timestamp'  : float
          'camera_id'  : str
          'conf'       : float  YOLO confidence
        """
        ts = time.time()
        results = self._yolo.track(
            frame_bgr,
            persist=self._persist,
            tracker=self.tracker_cfg,
            conf=self.conf_thresh,
            classes=[self.person_class],
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                track_id_t = box.id
                if track_id_t is None:
                    continue
                track_id = int(track_id_t.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                # Filter tiny detections
                bbox_h = y2 - y1
                if bbox_h < self.min_bbox_h:
                    continue

                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                detections.append({
                    "crop_bgr": crop,
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": ts,
                    "camera_id": self.camera_id,
                    "conf": conf,
                })

        return detections


def open_source(source) -> cv2.VideoCapture:
    """Open a video source (int webcam index or file path string)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source!r}")
    return cap
