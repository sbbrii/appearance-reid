"""
pipeline.py -- End-to-end SAFEPASS pipeline.

Orchestrates: video capture -> detection/tracking -> embedding -> quality
-> attributes -> color -> gallery search -> fusion re-rank -> display.

Targets 10+ FPS on CPU for 3-5 people.  Reports measured FPS honestly.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Colour palette for track_id annotation
_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31),
    (255, 178, 29), (207, 210, 49), (72, 249, 10),
    (146, 204, 23), (61, 219, 134), (26, 147, 52),
    (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236),
    (132, 56, 255), (82, 0, 133), (203, 56, 255),
]


def _track_colour(track_id: int) -> tuple:
    return _PALETTE[track_id % len(_PALETTE)]


def _draw_quality_bar(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    overall: float,
) -> None:
    bar_w = x2 - x1
    bar_h = 6
    bar_y = max(0, y1 - bar_h - 2)
    filled = int(bar_w * overall)
    color = (0, int(255 * overall), int(255 * (1 - overall)))
    cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (80, 80, 80), -1)
    if filled > 0:
        cv2.rectangle(frame, (x1, bar_y), (x1 + filled, bar_y + bar_h), color, -1)


def _draw_overlay(
    frame: np.ndarray,
    det: dict,
    quality_info: dict,
    color_info: dict,
    attr_info: dict,
    top_matches: list,
    cfg: dict,
) -> None:
    """Draw bbox, track_id, quality bar, color labels, attributes, top matches."""
    x1, y1, x2, y2 = det["bbox"]
    tid = det["track_id"]
    col = _track_colour(tid)

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    label = f"ID:{tid}"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    if cfg["pipeline"].get("draw_quality_bar") and quality_info:
        _draw_quality_bar(frame, x1, y1, x2, quality_info.get("overall", 0.5))

    # Color labels
    if cfg["pipeline"].get("draw_color_labels") and color_info:
        txt = f"U:{color_info.get('upper','?')}  L:{color_info.get('lower','?')}"
        cv2.putText(frame, txt, (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # Top gallery matches
    for rank, m in enumerate(top_matches[: cfg["pipeline"].get("top_k_display", 3)]):
        score = m.get("score", 0)
        match_tid = m["record"].track_id
        m_txt = f"#{rank+1} ID:{match_tid} {score:.2f}"
        cv2.putText(frame, m_txt, (x1, y2 + 28 + rank * 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1, cv2.LINE_AA)


class Pipeline:
    """
    Full SAFEPASS inference pipeline.

    Parameters
    ----------
    cfg : dict
        Full default.yaml config as a Python dict.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._extractor = None
        self._tracker = None
        self._gallery = None
        self._attr_model = None
        self._attr_loaded = False

    def _lazy_init(self, camera_id: str = "cam0") -> None:
        """Initialise all components (deferred to avoid import cost at import time)."""
        if self._extractor is not None:
            return
        from src.embed import EmbeddingExtractor
        from src.detect_track import DetectionTracker
        from src.gallery import GalleryIndex
        from src.attributes import AttributeInference

        logger.info("Initialising pipeline components...")
        self._extractor = EmbeddingExtractor(self.cfg["model"])
        self._tracker = DetectionTracker(self.cfg["detection"], camera_id=camera_id)
        self._gallery = GalleryIndex(self.cfg["gallery"], embedding_dim=self.cfg["model"]["embedding_dim"])
        self._attr_model = AttributeInference(self.cfg["attributes"])
        try:
            self._attr_model.load()
            self._attr_loaded = True
        except FileNotFoundError as exc:
            logger.warning("OpenVINO model not found: %s  (attributes disabled)", exc)
        logger.info("Pipeline ready.")

    def process_single_frame(self, frame: np.ndarray, camera_id: str = "cam0") -> list:
        """
        Process a single BGR frame programmatically through the entire pipeline.

        Returns
        -------
        list of dicts, each containing:
          'record'     : FeatureRecord
          'candidates' : list of re-ranked gallery candidates
          'detection'  : detection dict
        """
        from src.quality import estimate_quality
        from src.color import extract_colors
        from src.record import FeatureRecord, ColorInfo, AttributeInfo, QualityInfo
        from src.fusion import rerank

        self._lazy_init(camera_id=camera_id)
        detections = self._tracker.process_frame(frame)
        if not detections:
            return []

        crops_bgr = [d["crop_bgr"] for d in detections]
        embeddings = self._extractor.extract_batch(crops_bgr) if crops_bgr else []

        results = []
        for i, det in enumerate(detections):
            emb = embeddings[i] if len(embeddings) > i else None
            if emb is None:
                continue

            q_dict = estimate_quality(det["crop_bgr"], det["bbox"], self.cfg["quality"])
            color_dict = {"upper": "unknown", "upper_conf": 0.0, "lower": "unknown", "lower_conf": 0.0}
            attr_dict_probs = {n: 0.5 for n in self.cfg["attributes"]["attribute_names"]}
            if self._attr_loaded:
                try:
                    attr_out = self._attr_model.predict(det["crop_bgr"])
                    attr_dict_probs = attr_out["probs"]
                    upper_xy, lower_xy = attr_out["coords"]
                    color_dict = extract_colors(det["crop_bgr"], upper_xy, lower_xy, self.cfg["color"])
                except Exception as exc:
                    logger.debug("Attribute inference error: %s", exc)

            record = FeatureRecord(
                track_id=det["track_id"],
                camera_id=det["camera_id"],
                timestamp=det["timestamp"],
                bbox=det["bbox"],
                embedding=emb,
                colors=ColorInfo(**color_dict),
                attributes=AttributeInfo(**attr_dict_probs),
                quality=QualityInfo(**q_dict),
            )

            candidates = self._gallery.search(record)
            ranked = rerank(candidates, record, self.cfg["fusion"])
            self._gallery.add(record)
            results.append({
                "record": record,
                "candidates": ranked,
                "detection": det,
            })
        return results

    def run(self, source, camera_id: str = "cam0") -> None:
        """
        Run the pipeline on a video source.

        Parameters
        ----------
        source : int | str   webcam index or video file path.
        """
        import cv2
        from src.detect_track import open_source
        from src.quality import estimate_quality
        from src.color import extract_colors
        from src.record import FeatureRecord, ColorInfo, AttributeInfo, QualityInfo
        from src.fusion import rerank

        self._lazy_init(camera_id=camera_id)
        cap = open_source(source)

        fps_report_interval = 30  # report FPS every N frames
        frame_times: list = []
        frame_count = 0
        target_fps = float(self.cfg["pipeline"].get("target_fps", 10))
        display = bool(self.cfg["pipeline"].get("display", True))

        logger.info("Pipeline running.  Press Q to quit.")

        try:
            while True:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video stream ended.")
                    break

                # -- Detection + tracking --
                detections = self._tracker.process_frame(frame)

                crops_bgr = [d["crop_bgr"] for d in detections]
                embeddings = self._extractor.extract_batch(crops_bgr) if crops_bgr else []

                for i, det in enumerate(detections):
                    emb = embeddings[i] if len(embeddings) > i else None
                    if emb is None:
                        continue

                    # -- Quality --
                    q_dict = estimate_quality(det["crop_bgr"], det["bbox"], self.cfg["quality"])

                    # -- Attributes + color --
                    color_dict = {"upper": "unknown", "upper_conf": 0.0,
                                  "lower": "unknown", "lower_conf": 0.0}
                    attr_dict_probs = {n: 0.5 for n in self.cfg["attributes"]["attribute_names"]}
                    if self._attr_loaded:
                        try:
                            attr_out = self._attr_model.predict(det["crop_bgr"])
                            attr_dict_probs = attr_out["probs"]
                            upper_xy, lower_xy = attr_out["coords"]
                            color_dict = extract_colors(
                                det["crop_bgr"], upper_xy, lower_xy, self.cfg["color"]
                            )
                        except Exception as exc:
                            logger.debug("Attribute inference error: %s", exc)

                    # -- Build FeatureRecord --
                    record = FeatureRecord(
                        track_id=det["track_id"],
                        camera_id=det["camera_id"],
                        timestamp=det["timestamp"],
                        bbox=det["bbox"],
                        embedding=emb,
                        colors=ColorInfo(**color_dict),
                        attributes=AttributeInfo(**attr_dict_probs),
                        quality=QualityInfo(**q_dict),
                    )

                    # -- Gallery search + fusion re-rank --
                    candidates = self._gallery.search(record)
                    ranked = rerank(candidates, record, self.cfg["fusion"])

                    # Add current observation to gallery
                    self._gallery.add(record)

                    # -- Draw --
                    if display:
                        _draw_overlay(
                            frame, det,
                            quality_info=q_dict,
                            color_info=color_dict,
                            attr_info=attr_dict_probs,
                            top_matches=ranked,
                            cfg=self.cfg,
                        )

                # Periodic gallery TTL purge
                if frame_count % 300 == 0:
                    self._gallery.purge_expired()

                # FPS measurement
                elapsed = time.perf_counter() - t0
                frame_times.append(elapsed)
                frame_count += 1
                if frame_count % fps_report_interval == 0:
                    avg_t = sum(frame_times[-fps_report_interval:]) / fps_report_interval
                    fps_actual = 1.0 / max(avg_t, 1e-6)
                    logger.info("FPS: %.1f (target %.1f)", fps_actual, target_fps)
                    if display:
                        cv2.putText(
                            frame,
                            f"FPS: {fps_actual:.1f}",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        )

                if display:
                    cv2.imshow("SAFEPASS", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

            if frame_times:
                mean_fps = 1.0 / (sum(frame_times) / len(frame_times))
                logger.info(
                    "Session complete: %d frames processed  mean FPS: %.1f",
                    frame_count, mean_fps,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="SAFEPASS pipeline demo")
    parser.add_argument("--source", default="0",
                        help="Webcam index (int) or video file path")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--camera_id", default="cam0")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    source = int(args.source) if args.source.isdigit() else args.source
    pipeline = Pipeline(cfg)
    pipeline.run(source=source, camera_id=args.camera_id)


if __name__ == "__main__":
    main()
