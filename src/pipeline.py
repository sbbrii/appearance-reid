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
from typing import Any, Optional

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
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
    identity_res: Optional[Any] = None,
) -> None:
    """Draw bbox, persistent identity_id, track_id, quality bar, color labels, attributes, top matches."""
    x1, y1, x2, y2 = det["bbox"]
    tid = det["track_id"]

    # Determine identity label and display color
    wait_m = cfg.get("identity", {}).get("wait_frames_m", 6)
    if identity_res is not None and identity_res.identity_id is not None:
        iid = identity_res.identity_id
        col = _track_colour(iid)
        id_text = f"ID:{iid}"
    elif identity_res is not None and identity_res.status == "pending":
        col = (180, 180, 180)
        id_text = f"ID:? ({identity_res.frame_count}/{wait_m})"
    else:
        col = _track_colour(tid)
        id_text = f"ID:{tid}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    # Large ID label
    label_y = max(20, y1 - 10)
    cv2.putText(frame, id_text, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

    # Small T:track_id label
    id_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    tid_label = f"T:{tid}"
    cv2.putText(frame, tid_label, (x1 + id_size[0] + 8, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # Suggestion text (e.g. suggest-only or ambiguous match)
    if identity_res is not None and identity_res.suggestion:
        sug = identity_res.suggestion
        sug_text = f"possible: ID {sug['candidate_id']} ({sug['score']:.2f})"
        cv2.putText(frame, sug_text, (x1, max(36, label_y - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

    if cfg["pipeline"].get("draw_quality_bar") and quality_info:
        _draw_quality_bar(frame, x1, y1, x2, quality_info.get("overall", 0.5))

    # Color labels below bbox
    curr_y = y2 + 14
    if cfg["pipeline"].get("draw_color_labels") and color_info:
        txt = f"U:{color_info.get('upper','?')}  L:{color_info.get('lower','?')}"
        cv2.putText(frame, txt, (x1, curr_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        curr_y += 14

    # Top candidate matches with sub-scores
    candidates = identity_res.candidates if (identity_res and identity_res.candidates) else top_matches
    for rank, m in enumerate(candidates[: cfg["pipeline"].get("top_k_display", 3)]):
        if "candidate_id" in m:
            cand_id = m["candidate_id"]
            f_score = m.get("score", 0.0)
            e_score = m.get("emb_score", 0.0)
            c_score = m.get("color_score", 0.0)
            a_score = m.get("attr_score", 0.0)
            m_txt = f"#{rank+1} ID:{cand_id} F:{f_score:.2f} (E:{e_score:.2f} C:{c_score:.2f} A:{a_score:.2f})"
        else:
            score = m.get("score", 0.0)
            rec = m.get("record")
            match_tid = rec.track_id if rec else "?"
            m_txt = f"#{rank+1} ID:{match_tid} {score:.2f}"
        cv2.putText(frame, m_txt, (x1, curr_y + rank * 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 255, 180), 1, cv2.LINE_AA)


class Pipeline:
    """
    Full SAFEPASS inference pipeline.

    Parameters
    ----------
    cfg : dict
        Full default.yaml config as a Python dict.
    """

    def __init__(self, cfg: dict, score_log_path: Optional[str] = None) -> None:
        self.cfg = cfg
        self.score_log_path = score_log_path
        self._extractor = None
        self._tracker = None
        self._gallery = None
        self._attr_model = None
        self._attr_loaded = False
        self._identity = None
        self._seen_tracks: set[int] = set()

    def _lazy_init(self, camera_id: str = "cam0") -> None:
        """Initialise all components (deferred to avoid import cost at import time)."""
        if self._extractor is not None:
            return
        from src.embed import EmbeddingExtractor
        from src.detect_track import DetectionTracker
        from src.gallery import GalleryIndex
        from src.attributes import AttributeInference
        from src.identity import IdentityManager

        logger.info("Initialising pipeline components...")
        self._extractor = EmbeddingExtractor(self.cfg["model"])
        self._tracker = DetectionTracker(self.cfg["detection"], camera_id=camera_id)
        self._gallery = GalleryIndex(self.cfg["gallery"], embedding_dim=self.cfg["model"]["embedding_dim"])
        self._identity = IdentityManager(self.cfg, score_log_path=self.score_log_path)
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

        visible_track_ids = {d["track_id"] for d in detections}
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
                    coords = attr_out.get("coords")
                    upper_xy, lower_xy = coords if coords is not None else (None, None)
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

            id_res = self._identity.process_track(record, visible_track_ids)
            candidates = self._gallery.search(record)
            ranked = rerank(candidates, record, self.cfg["fusion"])
            self._gallery.add(record)

            results.append({
                "record": record,
                "candidates": ranked,
                "detection": det,
                "identity": id_res,
            })
        return results
        

    def run(self, source, camera_id: str = "cam0", duration: float = 0.0) -> None:
        """
        Run the pipeline on a video source.

        Parameters
        ----------
        source : int | str   webcam index or video file path.
        duration : float     Max runtime in seconds (0 for infinite).
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
        run_start_t = time.perf_counter()

        try:
            while True:
                if duration > 0 and (time.perf_counter() - run_start_t) >= duration:
                    logger.info("Target duration of %.1fs reached.", duration)
                    break
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video stream ended.")
                    break

                # -- Detection + tracking --
                detections = self._tracker.process_frame(frame)

                crops_bgr = [d["crop_bgr"] for d in detections]
                embeddings = self._extractor.extract_batch(crops_bgr) if crops_bgr else []

                visible_track_ids = {d["track_id"] for d in detections}
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
                            coords = attr_out.get("coords")
                            upper_xy, lower_xy = coords if coords is not None else (None, None)
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

                    # -- Persistent Identity Resolution --
                    id_res = self._identity.process_track(record, visible_track_ids)

                    # -- Draw --
                    if display:
                        _draw_overlay(
                            frame, det,
                            quality_info=q_dict,
                            color_info=color_dict,
                            attr_info=attr_dict_probs,
                            top_matches=id_res.candidates,
                            cfg=self.cfg,
                            identity_res=id_res,
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
            if self._identity and self._identity.score_logger:
                self._identity.score_logger.close()
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
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Max runtime in seconds (0 for infinite)")
    parser.add_argument("--log-scores", nargs="?", const="scores_log.csv", default=None,
                        help="Write linking scores and decisions to CSV file (default: scores_log.csv)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    source = int(args.source) if args.source.isdigit() else args.source
    pipeline = Pipeline(cfg, score_log_path=args.log_scores)
    pipeline.run(source=source, camera_id=args.camera_id, duration=args.duration)


if __name__ == "__main__":
    main()
