"""
identity.py -- Persistent identity tracking and gallery management.

Maintains identity_id separate from ByteTrack track_id.
Handles M-frame accumulation buffers, quality-weighted mean embeddings,
N-per-identity embedding banks, quality gating, temporal spacing,
and ambiguous match handling.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.fusion import fuse
from src.record import FeatureRecord

logger = logging.getLogger(__name__)


@dataclass
class IdentityBank:
    """Stores enrolled FeatureRecords for a persistent identity."""
    identity_id: int
    records: List[FeatureRecord] = field(default_factory=list)
    last_enrolled_time: float = 0.0


@dataclass
class IdentityResolution:
    """Result of identity assignment for a detection."""
    identity_id: Optional[int]
    track_id: int
    status: str  # "pending", "linked", "new_identity", "ambiguous", "suggest_only", "active"
    frame_count: int = 0
    best_candidate_id: Optional[int] = None
    best_score: float = 0.0
    margin: float = 0.0
    decision: str = ""
    suggestion: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)


class ScoreLogger:
    """Logs linking decision metrics to CSV for analysis and tuning."""

    HEADER = [
        "timestamp",
        "new_track_id",
        "candidate_identity",
        "emb_cosine",
        "color_sim",
        "attr_sim",
        "fused_score",
        "margin",
        "decision",
        "manual_label",
    ]

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = log_path
        self._file = None
        self._writer = None
        if self.log_path:
            p = Path(self.log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            file_exists = p.exists() and p.stat().st_size > 0
            self._file = open(p, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            if not file_exists:
                self._writer.writerow(self.HEADER)
                self._file.flush()
            logger.info("ScoreLogger initialized: writing to %s", p)

    def log(
        self,
        timestamp: float,
        new_track_id: int,
        candidate_identity: Optional[int],
        emb_cosine: float,
        color_sim: float,
        attr_sim: float,
        fused_score: float,
        margin: float,
        decision: str,
        manual_label: str = "",
    ) -> None:
        if self._writer is None:
            return
        cand_str = str(candidate_identity) if candidate_identity is not None else "None"
        self._writer.writerow([
            f"{timestamp:.3f}",
            new_track_id,
            cand_str,
            f"{emb_cosine:.4f}",
            f"{color_sim:.4f}",
            f"{attr_sim:.4f}",
            f"{fused_score:.4f}",
            f"{margin:.4f}",
            decision,
            manual_label,
        ])
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None


def mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """
    Compute average pairwise cosine similarity among a set of L2-normalized embeddings.
    Excludes self-similarity (diagonal). Returns 0.0 for fewer than 2 embeddings.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0
    sim = embeddings @ embeddings.T
    off_diag_sum = float(np.sum(sim) - np.trace(sim))
    return float(off_diag_sum / (n * (n - 1)))


class IdentityManager:
    """
    Manages persistent identities, track-to-identity mappings, multi-frame
    observation buffers, and gallery embedding banks.
    """

    def __init__(self, cfg: dict, score_log_path: Optional[str] = None) -> None:
        self.cfg = cfg
        id_cfg = cfg.get("identity", {})
        self.link_threshold: Optional[float] = id_cfg.get("link_threshold", None)
        if self.link_threshold is not None:
            self.link_threshold = float(self.link_threshold)
        self.wait_frames_m: int = int(id_cfg.get("wait_frames_m", 6))
        self.max_embeddings_n: int = int(id_cfg.get("max_embeddings_n", 10))
        self.temporal_spacing_s: float = float(id_cfg.get("temporal_spacing_s", 1.0))
        self.margin: float = float(id_cfg.get("margin", 0.05))
        self.match_metric: str = str(id_cfg.get("match_metric", "mean_top_k"))
        self.match_top_k: int = int(id_cfg.get("match_top_k", 3))

        # Quality gate configuration: absolute number or "auto"
        raw_q_gate = id_cfg.get("quality_threshold", 0.40)
        self.auto_gate_frames: int = int(id_cfg.get("auto_gate_frames", 50))
        self._session_enrolled_qualities: List[float] = []

        if isinstance(raw_q_gate, str) and raw_q_gate.strip().lower() == "auto":
            self.quality_gate_mode: str = "auto"
            self._effective_quality_threshold: float = 0.0
            self._auto_calibrated: bool = False
            logger.info(
                "Quality gate mode: AUTO (calibrating as 10th percentile over first %d enrolled frames)",
                self.auto_gate_frames,
            )
        else:
            self.quality_gate_mode = "absolute"
            self._effective_quality_threshold = float(raw_q_gate)
            self._auto_calibrated = True
            logger.info(
                "Quality gate mode: ABSOLUTE (threshold=%.4f)",
                self._effective_quality_threshold,
            )

        # Fusion config & live method
        self.fusion_cfg = cfg.get("fusion", {})
        raw_method = self.fusion_cfg.get("method", "adaptive_rule")
        if "learned" in raw_method or "lr" in raw_method:
            logger.warning(
                "fusion_lr.pkl is stale and disabled for live identity linking. Falling back to 'adaptive_rule'."
            )
            self.fusion_method = "adaptive_rule"
        else:
            self.fusion_method = raw_method

        logger.info(
            "IdentityManager ready: live fusion method='%s' | link_threshold=%s | M=%d | N=%d | Q_gate=%s | spacing=%.1fs | margin=%.2f",
            self.fusion_method,
            f"{self.link_threshold:.2f}" if self.link_threshold is not None else "None (suggest-only)",
            self.wait_frames_m,
            self.max_embeddings_n,
            f"{self._effective_quality_threshold:.2f} ({self.quality_gate_mode})" if self._auto_calibrated else "auto (calibrating)",
            self.temporal_spacing_s,
            self.margin,
        )

        self._next_identity_id: int = 1
        self._identities: Dict[int, IdentityBank] = {}
        self._track_to_identity: Dict[int, int] = {}
        self._track_buffers: Dict[int, List[FeatureRecord]] = {}
        self._track_ambiguous: Dict[int, bool] = {}
        self._track_suggestions: Dict[int, Dict[str, Any]] = {}
        self._track_candidates: Dict[int, List[Dict[str, Any]]] = {}

        self.score_logger = ScoreLogger(score_log_path)

    @property
    def quality_threshold(self) -> float:
        return self._effective_quality_threshold

    def reset(self) -> None:
        """Clear all tracks and identities."""
        self._next_identity_id = 1
        self._identities.clear()
        self._track_to_identity.clear()
        self._track_buffers.clear()
        self._track_ambiguous.clear()
        self._track_suggestions.clear()
        self._track_candidates.clear()
        self._session_enrolled_qualities.clear()
        if self.quality_gate_mode == "auto":
            self._effective_quality_threshold = 0.0
            self._auto_calibrated = False

    def get_identity_for_track(self, track_id: int) -> Optional[int]:
        return self._track_to_identity.get(track_id)

    def get_candidates_for_track(self, track_id: int) -> List[Dict[str, Any]]:
        return self._track_candidates.get(track_id, [])

    def get_suggestion_for_track(self, track_id: int) -> Optional[Dict[str, Any]]:
        return self._track_suggestions.get(track_id)

    def is_pending(self, track_id: int) -> bool:
        return track_id not in self._track_to_identity

    def get_bank(self, identity_id: int) -> List[FeatureRecord]:
        bank = self._identities.get(identity_id)
        return bank.records if bank else []

    def _maybe_enroll(self, identity_id: int, track_id: int, record: FeatureRecord) -> bool:
        """Enroll record into identity bank if quality and temporal spacing permit."""
        q_overall = float(record.quality.overall)
        gate = self._effective_quality_threshold if self._auto_calibrated else 0.0

        if q_overall < gate:
            logger.debug(
                "[ENROLL_SKIP] quality gate: %.3f < %.2f for track_id=%d identity_id=%d",
                q_overall, gate, track_id, identity_id
            )
            return False

        bank = self._identities.get(identity_id)
        if bank is None:
            bank = IdentityBank(identity_id=identity_id)
            self._identities[identity_id] = bank

        now = float(record.timestamp) if record.timestamp > 0 else time.time()
        elapsed = now - bank.last_enrolled_time
        if bank.records and elapsed < self.temporal_spacing_s:
            logger.debug(
                "[ENROLL_SKIP] spacing: %.2fs < %.2fs for track_id=%d identity_id=%d",
                elapsed, self.temporal_spacing_s, track_id, identity_id
            )
            return False

        # If in auto mode and not yet calibrated, record quality for calibration
        if self.quality_gate_mode == "auto" and not self._auto_calibrated:
            self._session_enrolled_qualities.append(q_overall)
            if len(self._session_enrolled_qualities) >= self.auto_gate_frames:
                self._effective_quality_threshold = float(
                    np.percentile(self._session_enrolled_qualities[:self.auto_gate_frames], 10)
                )
                self._auto_calibrated = True
                logger.info(
                    "Quality gate auto-calibrated: 10th percentile over first %d enrolled frames = %.4f",
                    self.auto_gate_frames, self._effective_quality_threshold,
                )

        # Case 1: Bank is not yet full (< N)
        if len(bank.records) < self.max_embeddings_n:
            bank.records.append(record)
            bank.last_enrolled_time = now
            logger.info(
                "[ENROLL] identity_id=%d | track_id=%d | quality.overall=%.4f (gate=%.2f [%s]) | enrolled=True | bank_size=%d",
                identity_id, track_id, q_overall, self.quality_threshold, self.quality_gate_mode, len(bank.records)
            )
            return True

        # Case 2: Bank is full (N): Redundancy-reduction replacement policy
        # 1. Pool N existing embeddings + candidate
        bank_embs = np.stack([r.get_embedding_vector().flatten() for r in bank.records], axis=0)  # (N, D)
        cand_emb = record.get_embedding_vector().flatten().reshape(1, -1)  # (1, D)
        pool_embs = np.vstack([bank_embs, cand_emb])  # (N+1, D)

        # 2. Compute pairwise cosine matrix among all N+1 embeddings
        sim_matrix = pool_embs @ pool_embs.T  # (N+1, N+1)

        # 3. For each embedding, compute mean similarity to the other N embeddings
        mean_sims = (np.sum(sim_matrix, axis=1) - np.diag(sim_matrix)) / float(self.max_embeddings_n)
        most_redundant_idx = int(np.argmax(mean_sims))

        # 4. If candidate itself is the most redundant in the pool, discard it
        if most_redundant_idx == self.max_embeddings_n:
            bank.last_enrolled_time = now
            logger.info(
                "[ENROLL_DISCARD] identity_id=%d | track_id=%d | candidate is most redundant in pool (mean_sim=%.4f) | bank_size=%d",
                identity_id, track_id, float(mean_sims[most_redundant_idx]), len(bank.records)
            )
            return False

        # 5. Otherwise, check if replacing that existing embedding with candidate reduces redundancy
        r_old = mean_pairwise_cosine(bank_embs)
        cand_bank_embs = np.delete(pool_embs, most_redundant_idx, axis=0)
        r_new = mean_pairwise_cosine(cand_bank_embs)

        if r_new < r_old:
            # Candidate reduces redundancy: replace
            bank.records[most_redundant_idx] = record
            bank.last_enrolled_time = now
            logger.info(
                "[ENROLL_REPLACE] identity_id=%d | track_id=%d | replaced idx=%d | redundancy: %.4f -> %.4f | bank_size=%d",
                identity_id, track_id, most_redundant_idx, r_old, r_new, len(bank.records)
            )
            return True
        else:
            # Candidate does not reduce redundancy: discard
            bank.last_enrolled_time = now
            logger.info(
                "[ENROLL_DISCARD] identity_id=%d | track_id=%d | candidate does not reduce redundancy (%.4f >= %.4f) | bank_size=%d",
                identity_id, track_id, r_new, r_old, len(bank.records)
            )
            return False

    def process_track(
        self,
        record: FeatureRecord,
        visible_track_ids: Set[int],
    ) -> IdentityResolution:
        """
        Process a single track detection record in the current frame.

        Parameters
        ----------
        record : FeatureRecord
        visible_track_ids : Set[int]
            Set of all track IDs currently detected in this frame.

        Returns
        -------
        IdentityResolution
        """
        tid = record.track_id

        # 1. Track is already assigned to an identity
        if tid in self._track_to_identity:
            assigned_id = self._track_to_identity[tid]
            # If not ambiguous, update the embedding bank
            if not self._track_ambiguous.get(tid, False):
                self._maybe_enroll(assigned_id, tid, record)
            return IdentityResolution(
                identity_id=assigned_id,
                track_id=tid,
                status="active",
                decision="active",
                suggestion=self._track_suggestions.get(tid),
                candidates=self._track_candidates.get(tid, []),
            )

        # 2. Track is new/unassigned: buffer M frames
        buf = self._track_buffers.setdefault(tid, [])
        buf.append(record)
        if len(buf) < self.wait_frames_m:
            return IdentityResolution(
                identity_id=None,
                track_id=tid,
                status="pending",
                frame_count=len(buf),
                decision="pending",
                candidates=[],
            )

        # 3. Buffer has reached M frames: resolve identity!
        # Compute quality-weighted mean embedding
        weights = np.array([max(float(r.quality.overall), 1e-4) for r in buf], dtype=np.float32)
        weights /= np.sum(weights)
        embs = np.stack([r.get_embedding_vector().flatten() for r in buf], axis=0)
        mean_emb = np.sum(embs * weights[:, None], axis=0)
        norm = float(np.linalg.norm(mean_emb))
        if norm > 1e-6:
            mean_emb = (mean_emb / norm).astype(np.float32)

        # Representative record (highest quality in buffer)
        best_buf_rec = max(buf, key=lambda r: r.quality.overall)
        query_record = FeatureRecord(
            track_id=tid,
            camera_id=record.camera_id,
            timestamp=record.timestamp,
            bbox=record.bbox,
            embedding=mean_emb,
            colors=best_buf_rec.colors,
            attributes=best_buf_rec.attributes,
            quality=best_buf_rec.quality,
        )

        # Rule 2: Active identities are currently visible on another track.
        # Never link two simultaneously visible tracks to the same identity!
        active_identities = {
            self._track_to_identity[t]
            for t in visible_track_ids
            if t in self._track_to_identity and t != tid
        }

        # Candidate identities: must be inactive and have enrolled records
        candidate_ids = [
            iid for iid, bank in self._identities.items()
            if iid not in active_identities and len(bank.records) > 0
        ]

        scored_candidates: List[Dict[str, Any]] = []
        for cand_id in candidate_ids:
            bank = self._identities[cand_id]
            item_scores = []
            for b_rec in bank.records:
                emb_cos = float(np.dot(mean_emb, b_rec.get_embedding_vector().flatten()))
                emb_cos = max(-1.0, min(1.0, emb_cos))
                if self.fusion_method == "osnet_only":
                    f_score = emb_cos
                    col_score = 0.0
                    attr_score = 0.0
                else:
                    f_res = fuse(
                        query_record=query_record,
                        gallery_record=b_rec,
                        emb_score=emb_cos,
                        cfg=self.fusion_cfg,
                        method=self.fusion_method,
                    )
                    f_score = float(f_res["score"])
                    col_score = float(f_res.get("color_score", 0.0))
                    attr_score = float(f_res.get("attr_score", 0.0))
                item_scores.append((f_score, emb_cos, col_score, attr_score))

            if not item_scores:
                continue

            item_scores.sort(key=lambda x: x[0], reverse=True)
            k = min(len(item_scores), self.match_top_k)
            if self.match_metric == "max":
                id_score = item_scores[0][0]
            else:  # mean_top_k
                id_score = float(np.mean([s[0] for s in item_scores[:k]]))

            top_sub = item_scores[0]
            scored_candidates.append({
                "candidate_id": cand_id,
                "score": id_score,
                "emb_score": top_sub[1],
                "color_score": top_sub[2],
                "attr_score": top_sub[3],
            })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        self._track_candidates[tid] = scored_candidates[:3]

        now_ts = float(record.timestamp) if record.timestamp > 0 else time.time()
        decision = "new_identity"
        best_cand_id: Optional[int] = None
        best_score = 0.0
        margin = 0.0
        suggestion: Optional[Dict[str, Any]] = None

        if not scored_candidates:
            # First person or no previous identities
            new_id = self._next_identity_id
            self._next_identity_id += 1
            self._track_to_identity[tid] = new_id
            self._identities[new_id] = IdentityBank(identity_id=new_id)
            decision = "new_identity"
        else:
            best_cand = scored_candidates[0]
            best_cand_id = best_cand["candidate_id"]
            best_score = best_cand["score"]
            second_score = scored_candidates[1]["score"] if len(scored_candidates) > 1 else 0.0
            margin = best_score - second_score

            if self.link_threshold is None:
                # Rule 7: Suggest-only mode until link_threshold is configured
                new_id = self._next_identity_id
                self._next_identity_id += 1
                self._track_to_identity[tid] = new_id
                self._identities[new_id] = IdentityBank(identity_id=new_id)
                decision = "suggest_only"
                suggestion = {"candidate_id": best_cand_id, "score": best_score}
                self._track_suggestions[tid] = suggestion
            else:
                if best_score >= self.link_threshold:
                    if margin >= self.margin:
                        # Rule 3: Clear link
                        self._track_to_identity[tid] = best_cand_id
                        decision = "link"
                    else:
                        # Rule 3: Ambiguous match -> create new identity, suggest match
                        new_id = self._next_identity_id
                        self._next_identity_id += 1
                        self._track_to_identity[tid] = new_id
                        self._identities[new_id] = IdentityBank(identity_id=new_id)
                        decision = "ambiguous"
                        self._track_ambiguous[tid] = True
                        suggestion = {"candidate_id": best_cand_id, "score": best_score}
                        self._track_suggestions[tid] = suggestion
                else:
                    # Below threshold -> genuinely new person
                    new_id = self._next_identity_id
                    self._next_identity_id += 1
                    self._track_to_identity[tid] = new_id
                    self._identities[new_id] = IdentityBank(identity_id=new_id)
                    decision = "new_identity"

        assigned_id = self._track_to_identity[tid]
        logger.info(
            "[IDENTITY RESOLVED] track_id=%d -> identity_id=%d | decision=%s | best_cand=%s | score=%.4f | margin=%.4f | threshold=%s",
            tid,
            assigned_id,
            decision,
            str(best_cand_id),
            best_score,
            margin,
            f"{self.link_threshold:.2f}" if self.link_threshold is not None else "None(suggest-only)",
        )

        # Log to CSV if configured (Rule 6)
        top_cand = scored_candidates[0] if scored_candidates else {}
        self.score_logger.log(
            timestamp=now_ts,
            new_track_id=tid,
            candidate_identity=best_cand_id,
            emb_cosine=top_cand.get("emb_score", 0.0),
            color_sim=top_cand.get("color_score", 0.0),
            attr_sim=top_cand.get("attr_score", 0.0),
            fused_score=best_score,
            margin=margin,
            decision=decision,
            manual_label="",
        )

        # Rule 4: Do not add embeddings on frames where match was ambiguous
        if decision != "ambiguous":
            self._maybe_enroll(assigned_id, tid, best_buf_rec)

        del self._track_buffers[tid]

        return IdentityResolution(
            identity_id=assigned_id,
            track_id=tid,
            status=decision,
            best_candidate_id=best_cand_id,
            best_score=best_score,
            margin=margin,
            decision=decision,
            suggestion=suggestion,
            candidates=self._track_candidates.get(tid, []),
        )
