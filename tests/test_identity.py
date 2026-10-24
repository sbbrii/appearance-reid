"""
test_identity.py -- Unit tests for persistent identity layer with synthetic embeddings.

Tests:
- Pending status during initial M frames
- Simultaneous tracks never share an identity
- Bank capped at N
- Spacing enforced between enrollments
- Ambiguous match creates a new identity and blocks bank enrollment
- Quality gate blocks low-quality frames from entering bank
- Suggest-only mode when link_threshold is None
- Clear link when above threshold and margin
- Score logger CSV generation
"""
import csv
import numpy as np
import pytest

from src.identity import IdentityManager, IdentityBank, ScoreLogger, mean_pairwise_cosine
from src.record import FeatureRecord, ColorInfo, AttributeInfo, QualityInfo


def make_record(
    track_id: int,
    embedding: np.ndarray,
    timestamp: float = 1000.0,
    overall_quality: float = 0.8,
    upper_color: str = "black",
    lower_color: str = "blue",
) -> FeatureRecord:
    # Ensure L2-normalized embedding
    emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb = emb / norm
    return FeatureRecord(
        track_id=track_id,
        camera_id="cam0",
        timestamp=timestamp,
        bbox=[10, 20, 100, 200],
        embedding=emb,
        colors=ColorInfo(upper=upper_color, lower=lower_color, upper_conf=0.9, lower_conf=0.9),
        attributes=AttributeInfo(is_male=0.8),
        quality=QualityInfo(overall=overall_quality, blur=overall_quality, visibility=1.0, size=1.0),
    )


@pytest.fixture
def base_cfg():
    return {
        "identity": {
            "link_threshold": 0.65,
            "wait_frames_m": 4,
            "max_embeddings_n": 5,
            "quality_threshold": 0.40,
            "temporal_spacing_s": 1.0,
            "margin": 0.05,
            "match_metric": "mean_top_k",
            "match_top_k": 3,
        },
        "fusion": {
            "method": "adaptive_rule",
            "fixed_weights": {"embedding": 0.7, "color": 0.15, "attributes": 0.15},
            "adaptive": {
                "quality_aggregation": "min",
                "temperature": 2.0,
                "mismatch_penalty": 0.0,
                "mismatch_requires_both_high": True,
                "mismatch_confidence_threshold": 0.6,
            },
        },
    }


def test_pending_until_m_frames(base_cfg):
    mgr = IdentityManager(base_cfg)
    v = np.random.randn(512).astype(np.float32)

    # First M-1 frames should be pending
    for f in range(1, 4):
        rec = make_record(track_id=1, embedding=v, timestamp=100.0 + f * 0.1)
        res = mgr.process_track(rec, visible_track_ids={1})
        assert res.status == "pending"
        assert res.identity_id is None
        assert res.frame_count == f

    # 4th frame reaches M=4 -> resolved!
    rec = make_record(track_id=1, embedding=v, timestamp=100.4)
    res = mgr.process_track(rec, visible_track_ids={1})
    assert res.status == "new_identity"
    assert res.identity_id == 1
    assert not mgr.is_pending(1)


def test_simultaneous_tracks_never_share_identity(base_cfg):
    mgr = IdentityManager(base_cfg)
    v = np.random.randn(512).astype(np.float32)

    # Establish Track 1 as Identity 1
    for f in range(4):
        rec1 = make_record(track_id=1, embedding=v, timestamp=100.0 + f * 0.1)
        mgr.process_track(rec1, visible_track_ids={1})

    assert mgr.get_identity_for_track(1) == 1

    # Now Track 2 appears simultaneously with Track 1 in the same frame
    # Even with IDENTICAL embedding, Track 2 must NEVER link to Identity 1!
    for f in range(4):
        rec2 = make_record(track_id=2, embedding=v, timestamp=105.0 + f * 0.1)
        res2 = mgr.process_track(rec2, visible_track_ids={1, 2})

    assert res2.identity_id == 2  # Created new identity, did not steal active identity 1
    assert res2.identity_id != mgr.get_identity_for_track(1)


def test_bank_capped_at_n(base_cfg):
    base_cfg["identity"]["max_embeddings_n"] = 3
    mgr = IdentityManager(base_cfg)
    v = np.random.randn(512).astype(np.float32)

    # Resolve track 1
    for f in range(4):
        rec = make_record(track_id=1, embedding=v, timestamp=100.0 + f * 0.1)
        mgr.process_track(rec, visible_track_ids={1})

    identity_id = mgr.get_identity_for_track(1)
    # Feed 10 additional frames well-spaced in time
    for i in range(10):
        t = 200.0 + i * 2.0  # 2.0s apart, > 1.0s spacing
        rec = make_record(track_id=1, embedding=v, timestamp=t, overall_quality=0.85)
        mgr.process_track(rec, visible_track_ids={1})

    bank = mgr.get_bank(identity_id)
    assert len(bank) == 3  # Capped strictly at N=3


def test_spacing_enforced(base_cfg):
    base_cfg["identity"]["temporal_spacing_s"] = 2.0
    mgr = IdentityManager(base_cfg)
    v = np.random.randn(512).astype(np.float32)

    # Resolve track 1
    for f in range(4):
        rec = make_record(track_id=1, embedding=v, timestamp=100.0 + f * 0.1)
        mgr.process_track(rec, visible_track_ids={1})

    identity_id = mgr.get_identity_for_track(1)
    initial_count = len(mgr.get_bank(identity_id))

    # Send 5 frames only 0.2s apart (should all be blocked by spacing gate)
    for i in range(5):
        rec = make_record(track_id=1, embedding=v, timestamp=100.4 + (i + 1) * 0.2)
        mgr.process_track(rec, visible_track_ids={1})

    assert len(mgr.get_bank(identity_id)) == initial_count

    # Send a frame 3.0s later (> 2.0s spacing) -> should be enrolled
    rec = make_record(track_id=1, embedding=v, timestamp=105.0)
    mgr.process_track(rec, visible_track_ids={1})
    assert len(mgr.get_bank(identity_id)) == initial_count + 1


def test_quality_gate_blocks_low_quality(base_cfg):
    base_cfg["identity"]["quality_threshold"] = 0.50
    mgr = IdentityManager(base_cfg)
    v = np.random.randn(512).astype(np.float32)

    # Resolve track 1 with good quality
    for f in range(4):
        rec = make_record(track_id=1, embedding=v, timestamp=100.0 + f * 0.1, overall_quality=0.7)
        mgr.process_track(rec, visible_track_ids={1})

    identity_id = mgr.get_identity_for_track(1)
    count_after_resolve = len(mgr.get_bank(identity_id))

    # Send low quality frame (0.35 < 0.50) spaced in time
    rec_low = make_record(track_id=1, embedding=v, timestamp=120.0, overall_quality=0.35)
    mgr.process_track(rec_low, visible_track_ids={1})

    # Bank must not have accepted low quality record
    assert len(mgr.get_bank(identity_id)) == count_after_resolve


def test_ambiguous_match_creates_new_identity(base_cfg):
    # Set margin requirement to 0.10
    base_cfg["identity"]["link_threshold"] = 0.50
    base_cfg["identity"]["margin"] = 0.10
    mgr = IdentityManager(base_cfg)

    # Create Identity 1 with vector v1
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    for f in range(4):
        rec = make_record(track_id=1, embedding=v1, timestamp=100.0 + f * 0.1)
        mgr.process_track(rec, visible_track_ids={1})

    # Create Identity 2 with vector v2
    v2 = np.zeros(512, dtype=np.float32)
    v2[1] = 1.0
    for f in range(4):
        rec = make_record(track_id=2, embedding=v2, timestamp=200.0 + f * 0.1)
        mgr.process_track(rec, visible_track_ids={2})

    # Both identities are now inactive (neither 1 nor 2 in visible_track_ids)
    # Track 3 appears with a vector almost equidistant to v1 and v2 (cosine ~0.70 to both)
    v3 = np.zeros(512, dtype=np.float32)
    v3[0] = 0.71
    v3[1] = 0.70
    v3 /= np.linalg.norm(v3)

    for f in range(4):
        rec3 = make_record(track_id=3, embedding=v3, timestamp=300.0 + f * 0.1)
        res3 = mgr.process_track(rec3, visible_track_ids={3})

    # Both candidates have high scores >= 0.50, but margin < 0.10 -> AMBIGUOUS
    assert res3.decision == "ambiguous"
    assert res3.identity_id == 3  # Assigned new identity
    assert res3.suggestion is not None
    assert "candidate_id" in res3.suggestion
    # Bank for Identity 3 must NOT have accepted ambiguous embeddings
    assert len(mgr.get_bank(3)) == 0


def test_clear_link_when_above_threshold_and_margin(base_cfg):
    base_cfg["identity"]["link_threshold"] = 0.60
    base_cfg["identity"]["margin"] = 0.05
    mgr = IdentityManager(base_cfg)

    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0

    # Person 1 stands in frame
    for f in range(4):
        rec1 = make_record(track_id=1, embedding=v1, timestamp=100.0 + f * 0.1)
        mgr.process_track(rec1, visible_track_ids={1})
    assert mgr.get_identity_for_track(1) == 1

    # Person 1 leaves frame, then returns as Track 2
    # Visible tracks contains only {2}
    for f in range(4):
        rec2 = make_record(track_id=2, embedding=v1, timestamp=200.0 + f * 0.1)
        res2 = mgr.process_track(rec2, visible_track_ids={2})

    assert res2.decision == "link"
    assert res2.identity_id == 1  # Linked to Identity 1!


def test_suggest_only_when_link_threshold_is_none(base_cfg):
    base_cfg["identity"]["link_threshold"] = None
    mgr = IdentityManager(base_cfg)

    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0

    for f in range(4):
        rec1 = make_record(track_id=1, embedding=v1, timestamp=100.0 + f * 0.1)
        mgr.process_track(rec1, visible_track_ids={1})

    # Re-entry with identical embedding
    for f in range(4):
        rec2 = make_record(track_id=2, embedding=v1, timestamp=200.0 + f * 0.1)
        res2 = mgr.process_track(rec2, visible_track_ids={2})

    assert res2.decision == "suggest_only"
    assert res2.identity_id == 2  # New identity assigned in suggest-only mode
    assert res2.suggestion is not None
    assert res2.suggestion["candidate_id"] == 1


def test_score_logger_writes_csv(tmp_path, base_cfg):
    csv_path = tmp_path / "test_scores.csv"
    mgr = IdentityManager(base_cfg, score_log_path=str(csv_path))

    v = np.random.randn(512).astype(np.float32)
    for f in range(4):
        rec = make_record(track_id=1, embedding=v, timestamp=100.0 + f * 0.1)
        mgr.process_track(rec, visible_track_ids={1})

    mgr.score_logger.close()
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ScoreLogger.HEADER
        row = next(reader)
        assert len(row) == len(ScoreLogger.HEADER)
        assert row[1] == "1"  # new_track_id


def test_diversity_metric_does_not_increase_after_replacement(base_cfg):
    """Verify that when a full bank replaces an embedding, mean pairwise cosine does not increase."""
    base_cfg["identity"]["max_embeddings_n"] = 3
    base_cfg["identity"]["temporal_spacing_s"] = 0.5
    mgr = IdentityManager(base_cfg)

    # Fill bank with 3 highly correlated embeddings (close cluster around dim 0)
    v0 = np.zeros(512, dtype=np.float32); v0[0] = 1.0
    v1 = np.zeros(512, dtype=np.float32); v1[0] = 0.96; v1[1] = 0.28
    v2 = np.zeros(512, dtype=np.float32); v2[0] = 0.96; v2[1] = -0.28

    for f in range(4):
        rec = make_record(track_id=1, embedding=v0, timestamp=100.0 + f * 0.1)
        mgr.process_track(rec, visible_track_ids={1})

    # Add v1 and v2 with spacing
    mgr.process_track(make_record(track_id=1, embedding=v1, timestamp=101.0), visible_track_ids={1})
    mgr.process_track(make_record(track_id=1, embedding=v2, timestamp=102.0), visible_track_ids={1})

    bank = mgr.get_bank(1)
    assert len(bank) == 3
    bank_embs_old = np.stack([r.get_embedding_vector().flatten() for r in bank], axis=0)
    r_old = mean_pairwise_cosine(bank_embs_old)

    # Now introduce a diverse orthogonal vector (e.g. concentrated in dim 2)
    v_diverse = np.zeros(512, dtype=np.float32); v_diverse[2] = 1.0
    rec_diverse = make_record(track_id=1, embedding=v_diverse, timestamp=103.0)
    enrolled = mgr._maybe_enroll(identity_id=1, track_id=1, record=rec_diverse)
    assert enrolled is True

    bank_new = mgr.get_bank(1)
    assert len(bank_new) == 3
    bank_embs_new = np.stack([r.get_embedding_vector().flatten() for r in bank_new], axis=0)
    r_new = mean_pairwise_cosine(bank_embs_new)

    # Diversity metric (mean pairwise cosine) must decrease (diversity increases)
    assert r_new < r_old


def test_repeated_near_duplicate_candidate_is_discarded(base_cfg):
    """Verify that a candidate that is a near-duplicate of an existing embedding is discarded."""
    base_cfg["identity"]["max_embeddings_n"] = 3
    base_cfg["identity"]["temporal_spacing_s"] = 0.5
    mgr = IdentityManager(base_cfg)

    # Establish bank with 3 diverse orthogonal vectors
    e0 = np.zeros(512, dtype=np.float32); e0[0] = 1.0
    e1 = np.zeros(512, dtype=np.float32); e1[1] = 1.0
    e2 = np.zeros(512, dtype=np.float32); e2[2] = 1.0

    for f in range(4):
        rec = make_record(track_id=1, embedding=e0, timestamp=100.0 + f * 0.1)
        mgr.process_track(rec, visible_track_ids={1})

    mgr.process_track(make_record(track_id=1, embedding=e1, timestamp=101.0), visible_track_ids={1})
    mgr.process_track(make_record(track_id=1, embedding=e2, timestamp=102.0), visible_track_ids={1})

    bank = mgr.get_bank(1)
    assert len(bank) == 3
    orig_embs = [r.get_embedding_vector().copy() for r in bank]

    # Candidate that is a near duplicate of e0
    e_dup = np.zeros(512, dtype=np.float32)
    e_dup[0] = 0.999
    e_dup[3] = 0.04
    e_dup /= np.linalg.norm(e_dup)

    rec_dup = make_record(track_id=1, embedding=e_dup, timestamp=103.0)
    enrolled = mgr._maybe_enroll(identity_id=1, track_id=1, record=rec_dup)

    # Must be discarded because it does not reduce redundancy
    assert enrolled is False
    bank_after = mgr.get_bank(1)
    assert len(bank_after) == 3
    for orig, after in zip(orig_embs, bank_after):
        assert np.allclose(orig, after.get_embedding_vector(), atol=1e-5)


def test_quality_gate_auto_mode(base_cfg):
    """Verify auto quality gate calibrates to 10th percentile over specified frames."""
    base_cfg["identity"]["quality_threshold"] = "auto"
    base_cfg["identity"]["auto_gate_frames"] = 10
    base_cfg["identity"]["temporal_spacing_s"] = 0.1
    mgr = IdentityManager(base_cfg)
    assert mgr.quality_gate_mode == "auto"
    assert mgr._auto_calibrated is False

    # Track 1 appears and resolves
    v = np.random.randn(512).astype(np.float32)
    for f in range(4):
        rec = make_record(track_id=1, embedding=v, timestamp=100.0 + f * 0.2, overall_quality=0.50)
        mgr.process_track(rec, visible_track_ids={1})

    # Enroll frames with known qualities: 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
    qualities = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    for i, q in enumerate(qualities):
        rec = make_record(track_id=1, embedding=v, timestamp=101.0 + i * 0.2, overall_quality=q)
        mgr.process_track(rec, visible_track_ids={1})

    # Should now be calibrated (1 frame from resolve + 9 frames = 10 frames)
    assert mgr._auto_calibrated is True
    expected_p10 = float(np.percentile(mgr._session_enrolled_qualities[:10], 10))
    assert np.isclose(mgr.quality_threshold, expected_p10, atol=1e-4)

    # Frame with quality well below 10th percentile should be skipped
    bank_len = len(mgr.get_bank(1))
    rec_bad = make_record(track_id=1, embedding=v, timestamp=110.0, overall_quality=0.10)
    enrolled = mgr._maybe_enroll(identity_id=1, track_id=1, record=rec_bad)
    assert enrolled is False
    assert len(mgr.get_bank(1)) == bank_len

