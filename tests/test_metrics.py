"""
test_metrics.py -- Unit tests for eval/metrics.py.

Verifies Rank-K, mAP, MRR on a tiny hand-crafted distance matrix.
Results double-checked manually.
"""
import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from eval.metrics import compute_cmc, compute_map, compute_mrr, compute_far_frr


def _tiny_setup():
    """
    Tiny example: 2 queries, 4 gallery items.
    q0 matches g0 (pid=1, different cam), q1 matches g2 (pid=2, different cam).
    g1 is junk for q0 (same pid, same cam).
    """
    # Queries: pid, camid
    q_pids   = np.array([1, 2])
    q_camids = np.array([0, 0])
    # Gallery: pid, camid
    g_pids   = np.array([1, 1, 2, 3])
    g_camids = np.array([1, 0, 1, 1])   # g1 is same cam as q0 -> junk for q0
    # Distance matrix: lower = more similar
    # q0 is closest to g0 (correct), then g1 (junk), then g2, g3
    # q1 is closest to g2 (correct), then g0, then g1, g3
    dist_mat = np.array([
        [0.1, 0.2, 0.5, 0.9],   # q0
        [0.4, 0.6, 0.1, 0.8],   # q1
    ], dtype=np.float32)
    return dist_mat, q_pids, g_pids, q_camids, g_camids


class TestCMC:
    def test_rank1_perfect(self):
        dist_mat, q_pids, g_pids, q_camids, g_camids = _tiny_setup()
        cmc = compute_cmc(dist_mat, q_pids, g_pids, q_camids, g_camids, max_rank=4)
        # q0: after removing junk g1, rank-1 is g0 (pid=1) -> correct
        # q1: rank-1 is g2 (pid=2) -> correct
        assert cmc[0] == pytest.approx(1.0), f"Rank-1 should be 1.0, got {cmc[0]}"

    def test_cmc_non_decreasing(self):
        dist_mat, q_pids, g_pids, q_camids, g_camids = _tiny_setup()
        cmc = compute_cmc(dist_mat, q_pids, g_pids, q_camids, g_camids, max_rank=4)
        for i in range(len(cmc) - 1):
            assert cmc[i] <= cmc[i + 1] + 1e-6, "CMC must be non-decreasing"

    def test_cmc_max_is_one(self):
        dist_mat, q_pids, g_pids, q_camids, g_camids = _tiny_setup()
        cmc = compute_cmc(dist_mat, q_pids, g_pids, q_camids, g_camids, max_rank=4)
        assert all(v <= 1.0 + 1e-6 for v in cmc)

    def test_cmc_shape(self):
        dist_mat, q_pids, g_pids, q_camids, g_camids = _tiny_setup()
        cmc = compute_cmc(dist_mat, q_pids, g_pids, q_camids, g_camids, max_rank=3)
        assert cmc.shape == (3,)


class TestMAP:
    def test_perfect_map(self):
        dist_mat, q_pids, g_pids, q_camids, g_camids = _tiny_setup()
        mAP = compute_map(dist_mat, q_pids, g_pids, q_camids, g_camids)
        assert mAP == pytest.approx(1.0, abs=0.01), f"Expected mAP≈1.0, got {mAP}"

    def test_map_in_bounds(self):
        dist_mat, q_pids, g_pids, q_camids, g_camids = _tiny_setup()
        mAP = compute_map(dist_mat, q_pids, g_pids, q_camids, g_camids)
        assert 0.0 <= mAP <= 1.0


class TestMRR:
    def test_perfect_mrr(self):
        dist_mat, q_pids, g_pids, q_camids, g_camids = _tiny_setup()
        mrr = compute_mrr(dist_mat, q_pids, g_pids, q_camids, g_camids)
        assert mrr == pytest.approx(1.0, abs=0.01)


class TestFARFRR:
    def test_shape(self):
        scores = np.array([0.9, 0.4, 0.8, 0.2])
        labels = np.array([1, 0, 1, 0])
        thresholds = np.linspace(0, 1, 5)
        far, frr = compute_far_frr(scores, labels, thresholds)
        assert far.shape == (5,)
        assert frr.shape == (5,)

    def test_far_frr_in_bounds(self):
        scores = np.random.RandomState(0).rand(100)
        labels = (np.random.RandomState(1).rand(100) > 0.5).astype(int)
        thresholds = np.linspace(0, 1, 20)
        far, frr = compute_far_frr(scores, labels, thresholds)
        assert (far >= 0).all() and (far <= 1).all()
        assert (frr >= 0).all() and (frr <= 1).all()
