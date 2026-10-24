"""
gallery.py -- FAISS inner-product index + metadata store.

FAISS gallery rewritten from understanding of Real-Time-Person-ReID/main.py
L21-79 (no licence; not copied verbatim).

Key differences from the reference:
- Stores full FeatureRecord metadata alongside FAISS IDs.
- Returns ranked top-K candidates, not a single identity assignment.
- Supports TTL-based privacy purge.
- Results are probabilistic candidates, not proof of identity.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SearchResult(dict):
    """Candidate dict supporting both dict lookup and tuple unpacking (record, emb_score)."""
    def __iter__(self):
        return iter((self["record"], self["emb_score"]))

    def __getitem__(self, key):
        if isinstance(key, int):
            return (self["record"], self["emb_score"])[key]
        return super().__getitem__(key)


class GalleryIndex:
    """
    FAISS-backed gallery for person re-ID.

    Inner-product search on L2-normalised embeddings is equivalent to
    cosine similarity.  Metadata (FeatureRecord objects) is stored in
    a parallel dict keyed by integer FAISS ID.

    Parameters
    ----------
    cfg : dict
        The ``gallery`` section of default.yaml.
    embedding_dim : int
        Dimensionality of embeddings (512 for OSNet x1_0).
    """

    def __init__(self, cfg: Optional[dict] = None, embedding_dim: int = 512, **kwargs) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed.  Run: pip install faiss-cpu"
            ) from exc
        import faiss as _faiss

        cfg = cfg or {}
        self.top_k_faiss: int = int(kwargs.get("top_k_faiss", cfg.get("top_k_faiss", 50)))
        self.top_k_output: int = int(kwargs.get("top_k_output", cfg.get("top_k_output", 10)))
        self.ttl_seconds: float = float(kwargs.get("ttl_seconds", cfg.get("ttl_seconds", 86400)))
        self.embedding_dim = int(kwargs.get("embedding_dim", embedding_dim))

        base = _faiss.IndexFlatIP(embedding_dim)
        self._index = _faiss.IndexIDMap(base)
        self._meta: dict = {}
        self._timestamps: dict = {}
        self._next_id: int = 0

        logger.info(
            "GalleryIndex: dim=%d  top_k_faiss=%d  ttl=%.0fs",
            embedding_dim, self.top_k_faiss, self.ttl_seconds,
        )

    @property
    def size(self) -> int:
        return self._index.ntotal

    def __len__(self) -> int:
        return self._index.ntotal

    def add(self, record: Any) -> int:
        """
        Add a FeatureRecord to the gallery.

        Returns the integer FAISS ID assigned.
        """
        fid = self._next_id
        self._next_id += 1
        import faiss as _faiss
        vec = record.embedding_f32()  # (1, 512) float32
        ids = np.array([fid], dtype=np.int64)
        self._index.add_with_ids(vec, ids)
        self._meta[fid] = record
        self._timestamps[fid] = time.time()
        return fid

    def search(self, query: Any, top_k: Optional[int] = None) -> list:
        """
        Retrieve top-K gallery candidates for a query.

        Parameters
        ----------
        query : FeatureRecord or np.ndarray
        top_k : int, optional

        Returns
        -------
        list of SearchResult (dict with tuple unpacking support: record, emb_score)
        """
        if self._index.ntotal == 0:
            return []

        if isinstance(query, np.ndarray):
            vec = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        elif hasattr(query, "embedding_f32"):
            vec = query.embedding_f32()
        elif hasattr(query, "get_embedding_vector"):
            vec = np.ascontiguousarray(query.get_embedding_vector().reshape(1, -1), dtype=np.float32)
        else:
            vec = np.ascontiguousarray(np.array(query, dtype=np.float32).reshape(1, -1))

        k_limit = top_k if top_k is not None else self.top_k_faiss
        k = min(k_limit, self._index.ntotal)
        scores, ids = self._index.search(vec, k)
        scores, ids = scores[0], ids[0]

        results = []
        for score, fid in zip(scores, ids):
            if fid < 0:
                continue
            rec = self._meta.get(int(fid))
            if rec is None:
                continue
            results.append(SearchResult({
                "record": rec,
                "faiss_id": int(fid),
                "emb_score": float(score),
            }))
        out_k = top_k if top_k is not None else self.top_k_output
        return results[: out_k]

    def purge_expired(self) -> int:
        """Remove records older than ttl_seconds.  Returns count removed."""
        if self.ttl_seconds <= 0:
            return 0
        now = time.time()
        expired = [fid for fid, ts in self._timestamps.items() if (now - ts) > self.ttl_seconds]
        if not expired:
            return 0
        import faiss as _faiss
        self._index.remove_ids(np.array(expired, dtype=np.int64))
        for fid in expired:
            self._meta.pop(fid, None)
            self._timestamps.pop(fid, None)
        logger.info("Gallery TTL purge: removed %d records", len(expired))
        return len(expired)

    def clear(self) -> None:
        """Remove all records."""
        self._index.reset()
        self._meta.clear()
        self._timestamps.clear()
        self._next_id = 0
