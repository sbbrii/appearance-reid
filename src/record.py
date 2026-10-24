"""
record.py -- FeatureRecord dataclass and JSON serialisation.

A FeatureRecord captures one observation of a person from one camera at one
moment in time.  It carries all modalities needed by fusion.py and gallery.py.
Raw crops are NOT stored here unless the debug flag is set (privacy by design).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class ColorInfo:
    """Per-region color label with confidence."""
    upper: str = "unknown"
    lower: str = "unknown"
    upper_conf: float = 0.0
    lower_conf: float = 0.0
    sampling_path: str = "fallback"


@dataclass
class AttributeInfo:
    """Soft probabilities (0-1) for each person attribute."""
    is_male: float = 0.5
    has_bag: float = 0.0
    has_backpack: float = 0.0
    has_hat: float = 0.0
    has_longsleeves: float = 0.5
    has_longpants: float = 0.5
    has_longhair: float = 0.0
    has_coat_jacket: float = 0.0

    def to_bool_dict(self, thresholds: dict) -> dict:
        """Apply per-attribute thresholds -> boolean dict."""
        return {
            k: getattr(self, k) >= thresholds.get(k, 0.5)
            for k in self.__dataclass_fields__
        }


@dataclass
class QualityInfo:
    """Quality sub-scores and overall score, all in [0, 1]."""
    blur: float = 0.5        # 1 = sharp, 0 = very blurry
    visibility: float = 1.0  # 1 = fully visible, 0 = heavily occluded
    size: float = 1.0        # 1 = large enough, 0 = tiny
    overall: float = 0.5     # weighted combination


@dataclass
class FeatureRecord:
    """
    One observation of one person from one camera.

    Fields
    ------
    track_id    : tracker-assigned integer ID (ByteTrack).
    camera_id   : string identifier for the camera source.
    timestamp   : Unix epoch float (seconds).
    bbox        : [x1, y1, x2, y2] in pixel coordinates.
    embedding   : L2-normalised 512-D float32 array (OSNet).
    colors      : ColorInfo.
    attributes  : AttributeInfo.
    quality     : QualityInfo.
    record_id   : unique UUID string (auto-generated).
    """
    track_id: int
    camera_id: str = "cam_0"
    timestamp: float = 0.0
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    colors: ColorInfo = field(default_factory=ColorInfo)
    attributes: AttributeInfo = field(default_factory=AttributeInfo)
    quality: QualityInfo = field(default_factory=QualityInfo)
    record_id: str = field(default="")

    def get_embedding_vector(self) -> np.ndarray:
        return self.embedding

    def __post_init__(self) -> None:
        if not self.record_id:
            self.record_id = str(uuid.uuid4())
        self.embedding = np.asarray(self.embedding, dtype=np.float32)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict (embedding as list)."""
        d = asdict(self)
        d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureRecord":
        """Reconstruct from the dict produced by to_dict()."""
        d = dict(d)
        d["embedding"] = np.array(d["embedding"], dtype=np.float32)
        d["colors"] = ColorInfo(**d["colors"])
        d["attributes"] = AttributeInfo(**d["attributes"])
        d["quality"] = QualityInfo(**d["quality"])
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "FeatureRecord":
        return cls.from_dict(json.loads(s))

    def embedding_f32(self) -> np.ndarray:
        """Contiguous float32 row vector (1, 512) ready for FAISS."""
        return np.ascontiguousarray(self.embedding.reshape(1, -1), dtype=np.float32)
