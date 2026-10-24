"""
test_pipeline_modules.py -- Tests for embed, attributes, gallery, and record modules.
"""
import numpy as np
import pytest
import yaml

from src.record import FeatureRecord, ColorInfo, AttributeInfo, QualityInfo
from src.gallery import GalleryIndex
from src.attributes import AttributeInference
from src.embed import EmbeddingExtractor


@pytest.fixture
def cfg():
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestFeatureRecord:
    def test_record_serialization(self):
        rec = FeatureRecord(
            track_id=1,
            camera_id="cam_01",
            timestamp=1700000000.0,
            bbox=[10, 20, 100, 200],
            embedding=np.random.randn(512).astype(np.float32),
            quality=QualityInfo(overall=0.85, blur=0.8, visibility=0.9, size=0.85),
            colors=ColorInfo(upper="blue", lower="black", upper_conf=0.9, lower_conf=0.8),
            attributes=AttributeInfo(is_male=0.9, has_backpack=0.8),
        )
        d = rec.to_dict()
        assert isinstance(d, dict)
        assert d["track_id"] == 1
        assert d["camera_id"] == "cam_01"
        assert d["bbox"] == [10, 20, 100, 200]
        assert len(d["embedding"]) == 512

        rec2 = FeatureRecord.from_dict(d)
        assert rec2.track_id == rec.track_id
        assert rec2.camera_id == rec.camera_id
        assert rec2.timestamp == rec.timestamp
        assert rec2.bbox == rec.bbox
        assert np.allclose(rec.embedding_f32(), rec2.embedding_f32(), atol=1e-5)
        assert rec2.colors.upper == "blue"
        assert rec2.attributes.is_male == 0.9


class TestEmbeddingExtractor:
    def test_extractor_forward(self, cfg):
        extractor = EmbeddingExtractor(cfg["model"], device="cpu")
        crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
        emb = extractor.extract(crop)
        assert emb.shape == (512,)
        norm = np.linalg.norm(emb)
        assert np.isclose(norm, 1.0, atol=1e-4)

    def test_extractor_batch(self, cfg):
        extractor = EmbeddingExtractor(cfg["model"], device="cpu")
        crops = [np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8) for _ in range(3)]
        embs = extractor.extract_batch(crops)
        assert embs.shape == (3, 512)
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)


class TestAttributeInference:
    def test_attribute_inference_live(self, cfg):
        ai = AttributeInference(cfg["attributes"])
        ai.load()
        crop = np.random.randint(0, 255, (160, 80, 3), dtype=np.uint8)
        res = ai.predict(crop)
        assert "probs" in res
        assert "bools" in res
        assert "coords" in res
        assert len(res["probs"]) == 8
        assert len(res["coords"]) == 2
        for v in res["probs"].values():
            assert 0.0 <= v <= 1.0


class TestGalleryIndex:
    def test_gallery_add_and_search(self, cfg):
        gallery = GalleryIndex(cfg["gallery"], embedding_dim=512)
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)

        rec1 = FeatureRecord(
            track_id=101,
            camera_id="cam_01",
            timestamp=1700000000.0,
            bbox=[10, 20, 100, 200],
            embedding=emb1,
            quality=QualityInfo(overall=0.9),
            colors=ColorInfo(upper="red", lower="blue"),
        )
        gallery.add(rec1)
        assert len(gallery) == 1

        results = gallery.search(rec1)
        assert len(results) == 1
        top_item = results[0]
        assert top_item["record"].track_id == 101
        assert np.isclose(top_item["emb_score"], 1.0, atol=1e-4)
