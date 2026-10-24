"""
embed.py -- OSNet feature extractor.

Produces L2-normalised 512-D embeddings from BGR person crops.
Preprocessing pipeline rewritten from understanding of reference repo
Real-Time-Person-ReID/OSNet_Extractor.py (no licence; not copied verbatim).

GPU policy (RTX 2050, 4 GB VRAM):
- fp16 autocast when cfg['fp16'] is True.
- Auto-halves batch and retries on CUDA OOM, logs the event.
- Caller must not run YOLO + OSNet simultaneously on the GPU.
"""
from __future__ import annotations

import logging
import pathlib
import warnings
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


def _build_osnet(model_name: str, pretrained: bool, num_classes: int = 1000):
    """Build an OSNet model via torchreid / pyppbox-torchreid."""
    try:
        import torchreid
    except ImportError as exc:
        raise ImportError(
            "pyppbox-torchreid is not installed.  "
            "Run: pip install pyppbox-torchreid"
        ) from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torchreid.models.build_model(
            name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    return model


class EmbeddingExtractor:
    """
    Extract L2-normalised 512-D embeddings from BGR person crops via OSNet.

    Parameters
    ----------
    cfg : dict
        The ``model`` section of default.yaml.
    device : str | None
        Force device; None = auto-detect CUDA.
    """

    def __init__(self, cfg: dict, device: Optional[str] = None) -> None:
        self.model_name: str = cfg["osnet_name"]
        self.embedding_dim: int = cfg["embedding_dim"]
        h, w = cfg["image_size"]
        self.image_size: tuple = (h, w)   # (H, W)
        self.fp16: bool = cfg.get("fp16", True)
        self.batch_size: int = cfg.get("batch_size", 32)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(
            "EmbeddingExtractor: device=%s  fp16=%s  batch=%d",
            self.device, self.fp16, self.batch_size,
        )

        raw_weights = cfg.get("weights_path")
        if raw_weights in ("imagenet", "none", "None", ""):
            use_pretrained = (raw_weights == "imagenet") or cfg.get("osnet_pretrained", True)
            weights_path = None
        elif raw_weights is None:
            candidate = pathlib.Path("models/osnet_x1_0_market1501.pth")
            if candidate.exists():
                weights_path = str(candidate)
                use_pretrained = False
            else:
                weights_path = None
                use_pretrained = cfg.get("osnet_pretrained", True)
        else:
            weights_path = raw_weights
            use_pretrained = False

        model = _build_osnet(
            model_name=self.model_name,
            pretrained=use_pretrained,
        )
        if weights_path and pathlib.Path(weights_path).exists():
            try:
                from torchreid.reid.utils.torchtools import load_pretrained_weights
            except ImportError:
                from pyppbox_torchreid.utils.torchtools import load_pretrained_weights
            load_pretrained_weights(model, weights_path)

        self.model = model.to(self.device).eval()

        # Transform: resize -> tensor -> ImageNet normalise.
        # Values from reference repo understanding (standard ImageNet stats).
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # -- public API -----------------------------------------------------------

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Extract embedding for one BGR crop.

        Returns shape (512,) float32, L2-normalised.
        """
        return self.extract_batch([crop_bgr])[0]

    def extract_batch(
        self,
        crops_bgr: list,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract embeddings for a list of BGR crops.

        Returns shape (N, 512) float32, each row L2-normalised.
        """
        if not crops_bgr:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        bsz = batch_size if batch_size is not None else self.batch_size
        tensors = [self._preprocess(c) for c in crops_bgr]
        parts: list = []
        for start in range(0, len(tensors), bsz):
            chunk = tensors[start : start + bsz]
            parts.append(self._forward(chunk))

        feats = np.concatenate(parts, axis=0)
        return self._l2_norm_batch(feats)

    # -- private --------------------------------------------------------------

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """BGR ndarray -> normalised float tensor (3, H, W)."""
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Empty or None crop.")
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.transform(pil)

    def _forward(self, tensors: list) -> np.ndarray:
        """Single OSNet forward pass; auto-retry with half batch on OOM."""
        batch = torch.stack(tensors, dim=0).to(self.device)
        use_fp16 = self.fp16 and self.device.type == "cuda"
        try:
            with torch.no_grad():
                if use_fp16:
                    with torch.amp.autocast('cuda'):
                        feats = self.model(batch)
                else:
                    feats = self.model(batch)
            return feats.detach().float().cpu().numpy()
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                torch.cuda.empty_cache()
                half = max(1, len(tensors) // 2)
                logger.warning("CUDA OOM on batch=%d, retrying with %d", len(tensors), half)
                left = self._forward(tensors[:half])
                right = self._forward(tensors[half:])
                return np.concatenate([left, right], axis=0)
            raise

    @staticmethod
    def _l2_norm_batch(features: np.ndarray) -> np.ndarray:
        """Per-row L2 normalisation."""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return features / norms
