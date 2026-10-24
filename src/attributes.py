"""
attributes.py -- OpenVINO person-attributes inference.

Loads person-attributes-recognition-crossroad-0230 IR model and runs
inference on BGR person crops.  Returns attribute probabilities and the
two sampling-point coordinates consumed by color.py.

Model spec (verified at runtime; raises if shapes differ):
  Input  shape : [1, 3, 160, 80]  BGR NCHW
  Output (attr): [1, 8, 1, 1]     8 attribute probabilities
  Output (coord): [1, 4, 1, 1]    4 floats [x_top, y_top, x_bot, y_bot]

Model runs on CPU only (per project requirement).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_EXPECTED_ATTR_DIM = 8
_EXPECTED_COORD_DIM = 4


class AttributeInference:
    """
    Wraps OpenVINO person-attributes-recognition-crossroad-0230.

    Parameters
    ----------
    cfg : dict
        The ``attributes`` section of default.yaml.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.input_h: int = int(cfg["input_height"])
        self.input_w: int = int(cfg["input_width"])
        self.attr_names: list = list(cfg["attribute_names"])
        self.thresholds: dict = dict(cfg["thresholds"])
        self._model_ready = False
        self._compiled = None
        self._infer_req = None
        self._attr_out_idx: Optional[int] = None
        self._coord_out_idx: Optional[int] = None

    def load(self, model_xml: Optional[str] = None) -> None:
        """
        Load and compile the OpenVINO IR model.  Call once before inference.

        Parameters
        ----------
        model_xml : path to .xml.  If None, uses cfg['model_xml'].
        """
        try:
            try:
                from openvino.runtime import Core
            except ImportError:
                from openvino import Core
        except ImportError as exc:
            raise ImportError(
                "openvino is not installed.  Run: pip install openvino"
            ) from exc

        xml_path = Path(model_xml or self.cfg["model_xml"])
        if not xml_path.exists():
            raise FileNotFoundError(
                f"OpenVINO model not found: {xml_path}\n"
                f"Run: python scripts/download_models.py"
            )

        core = Core()
        model = core.read_model(str(xml_path))
        self._compiled = core.compile_model(model, device_name="CPU")
        self._infer_req = self._compiled.create_infer_request()

        # Discover and validate output layers at load time.
        outputs = self._compiled.outputs
        logger.info("AttributeInference: %d output(s) found", len(outputs))
        attr_idx = coord_idx = None
        coord_2d_indices = []
        for out in outputs:
            shape = tuple(out.partial_shape.to_shape())
            flat_c = shape[1] if len(shape) >= 2 else shape[0]
            logger.info("  output idx=%d  name=%s  shape=%s", out.index, out.any_name, shape)
            if flat_c == _EXPECTED_ATTR_DIM:
                attr_idx = out.index
            elif flat_c == _EXPECTED_COORD_DIM:
                coord_idx = out.index
            elif flat_c == 2:
                coord_2d_indices.append(out.index)

        if attr_idx is None or (coord_idx is None and len(coord_2d_indices) != 2):
            shapes = [(o.any_name, tuple(o.partial_shape.to_shape())) for o in outputs]
            raise RuntimeError(
                f"Could not identify attribute (dim={_EXPECTED_ATTR_DIM}) or "
                f"coordinate (dim={_EXPECTED_COORD_DIM} or 2x2) outputs.\n"
                f"Actual outputs: {shapes}\n"
                f"Stopping -- verify model identity."
            )

        self._attr_out_idx = attr_idx
        self._coord_out_idx = coord_idx
        self._coord_2d_indices = coord_2d_indices
        logger.info(
            "AttributeInference ready: attr_out_idx=%d  coord_out_idx=%s  coord_2d=%s",
            attr_idx, coord_idx, coord_2d_indices,
        )
        self._model_ready = True

    def predict(self, crop_bgr: np.ndarray) -> dict:
        """
        Run inference on one BGR crop.

        Returns
        -------
        dict:
          probs  : dict[str, float]  attribute probabilities in [0,1]
          bools  : dict[str, bool]   thresholded booleans
          coords : tuple((x_top, y_top), (x_bot, y_bot)) in [0,1]
        """
        if not self._model_ready:
            raise RuntimeError("Call AttributeInference.load() first.")

        tensor = self._preprocess(crop_bgr)
        self._infer_req.infer({0: tensor})

        attr_raw = self._infer_req.get_output_tensor(self._attr_out_idx).data.flatten()
        if self._coord_out_idx is not None:
            coord_raw = self._infer_req.get_output_tensor(self._coord_out_idx).data.flatten()
            x_top, y_top, x_bot, y_bot = coord_raw[:4]
        else:
            top_raw = self._infer_req.get_output_tensor(self._coord_2d_indices[0]).data.flatten()
            bot_raw = self._infer_req.get_output_tensor(self._coord_2d_indices[1]).data.flatten()
            x_top, y_top = top_raw[:2]
            x_bot, y_bot = bot_raw[:2]

        probs = {name: float(attr_raw[i]) for i, name in enumerate(self.attr_names)}
        bools = {
            name: float(attr_raw[i]) >= self.thresholds.get(name, 0.5)
            for i, name in enumerate(self.attr_names)
        }
        coords = ((float(x_top), float(y_top)), (float(x_bot), float(y_bot)))
        return {"probs": probs, "bools": bools, "coords": coords}

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Resize crop to model input size, return NCHW float32 tensor."""
        resized = cv2.resize(
            crop_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
        )
        tensor = resized.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
        return tensor[np.newaxis, ...]  # (1, 3, H, W)
