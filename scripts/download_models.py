"""
download_models.py -- Download and verify all runtime models for SAFEPASS.

Downloads:
1. OpenVINO FP32 IR model (.xml + .bin) for person attributes recognition
   from Intel's Open Model Zoo storage (Apache 2.0).
2. OSNet x1.0 Market-1501 feature extractor checkpoint (.pth)
   from Torchreid Model Zoo with SHA256 verification.
3. YOLOv8n detector (.pt) for offline-first readiness with size check.

Usage:
------
python scripts/download_models.py [--models_dir models] [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# OpenVINO person-attributes-recognition-crossroad-0230 (Apache 2.0)
_OPENVINO_BASE_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
    "2023.0/models_bin/1/person-attributes-recognition-crossroad-0230/FP32/"
)
_OPENVINO_MODEL_NAME = "person-attributes-recognition-crossroad-0230"
_OPENVINO_FILES = {
    f"{_OPENVINO_MODEL_NAME}.xml": {
        "url": _OPENVINO_BASE_URL + f"{_OPENVINO_MODEL_NAME}.xml",
        "expected_size": 183946,
    },
    f"{_OPENVINO_MODEL_NAME}.bin": {
        "url": _OPENVINO_BASE_URL + f"{_OPENVINO_MODEL_NAME}.bin",
        "expected_size": 2939232,
    },
}

# OSNet Market-1501 weights (Kaiyang Zhou, deep-person-reid Model Zoo)
_OSNET_GDRIVE_ID = "1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA"
_OSNET_FILENAME = "osnet_x1_0_market1501.pth"
_OSNET_SHA256 = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"
_OSNET_EXPECTED_SIZE = 10399605

# YOLOv8n (Ultralytics AGPL-3.0)
_YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
_YOLO_FILENAME = "yolov8n.pt"
_YOLO_MIN_SIZE = 6000000


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_size: int | None = None) -> bool:
    """Download a single file via HTTP GET with streaming and size check."""
    if dest.exists():
        actual_size = dest.stat().st_size
        if expected_size and actual_size != expected_size:
            logger.warning(
                "Existing file %s size mismatch (got %d, expected %d). Re-downloading.",
                dest.name, actual_size, expected_size,
            )
        else:
            logger.info("SKIP (already exists): %s (%d bytes)", dest.name, actual_size)
            return True

    logger.info("Downloading: %s -> %s", url, dest)
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("HTTP error %d for URL: %s", exc.response.status_code, url)
        return False
    except requests.ConnectionError as exc:
        logger.error("Connection error while connecting to %s: %s", url, exc)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            total += len(chunk)

    logger.info("Downloaded: %s (%d bytes)", dest.name, total)

    if expected_size and total != expected_size:
        logger.error(
            "Size mismatch for %s: got=%d, expected=%d. File might be corrupt.",
            dest.name, total, expected_size,
        )
        return False

    if total == 0:
        logger.error("Downloaded file is empty: %s", dest.name)
        return False

    return True


def download_osnet(dest: Path) -> bool:
    """Download OSNet Market-1501 checkpoint with SHA256 verification."""
    if dest.exists():
        if dest.stat().st_size == _OSNET_EXPECTED_SIZE:
            h = compute_sha256(dest)
            if h == _OSNET_SHA256:
                logger.info("SKIP (verified): %s (SHA256: %s)", dest.name, h[:16])
                return True
            logger.warning("Checksum mismatch for existing %s. Re-downloading.", dest.name)
        else:
            logger.warning("Size mismatch for existing %s. Re-downloading.", dest.name)

    logger.info("Downloading OSNet weights (%s) from Torchreid Google Drive zoo...", dest.name)
    try:
        import gdown
        dest.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=_OSNET_GDRIVE_ID, output=str(dest), quiet=False)
    except Exception as exc:
        logger.error("Failed to download via gdown: %s", exc)
        logger.error("You can manually download OSNet weights from: https://drive.google.com/uc?id=%s", _OSNET_GDRIVE_ID)
        return False

    if not dest.exists() or dest.stat().st_size == 0:
        logger.error("OSNet download failed: file is missing or empty.")
        return False

    actual_hash = compute_sha256(dest)
    if actual_hash != _OSNET_SHA256:
        logger.error(
            "OSNet SHA256 verification failed!\n  Got:      %s\n  Expected: %s",
            actual_hash, _OSNET_SHA256,
        )
        return False

    logger.info("OSNet verified successfully (SHA256: %s).", actual_hash[:16])
    return True


def download_yolo(dest: Path) -> bool:
    """Verify or download YOLOv8n weights for offline-ready first run."""
    if dest.exists() and dest.stat().st_size > _YOLO_MIN_SIZE:
        logger.info("SKIP (already exists): %s (%d bytes)", dest.name, dest.stat().st_size)
        return True

    logger.info("Downloading YOLOv8n detector (%s)...", dest.name)
    try:
        from ultralytics import YOLO
        # Ultralytics will fetch to cwd if missing
        _ = YOLO("yolov8n.pt")
        if dest.exists() and dest.stat().st_size > _YOLO_MIN_SIZE:
            logger.info("YOLOv8n successfully initialized: %s", dest.name)
            return True
    except Exception as exc:
        logger.warning("Ultralytics auto-download notice: %s. Trying direct download...", exc)

    return download_file(_YOLO_URL, dest)


def verify_openvino_model(xml_path: Path, cfg: dict) -> bool:
    """Load the OpenVINO attribute model and check outputs."""
    try:
        from openvino import Core
    except ImportError:
        logger.error("openvino not installed. Run: pip install openvino")
        return False

    core = Core()
    try:
        model = core.read_model(str(xml_path))
        compiled = core.compile_model(model, "CPU")
    except Exception as exc:
        logger.error("Failed to read/compile OpenVINO model: %s", exc)
        return False

    outputs = compiled.outputs
    found_attr_out = False
    for out in outputs:
        shape = tuple(out.partial_shape.to_shape())
        flat_c = shape[1] if len(shape) >= 2 else shape[0]
        if flat_c == 8:
            found_attr_out = True
            break
    if not found_attr_out:
        logger.error("Could not find expected 8-channel attribute output.")
        return False

    logger.info("OpenVINO attribute model verified successfully.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download runtime models for SAFEPASS")
    parser.add_argument("--models_dir", default="models", help="Directory for downloaded models")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file path")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    fp32_dir = models_dir / "FP32"
    fp32_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. OpenVINO Attributes Model
    logger.info("\n=== 1/3 Checking OpenVINO Attribute Recognition Model ===")
    for filename, info in _OPENVINO_FILES.items():
        dest = fp32_dir / filename
        if not download_file(info["url"], dest, expected_size=info["expected_size"]):
            logger.error("Failed to download OpenVINO file: %s", filename)
            sys.exit(1)

    xml_path = fp32_dir / f"{_OPENVINO_MODEL_NAME}.xml"
    if not verify_openvino_model(xml_path, cfg):
        sys.exit(1)

    # 2. OSNet Re-ID Model
    logger.info("\n=== 2/3 Checking OSNet x1.0 Re-ID Checkpoint ===")
    osnet_dest = models_dir / _OSNET_FILENAME
    if not download_osnet(osnet_dest):
        sys.exit(1)

    # 3. YOLOv8n Detection Model
    logger.info("\n=== 3/3 Checking YOLOv8n Detector Checkpoint ===")
    yolo_dest = Path(_YOLO_FILENAME)
    if not download_yolo(yolo_dest):
        sys.exit(1)

    logger.info("\n[SUCCESS] All runtime models are downloaded and verified.")
    logger.info("You can now run the live webcam demo without internet access:")
    logger.info("  python -m src.pipeline --source 0")


if __name__ == "__main__":
    main()
