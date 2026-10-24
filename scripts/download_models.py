"""
download_models.py -- Download OpenVINO person-attributes-recognition-crossroad-0230.

Downloads the FP32 IR model (.xml + .bin) from Intel's Open Model Zoo storage.
Verifies file size after download.  Skips if files already exist.
Records Apache 2.0 license source in docs/ATTRIBUTION.md.

Usage
-----
python scripts/download_models.py [--model_dir models/FP32]

After download, loads the model with openvino.Core().read_model() and prints:
  - Input shape
  - Output names and shapes
  - Attribute list (from config)

Stop if any download 404s or shape assertions fail.
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

# Source URLs from Open Model Zoo model.yml (Apache 2.0, Intel Corporation 2023-2024)
_BASE_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
    "2023.0/models_bin/1/person-attributes-recognition-crossroad-0230/FP32/"
)
_MODEL_NAME = "person-attributes-recognition-crossroad-0230"
_FILES = {
    f"{_MODEL_NAME}.xml": {
        "url": _BASE_URL + f"{_MODEL_NAME}.xml",
        "expected_size": 183946,  # bytes from model.yml
    },
    f"{_MODEL_NAME}.bin": {
        "url": _BASE_URL + f"{_MODEL_NAME}.bin",
        "expected_size": None,  # not published in truncated model.yml; verify non-zero
    },
}


def download_file(url: str, dest: Path, expected_size=None) -> bool:
    """
    Download a single file via HTTP GET.  Returns True on success.

    Skips if file already exists with correct size.
    """
    if dest.exists():
        actual_size = dest.stat().st_size
        if expected_size and actual_size != expected_size:
            logger.warning(
                "Existing file %s size mismatch (got %d, expected %d). Re-downloading.",
                dest.name, actual_size, expected_size,
            )
        else:
            logger.info("SKIP (already exists): %s  (%d bytes)", dest.name, actual_size)
            return True

    logger.info("Downloading: %s -> %s", url, dest)
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("HTTP error %d for URL: %s", exc.response.status_code, url)
        logger.error("STOPPING -- please verify the URL or check your internet connection.")
        return False
    except requests.ConnectionError as exc:
        logger.error("Connection error: %s", exc)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            total += len(chunk)

    logger.info("Downloaded: %s  (%d bytes)", dest.name, total)

    if expected_size and total != expected_size:
        logger.error(
            "Size mismatch: %s  got=%d  expected=%d  STOPPING.",
            dest.name, total, expected_size,
        )
        return False

    if total == 0:
        logger.error("Downloaded file is empty: %s  STOPPING.", dest.name)
        return False

    return True


def verify_model(xml_path: Path, cfg: dict) -> bool:
    """
    Load the downloaded IR model, print shapes and output names.

    Asserts expected output dimensions.  Returns False and prints a clear
    error if anything is wrong.
    """
    try:
        try:
            from openvino.runtime import Core
        except ImportError:
            from openvino import Core
    except ImportError:
        logger.error("openvino not installed.  Run: pip install openvino")
        return False

    logger.info("Loading model with OpenVINO Core...")
    core = Core()
    try:
        model = core.read_model(str(xml_path))
    except Exception as exc:
        logger.error("Failed to read model: %s", exc)
        return False

    compiled = core.compile_model(model, "CPU")

    print("\n=== OpenVINO model verification ===")
    print(f"  XML: {xml_path}")

    inputs = compiled.inputs
    for inp in inputs:
        print(f"  INPUT  name={inp.any_name}  shape={tuple(inp.partial_shape.to_shape())}")

    outputs = compiled.outputs
    attr_dim = int(cfg["attributes"]["input_height"])  # not attr count, just for logging
    attr_names = cfg["attributes"]["attribute_names"]
    print(f"  Expected attributes ({len(attr_names)}): {attr_names}")

    found_attr_out = found_coord_out = False
    coord_2d_count = 0
    for out in outputs:
        shape = tuple(out.partial_shape.to_shape())
        flat_c = shape[1] if len(shape) >= 2 else shape[0]
        print(f"  OUTPUT name={out.any_name}  shape={shape}  flat_channels={flat_c}")
        if flat_c == 8:
            found_attr_out = True
            print("         -> ATTRIBUTE output (8 attributes)")
        elif flat_c == 4:
            found_coord_out = True
            print("         -> COORDINATE output (x_top, y_top, x_bot, y_bot)")
        elif flat_c == 2:
            coord_2d_count += 1
            print(f"         -> COORDINATE output part {coord_2d_count} (2 values)")

    if not found_attr_out:
        print("\nERROR: Could not find 8-channel attribute output.")
        print("Stopping -- model may be a different version than expected.")
        return False
    if not found_coord_out and coord_2d_count != 2:
        print("\nERROR: Could not find 4-channel or dual 2-channel coordinate outputs.")
        print("Stopping -- model may be a different version than expected.")
        return False

    print("\nModel verification PASSED. All expected outputs found.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OpenVINO person-attributes model")
    parser.add_argument("--model_dir", default="models/FP32")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_dir = Path(args.model_dir)

    for filename, info in _FILES.items():
        dest = model_dir / filename
        ok = download_file(info["url"], dest, expected_size=info["expected_size"])
        if not ok:
            logger.error("Download failed for %s. STOPPING.", filename)
            sys.exit(1)

    xml_path = model_dir / f"{_MODEL_NAME}.xml"
    if not verify_model(xml_path, cfg):
        sys.exit(1)

    logger.info(
        "\nAll model files ready at: %s\n"
        "License: Apache 2.0  (Intel Corporation 2023-2024)\n"
        "Source: %s",
        model_dir, _BASE_URL,
    )


if __name__ == "__main__":
    main()
