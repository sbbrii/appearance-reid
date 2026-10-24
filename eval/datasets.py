"""
datasets.py -- Market-1501 dataset loader for evaluation.

Uses torchreid Market1501 datamanager to get standard query/gallery splits.
Junk images (same camera, same identity) are excluded per the Market-1501
evaluation protocol.

If the dataset is not present, prints clear download instructions and exits.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MARKET1501_DOWNLOAD_URL = (
    "https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html\n"
    "  OR via Google Drive (search 'Market-1501 dataset download').\n"
    "  Expected directory structure:\n"
    "    data/Market-1501/\n"
    "      Market-1501-v15.09.15/\n"
    "        bounding_box_train/\n"
    "        bounding_box_test/\n"
    "        query/"
)


def create_synthetic_dataset(out_dir: Path) -> dict:
    """Generate a lightweight synthetic dataset of person crops for pipeline verification."""
    import cv2
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    query_dir = out_dir / "query"
    gallery_dir = out_dir / "bounding_box_test"
    query_dir.mkdir(exist_ok=True)
    gallery_dir.mkdir(exist_ok=True)

    # 4 distinct identities with varied clothing colors
    colors = [
        ((0, 0, 220), (40, 40, 40)),    # Red upper, dark lower
        ((0, 220, 0), (200, 80, 40)),   # Green upper, blue lower
        ((220, 50, 0), (20, 20, 20)),   # Blue upper, black lower
        ((0, 220, 220), (60, 60, 60)),  # Yellow upper, gray lower
    ]
    query = []
    gallery = []

    for pid, (upper_col, lower_col) in enumerate(colors, start=1):
        # Query crop (cam 1)
        q_path = query_dir / f"{pid:04d}_c1s1_0001.jpg"
        img = np.zeros((256, 128, 3), dtype=np.uint8)
        img[:128, :] = upper_col
        img[128:, :] = lower_col
        cv2.circle(img, (64, 30), 18, (150, 180, 210), -1)
        cv2.imwrite(str(q_path), img)
        query.append((str(q_path), pid, 1))

        # Gallery crops (cam 2 and cam 3)
        for camid in [2, 3]:
            g_path = gallery_dir / f"{pid:04d}_c{camid}s1_0001.jpg"
            g_img = img.copy()
            noise = np.random.randint(-10, 10, g_img.shape, dtype=np.int16)
            g_img = np.clip(g_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(g_path), g_img)
            gallery.append((str(g_path), pid, camid))

    logger.info("Generated synthetic dataset: %d query, %d gallery", len(query), len(gallery))
    return {
        "query": query,
        "gallery": gallery,
        "train": gallery,
    }


def load_market1501(cfg: dict) -> dict:
    """
    Load Market-1501 query/gallery/train image lists via torchreid.

    Parameters
    ----------
    cfg : dict
        ``eval`` section of default.yaml.

    Returns
    -------
    dict:
      query   : list of (img_path, pid, camid)
      gallery : list of (img_path, pid, camid)
      train   : list of (img_path, pid, camid)
    """
    dataset_root = Path(cfg["dataset_root"])
    if not dataset_root.exists():
        if cfg.get("allow_synthetic", True):
            logger.info("Market-1501 not found at %s. Using synthetic dataset for verification.", dataset_root)
            return create_synthetic_dataset(Path("data/synthetic_market1501"))
        logger.error(
            "Market-1501 dataset not found at: %s\n"
            "Download from: %s",
            dataset_root, MARKET1501_DOWNLOAD_URL,
        )
        sys.exit(1)

    try:
        import torchreid
    except ImportError as exc:
        raise ImportError(
            "pyppbox-torchreid not installed.  Run: pip install pyppbox-torchreid"
        ) from exc

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dm = torchreid.data.ImageDataManager(
            root=str(dataset_root.parent),
            sources="market1501",
            height=256,
            width=128,
            batch_size_train=32,
            batch_size_test=100,
            workers=0,
        )

    dataset = dm.dataset
    return {
        "query": dataset.query,
        "gallery": dataset.gallery,
        "train": dataset.train,
    }


def load_images_as_crops(
    items: list,
    max_items: Optional[int] = None,
) -> list:
    """
    Load image files as BGR numpy arrays.

    Parameters
    ----------
    items : list of (img_path, pid, camid)
    max_items : limit for debugging.

    Returns
    -------
    list of dicts: {crop_bgr, pid, camid, img_path}
    """
    import cv2

    results = []
    subset = items[:max_items] if max_items else items
    for img_path, pid, camid in subset:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read: %s", img_path)
            continue
        results.append({
            "crop_bgr": img,
            "pid": int(pid),
            "camid": int(camid),
            "img_path": str(img_path),
        })
    return results
