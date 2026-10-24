# SAFEPASS — Appearance + Quality-Aware Fusion Module

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-70%20passed-green.svg)](tests/)

Multi-modal Person Re-Identification with quality-aware adaptive appearance fusion, persistent multi-embedding identity tracking, and real-time CCTV webcam inference.

---

## Quickstart (Windows PowerShell)

Follow these steps to clone and run the real-time webcam demo with zero external files:

### 1. Clone & Set Up Environment

Open **PowerShell** and run:

```powershell
# Clone the repository
git clone https://github.com/sbbrii/appearance-reid.git
cd appearance-reid

# Create a virtual environment (Python 3.10 - 3.14)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip and install PyTorch with CUDA (or CPU)
# For NVIDIA GPU (CUDA 12.8 / 12.9):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# Or for CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Download Models (One Command)

Download and verify the required neural network weights (OpenVINO attribute model, OSNet Re-ID checkpoint, and YOLOv8n detector):

```powershell
python scripts/download_models.py
```
*All weights are verified with file size and SHA256 checksums before proceeding.*

### 3. Run the Live Webcam Demo

```powershell
python -m src.pipeline --source 0
```

#### Camera Selection & Keys
- **Select Camera**: `--source 0` selects default webcam. If you have multiple webcams (e.g. external USB camera or virtual camera), pass `--source 1` or `--source 2`.
- **Run on a Video File**: `python -m src.pipeline --source path\to\cctv.mp4`
- **Interactive Controls**:
  - `q`: Quit the demo.
  - `f`: Toggle full-screen display.

#### Expected First-Run Behavior
1. YOLOv8n detects persons in the scene and ByteTrack assigns a transient `track_id` (`T:X`).
2. The pipeline collects high-quality crops into an M-frame rolling buffer (`ID:?`).
3. Once enrolled, the persistent identity engine creates or links to an `ID:X` badge.
4. If a person leaves the camera frame and returns after several seconds, the system queries the in-memory gallery and re-identifies them with their persistent `ID` based on cosine similarity and multi-modal appearance fusion.
5. Live FPS, latency breakdown, bounding boxes, dominant upper/lower body colors, and attribute predictions are displayed on the real-time HUD.

---

## Privacy & Data Handling Note

- **In-Memory Only**: All feature embeddings, color descriptors, and gallery indices are stored strictly in volatile RAM for the duration of the process.
- **No Disk Storage by Default**: No video frames, face crops, or personal recordings are saved to disk unless `--record 1` is explicitly supplied.
- **Log Hygiene**: Session score logs (`*.csv`, `*.log`) are ignored by version control and must not be committed to GitHub.

---

## System Architecture

```
                      Video Frame / CCTV Camera / Video File
                                         │
                                         ▼
                          ┌─────────────────────────────┐
                          │   Detection & Tracking      │
                          │ YOLOv8n + ByteTrack (CPU/GPU)│
                          └──────────────┬──────────────┘
                                         │  Crop BGR + Track ID + BBox
                 ┌───────────────────────┼──────────────────────┐
                 ▼                       ▼                      ▼
       ┌──────────────────┐    ┌──────────────────┐   ┌──────────────────┐
       │     embed.py     │    │  attributes.py   │   │    quality.py    │
       │ OSNet (fp16 GPU) │    │ OpenVINO IR CPU  │   │ Blur, Visibility,│
       │ 512-D L2 Embed   │    │ 8 Attributes +   │   │ Size Proxy       │
       └────────┬─────────┘    │ 2 Color Pts      │   └────────┬─────────┘
                │              └─────────┬────────┘            │
                │                        │ Sampling Points     │
                │              ┌─────────▼────────┐            │
                │              │     color.py     │            │
                │              │ CIELAB Centroids │            │
                │              │ Upper/Lower + Conf│           │
                │              └─────────┬────────┘            │
                │                        │                     │
                └────────────────────────┼─────────────────────┘
                                         │
                                    FeatureRecord
                                         │
                               ┌─────────▼────────┐
                               │   identity.py    │
                               │ Multi-Embedding  │
                               │ Persistent Bank  │
                               └─────────┬────────┘
                                         │
                               ┌─────────▼────────┐
                               │    gallery.py    │
                               │ FAISS IndexFlatIP│
                               │ Top-K Candidates │
                               └─────────┬────────┘
                                         │ Top-K Candidates
                               ┌─────────▼────────┐
                               │    fusion.py     │
                               │ Quality-Aware    │
                               │ Adaptive Re-Rank │
                               └─────────┬────────┘
                                         │
                               ┌─────────▼────────┐
                               │   pipeline.py    │
                               │ Display / HUD    │
                               └──────────────────┘
```

### Module Overview

- **`src/detect_track.py`**: YOLOv8 person detection coupled with ByteTrack association.
- **`src/embed.py`**: OSNet x1.0 feature extractor producing L2-normalized 512-D appearance embeddings. Supports fp16 GPU autocast and automatic OOM recovery.
- **`src/identity.py`**: Persistent identity layer maintaining separate `identity_id` from transient `track_id`, an M-frame enrollment buffer, an N-embedding diversity bank with pairwise-cosine redundancy replacement, and ambiguity guards.
- **`src/quality.py`**: Multi-factor crop quality estimation assessing sharpness (Laplacian variance), visibility (aspect ratio & border proximity), and pixel resolution.
- **`src/attributes.py`**: Intel OpenVINO inference engine extracting 8 pedestrian attributes (`is_male`, `has_bag`, `has_backpack`, `has_hat`, `has_longsleeves`, `has_longpants`, `has_longhair`, `has_coat_jacket`) and color sampling points.
- **`src/color.py`**: CIELAB dominant upper/lower color classification with spatial sampling.
- **`src/gallery.py`**: High-performance in-memory gallery powered by FAISS `IndexFlatIP`.
- **`src/fusion.py`**: Quality-aware appearance fusion engine implementing baseline cosine, fixed weighting, dynamic softmax gating, and learned logistic calibration.
- **`src/pipeline.py`**: Live video pipeline integrating capture, inference, overlay rendering, and real-time performance tracking (17+ FPS on RTX 2050).

---

## Running Unit Tests

Run the complete test suite:

```powershell
pytest -v
```

All 70 tests across identity banking, quality gating, color spaces, metrics, and fusion ranking will execute.

---

## Offline Evaluation Reproduction (Optional)

> **Note**: Evaluation reproduction is optional and requires the Market-1501 dataset. The trained fusion model (`models/fusion_lr.pkl`) and some benchmark metrics are provisional research artifacts.

### Dataset Requirement
The **Market-1501** dataset is not distributed with this repository due to academic research licensing restrictions. To run offline evaluation:
1. Request access from the [Zheng Lab Market-1501 Project Page](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html).
2. Extract the dataset into `data/market1501/` such that `bounding_box_test/` and `query/` subdirectories are present.

### Running Benchmarks
```powershell
# Run baseline OSNet evaluation on Market-1501
python -m eval.run_eval --methods osnet_only --config configs/default.yaml

# Run full comparative study across all fusion formulations
python -m eval.run_eval --methods all --config configs/default.yaml
```

---

## Directory Structure

```
appearance-reid/
├── configs/
│   └── default.yaml         # Central configuration parameters
├── data/                    # Dataset directory (Market-1501 placed here for eval)
├── docs/
│   └── ATTRIBUTION.md       # Licensing, model origins, and third-party citations
├── eval/                    # Offline evaluation & degradation test scripts
│   ├── datasets.py          # Market-1501 loader
│   ├── degrade.py           # Synthetic blur/occlusion harness
│   ├── metrics.py           # CMC Rank-1/5/10 and mAP calculators
│   └── run_eval.py          # Comparative evaluation runner
├── models/                  # Downloaded model weights (OSNet, OpenVINO IR)
├── results/                 # Evaluation output plots and metric logs
├── scripts/
│   └── download_models.py   # Checksum-verified model download utility
├── src/                     # Core real-time pipeline and ML modules
│   ├── attributes.py
│   ├── color.py
│   ├── detect_track.py
│   ├── embed.py
│   ├── fusion.py
│   ├── gallery.py
│   ├── identity.py
│   ├── pipeline.py
│   ├── quality.py
│   └── record.py
├── tests/                   # Pytest test suite (70 unit tests)
├── LICENSE                  # MIT License & third-party terms
├── README.md                # Project documentation & quickstart
└── requirements.txt         # Pinned python dependencies
```

---

## License & Attribution

- **SAFEPASS**: Distributed under the [MIT License](LICENSE).
- **YOLOv8**: Licensed under AGPL-3.0 by Ultralytics.
- **OSNet Checkpoint**: Distributed under Apache 2.0 / Market-1501 academic research terms.
- **OpenVINO Attribute Model**: Distributed under Apache 2.0 by Intel Corporation.
- For complete citations and license texts, see [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md).
