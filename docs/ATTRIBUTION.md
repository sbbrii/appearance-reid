# ATTRIBUTION & LICENSING -- SAFEPASS Appearance Module

## Third-Party Models & Weights

### 1. OSNet x1.0 (Market-1501 Pretrained Checkpoint)
- **Model Checkpoint:** `models/osnet_x1_0_market1501.pth`
- **Architecture:** Omni-Scale Network (OSNet x1.0)
- **Reference Paper:** *"Omni-Scale Feature Learning for Person Re-Identification"*, Kaiyang Zhou, Yongxin Yang, Andrea Cavallaro, Tao Xiang (ICCV 2019).
- **Source Repository:** [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
- **Codebase License:** Apache License 2.0
- **Training Dataset Terms:** Weights were trained on the Market-1501 dataset. Market-1501 is released for non-commercial academic research only. Model weights are downloaded at setup via `scripts/download_models.py` and are not distributed inside this git repository.

### 2. OpenVINO Person Attributes Recognition
- **Model Files:** `models/FP32/person-attributes-recognition-crossroad-0230.xml` and `.bin`
- **Source:** Intel Open Model Zoo (2023.0)
- **License:** Apache License 2.0
- **Copyright:** Copyright (c) 2018-2024 Intel Corporation
- **Download Source:** [storage.openvinotoolkit.org](https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-attributes-recognition-crossroad-0230/FP32/)
- **Terms:** Freely redistributable under Apache 2.0 terms; fetched automatically via `scripts/download_models.py` to keep repository size minimal.

### 3. Ultralytics YOLOv8n
- **Model Checkpoint:** `yolov8n.pt`
- **Source:** [Ultralytics](https://github.com/ultralytics/ultralytics)
- **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
- **Terms:** Used here for academic research and evaluation. Commercial deployment requires an Ultralytics enterprise license or replacing the detector with a permissively licensed alternative. Weights are auto-fetched by Ultralytics / `scripts/download_models.py`.

---

## Datasets

### Market-1501 Dataset
- **Paper:** *"Scalable Person Re-identification: A Benchmark"*, Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang, Qi Tian (ICCV 2015).
- **Website:** [Zheng Lab - Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
- **License / Terms:** Non-commercial academic research use only.
- **IMPORTANT:** **Market-1501 is NOT included or distributed in this repository.** Users who wish to run offline benchmark evaluation (`eval/run_eval.py`) must request and download the dataset directly from the dataset authors and place it in `data/market1501/`.

---

## Third-Party Libraries

| Library | License | Usage |
|---|---|---|
| **Torchreid** (`deep-person-reid`) | Apache-2.0 / MIT | Deep person re-identification framework & OSNet architecture |
| **Ultralytics YOLOv8** | AGPL-3.0 | Pedestrian detection and ByteTrack tracking |
| **OpenVINO Runtime** | Apache-2.0 | CPU-optimized inference for pedestrian attribute extraction |
| **FAISS** (`faiss-cpu`) | MIT | In-memory L2/Inner-Product similarity search for identity gallery |
| **PyTorch** & **Torchvision** | BSD-3-Clause | Tensor computing and deep neural networks |
| **OpenCV** (`opencv-python`) | Apache-2.0 | Video I/O, image processing, and color space transformations |
| **scikit-learn** | BSD-3-Clause | Fusion calibration and offline evaluation metrics |
| **NumPy** & **SciPy** | BSD-3-Clause | Numerical arrays and scientific algorithms |
| **PyYAML** | MIT | Configuration file parsing |
| **gdown** | Apache-2.0 | Automated checkpoint fetching with integrity verification |

---

## Ported Concepts (Rewritten from Understanding)

**Reference:** `Real-Time-Person-ReID` (Yusuf Ortatepe)
- **URL:** https://github.com/ortatepeyusuf/Real-Time-Person-ReID
- **Status:** Evaluated during system design. No code was copied verbatim.
- **Implemented Differences:**
  - The reference repository tracked only a single person at a time using a single crop per frame without multi-target re-identification or temporal identity tracking.
  - SAFEPASS completely reimplements the pipeline with YOLOv8 multi-target detection, ByteTrack tracking, multi-frame identity banks, auto-gated quality filtering, color histograms, and Bayesian attribute fusion.
