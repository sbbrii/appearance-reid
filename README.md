# SAFEPASS — Appearance + Quality-Aware Fusion Module

> **B.Tech Capstone / Research Project:** Multi-modal Person Re-Identification for CCTV Missing-Person Search.  
> **Module Scope:** OSNet Re-ID embeddings, body quality estimation, OpenVINO attribute extraction, CIELAB color classification, FAISS gallery index, and quality-aware adaptive fusion re-ranking.

---

## 1. System Architecture & Data Flow

```
                      Video Frame / CCTV Camera / Video File
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Detection & Tracking      │
                         │ YOLOv8n + ByteTrack (CPU/GPU│
                         └──────────────┬──────────────┘
                                        │  Crop BGR + Track ID + BBox
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
       ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
       │     embed.py     │   │  attributes.py   │   │    quality.py    │
       │ OSNet (fp16 GPU) │   │ OpenVINO IR CPU  │   │ Blur, Visibility,│
       │ 512-D L2 Embed   │   │ 8 Attributes +   │   │ Size Proxy       │
       └────────┬─────────┘   │ 2 Color Pts      │   └────────┬─────────┘
                │             └─────────┬────────┘            │
                │                       │ Sampling Points     │
                │             ┌─────────▼────────┐            │
                │             │     color.py     │            │
                │             │ CIELAB Centroids │            │
                │             │ Upper/Lower + Conf│           │
                │             └─────────┬────────┘            │
                │                       │                     │
                └───────────────────────┼─────────────────────┘
                                        │
                                  FeatureRecord
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
                              │ Display / Stream │
                              └──────────────────┘
```

---

## 2. Research Core: Quality-Aware Fusion

When CCTV queries suffer from blur, occlusion, or low resolution, raw visual embeddings degrade significantly. SAFEPASS implements four fusion methods under identical experimental conditions:

1. **`osnet_only` (Baseline A):** Pure cosine similarity on OSNet 512-D L2-normalized vectors.
2. **`fixed_weight` (Baseline B):** Static weighted linear combination:
   $$\text{Score} = 0.6 \cdot S_{\text{emb}} + 0.2 \cdot S_{\text{color}} + 0.2 \cdot S_{\text{attr}}$$
3. **`adaptive_rule` (Proposed Core):** Dynamic softmax gating weighted by aggregated query-gallery quality:
   $$Q = \min(Q_{\text{query}}, Q_{\text{gallery}})$$
   $$\mathbf{w} = \text{Softmax}\left(\frac{[w_{\text{emb}} \cdot Q, w_{\text{color}} \cdot (1 - Q), w_{\text{attr}} \cdot (1 - Q)]}{T}\right)$$
   With mismatch penalties applied when both query and gallery are confident.
4. **`adaptive_learned`:** Logistic regression model trained on feature vectors $[Q_q, Q_g, \text{blur}_q, \text{blur}_g, \text{vis}_q, \text{vis}_g, S_{\text{emb}}, S_{\text{color}}, S_{\text{attr}}]$.

---

## 3. Directory Layout

```
safepass_appearance/
├── configs/
│   └── default.yaml         # Central configuration (all parameters tuned here)
├── docs/
│   └── ATTRIBUTION.md       # Clean attribution for all ported concepts & model licenses
├── models/
│   └── FP32/
│       ├── person-attributes-recognition-crossroad-0230.xml
│       └── person-attributes-recognition-crossroad-0230.bin
├── src/
│   ├── detect_track.py      # YOLOv8 + ByteTrack multi-person tracking
│   ├── embed.py             # OSNet feature extractor (GPU fp16 / CPU)
│   ├── quality.py           # Body quality estimator (blur, visibility, size)
│   ├── attributes.py        # OpenVINO person-attributes inference
│   ├── color.py             # CIELAB color centroid mapping
│   ├── record.py            # FeatureRecord dataclass & serialization
│   ├── gallery.py           # FAISS IndexFlatIP + metadata gallery
│   ├── fusion.py            # Quality-aware adaptive re-ranking engine
│   └── pipeline.py          # End-to-end real-time pipeline (17.3 FPS measured)
├── eval/
│   ├── datasets.py          # Market-1501 dataset loader
│   ├── degrade.py           # Synthetic degradation harness (blur, occlusion, scale)
│   ├── metrics.py           # CMC (Rank-1/5/10), mAP, MRR, FAR/FRR
│   └── run_eval.py          # Comparative evaluation script across all methods
├── tests/
│   ├── test_quality.py      # Quality estimator unit tests
│   ├── test_color.py        # Color classifier unit tests
│   ├── test_metrics.py      # CMC, mAP, MRR metric tests
│   ├── test_fusion.py       # Fusion gating and ranking tests
│   └── test_pipeline_modules.py # Live model & pipeline component tests
└── requirements.txt         # Pinned project dependencies
```

---

## 4. Environment & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- NVIDIA GPU with CUDA support (tested on RTX 2050, CUDA 12.8)

### Run Unit Tests
```bash
pytest -v
```
*Current test suite: **54 passed in ~5.7s** (100% pass rate).*

### Download / Verify Models
Downloads the Intel Open Model Zoo FP32 attribute recognition model (`Apache 2.0`):
```bash
python scripts/download_models.py
```

### Run Real-Time Pipeline
```bash
# Live webcam
python src/pipeline.py --source 0

# Video file
python src/pipeline.py --source path/to/cctv_video.mp4
```

### Run Evaluation Harness
```bash
# Single method on Market-1501
python eval/run_eval.py --methods osnet_only --config configs/default.yaml

# Full comparative ablation study
python eval/run_eval.py --methods all --ablations all --config configs/default.yaml
```

---

## 5. Performance Benchmark
Measured on NVIDIA GeForce RTX 2050 (4 GB VRAM) with Python 3.14:
- **Pipeline Speed:** **17.3 FPS** (exceeds the 10.0 FPS project target)
- **OSNet Inference:** GPU fp16 batch acceleration
- **Attribute Inference:** OpenVINO CPU runtime
- **Gallery Search:** FAISS IndexFlatIP (exact cosine similarity)

---

## 6. Known Limitations & Clean Out-of-Scope Interfaces
- **Out of Scope (By Spec):** Face/ArcFace recognition, gait dynamics, camera topology graphs, and web dashboards are intentionally excluded. Modular interfaces (`FeatureRecord` and `rerank()`) are provided for straightforward future plug-in.
- **Dataset Dependency:** Evaluation scripts require Market-1501 dataset placed under `data/Market-1501/`. If absent, `eval/datasets.py` outputs download instructions and exits safely.
