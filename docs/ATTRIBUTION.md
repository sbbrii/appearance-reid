# ATTRIBUTION -- SAFEPASS Appearance + Quality-Aware Fusion Module

## Ported Concepts (rewritten from understanding, NOT copied verbatim)

**Repository:** Real-Time-Person-ReID
**URL:** https://github.com/ortatepeyusuf/Real-Time-Person-ReID
**Commit cloned:** 2026-10-24
**License:** NONE PRESENT -- No LICENSE file found at clone time.

No code was copied verbatim.  All concepts listed below were understood
from reading the source and then independently reimplemented in Python.

| Concept | Source file / approx lines | Our target file |
|---|---|---|
| OSNet preprocessing: resize (256x128), ToTensor, ImageNet normalise | OSNet_Extractor.py L96-107 | src/embed.py |
| Single-image + batch forward pass; L2-normalisation | OSNet_Extractor.py L146-187 | src/embed.py |
| FAISS IndexFlatIP + IndexIDMap gallery structure | main.py L21-23 | src/gallery.py |
| Crop-from-bbox helper frame[y1:y2, x1:x2] | main.py L60-61 | src/detect_track.py |
| Main loop: frame -> detect -> crop -> embed -> FAISS -> annotate | main.py L105-153 | src/pipeline.py |

### Known limitation documented from reference
> "The current real-time loop tracks one target at a time and uses the first
> accepted detection returned for a frame. It is not a multi-object tracker."
> (Real-Time-Person-ReID README.md, line 38)

Our implementation resolves this with YOLOv8 + ByteTrack.

---

## Third-Party Models

### person-attributes-recognition-crossroad-0230
| Field | Value |
|---|---|
| Source | Intel Open Model Zoo |
| Download URL | https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-attributes-recognition-crossroad-0230/FP32/ |
| License | **Apache License 2.0** |
| Copyright | Copyright (c) 2023-2024 Intel Corporation |
| Full license | https://www.apache.org/licenses/LICENSE-2.0 |

### OSNet x1_0 pretrained weights
| Field | Value |
|---|---|
| Source | pyppbox-torchreid (fork of KaiyangZhou/deep-person-reid) |
| Paper | "Omni-Scale Feature Learning for Person Re-Identification", Zhou et al., ICCV 2019 |
| License | MIT |

---

## Third-Party Libraries (key)

| Library | License |
|---|---|
| PyTorch | BSD-3-Clause |
| Ultralytics YOLOv8 | AGPL-3.0 (academic research use only) |
| FAISS | MIT |
| OpenVINO Runtime | Apache 2.0 |
| OpenCV | Apache 2.0 |
| scikit-learn | BSD-3-Clause |
| pyppbox-torchreid | MIT |
| NumPy | BSD-3-Clause |

> **Note on Ultralytics AGPL-3.0:** YOLOv8 is used for academic B.Tech
> research only.  Commercial deployment requires a separate Ultralytics
> commercial license or a non-AGPL detector replacement.
