# Models Directory

This directory stores model weights and neural network definitions used by SAFEPASS.

All models are downloaded automatically via:
```powershell
python scripts/download_models.py
```

Models downloaded:
1. `models/osnet_x1_0_market1501.pth` (~9.9 MB, OSNet Re-ID feature extractor)
2. `models/FP32/person-attributes-recognition-crossroad-0230.xml` and `.bin` (~2.8 MB, OpenVINO attribute classifier)
3. `yolov8n.pt` (~6.2 MB, YOLOv8 detector, auto-downloaded by Ultralytics or `download_models.py`)

See `docs/ATTRIBUTION.md` for full licensing, model origin, and attribution details.
