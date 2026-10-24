import sys
from pathlib import Path
import cv2
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attributes import AttributeInference
from src.color import extract_colors, FALLBACK_UPPER_XY, FALLBACK_LOWER_XY

with open("configs/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model = AttributeInference(cfg["attributes"])
model.load()

query_dir = Path("data/market1501/Market-1501-v15.09.15/query")
sample_files = sorted(list(query_dir.glob("*.jpg")))[:5]

print("| Image | Fallback Sampling Points | Model Predicted Points | Before (Fallback) [Upper / Lower] | After (Model Points) [Upper / Lower] | Path Used |")
print("|---|---|---|---|---|---|")

for img_p in sample_files:
    crop = cv2.imread(str(img_p))
    res = model.predict(crop)
    coords = res["coords"]
    
    before = extract_colors(crop, FALLBACK_UPPER_XY, FALLBACK_LOWER_XY, cfg["color"])
    after = extract_colors(crop, coords[0], coords[1], cfg["color"])
    
    u_b = f"{before['upper']} ({before['upper_conf']:.2f})"
    l_b = f"{before['lower']} ({before['lower_conf']:.2f})"
    u_a = f"{after['upper']} ({after['upper_conf']:.2f})"
    l_a = f"{after['lower']} ({after['lower_conf']:.2f})"
    
    c_str = f"top=({coords[0][0]:.3f}, {coords[0][1]:.3f}), bot=({coords[1][0]:.3f}, {coords[1][1]:.3f})"
    fb_str = f"top=({FALLBACK_UPPER_XY[0]:.2f}, {FALLBACK_UPPER_XY[1]:.2f}), bot=({FALLBACK_LOWER_XY[0]:.2f}, {FALLBACK_LOWER_XY[1]:.2f})"
    
    print(f"| `{img_p.name}` | {fb_str} | {c_str} | {u_b} / {l_b} | {u_a} / {l_a} | `{after['sampling_path']}` |")
