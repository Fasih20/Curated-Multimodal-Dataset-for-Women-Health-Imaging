"""
Source: original Colab notebook, cell index [33]
Auto-extracted -- review before treating as final.
"""


# Runtime -> Change runtime type -> T4 GPU
from pathlib import Path
from ultralytics import YOLO

YOLO_ROOT = Path("/content/pipeline_data/medicat_yolo")
(YOLO_ROOT / "data.yaml").write_text(f"""path: {YOLO_ROOT}
train: images/train
val: images/valid
test: images/test

names:
  0: panel
""")

model = YOLO("yolov8n.pt")
model.train(data=str(YOLO_ROOT / "data.yaml"), epochs=30, imgsz=640, batch=16,
            patience=8, seed=42, device=0,
            project="/content/pipeline_data/yolo_runs", name="medicat_panel_v1", exist_ok=True)

m = model.val(split="test")   # held-out official MedICaT test split
print(f"Precision: {m.box.mp:.4f} | Recall: {m.box.mr:.4f}")
print(f"mAP@50: {m.box.map50:.4f} | mAP@50:95: {m.box.map:.4f}")