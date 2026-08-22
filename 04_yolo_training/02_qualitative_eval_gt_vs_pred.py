"""
Source: original Colab notebook, cell index [34]
Auto-extracted -- review before treating as final.
"""

# Qualitative GT (green) vs prediction (red) on test figures
import random, matplotlib.pyplot as plt
from PIL import Image, ImageDraw

best = YOLO("/content/pipeline_data/yolo_runs/medicat_panel_v1/weights/best.pt")
sample = random.sample(sorted((YOLO_ROOT / "images" / "test").glob("*")), 6)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, ip in zip(axes.flatten(), sample):
    img = Image.open(ip).convert("RGB"); w, h = img.size
    d = ImageDraw.Draw(img)
    lp = YOLO_ROOT / "labels" / "test" / (ip.stem + ".txt")
    for line in lp.read_text().splitlines():
        cx, cy, bw, bh = map(float, line.split()[1:])
        d.rectangle([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h], outline="lime", width=3)
    for r in best.predict(str(ip), conf=0.25, verbose=False)[0].boxes:
        d.rectangle(r.xyxy[0].tolist(), outline="red", width=3)
    ax.imshow(img); ax.axis("off"); ax.set_title(ip.name[:28], fontsize=9)
plt.tight_layout(); plt.show()