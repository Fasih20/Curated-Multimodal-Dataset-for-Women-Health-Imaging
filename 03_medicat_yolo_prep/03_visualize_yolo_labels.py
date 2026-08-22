"""
Source: original Colab notebook, cell index [28]
Auto-extracted -- review before treating as final.
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

YOLO_ROOT = Path("/content/pipeline_data/medicat_yolo")
IMAGES_ROOT = YOLO_ROOT / "images"
LABELS_ROOT = YOLO_ROOT / "labels"
VERIFY_DIR = YOLO_ROOT / "visual_verification"
VERIFY_DIR.mkdir(parents=True, exist_ok=True)

# 1. Gather all extracted image-label pairs
pairs = []
for split in ("train", "valid", "test"):
    lbl_dir = LABELS_ROOT / split
    img_dir = IMAGES_ROOT / split
    if lbl_dir.exists() and img_dir.exists():
        for lbl_path in lbl_dir.glob("*.txt"):
            # Find the corresponding image (try common extensions)
            for ext in [".png", ".jpg", ".jpeg", ".gif"]:
                img_path = img_dir / (lbl_path.stem + ext)
                if img_path.exists():
                    pairs.append((img_path, lbl_path, split))
                    break

print(f"Found {len(pairs)} matched image-label pairs for verification.")

# 2. Sample 12 random pairs for visual inspection
sample_size = min(12, len(pairs))
samples = random.sample(pairs, sample_size)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for i, (img_path, lbl_path, split) in enumerate(samples):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Read YOLO labels
    with open(lbl_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls, cx, cy, bw, bh = parts
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)

            # Convert YOLO format back to absolute pixel coordinates
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            # Draw red bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Save to disk so you can browse them in the Colab sidebar
    out_path = VERIFY_DIR / f"{split}_{img_path.name}"
    img.save(out_path)

    # Plot in notebook
    axes[i].imshow(img)
    axes[i].set_title(f"{split} | {img_path.name[:25]}...", fontsize=10)
    axes[i].axis("off")

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

print(f"\nSaved {sample_size} annotated verification images to: {VERIFY_DIR}")
print("Check the Colab file browser (left sidebar) under 'visual_verification' to zoom in on the red boxes!")