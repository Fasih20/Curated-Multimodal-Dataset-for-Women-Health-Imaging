"""
Source: original Colab notebook, cell index [29]
Auto-extracted -- review before treating as final.
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image

YOLO_ROOT = Path("/content/pipeline_data/medicat_yolo")
IMAGES_ROOT = YOLO_ROOT / "images"
LABELS_ROOT = YOLO_ROOT / "labels"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif"}
TINY_BOX = 0.05  # relative w or h below this = "extremely small"

images_per_split, boxes_per_split = Counter(), Counter()
panel_dist = Counter()
box_ws, box_hs, img_short, img_dims = [], [], [], []
invalid_lines = zero_box_imgs = tiny_boxes = 0
missing_image, missing_label = [], []

for split in ("train", "valid", "test"):
    img_stems = {p.stem for p in (IMAGES_ROOT / split).iterdir() if p.suffix.lower() in IMG_EXTS}
    lbl_paths = list((LABELS_ROOT / split).glob("*.txt"))
    lbl_stems = {p.stem for p in lbl_paths}

    images_per_split[split] = len(img_stems)
    missing_image += [f"{split}/{s}" for s in lbl_stems - img_stems]
    missing_label += [f"{split}/{s}" for s in img_stems - lbl_stems]

    for lp in lbl_paths:
        n = 0
        for line in lp.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                invalid_lines += 1; continue
            try:
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                invalid_lines += 1; continue
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                invalid_lines += 1; continue
            n += 1
            box_ws.append(bw); box_hs.append(bh)
            if bw < TINY_BOX or bh < TINY_BOX:
                tiny_boxes += 1
        if n == 0:
            zero_box_imgs += 1
        else:
            panel_dist[n] += 1
            boxes_per_split[split] += n

    for stem in (lbl_stems & img_stems):
        ip = next(IMAGES_ROOT / split / (stem + e) for e in IMG_EXTS
                  if (IMAGES_ROOT / split / (stem + e)).exists())
        with Image.open(ip) as im:
            w, h = im.size
        img_short.append(min(w, h)); img_dims.append((w, h))

def q(vals):
    s = pd.Series(vals)
    return {f"p{k}": round(float(s.quantile(k / 100)), 3) for k in (5, 25, 50, 75, 95)}

buckets = Counter()
for n, c in panel_dist.items():
    buckets["1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4+"] += c

total_boxes = sum(boxes_per_split.values())
print("=== PER-SPLIT COUNTS ===")
for s in ("train", "valid", "test"):
    print(f"  {s}: images={images_per_split[s]}, labels paired, boxes={boxes_per_split[s]}")
print(f"  TOTAL boxes: {total_boxes} (expected 8,082)")

print("\n=== PANELS PER IMAGE ===")
for k in ("1", "2", "3", "4+"):
    print(f"  {k} panels: {buckets.get(k, 0)}")
print(f"  mean panels/image: {total_boxes / sum(panel_dist.values()):.2f}")

print("\n=== BOX SIZE (relative) ===")
print(f"  width : {q(box_ws)}")
print(f"  height: {q(box_hs)}")
print(f"  tiny boxes (<{TINY_BOX}): {tiny_boxes}")

print("\n=== IMAGE RESOLUTION ===")
print(f"  short side: {q(img_short)}")
print(f"  min dims: {min(img_dims, key=lambda d: d[0] * d[1])}")

print("\n=== INTEGRITY ===")
print(f"  invalid label lines:      {invalid_lines}")
print(f"  label files w/ 0 boxes:   {zero_box_imgs}")
print(f"  labels missing image:     {len(missing_image)}")
print(f"  images missing label:     {len(missing_label)}")
print("\nPASS" if (invalid_lines == zero_box_imgs == len(missing_image) == len(missing_label) == 0)
      else "\nCHECK THE NON-ZERO ITEMS ABOVE BEFORE TRAINING")