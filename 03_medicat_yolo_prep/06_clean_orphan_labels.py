"""
Source: original Colab notebook, cell index [31]
Auto-extracted -- review before treating as final.
"""

import json
from pathlib import Path
from collections import Counter
import pandas as pd

YOLO_ROOT = Path("/content/pipeline_data/medicat_yolo")
IMAGES_ROOT, LABELS_ROOT = YOLO_ROOT / "images", YOLO_ROOT / "labels"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif"}

# ── 1. Delete label files with no matching image in the same split ──
removed = []
for split in ("train", "valid", "test"):
    img_stems = {p.stem for p in (IMAGES_ROOT / split).iterdir() if p.suffix.lower() in IMG_EXTS}
    for lp in list((LABELS_ROOT / split).glob("*.txt")):
        if lp.stem not in img_stems:
            removed.append(split)
            lp.unlink()
print(f"Removed {len(removed)} stale labels: {Counter(removed)}")

# ── 2. Re-run the integrity stats ──
boxes_per_split, panel_dist = Counter(), Counter()
bad = tiny = 0
for split in ("train", "valid", "test"):
    img_stems = {p.stem for p in (IMAGES_ROOT / split).iterdir() if p.suffix.lower() in IMG_EXTS}
    lbl_stems = {p.stem for p in (LABELS_ROOT / split).glob("*.txt")}
    miss_img = len(lbl_stems - img_stems)
    miss_lbl = len(img_stems - lbl_stems)
    n_boxes = 0
    for lp in (LABELS_ROOT / split).glob("*.txt"):
        n = 0
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) != 5: bad += 1; continue
            v = [float(x) for x in p[1:]]
            if not all(0 <= x <= 1 for x in v): bad += 1; continue
            if v[2] < 0.05 or v[3] < 0.05: tiny += 1
            n += 1
        panel_dist[n] += 1
        n_boxes += n
    boxes_per_split[split] = n_boxes
    print(f"{split}: images={len(img_stems)}, labels={len(lbl_stems)}, "
          f"boxes={n_boxes}, missing_image={miss_img}, missing_label={miss_lbl}")

total = sum(boxes_per_split.values())
print(f"\nTOTAL boxes: {total} (expected 8,082)")
print(f"Panels/image: {dict(sorted(panel_dist.items()))}, mean={total/sum(panel_dist.values()):.2f}")
print(f"invalid lines={bad}, tiny boxes={tiny}")
print("\nPASS" if total == 8082 and bad == 0 and len(removed) == 741 else "\nSTILL SOMETHING OFF")