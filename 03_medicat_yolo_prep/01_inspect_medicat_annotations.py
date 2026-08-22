"""
Source: original Colab notebook, cell index [25]
Auto-extracted -- review before treating as final.
"""

import json
from pathlib import Path
from collections import Counter

ANNOT_PATH = Path("/content/pipeline_data/medicat/subcaptions_public.jsonl")
SPLIT_PATH = Path("/content/pipeline_data/medicat/subcaption_split_keys.json")

OUT_ROOT = Path("/content/pipeline_data/medicat_yolo")
LABELS_ROOT = OUT_ROOT / "labels"
NEEDED_IMAGES_PATH = OUT_ROOT / "needed_images.json"
CLASS_MAP_PATH = OUT_ROOT / "class_map.txt"

# ── 1. Load split keys using pdf_hash + fig_key ──────────────────────
with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split_keys = json.load(f)

key_to_split = {}
for split_name, entries in split_keys.items():
    for entry in entries:
        key = f"{entry['pdf_hash']}_{entry['fig_key']}"
        key_to_split[key] = split_name

print(f"Loaded {len(key_to_split)} split keys.")

# ── 2. Prepare output directories ────────────────────────────────────
for split_name in ("train", "valid", "test"):
    (LABELS_ROOT / split_name).mkdir(parents=True, exist_ok=True)

# ── 3. Parse annotations and write YOLO labels ──────────────────────
needed_images = []
missing_split = []       # records with no split assignment
skipped_no_boxes = 0     # records with subfigures but no valid boxes
n_records = 0
n_boxes_total = 0
split_counts = Counter()

with open(ANNOT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        n_records += 1

        pdf_hash = rec["pdf_hash"]
        fig_key  = rec["fig_key"]
        fig_uri  = rec["fig_uri"]
        w, h     = rec["width"], rec["height"]

        lookup_key = f"{pdf_hash}_{fig_key}"

        # ── THE FIX: no silent fallback ──
        split_name = key_to_split.get(lookup_key)
        if split_name is None:
            missing_split.append(lookup_key)
            continue

        subfigs = rec.get("subfigures", [])
        if not subfigs:
            continue

        # Convert subfigure polygons to YOLO boxes
        lines = []
        for sf in subfigs:
            xs = [p[0] for p in sf["points"]]
            ys = [p[1] for p in sf["points"]]
            x_min, x_max = max(0.0, min(xs)), min(w, max(xs))
            y_min, y_max = max(0.0, min(ys)), min(h, max(ys))
            if x_max <= x_min or y_max <= y_min:
                continue
            cx = (x_min + x_max) / 2 / w
            cy = (y_min + y_max) / 2 / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            skipped_no_boxes += 1
            continue

        out_name = f"{pdf_hash}_{fig_uri}"

        label_path = LABELS_ROOT / split_name / (Path(out_name).stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines) + "\n")

        needed_images.append({
            "pdf_hash": pdf_hash,
            "fig_uri": fig_uri,
            "out_name": out_name,
            "split": split_name,
        })
        split_counts[split_name] += 1
        n_boxes_total += len(lines)

# ── 4. Save manifest and class map ──────────────────────────────────
OUT_ROOT.mkdir(parents=True, exist_ok=True)
with open(NEEDED_IMAGES_PATH, "w", encoding="utf-8") as f:
    json.dump(needed_images, f, indent=2)
with open(CLASS_MAP_PATH, "w", encoding="utf-8") as f:
    f.write("0: panel\n")

# ── 5. Report ────────────────────────────────────────────────────────
print(f"\nTotal annotation records:        {n_records}")
print(f"Records with valid YOLO boxes:   {len(needed_images)}")
print(f"Total boxes written:             {n_boxes_total}")
print(f"Records skipped (no valid box):  {skipped_no_boxes}")
print(f"\nSplit distribution:")
for s in ("train", "valid", "test"):
    print(f"  {s}: {split_counts[s]}")
print(f"\nMissing split assignments: {len(missing_split)}")
for m in missing_split:
    print(f"  EXCLUDED: {m}")
print(f"\nWrote labels under: {LABELS_ROOT}")
print(f"Wrote manifest:     {NEEDED_IMAGES_PATH}")