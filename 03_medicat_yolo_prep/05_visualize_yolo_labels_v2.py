"""
Source: original Colab notebook, cell index [30]
Auto-extracted -- review before treating as final.
"""

import random
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

YOLO_ROOT = Path("/content/pipeline_data/medicat_yolo")
IMAGES_ROOT, LABELS_ROOT = YOLO_ROOT / "images", YOLO_ROOT / "labels"
VERIFY_DIR = YOLO_ROOT / "visual_verification_v2"
random.seed(42)
TARGET = {"train": 20, "valid": 15, "test": 15}

def cat(n): return "1" if n <= 1 else "2" if n == 2 else "3+"

def annotate(img_path, lbl_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5: continue
        cx, cy, bw, bh = map(float, p[1:])
        d.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h], outline="red", width=3)
    return img

manifest = []
for split, n_target in TARGET.items():
    (VERIFY_DIR / split).mkdir(parents=True, exist_ok=True)
    by_cat = {"1": [], "2": [], "3+": []}
    for lp in (LABELS_ROOT / split).glob("*.txt"):
        n = sum(1 for l in lp.read_text().splitlines() if l.strip())
        for ext in (".png", ".jpg", ".jpeg"):
            ip = IMAGES_ROOT / split / (lp.stem + ext)
            if ip.exists():
                by_cat[cat(n)].append((ip, lp, n)); break

    picked, per = [], max(3, n_target // 3)
    for items in by_cat.values():
        picked += random.sample(items, min(per, len(items)))
    if len(picked) < n_target:
        pool = [x for v in by_cat.values() for x in v if x not in set(picked)]
        picked += random.sample(pool, min(n_target - len(picked), len(pool)))
    random.shuffle(picked)
    picked = picked[:n_target]

    fig, axes = plt.subplots((len(picked) + 4) // 5, 5, figsize=(20, 4.2 * ((len(picked) + 4) // 5)))
    for ax, (ip, lp, n) in zip(axes.flatten(), picked):
        ann = annotate(ip, lp)
        ann.save(VERIFY_DIR / split / ip.name)          # full-size for zooming
        thumb = ann.copy(); thumb.thumbnail((384, 384))
        ax.imshow(thumb); ax.set_title(f"{cat(n)}-panel | n={n}", fontsize=9); ax.axis("off")
        manifest.append({"split": split, "panel_cat": cat(n), "n_boxes": n, "image": ip.name})
    for ax in axes.flatten()[len(picked):]:
        ax.axis("off")
    plt.suptitle(f"{split.upper()} validation sample ({len(picked)})", y=1.0)
    plt.tight_layout(); plt.show()

    sel = Counter(m["panel_cat"] for m in manifest if m["split"] == split)
    print(f"{split}: sampled {len(picked)} -> categories {dict(sel)}")

pd.DataFrame(manifest).to_csv(VERIFY_DIR / "validation_manifest.csv", index=False)
print(f"\nFull-size annotated images + manifest saved to: {VERIFY_DIR}")