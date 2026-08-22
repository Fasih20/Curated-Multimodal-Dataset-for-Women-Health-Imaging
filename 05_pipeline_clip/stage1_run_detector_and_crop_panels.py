"""
Source: original Colab notebook, cell index [36]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 1 (PATCHED) -- run the trained YOLOv8 panel detector over all compound
figures, sort boxes into reading order (A, B, C...), and persist crops + manifest.
"""
from __future__ import annotations
import csv
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import pandas as pd

LOCAL_DOWNLOAD_ROOT = Path("/content/local_download")
WORKLIST_PATH = Path("/content/pipeline_data/compound_worklist_v1/compound_figures_manifest.parquet")
WEIGHTS_PATH = Path("/content/pipeline_data/yolo_runs/medicat_panel_v1/weights/best.pt")

OUT_DIR = Path("/content/pipeline_data/panels_v1")
CROPS_DIR = OUT_DIR / "crops"
MANIFEST_PATH = OUT_DIR / "panel_manifest.csv"
CONF_THRESH = 0.25

FIELDNAMES = [
    "figure_id", "panel_id", "source_image", "x1", "y1", "x2", "y2",
    "crop_path", "width", "height", "aspect_ratio", "confidence",
]

def order_boxes(boxes, img_height):
    """Reading order: cluster into rows by y-center, sort each row left->right."""
    rows = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        yc = (b[1] + b[3]) / 2
        for row in rows:
            row_yc = sum((x[1] + x[3]) / 2 for x in row) / len(row)
            if abs(yc - row_yc) < 0.12 * img_height: # 12% tolerance for row clustering
                row.append(b); break
        else:
            rows.append([b])
    out = []
    for row in rows:
        out += sorted(row, key=lambda b: b[0]) # sort left-to-right
    return out

def already_processed_figure_ids() -> set[str]:
    if not MANIFEST_PATH.exists(): return set()
    return set(pd.read_csv(MANIFEST_PATH)["figure_id"].astype(str).unique())

def main() -> None:
    worklist = pd.read_parquet(WORKLIST_PATH)
    print(f"Compound figures to process: {len(worklist)}")
    done = already_processed_figure_ids()
    todo = worklist[~worklist["figure_id"].astype(str).isin(done)]
    print(f"Already processed (resume): {len(done)} | Remaining: {len(todo)}")

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not MANIFEST_PATH.exists()
    model = YOLO(str(WEIGHTS_PATH))

    n_figs, n_panels, n_no_detect = 0, 0, 0
    with open(MANIFEST_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header: writer.writeheader()

        for _, row in todo.iterrows():
            fig_id = str(row["figure_id"])
            img_path = LOCAL_DOWNLOAD_ROOT / row["image_path"]
            if not img_path.is_file(): continue

            try:
                result = model.predict(str(img_path), conf=CONF_THRESH, verbose=False)[0]
            except Exception as e:
                print(f"[WARN] detection failed for {fig_id}: {e}"); continue

            n_figs += 1
            if len(result.boxes) == 0:
                n_no_detect += 1; continue

            # Extract raw boxes
            boxes_data = []
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                boxes_data.append((x1, y1, x2, y2, conf))

            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                ordered_boxes = order_boxes(boxes_data, H)

                for panel_idx, (x1, y1, x2, y2, conf) in enumerate(ordered_boxes):
                    # Assign A, B, C... based on reading order
                    letter = chr(ord("A") + panel_idx) if panel_idx < 26 else f"p{panel_idx}"
                    panel_id = f"{fig_id}_{letter}"

                    crop = im.crop((x1, y1, x2, y2))
                    crop_path = CROPS_DIR / f"{panel_id}.png"
                    crop.save(crop_path)
                    w, h = crop.size

                    writer.writerow({
                        "figure_id": fig_id, "panel_id": panel_id,
                        "source_image": str(img_path),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "crop_path": str(crop_path),
                        "width": w, "height": h,
                        "aspect_ratio": round(w / h, 4) if h else None,
                        "confidence": round(conf, 4),
                    })
                    n_panels += 1

            if n_figs % 200 == 0:
                f.flush()
                print(f"  ...{n_figs} figures processed, {n_panels} panels so far")

    print("\n[STAGE 1 COMPLETE]")
    print(f"Figures processed this run: {n_figs}")
    print(f"Figures with zero detections: {n_no_detect}")
    print(f"Panels extracted this run: {n_panels}")
    print(f"Output: {MANIFEST_PATH}")

if __name__ == "__main__":
    main()