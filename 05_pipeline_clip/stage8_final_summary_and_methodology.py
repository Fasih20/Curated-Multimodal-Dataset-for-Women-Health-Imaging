"""
Source: original Colab notebook, cell index [51]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 8 -- final dataset summary JSON + METHODOLOGY.md.

Run this last. Pulls stats out of every stage's already-saved output
files (does not recompute anything).
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path("/content/pipeline_data")
OUT_DIR = PIPELINE_ROOT / "final_summary_v1"
SUMMARY_JSON_PATH = OUT_DIR / "dataset_summary.json"
METHODOLOGY_PATH = OUT_DIR / "METHODOLOGY.md"

SEED = 42


def safe_read_parquet(path: Path):
    return pd.read_parquet(path) if path.exists() else None


def safe_read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    worklist = safe_read_parquet(PIPELINE_ROOT / "compound_worklist_v1/compound_figures_manifest.parquet")
    panels = pd.read_csv(PIPELINE_ROOT / "panels_v1/panel_manifest.csv") if (PIPELINE_ROOT / "panels_v1/panel_manifest.csv").exists() else None
    alignment = safe_read_parquet(PIPELINE_ROOT / "alignment_v1/alignment_manifest.parquet")
    quality = safe_read_json(PIPELINE_ROOT / "quality_v1/quality_summary.json")
    split_report = safe_read_json(PIPELINE_ROOT / "splits_final_v1/split_report.json")
    retrieval = safe_read_json(PIPELINE_ROOT / "retrieval_v1/retrieval_results.json")
    vqa = safe_read_parquet(PIPELINE_ROOT / "vqa_v1/vqa_dataset.parquet")
    captioning = safe_read_parquet(PIPELINE_ROOT / "captioning_v1/captioning_dataset.parquet")

    yolo_metrics = {
        "precision": 0.943, "recall": 0.943, "mAP50": 0.974, "mAP50_95": 0.910,
        "train_images": 1375, "train_boxes": 5274,
        "valid_images": 317, "valid_boxes": 1174,
        "test_images": 422, "test_boxes": 1634,
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_seed": SEED,
        "yolo_panel_detector": yolo_metrics,
        "compound_figures_input": int(len(worklist)) if worklist is not None else None,
        "panels_extracted": int(len(panels)) if panels is not None else None,
        "aligned_panels": int(len(alignment)) if alignment is not None else None,
        "quality": quality,
        "splits": split_report,
        "retrieval": retrieval,
        "vqa_pairs": int(len(vqa)) if vqa is not None else None,
        "vqa_question_types": vqa["question_type"].value_counts().to_dict() if vqa is not None else None,
        "captioning_pairs": int(len(captioning)) if captioning is not None else None,
        "models_used": {
            "panel_detector": "yolov8n (Stage 0-1 of prior pass)",
            "alignment_and_retrieval": "openai/clip-vit-base-patch32 (transformers)",
            "retrieval_index": "faiss IndexFlatIP",
        },
    }
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    md = f"""# Methodology: Compound-Figure Panel Dataset Pipeline

Generated: {summary['generated_at_utc']}

## 1. Panel detector
YOLOv8n trained on MedICaT subfigure annotations (2,114 figures, 8,082
boxes). Test-split metrics: P={yolo_metrics['precision']}, R={yolo_metrics['recall']},
mAP50={yolo_metrics['mAP50']}, mAP50:95={yolo_metrics['mAP50_95']}.

## 2. Panel extraction (Stage 1)
Detector applied to {summary['compound_figures_input']} compound figures
(compound_flag_v2 == True from the tiered compound-detection policy,
measured precision 82.5-100% depending on signal tier). Extracted
{summary['panels_extracted']} panel crops with bounding boxes and
detection confidence, saved to `panels_v1/panel_manifest.csv`.

## 3. Image-text alignment (Stage 2)
CLIP (ViT-B/32) cosine similarity between panel crops and candidate
caption segments. Segments are split on panel-letter markers (e.g. "(A)")
where present; otherwise the whole caption is used as a single
figure-level candidate (weaker match, labeled `match_type =
figure_level_caption`). This is **automatic alignment, not ground
truth** -- distinguish from the manually-validated 200-sample compound
review used upstream.

## 4. Quality analysis (Stage 3)
Panels flagged (not deleted) for: too-small crops, extreme aspect ratio,
missing/low-similarity alignment, or ambiguous alignment margin. See
`quality_v1/quality_summary.json` for exact thresholds and counts.

## 5. Split (Stage 4)
Paper-level (pmcid) split, reusing the existing split assignment from
upstream where available; falls back to a fresh seed=42 GroupShuffleSplit
(80/10/10) only if that column is missing. No panel/figure crosses splits.

## 6. Retrieval baseline (Stage 5)
FAISS flat index over CLIP embeddings, evaluated on the test split.
Recall@{{1,5,10}} for text->panel retrieval uses Stage 2's own alignment
as the ground truth proxy -- **not human-validated**, report accordingly.
Panel->panel retrieval has no defensible ground truth in the metadata,
so only the index (for qualitative inspection) is produced, no invented
metric.

## 7. VQA dataset (Stage 6)
Template questions (modality, anatomy, panel-specific caption, panel
count) grounded in Stage 2 aligned text or simple keyword lookups --
no hallucinated visual facts. {summary['vqa_pairs']} pairs generated;
a random 60-row sample saved for manual validation.

## 8. Captioning dataset (Stage 7)
Reuses Stage 2's best_match_text as-is (no synthetic caption model).
{summary['captioning_pairs']} image-caption pairs.

## Reproducibility
- Random seed: {SEED} throughout.
- All embeddings cached to disk (`alignment_v1/embed_cache/`), never
  recomputed on rerun.
- Every stage script checks for its own output file and skips if present
  (safe to resume after a Colab disconnect).
"""
    METHODOLOGY_PATH.write_text(md)

    print("\n[STAGE 8 COMPLETE]")
    print(f"Summary: {SUMMARY_JSON_PATH}")
    print(f"Methodology: {METHODOLOGY_PATH}")


if __name__ == "__main__":
    main()