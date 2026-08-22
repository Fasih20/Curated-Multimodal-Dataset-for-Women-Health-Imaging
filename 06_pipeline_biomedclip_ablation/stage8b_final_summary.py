"""
Source: original Colab notebook, cell index [67]
Auto-extracted -- review before treating as final.
"""

"""Colab: Stage 8b (ABLATION) -- final summary for the BiomedCLIP track."""
from __future__ import annotations
import json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

PIPELINE_ROOT = Path("/content/pipeline_data")
OUT_DIR = PIPELINE_ROOT / "final_summary_biomedclip_v1"
SUMMARY_JSON_PATH = OUT_DIR / "dataset_summary.json"

SEED = 42


def safe_read_parquet(path):
    return pd.read_parquet(path) if path.exists() else None


def safe_read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    worklist = safe_read_parquet(PIPELINE_ROOT / "compound_worklist_v1/compound_figures_manifest.parquet")
    panels = pd.read_csv(PIPELINE_ROOT / "panels_v1/panel_manifest.csv") if (PIPELINE_ROOT / "panels_v1/panel_manifest.csv").exists() else None
    alignment = safe_read_parquet(PIPELINE_ROOT / "alignment_biomedclip_v1/alignment_manifest.parquet")
    quality = safe_read_json(PIPELINE_ROOT / "quality_biomedclip_v1/quality_summary.json")
    split_report = safe_read_json(PIPELINE_ROOT / "splits_biomedclip_v1/split_report.json")
    retrieval = safe_read_json(PIPELINE_ROOT / "retrieval_biomedclip_v1/retrieval_results.json")
    vqa = safe_read_parquet(PIPELINE_ROOT / "vqa_biomedclip_v1/vqa_dataset.parquet")
    captioning = safe_read_parquet(PIPELINE_ROOT / "captioning_biomedclip_v1/captioning_dataset.parquet")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version, "platform": platform.platform(), "random_seed": SEED,
        "compound_figures_input": int(len(worklist)) if worklist is not None else None,
        "panels_extracted": int(len(panels)) if panels is not None else None,
        "aligned_panels": int(len(alignment)) if alignment is not None else None,
        "quality": quality, "splits": split_report, "retrieval": retrieval,
        "vqa_pairs": int(len(vqa)) if vqa is not None else None,
        "vqa_question_types": vqa["question_type"].value_counts().to_dict() if vqa is not None else None,
        "captioning_pairs": int(len(captioning)) if captioning is not None else None,
        "models_used": {
            "panel_detector": "yolov8n (shared with CLIP track, Stage 1 is common)",
            "alignment_and_retrieval": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (open_clip)",
            "retrieval_index": "faiss IndexFlatIP",
        },
    }
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n[STAGE 8b COMPLETE] (BiomedCLIP)")
    print(f"Summary: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()