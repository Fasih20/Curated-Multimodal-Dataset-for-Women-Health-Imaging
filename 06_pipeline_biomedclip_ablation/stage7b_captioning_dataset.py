"""
Source: original Colab notebook, cell index [65]
Auto-extracted -- review before treating as final.
"""

"""Colab: Stage 7b (ABLATION) -- captioning dataset from BiomedCLIP's accepted panels."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ACCEPTED_PATH = Path("/content/pipeline_data/quality_biomedclip_v1/accepted_dataset.parquet")
SPLITS_DIR = Path("/content/pipeline_data/splits_biomedclip_v1")

OUT_DIR = Path("/content/pipeline_data/captioning_biomedclip_v1")
CAPTIONING_PATH = OUT_DIR / "captioning_dataset.parquet"


def main() -> None:
    if CAPTIONING_PATH.exists():
        print(f"[SKIP] Stage 7b already done -- {CAPTIONING_PATH}")
        return

    accepted = pd.read_parquet(ACCEPTED_PATH)
    split_map = {}
    for split_name in ("train", "val", "test"):
        p = SPLITS_DIR / f"{split_name}.parquet"
        if p.exists():
            for pid in pd.read_parquet(p)["panel_id"].astype(str):
                split_map[pid] = split_name

    df = accepted[accepted["best_match_text"].notna() & (accepted["best_match_text"].str.strip() != "")].copy()
    df["split"] = df["panel_id"].astype(str).map(split_map).fillna("unassigned")
    out = df[["crop_path", "best_match_text", "figure_id", "panel_id", "split"]].rename(
        columns={"crop_path": "image_path", "best_match_text": "caption"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CAPTIONING_PATH, index=False)
    out.to_csv(CAPTIONING_PATH.with_suffix(".csv"), index=False)

    print("\n[STAGE 7b COMPLETE] (BiomedCLIP)")
    print(f"Caption pairs: {len(out)}")
    print(out["split"].value_counts())
    print(f"Output: {CAPTIONING_PATH}")


if __name__ == "__main__":
    main()