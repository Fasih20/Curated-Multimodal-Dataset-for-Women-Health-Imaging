"""
Source: original Colab notebook, cell index [11]
Auto-extracted -- review before treating as final.
"""

"""
Colab: fast version of 03_build_working_manifest.py.

Same output as before, but avoids the slow part: instead of opening ~16K
individual caption files over the mounted Drive filesystem (high per-file
latency, can take an hour+), this copies the 'relevant' images/ and
captions/ folders to local Colab disk in one bulk operation first, then
reads everything locally (fast).

Run this INSTEAD of 03_build_working_manifest.py, or after killing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# Config -- now pointing at the LOCAL copy, not Drive
# -----------------------------------------------------------------------------
LOCAL_DOWNLOAD_ROOT = Path("/content/local_download")
LOCAL_IMAGES_RELEVANT = LOCAL_DOWNLOAD_ROOT / "images/relevant"
LOCAL_CAPTIONS_RELEVANT = LOCAL_DOWNLOAD_ROOT / "captions/relevant"

SPLIT_MANIFEST_PATH = Path("/content/pipeline_data/splits_pmcid_v1/labeled_split_manifest_v1.parquet")

OUT_DIR = Path("/content/pipeline_data/working_v1")
WORKING_MANIFEST_PATH = OUT_DIR / "relevant_working_manifest_v1.parquet"
WORKING_SUMMARY_PATH = OUT_DIR / "relevant_working_manifest_v1.summary.json"


def build_pmcid_to_split(split_df: pd.DataFrame) -> dict[str, str]:
    per_pmcid_splits = split_df.groupby("pmcid")["split"].nunique()
    bad = per_pmcid_splits[per_pmcid_splits > 1]
    if len(bad) > 0:
        raise RuntimeError(
            f"{len(bad)} PMCIDs have rows in more than one split -- "
            "split manifest is inconsistent, stop and investigate."
        )
    return split_df.drop_duplicates("pmcid").set_index("pmcid")["split"].to_dict()


def parse_pmcid_from_stem(stem: str) -> str | None:
    # NNNNN-<pmcid>__<basename>
    if "__" not in stem:
        return None
    head = stem.split("__", 1)[0]
    if "-" not in head:
        return None
    return head.split("-", 1)[1]


def main() -> None:
    if not LOCAL_IMAGES_RELEVANT.is_dir():
        raise FileNotFoundError(
            f"{LOCAL_IMAGES_RELEVANT} not found -- run the bulk copy cell at "
            "the top of this file first."
        )

    print(f"Scanning local images at {LOCAL_IMAGES_RELEVANT} ...")
    image_files = sorted(p for p in LOCAL_IMAGES_RELEVANT.rglob("*") if p.is_file())
    print(f"Found {len(image_files)} local relevant image files")

    print(f"Loading split manifest from {SPLIT_MANIFEST_PATH} ...")
    split_df = pd.read_parquet(SPLIT_MANIFEST_PATH)
    pmcid_to_split = build_pmcid_to_split(split_df)

    rows = []
    n_no_caption = 0
    n_unmapped = 0
    for img_path in image_files:
        stem = img_path.stem
        pmcid = parse_pmcid_from_stem(stem)
        split = pmcid_to_split.get(pmcid) if pmcid else None
        if split is None:
            n_unmapped += 1
            continue

        caption_path = LOCAL_CAPTIONS_RELEVANT / f"{stem}.txt"
        caption_text = None
        if caption_path.is_file():
            try:
                caption_text = caption_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                caption_text = None
        else:
            n_no_caption += 1

        rows.append(
            {
                "pmcid": pmcid,
                "split": split,
                "image_path": str(img_path.relative_to(LOCAL_DOWNLOAD_ROOT)),
                "filename_stem": stem,
                "caption_text": caption_text,
            }
        )

    final_df = pd.DataFrame(rows)
    rows_by_split = final_df["split"].value_counts().to_dict()
    papers_by_split = final_df.groupby("split")["pmcid"].nunique().to_dict()

    summary = {
        "total_local_relevant_images": len(image_files),
        "images_with_pmcid_mapped_to_split": int(len(final_df)),
        "images_dropped_unmapped_pmcid": n_unmapped,
        "images_missing_caption_file": n_no_caption,
        "rows_by_split": {k: int(v) for k, v in rows_by_split.items()},
        "papers_by_split": {k: int(v) for k, v in papers_by_split.items()},
    }
    print(json.dumps(summary, indent=2, default=str))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(WORKING_MANIFEST_PATH, index=False)
    with open(WORKING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote canonical working manifest (local, Colab-only): {WORKING_MANIFEST_PATH}")
    print(f"Wrote summary: {WORKING_SUMMARY_PATH}")


if __name__ == "__main__":
    main()