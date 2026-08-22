"""
Source: original Colab notebook, cell index [4]
Auto-extracted -- review before treating as final.
"""

"""
Colab: resolve the PMCID split manifest against real files on disk.

Takes labeled_split_manifest_v1.parquet (41,772 rows, url-level, no guarantee
a local file exists for each) and joins it against the actual downloaded
images/captions folders, using the same item-key logic as
04-download-labeled.py's manifest.jsonl (pmcid, url, label).

Output stays in /content (Colab local disk) -- NOT written back to Drive.
Download/export it yourself when ready.

-----------------------------------------------------------------------------
COLAB SETUP (CPU runtime)
-----------------------------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')
!pip install -q pandas pyarrow
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive/dataset-women-health-imaging-ml")
SPLIT_MANIFEST_PATH = Path("/content/pipeline_data/splits_pmcid_v1/labeled_split_manifest_v1.parquet")

# Adjust to wherever the downloaded images/captions actually live on Drive.
# Per 04-download-labeled.py this is <output_dir>/manifest.jsonl plus
# images/<label>/... and captions/<label>/... -- check your real folder name,
# it may be data/labeled/downloaded_images or data/labelled/downloaded_labelled_images
DOWNLOAD_ROOT = DRIVE_ROOT / "data/labeled/downloaded_images"
IMAGES_ROOT = DOWNLOAD_ROOT / "images"
CAPTIONS_ROOT = DOWNLOAD_ROOT / "captions"

# Everything written locally in Colab -- upload/export yourself later.
OUT_DIR = Path("/content/pipeline_data/resolved_v1")
RESOLVED_PARQUET = OUT_DIR / "labeled_split_resolved_v1.parquet"
RESOLVED_SUMMARY = OUT_DIR / "labeled_split_resolved_v1.summary.json"

# Manifest known to reliably cover 'not_relevant' only (see technical-report.md
# section 6.7 / 15.2) -- used as a fallback join, not the primary source of truth.
DOWNLOAD_MANIFEST_JSONL = DOWNLOAD_ROOT / "manifest.jsonl"

# naming convention from 04-download-labeled.py:
#   images/<label>/NNNNN-<pmcid>__<url_basename>.<ext>
#   captions/<label>/NNNNN-<pmcid>__<url_basename>.txt
SEQ_PREFIX_RE = None  # set below after re import


def item_key(pmcid: str, url: str, label) -> tuple[str, str, str]:
    label_str = "" if label is None else str(label).strip()
    return (str(pmcid).strip(), str(url).strip(), label_str)


def load_download_manifest(path: Path) -> pd.DataFrame:
    """
    Read manifest.jsonl from 04-download-labeled.py. Known caveat: this
    manifest reliably records 'not_relevant' downloads but the 'relevant'
    side has incomplete provenance here -- use scan_disk_by_pmcid as the
    primary source for 'relevant' rows instead.
    """
    if not path.is_file():
        return pd.DataFrame(columns=["pmcid", "url", "label", "image_path", "caption_path"])

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or rec.get("type") == "run_header":
                continue
            if rec.get("ok") is not True:
                continue
            rows.append(
                {
                    "pmcid": str(rec.get("pmcid") or "").strip(),
                    "url": str(rec.get("url") or "").strip(),
                    "label": rec.get("label"),
                    "image_path": rec.get("image_path"),
                    "caption_path": rec.get("caption_path"),
                }
            )
    return pd.DataFrame(rows)


def scan_disk_by_pmcid(images_root: Path) -> pd.DataFrame:
    """
    Walk images/<label>/*  directly, no manifest trust required. Parses
    pmcid out of the filename (NNNNN-<pmcid>__<rest>) per the naming
    convention in 04-download-labeled.py's _safe_segment/name_core logic.
    Returns one row per file found: pmcid, label, image_path (relative to
    DOWNLOAD_ROOT), filename_stem (for matching against a caption file).
    """
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    rows = []
    for label_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for img_path in sorted(label_dir.rglob("*")):
            if not img_path.is_file():
                continue
            stem = img_path.stem  # NNNNN-<pmcid>__<basename>
            # pmcid is the token between the first '-' and the '__'
            pmcid = None
            if "__" in stem:
                head = stem.split("__", 1)[0]
                if "-" in head:
                    pmcid_candidate = head.split("-", 1)[1]
                    pmcid = pmcid_candidate
            rows.append(
                {
                    "label": label_dir.name,
                    "pmcid_from_filename": pmcid,
                    "image_path": str(img_path.relative_to(DOWNLOAD_ROOT)),
                    "filename_stem": stem,
                }
            )
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    print(f"Loading split manifest from {SPLIT_MANIFEST_PATH} ...")
    split_df = pd.read_parquet(SPLIT_MANIFEST_PATH)
    print(f"Split manifest: {len(split_df)} rows / {split_df['pmcid'].nunique()} PMCIDs")

    print(f"\nScanning images on disk under {IMAGES_ROOT} ...")
    disk_df = scan_disk_by_pmcid(IMAGES_ROOT)
    print(f"Image files found on disk: {len(disk_df)}")
    print(disk_df["label"].value_counts())

    # Primary join: split rows to disk files, matched by pmcid only (not url --
    # the manifest-recorded url provenance is unreliable for 'relevant', but
    # pmcid parsed straight from the filename is not).
    disk_by_pmcid = disk_df.groupby("pmcid_from_filename").size().rename("n_images_on_disk")

    split_pmcids = set(split_df["pmcid"].astype(str))
    disk_pmcids = set(disk_df["pmcid_from_filename"].dropna().astype(str))
    overlap_pmcids = split_pmcids & disk_pmcids

    print(f"\nPMCIDs in split manifest: {len(split_pmcids)}")
    print(f"PMCIDs with >=1 image file on disk: {len(disk_pmcids)}")
    print(f"PMCIDs present in both: {len(overlap_pmcids)}")

    # Row-level result: for each split row, does its pmcid have disk images,
    # and specifically how many 'relevant'-labeled images does that pmcid have.
    relevant_disk = (
        disk_df[disk_df["label"] == "relevant"]
        .groupby("pmcid_from_filename")
        .size()
        .rename("n_relevant_images_on_disk")
    )

    split_df = split_df.merge(
        relevant_disk, left_on="pmcid", right_index=True, how="left"
    )
    split_df["n_relevant_images_on_disk"] = split_df["n_relevant_images_on_disk"].fillna(0).astype(int)

    usable_relevant = split_df[
        (split_df["label"] == "relevant") & (split_df["n_relevant_images_on_disk"] > 0)
    ]

    by_split = usable_relevant.groupby("split")["pmcid"].nunique().to_dict()
    by_split_rows = usable_relevant.groupby("split").size().to_dict()

    summary = {
        "total_split_rows": int(len(split_df)),
        "total_pmcids_in_split": len(split_pmcids),
        "pmcids_with_any_image_on_disk": len(disk_pmcids),
        "pmcids_in_both": len(overlap_pmcids),
        "total_image_files_on_disk_by_label": disk_df["label"].value_counts().to_dict(),
        "relevant_labeled_rows_with_pmcid_having_disk_image": int(len(usable_relevant)),
        "usable_relevant_papers_by_split": {k: int(v) for k, v in by_split.items()},
        "usable_relevant_rows_by_split": {k: int(v) for k, v in by_split_rows.items()},
    }
    print(json.dumps(summary, indent=2, default=str))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_df.to_parquet(RESOLVED_PARQUET, index=False)
    disk_df.to_parquet(OUT_DIR / "disk_image_inventory_v1.parquet", index=False)
    with open(RESOLVED_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote resolved split (local, Colab-only): {RESOLVED_PARQUET}")
    print(f"Wrote disk image inventory: {OUT_DIR / 'disk_image_inventory_v1.parquet'}")
    print(f"Wrote summary: {RESOLVED_SUMMARY}")
    print(
        "\nNote: this joins by PMCID, not exact url/file, because the pmcid "
        "parsed from each filename is reliable while the manifest.jsonl url "
        "provenance for 'relevant' downloads is known-incomplete (see "
        "technical-report.md 6.7/15.2). A pmcid can have multiple images -- "
        "if you need row-exact (one split row = one specific file) matching, "
        "we'll need to match on caption text similarity or re-derive it from "
        "the caption .txt files next, since url matching alone won't cover it."
    )


if __name__ == "__main__":
    main()