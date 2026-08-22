"""
Source: original Colab notebook, cell index [13]
Auto-extracted -- review before treating as final.
"""

"""
Colab: image integrity check + duplicate detection.

Runs against the canonical working manifest (16,171 relevant images, local
Colab disk copy). Adds columns for:
  - integrity: openable, width, height, format, is_low_res
  - dedup: sha256 (exact dup), phash (near-dup, catches re-compressed /
    resized copies of the same figure)

Does NOT delete or drop anything -- only flags, same conservative approach
your collaborator used for cleaning_v1. You decide what to exclude after
reviewing the flagged groups.

Needs Pillow + imagehash:
  !pip install -q pillow imagehash pandas pyarrow

Everything stays local to Colab (/content), nothing written to Drive.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile

try:
    import imagehash
except ImportError as e:
    raise SystemExit("Run: !pip install -q imagehash") from e

# Some PMC figures are truncated/odd -- don't hard-fail on load, but we still
# want to know when this happens (see is_truncated below).
ImageFile.LOAD_TRUNCATED_IMAGES = False

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
LOCAL_DOWNLOAD_ROOT = Path("/content/local_download")
WORKING_MANIFEST_PATH = Path("/content/pipeline_data/working_v1/relevant_working_manifest_v1.parquet")

OUT_DIR = Path("/content/pipeline_data/cleaning_v1_local")
CLEANED_MANIFEST_PATH = OUT_DIR / "relevant_manifest_with_flags_v1.parquet"
DUP_GROUPS_PATH = OUT_DIR / "duplicate_groups_v1.json"
SUMMARY_PATH = OUT_DIR / "cleaning_summary_v1.json"

# Below this on the shorter side, flag as low-res (adjust based on what your
# downstream VLMs expect -- 224 is a common minimum for ViT-style encoders).
MIN_SHORT_SIDE = 224

# Hamming distance threshold for near-duplicate phash matches. 0 = identical
# hash; small values (<=4) catch re-saves/re-compressions of the same image.
PHASH_NEAR_DUP_THRESHOLD = 4


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_image(path: Path) -> dict:
    result = {
        "openable": False,
        "width": None,
        "height": None,
        "format": None,
        "mode": None,
        "is_low_res": None,
        "sha256": None,
        "phash": None,
        "error": None,
    }
    try:
        result["sha256"] = sha256_of_file(path)
    except OSError as e:
        result["error"] = f"read_error: {e}"
        return result

    try:
        with Image.open(path) as im:
            im.verify()  # cheap structural check
        with Image.open(path) as im:  # re-open, verify() invalidates the handle
            im.load()  # force full decode -- catches truncated files
            w, h = im.size
            result["width"] = w
            result["height"] = h
            result["format"] = im.format
            result["mode"] = im.mode
            result["openable"] = True
            result["is_low_res"] = bool(min(w, h) < MIN_SHORT_SIDE)
            result["phash"] = str(imagehash.phash(im))
    except Exception as e:  # noqa: BLE001 - want to flag, not crash, on bad images
        result["error"] = f"decode_error: {e}"

    return result


def find_exact_duplicate_groups(df: pd.DataFrame) -> list[list[str]]:
    groups = df.dropna(subset=["sha256"]).groupby("sha256")["image_path"].apply(list)
    return [g for g in groups if len(g) > 1]


def find_near_duplicate_groups(df: pd.DataFrame, threshold: int) -> list[list[str]]:
    """
    Naive O(n^2) phash comparison. Fine for ~16K images (a few minutes);
    switch to a BK-tree or LSH bucket approach if this ever needs to scale
    to the large corpus.
    """
    valid = df.dropna(subset=["phash"])[["image_path", "phash"]].reset_index(drop=True)
    hashes = [imagehash.hex_to_hash(h) for h in valid["phash"]]
    n = len(hashes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= threshold:
                union(i, j)

    groups_by_root: dict[int, list[str]] = {}
    for i in range(n):
        root = find(i)
        groups_by_root.setdefault(root, []).append(valid.loc[i, "image_path"])

    return [g for g in groups_by_root.values() if len(g) > 1]


def main() -> None:
    print(f"Loading working manifest from {WORKING_MANIFEST_PATH} ...")
    df = pd.read_parquet(WORKING_MANIFEST_PATH)
    print(f"Rows: {len(df)}")

    print("Inspecting images (integrity + hashes) -- this is the slow part, "
          "expect a few minutes for ~16K local files ...")
    records = []
    for i, row in enumerate(df.itertuples(index=False)):
        abs_path = LOCAL_DOWNLOAD_ROOT / row.image_path
        info = inspect_image(abs_path)
        records.append(info)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(df)} inspected")

    info_df = pd.DataFrame(records)
    out_df = pd.concat([df.reset_index(drop=True), info_df], axis=1)

    n_unopenable = int((~out_df["openable"]).sum())
    n_low_res = int(out_df["is_low_res"].fillna(False).sum())
    print(f"\nUnopenable/corrupt images: {n_unopenable}")
    print(f"Low-res images (shorter side < {MIN_SHORT_SIDE}px): {n_low_res}")

    print("\nFinding exact duplicates (sha256) ...")
    exact_groups = find_exact_duplicate_groups(out_df)
    print(f"Exact-duplicate groups: {len(exact_groups)} "
          f"({sum(len(g) for g in exact_groups)} images involved)")

    print("Finding near-duplicates (phash, may take a few minutes for ~16K images) ...")
    openable_df = out_df[out_df["openable"]]
    near_groups = find_near_duplicate_groups(openable_df, PHASH_NEAR_DUP_THRESHOLD)
    print(f"Near-duplicate groups: {len(near_groups)} "
          f"({sum(len(g) for g in near_groups)} images involved)")

    # Flag: keep first image per exact-dup group and per near-dup group,
    # flag the rest as duplicate candidates. Does not drop rows.
    dup_flag = {}
    for g in exact_groups:
        for p in g[1:]:
            dup_flag[p] = "exact_duplicate"
    for g in near_groups:
        for p in g[1:]:
            dup_flag.setdefault(p, "near_duplicate")
    out_df["duplicate_flag"] = out_df["image_path"].map(dup_flag)

    summary = {
        "total_images": int(len(out_df)),
        "unopenable_images": n_unopenable,
        "low_res_images": n_low_res,
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_images_flagged": sum(len(g) - 1 for g in exact_groups),
        "near_duplicate_groups": len(near_groups),
        "near_duplicate_images_flagged": sum(len(g) - 1 for g in near_groups),
        "min_short_side_threshold": MIN_SHORT_SIDE,
        "phash_near_dup_threshold": PHASH_NEAR_DUP_THRESHOLD,
        "rows_by_split_after_flags": out_df.groupby("split").size().to_dict(),
        "clean_estimate_rows": int(
            len(out_df)
            - n_unopenable
            - out_df["duplicate_flag"].notna().sum()
        ),
    }
    print(json.dumps(summary, indent=2, default=str))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(CLEANED_MANIFEST_PATH, index=False)
    with open(DUP_GROUPS_PATH, "w", encoding="utf-8") as f:
        json.dump({"exact": exact_groups, "near": near_groups}, f, indent=2)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote flagged manifest: {CLEANED_MANIFEST_PATH}")
    print(f"Wrote duplicate groups: {DUP_GROUPS_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")
    print(
        "\nNothing was deleted -- review duplicate_groups_v1.json and the "
        "duplicate_flag / openable / is_low_res columns before deciding what "
        "to exclude from the modeling subset. Next: compound-image detection "
        "on the surviving (non-duplicate, openable) rows."
    )


if __name__ == "__main__":
    main()