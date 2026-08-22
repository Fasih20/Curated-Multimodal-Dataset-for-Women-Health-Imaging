"""
Source: original Colab notebook, cell index [14]
Auto-extracted -- review before treating as final.
"""

"""
Colab: diagnose the exact-duplicate spike (76% of images in dup groups).

Before excluding anything, answer:
  1. Are duplicates within the same PMCID (same article, same figure cited
     by multiple caption rows -- benign/expected) or across different
     PMCIDs (same file content shared across articles -- suspicious)?
  2. Are the duplicated files suspiciously small/uniform (a broken-image
     placeholder saved on every failed download -- a real bug) or normal-
     sized medical figures?
  3. What do a few duplicate groups actually look like?
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

LOCAL_DOWNLOAD_ROOT = Path("/content/local_download")
FLAGGED_MANIFEST_PATH = Path("/content/pipeline_data/cleaning_v1_local/relevant_manifest_with_flags_v1.parquet")
DUP_GROUPS_PATH = Path("/content/pipeline_data/cleaning_v1_local/duplicate_groups_v1.json")

OUT_DIR = Path("/content/pipeline_data/cleaning_v1_local")
SAMPLE_THUMBS_DIR = OUT_DIR / "dup_group_samples"


def main() -> None:
    df = pd.read_parquet(FLAGGED_MANIFEST_PATH)
    with open(DUP_GROUPS_PATH, encoding="utf-8") as f:
        groups = json.load(f)
    exact_groups = groups["exact"]

    print(f"Total exact-duplicate groups: {len(exact_groups)}")

    # Map image_path -> pmcid for quick lookup
    path_to_pmcid = dict(zip(df["image_path"], df["pmcid"]))

    same_pmcid_groups = 0
    cross_pmcid_groups = 0
    group_sizes = Counter()
    for g in exact_groups:
        pmcids = {path_to_pmcid.get(p) for p in g}
        group_sizes[len(g)] += 1
        if len(pmcids) == 1:
            same_pmcid_groups += 1
        else:
            cross_pmcid_groups += 1

    print(f"\nGroups where all images share the SAME pmcid (benign -- same "
          f"figure cited by multiple caption rows): {same_pmcid_groups}")
    print(f"Groups spanning DIFFERENT pmcids (same file content reused "
          f"across articles -- needs a closer look): {cross_pmcid_groups}")
    print(f"\nGroup size distribution (size -> count): {dict(sorted(group_sizes.items()))}")

    # File size + dimensions for a sample of duplicated files, to check for
    # a broken-image-placeholder pattern (suspiciously small/uniform).
    print("\nFile size stats for images inside exact-duplicate groups:")
    dup_paths = [p for g in exact_groups for p in g]
    sizes = []
    dims = []
    for p in dup_paths[:2000]:  # sample, full scan of 12K is unnecessary here
        abs_path = LOCAL_DOWNLOAD_ROOT / p
        try:
            sizes.append(abs_path.stat().st_size)
            with Image.open(abs_path) as im:
                dims.append(im.size)
        except OSError:
            continue
    if sizes:
        s = pd.Series(sizes)
        print(f"  file size (bytes): min={s.min()} p25={s.quantile(.25):.0f} "
              f"median={s.median():.0f} p75={s.quantile(.75):.0f} max={s.max()}")
    if dims:
        dim_counts = Counter(dims)
        print(f"  most common dimensions in sample: {dim_counts.most_common(5)}")

    # Print full detail for the 5 largest duplicate groups so you can eyeball
    # whether they're a real repeated figure or a broken placeholder.
    print("\n--- Largest duplicate groups (inspect these manually) ---")
    largest = sorted(exact_groups, key=len, reverse=True)[:5]
    for i, g in enumerate(largest):
        print(f"\nGroup {i+1}: {len(g)} images")
        for p in g[:5]:
            abs_path = LOCAL_DOWNLOAD_ROOT / p
            pmcid = path_to_pmcid.get(p)
            try:
                size_bytes = abs_path.stat().st_size
                with Image.open(abs_path) as im:
                    dims_str = f"{im.size[0]}x{im.size[1]}"
            except OSError:
                size_bytes, dims_str = None, None
            row = df[df["image_path"] == p].iloc[0]
            caption_preview = (row.get("caption_text") or "")[:80]
            print(f"  {p} | pmcid={pmcid} | {size_bytes}B | {dims_str} | caption: {caption_preview!r}")
        if len(g) > 5:
            print(f"  ... and {len(g) - 5} more")

    # Save a couple of the largest groups' images as viewable thumbnails so
    # you can look at them directly in the Colab file browser.
    SAMPLE_THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    for gi, g in enumerate(largest[:3]):
        for pi, p in enumerate(g[:4]):
            abs_path = LOCAL_DOWNLOAD_ROOT / p
            try:
                with Image.open(abs_path) as im:
                    im.thumbnail((256, 256))
                    out_path = SAMPLE_THUMBS_DIR / f"group{gi}_img{pi}_{Path(p).name}"
                    im.convert("RGB").save(out_path, "JPEG")
            except OSError:
                continue
    print(f"\nSaved thumbnails for the 3 largest groups to: {SAMPLE_THUMBS_DIR}")
    print("Open these in the Colab file browser (left sidebar) to visually confirm.")


if __name__ == "__main__":
    main()