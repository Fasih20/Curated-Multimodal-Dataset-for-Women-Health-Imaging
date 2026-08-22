"""
Source: original Colab notebook, cell index [15]
Auto-extracted -- review before treating as final.
"""

"""
Colab: resolve exact-duplicate groups to one canonical row each.

Confirmed benign: all 6,120 exact-duplicate groups share the same pmcid
(multilingual duplicate articles -- same image, English/German/Portuguese
captions -- plus a few double-extracted figures with one empty caption).
Group sizes are small (2-4). Safe to keep exactly one row per group.

Selection rule per group, in priority order:
  1. Prefer a row with a non-empty caption over an empty/missing one.
  2. Among non-empty captions, prefer the one that looks most like English
     (lowest ratio of non-ASCII characters -- catches German umlauts,
     Portuguese diacritics etc. without needing a language-detection lib).
  3. Tie-break: longest caption, then lowest filename_stem (earliest
     sequence number) for determinism.

Also drops the 3 unopenable/corrupt images and flags low-res ones (kept,
just flagged -- your call whether to exclude those for training).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FLAGGED_MANIFEST_PATH = Path("/content/pipeline_data/cleaning_v1_local/relevant_manifest_with_flags_v1.parquet")

OUT_DIR = Path("/content/pipeline_data/cleaning_v1_local")
DEDUPED_MANIFEST_PATH = OUT_DIR / "relevant_manifest_deduped_v1.parquet"
DEDUP_SUMMARY_PATH = OUT_DIR / "dedup_resolution_summary_v1.json"


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(len(text), 1)


def pick_canonical(group_df: pd.DataFrame) -> pd.Series:
    g = group_df.copy()
    g["_has_caption"] = g["caption_text"].fillna("").str.strip().str.len() > 0
    g["_non_ascii_ratio"] = g["caption_text"].fillna("").apply(non_ascii_ratio)
    g["_caption_len"] = g["caption_text"].fillna("").str.len()
    g_sorted = g.sort_values(
        by=["_has_caption", "_non_ascii_ratio", "_caption_len", "filename_stem"],
        ascending=[False, True, False, True],
    )
    return g_sorted.iloc[0]


def main() -> None:
    df = pd.read_parquet(FLAGGED_MANIFEST_PATH)
    print(f"Loaded {len(df)} rows")

    # Drop corrupt/unopenable outright -- not a judgment call.
    before = len(df)
    df = df[df["openable"] == True]  # noqa: E712
    print(f"Dropped {before - len(df)} unopenable/corrupt images")

    exact_dup_sha = df.dropna(subset=["sha256"])
    groups = exact_dup_sha.groupby("sha256")

    kept_rows = []
    n_groups_with_dupes = 0
    n_dropped_as_dupe = 0
    for sha, group_df in groups:
        if len(group_df) == 1:
            kept_rows.append(group_df.iloc[0])
            continue
        n_groups_with_dupes += 1
        canonical = pick_canonical(group_df)
        kept_rows.append(canonical)
        n_dropped_as_dupe += len(group_df) - 1

    deduped_df = pd.DataFrame(kept_rows).drop(
        columns=[c for c in ["_has_caption", "_non_ascii_ratio", "_caption_len"]
                 if c in kept_rows[0].index],
        errors="ignore",
    )
    # pd.DataFrame(list of Series) can carry the helper cols through; drop cleanly
    deduped_df = deduped_df.loc[:, ~deduped_df.columns.str.startswith("_")]

    rows_by_split = deduped_df["split"].value_counts().to_dict()
    papers_by_split = deduped_df.groupby("split")["pmcid"].nunique().to_dict()
    still_no_caption = int((deduped_df["caption_text"].fillna("").str.strip() == "").sum())
    low_res_remaining = int(deduped_df["is_low_res"].fillna(False).sum())

    summary = {
        "rows_before_dedup": int(before),
        "rows_after_dropping_unopenable": int(len(df)),
        "duplicate_groups_resolved": n_groups_with_dupes,
        "rows_dropped_as_duplicate": n_dropped_as_dupe,
        "rows_after_dedup": int(len(deduped_df)),
        "rows_still_missing_caption_after_dedup": still_no_caption,
        "low_res_rows_remaining_flagged_not_dropped": low_res_remaining,
        "rows_by_split": {k: int(v) for k, v in rows_by_split.items()},
        "papers_by_split": {k: int(v) for k, v in papers_by_split.items()},
    }
    print(json.dumps(summary, indent=2, default=str))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deduped_df.to_parquet(DEDUPED_MANIFEST_PATH, index=False)
    with open(DEDUP_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote deduped manifest: {DEDUPED_MANIFEST_PATH}")
    print(f"Wrote summary: {DEDUP_SUMMARY_PATH}")
    print(
        "\nNote: near-duplicate (phash) groups were NOT auto-resolved here -- "
        "those can be legitimately different images (e.g. two different "
        "patients' scans that just look visually similar), unlike exact sha256 "
        "matches which are byte-identical. Review near_duplicate_flag rows "
        "manually if you want to prune those too.\n"
        "Next: compound-image detection on this deduped set."
    )


if __name__ == "__main__":
    main()