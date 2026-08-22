"""
Source: original Colab notebook, cell index [35]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 0 -- build the canonical compound-figure worklist.

Merges two manifests that were never joined in the notebook so far:
  - compound_v2/relevant_manifest_with_compound_flags_v2.parquet
        (has compound_flag_v2, compound_tier -- from the tiered policy pass)
  - working_v1/relevant_working_manifest_v1.parquet
        (has image_path, caption_text, split, pmcid, filename_stem)

ASSUMPTION (document this in your methods section): the join key is
`filename_stem` if compound_v2 has it, else `pmcid` (falls back to a
pmcid-level join, which is coarser -- logged explicitly so you know
which path was taken). Adjust JOIN_KEY below if your actual columns
differ; the script prints available columns from both frames before
merging so you can sanity check.

Output: one row per compound figure with everything downstream stages
need (figure_id, image_path, caption_text, split, pmcid, compound_tier).
Restartable: skips the merge if the output already exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COMPOUND_PATH = Path(
    "/content/pipeline_data/compound_v2/relevant_manifest_with_compound_flags_v2.parquet"
)
WORKING_PATH = Path(
    "/content/pipeline_data/working_v1/relevant_working_manifest_v1.parquet"
)

OUT_DIR = Path("/content/pipeline_data/compound_worklist_v1")
OUT_PATH = OUT_DIR / "compound_figures_manifest.parquet"
SUMMARY_PATH = OUT_DIR / "compound_figures_manifest.summary.json"


def main() -> None:
    if OUT_PATH.exists():
        df = pd.read_parquet(OUT_PATH)
        print(f"[SKIP] Stage 0 already done -- {len(df)} compound figures at {OUT_PATH}")
        return

    compound_df = pd.read_parquet(COMPOUND_PATH)
    working_df = pd.read_parquet(WORKING_PATH)

    print("compound_v2 columns:", list(compound_df.columns))
    print("working_v1 columns:", list(working_df.columns))

    flagged = compound_df[compound_df["compound_flag_v2"] == True].copy()  # noqa: E712
    print(f"Compound-flagged rows (v2 policy): {len(flagged)}")

    if "filename_stem" in flagged.columns and "filename_stem" in working_df.columns:
        join_key = "filename_stem"
    elif "pmcid" in flagged.columns and "pmcid" in working_df.columns:
        join_key = "pmcid"
    else:
        raise RuntimeError(
            "Neither filename_stem nor pmcid present in both frames -- "
            "inspect columns printed above and set JOIN_KEY manually."
        )
    print(f"Joining on: {join_key}")

    merged = flagged.merge(
        working_df[[join_key, "image_path", "caption_text", "split", "pmcid"]]
        if join_key != "pmcid"
        else working_df[[join_key, "image_path", "caption_text", "split"]],
        on=join_key,
        how="inner",
        suffixes=("", "_work"),
    )

    n_dropped = len(flagged) - len(merged)
    if n_dropped:
        print(f"[WARN] {n_dropped} compound-flagged rows had no match in working manifest "
              f"(no local image/caption) -- excluded, not silently kept.")

    merged = merged.dropna(subset=["image_path"]).reset_index(drop=True)
    merged["figure_id"] = merged.get("pmcid", merged[join_key]).astype(str) + "__" + merged.index.astype(str)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)

    summary = {
        "join_key": join_key,
        "compound_flagged_input": int(len(flagged)),
        "resolved_to_local_files": int(len(merged)),
        "dropped_no_match": int(n_dropped),
        "by_split": merged["split"].value_counts().to_dict() if "split" in merged else {},
        "by_tier": merged["compound_tier"].value_counts().to_dict() if "compound_tier" in merged else {},
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n[STAGE 0 COMPLETE]")
    print(f"Compound figures resolved: {len(merged)}")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()