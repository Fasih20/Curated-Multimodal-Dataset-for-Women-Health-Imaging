"""
Source: original Colab notebook, cell index [40]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 3 -- quality analysis over panels + alignment.

Nothing is deleted. Every panel is labeled accepted/flagged with an
explicit reason list, so you can loosen/tighten thresholds later without
recomputing anything upstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PANEL_MANIFEST_PATH = Path("/content/pipeline_data/panels_v1/panel_manifest.csv")
ALIGNMENT_MANIFEST_PATH = Path("/content/pipeline_data/alignment_v1/alignment_manifest.parquet")

OUT_DIR = Path("/content/pipeline_data/quality_v1")
ACCEPTED_PATH = OUT_DIR / "accepted_dataset.parquet"
FLAGGED_PATH = OUT_DIR / "flagged_dataset.parquet"
STATS_PATH = OUT_DIR / "quality_summary.json"
PLOTS_DIR = OUT_DIR / "plots"

# Thresholds -- documented here, not buried in code. Conservative on
# purpose: flag, don't drop, anything borderline.
MIN_PANEL_SIDE_PX = 32
MIN_SIMILARITY = 0.18          # CLIP cosine sim below this = likely bad match
MIN_MARGIN = 0.01              # near-zero margin = ambiguous between candidates
MAX_ASPECT_RATIO = 8.0         # extreme aspect ratio = likely a mis-crop


def main() -> None:
    if ACCEPTED_PATH.exists() and FLAGGED_PATH.exists():
        print(f"[SKIP] Stage 3 already done -- see {OUT_DIR}")
        return

    panels = pd.read_csv(PANEL_MANIFEST_PATH)
    align = pd.read_parquet(ALIGNMENT_MANIFEST_PATH)
    df = panels.merge(align, on=["figure_id", "panel_id"], how="left")

    n_panels_per_fig = df.groupby("figure_id")["panel_id"].transform("count")
    df["panels_in_figure"] = n_panels_per_fig

    reasons = []
    for _, r in df.iterrows():
        row_reasons = []
        if r["width"] < MIN_PANEL_SIDE_PX or r["height"] < MIN_PANEL_SIDE_PX:
            row_reasons.append("panel_too_small")
        if r["aspect_ratio"] and (r["aspect_ratio"] > MAX_ASPECT_RATIO or r["aspect_ratio"] < 1 / MAX_ASPECT_RATIO):
            row_reasons.append("extreme_aspect_ratio")
        if pd.isna(r.get("best_similarity")):
            row_reasons.append("missing_alignment")
        elif r["best_similarity"] < MIN_SIMILARITY:
            row_reasons.append("low_clip_similarity")
        if pd.notna(r.get("similarity_margin")) and r["similarity_margin"] < MIN_MARGIN and r.get("n_candidate_segments", 1) > 1:
            row_reasons.append("ambiguous_alignment")
        if r.get("match_type") == "figure_level_caption":
            row_reasons.append("figure_level_caption_only")  # not disqualifying alone, just noted
        reasons.append(row_reasons)

    df["quality_flags"] = reasons
    disqualifying = {"panel_too_small", "extreme_aspect_ratio", "missing_alignment",
                      "low_clip_similarity", "ambiguous_alignment"}
    df["is_flagged"] = df["quality_flags"].apply(lambda rs: bool(set(rs) & disqualifying))

    accepted = df[~df["is_flagged"]].reset_index(drop=True)
    flagged = df[df["is_flagged"]].reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    accepted.to_parquet(ACCEPTED_PATH, index=False)
    flagged.to_parquet(FLAGGED_PATH, index=False)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    df["aspect_ratio"].clip(0, 10).hist(bins=40, ax=axes[0, 0])
    axes[0, 0].set_title("Panel aspect ratio")
    df["best_similarity"].dropna().hist(bins=40, ax=axes[0, 1])
    axes[0, 1].set_title("CLIP best_similarity")
    df["panels_in_figure"].hist(bins=30, ax=axes[1, 0])
    axes[1, 0].set_title("Panels per figure")
    df["is_flagged"].value_counts().plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Accepted vs Flagged")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "quality_overview.png", dpi=120)
    plt.close(fig)

    reason_counts = pd.Series([r for rs in df["quality_flags"] for r in rs]).value_counts().to_dict()
    summary = {
        "total_panels": int(len(df)),
        "accepted": int(len(accepted)),
        "flagged": int(len(flagged)),
        "flag_reason_counts": {k: int(v) for k, v in reason_counts.items()},
        "thresholds": {
            "MIN_PANEL_SIDE_PX": MIN_PANEL_SIDE_PX,
            "MIN_SIMILARITY": MIN_SIMILARITY,
            "MIN_MARGIN": MIN_MARGIN,
            "MAX_ASPECT_RATIO": MAX_ASPECT_RATIO,
        },
    }
    with open(STATS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[STAGE 3 COMPLETE]")
    print(f"Accepted: {len(accepted)} | Flagged: {len(flagged)}")
    print(f"Flag reasons: {json.dumps(reason_counts, indent=2)}")
    print(f"Output: {ACCEPTED_PATH}, {FLAGGED_PATH}")


if __name__ == "__main__":
    main()