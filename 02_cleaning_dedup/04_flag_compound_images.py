"""
Source: original Colab notebook, cell index [16]
Auto-extracted -- review before treating as final.
"""

"""
Colab: flag likely compound (multi-panel) images.

Same conservative approach as before: flag, don't auto-split or auto-drop.
Two independent signals, matching what your collaborator's original
integrity-and-compound-filter used (per docs/03-integrity-and-compound-filter.md):

  1. Caption language cues: panel markers like "(A)", "(a) and (b)",
     "top: ... bottom: ...", "left, right", multiple sentences each
     describing a distinct sub-image.
  2. Aspect ratio outliers: multi-panel figures are very often wide
     (side-by-side panels) or tall (stacked panels) relative to the bulk
     of single-panel figures in this dataset.

Output: same manifest with a new `compound_flag` column + reasons, plus a
sample gallery of flagged images saved as thumbnails so you can eyeball
how good the heuristic actually is before deciding whether to build an
automatic panel-splitter.

Nothing is dropped or split -- this stage is detection only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image

LOCAL_DOWNLOAD_ROOT = Path("/content/local_download")
DEDUPED_MANIFEST_PATH = Path("/content/pipeline_data/cleaning_v1_local/relevant_manifest_deduped_v1.parquet")

OUT_DIR = Path("/content/pipeline_data/compound_v1")
FLAGGED_PATH = OUT_DIR / "relevant_manifest_with_compound_flags_v1.parquet"
SUMMARY_PATH = OUT_DIR / "compound_detection_summary_v1.json"
SAMPLE_DIR = OUT_DIR / "compound_flagged_samples"

# -----------------------------------------------------------------------------
# Caption-based panel-language patterns
# -----------------------------------------------------------------------------
PANEL_LETTER_RE = re.compile(
    r"\(\s*[a-hA-H]\s*\)"  # (A), (b), (C) etc.
)
PANEL_WORD_RE = re.compile(
    r"\b(top|bottom|left|right|upper|lower|inset|panel[s]?)\b",
    re.IGNORECASE,
)
MULTI_LETTER_LIST_RE = re.compile(
    r"\b[a-h]\b(?:\s*(?:,|and|&)\s*\b[a-h]\b){1,}",  # "a and b", "a, b, c"
    re.IGNORECASE,
)

# -----------------------------------------------------------------------------
# Aspect ratio thresholds -- tune after looking at the distribution printed
# below; these are reasonable starting points for "unusually wide/tall".
# -----------------------------------------------------------------------------
WIDE_ASPECT_THRESHOLD = 1.8   # width / height
TALL_ASPECT_THRESHOLD = 0.5   # width / height (i.e. height > 2x width)

SAMPLE_N = 24


def caption_panel_cues(caption: str) -> list[str]:
    if not caption:
        return []
    reasons = []
    n_letter_markers = len(PANEL_LETTER_RE.findall(caption))
    if n_letter_markers >= 2:
        reasons.append(f"panel_letters_x{n_letter_markers}")
    if MULTI_LETTER_LIST_RE.search(caption):
        reasons.append("panel_letter_list")
    if PANEL_WORD_RE.search(caption):
        reasons.append("panel_word")
    return reasons


def aspect_ratio_cue(width: int, height: int) -> str | None:
    if not width or not height:
        return None
    ratio = width / height
    if ratio >= WIDE_ASPECT_THRESHOLD:
        return f"wide_aspect_{ratio:.2f}"
    if ratio <= TALL_ASPECT_THRESHOLD:
        return f"tall_aspect_{ratio:.2f}"
    return None


def main() -> None:
    df = pd.read_parquet(DEDUPED_MANIFEST_PATH)
    print(f"Loaded {len(df)} rows")

    print("\nAspect ratio distribution (width/height) over this dataset:")
    ratios = (df["width"] / df["height"]).dropna()
    print(ratios.describe())

    reasons_col = []
    for row in df.itertuples(index=False):
        reasons = []
        reasons.extend(caption_panel_cues(row.caption_text or ""))
        ar_cue = aspect_ratio_cue(row.width, row.height)
        if ar_cue:
            reasons.append(ar_cue)
        reasons_col.append(reasons)

    df["compound_reasons"] = reasons_col
    df["compound_flag"] = df["compound_reasons"].apply(lambda r: len(r) > 0)

    n_flagged = int(df["compound_flag"].sum())
    n_caption_only = int(
        df["compound_reasons"].apply(
            lambda r: any(x for x in r if not x.startswith(("wide_", "tall_")))
        ).sum()
    )
    n_aspect_only = int(
        df["compound_reasons"].apply(
            lambda r: any(x.startswith(("wide_", "tall_")) for x in r)
        ).sum()
    )
    n_both = int(
        df["compound_reasons"].apply(
            lambda r: any(x.startswith(("wide_", "tall_")) for x in r)
            and any(x for x in r if not x.startswith(("wide_", "tall_")))
        ).sum()
    )

    summary = {
        "total_rows": int(len(df)),
        "flagged_as_likely_compound": n_flagged,
        "flagged_by_caption_cue": n_caption_only,
        "flagged_by_aspect_ratio_cue": n_aspect_only,
        "flagged_by_both_signals": n_both,
        "flagged_fraction": round(n_flagged / len(df), 4),
        "wide_aspect_threshold": WIDE_ASPECT_THRESHOLD,
        "tall_aspect_threshold": TALL_ASPECT_THRESHOLD,
    }
    print(json.dumps(summary, indent=2, default=str))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FLAGGED_PATH, index=False)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save a random sample of flagged images (prioritize ones flagged by BOTH
    # signals -- highest confidence) as thumbnails for visual review.
    both_df = df[
        df["compound_reasons"].apply(
            lambda r: any(x.startswith(("wide_", "tall_")) for x in r)
            and any(x for x in r if not x.startswith(("wide_", "tall_")))
        )
    ]
    sample_df = both_df.sample(min(SAMPLE_N, len(both_df)), random_state=42) if len(both_df) else df[df["compound_flag"]].sample(min(SAMPLE_N, n_flagged), random_state=42)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for row in sample_df.itertuples(index=False):
        abs_path = LOCAL_DOWNLOAD_ROOT / row.image_path
        try:
            with Image.open(abs_path) as im:
                im.thumbnail((320, 320))
                safe_name = Path(row.image_path).name
                im.convert("RGB").save(SAMPLE_DIR / safe_name, "JPEG")
        except OSError:
            continue

    print(f"\nWrote flagged manifest: {FLAGGED_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")
    print(f"Saved {len(sample_df)} sample thumbnails to: {SAMPLE_DIR}")
    print(
        "\nOpen the sample thumbnails in Colab's file browser and check how "
        "many are genuinely multi-panel vs false positives. That tells us "
        "whether these heuristics are good enough to trust, or whether we "
        "need a VLM-based panel classifier instead before deciding how to "
        "handle (split vs exclude vs keep-as-is) the ~"
        f"{n_flagged} flagged images."
    )


if __name__ == "__main__":
    main()