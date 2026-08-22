"""
Source: original Colab notebook, cell index [17]
Auto-extracted -- review before treating as final.
"""

"""
Colab: stratified precision check on compound-image flags.

Your compound_flag heuristic fired on 7,392/10,042 rows (73.6%) using two
independent signals (caption panel-cues, aspect-ratio outliers). Before
deciding what to do with that many rows, we need per-signal precision:
the "both signals" bucket (2,640 rows) is your highest-confidence bucket
and you've already visually confirmed it looks good -- but that tells you
nothing about the other ~4,750 rows that were flagged by only ONE weak
signal.

Known risk specific to this medical corpus: PANEL_WORD_RE matches
"left"/"right"/"upper"/"lower" as standalone words. In medical captions
these very often describe ANATOMY/LATERALITY ("left ovary", "right lobe",
"upper pole of kidney") rather than a multi-panel figure layout. That is
a likely source of false positives concentrated in the caption-only
bucket specifically.

This script:
  1. Splits flagged rows into three disjoint buckets: caption-only,
     aspect-only, both-signals.
  2. Further splits caption-only into "letter-marker" hits (much more
     reliable -- "(A)" "(B)" almost never means laterality) vs
     "word-only" hits (the laterality-risk bucket).
  3. Draws a stratified random sample (default 40 per bucket) and saves
     thumbnails + a review CSV so you can eyeball true/false positive
     rate per bucket in ~15-20 minutes of manual review.
  4. Writes counts so you know exactly how big each bucket is before
     you decide policy.

Nothing is dropped, split, or excluded here -- this is measurement only.
Fill in the `is_compound` column in the review CSV by hand (1/0), rerun
`08_apply_compound_policy.py` after with your findings to set thresholds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image

LOCAL_DOWNLOAD_ROOT = Path("/content/local_download")
FLAGGED_MANIFEST_PATH = Path(
    "/content/pipeline_data/compound_v1/relevant_manifest_with_compound_flags_v1.parquet"
)

OUT_DIR = Path("/content/pipeline_data/compound_v1/precision_check")
REVIEW_CSV_PATH = OUT_DIR / "compound_precision_review.csv"
SUMMARY_PATH = OUT_DIR / "compound_precision_bucket_counts.json"
SAMPLE_DIR = OUT_DIR / "review_thumbnails"

SAMPLE_PER_BUCKET = 40
RANDOM_STATE = 42

PANEL_LETTER_RE = re.compile(r"panel_letters_x\d+|panel_letter_list")
PANEL_WORD_ONLY = "panel_word"
ASPECT_PREFIXES = ("wide_aspect_", "tall_aspect_")


def bucket_row(reasons: list[str]) -> str:
    has_letter = any(PANEL_LETTER_RE.fullmatch(r) for r in reasons)
    has_word_only = PANEL_WORD_ONLY in reasons
    has_aspect = any(r.startswith(ASPECT_PREFIXES) for r in reasons)
    has_caption = has_letter or has_word_only

    if has_caption and has_aspect:
        return "both_signals"
    if has_letter and not has_word_only:
        return "caption_letter_only"
    if has_word_only and not has_letter:
        return "caption_word_only"  # <-- laterality-risk bucket
    if has_letter and has_word_only:
        return "caption_letter_and_word"
    if has_aspect:
        return "aspect_only"
    return "unflagged"  # shouldn't occur among flagged rows


def main() -> None:
    df = pd.read_parquet(FLAGGED_MANIFEST_PATH)
    flagged = df[df["compound_flag"]].copy()
    print(f"Loaded {len(df)} rows total, {len(flagged)} flagged as compound")

    flagged["review_bucket"] = flagged["compound_reasons"].apply(bucket_row)

    bucket_counts = flagged["review_bucket"].value_counts().to_dict()
    print("\nBucket sizes among flagged rows:")
    for k, v in bucket_counts.items():
        print(f"  {k}: {v}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({k: int(v) for k, v in bucket_counts.items()}, f, indent=2)

    # Stratified sample: up to SAMPLE_PER_BUCKET rows per bucket
    samples = []
    for bucket, group in flagged.groupby("review_bucket"):
        n = min(SAMPLE_PER_BUCKET, len(group))
        samples.append(group.sample(n, random_state=RANDOM_STATE))
    sample_df = pd.concat(samples, ignore_index=True)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    review_rows = []
    for row in sample_df.itertuples(index=False):
        abs_path = LOCAL_DOWNLOAD_ROOT / row.image_path
        thumb_name = f"{row.review_bucket}__{Path(row.image_path).name}"
        try:
            with Image.open(abs_path) as im:
                im.thumbnail((320, 320))
                im.convert("RGB").save(SAMPLE_DIR / thumb_name, "JPEG")
            saved = True
        except OSError:
            saved = False

        review_rows.append(
            {
                "review_bucket": row.review_bucket,
                "image_path": row.image_path,
                "thumbnail": thumb_name if saved else "",
                "compound_reasons": ";".join(row.compound_reasons),
                "caption_text": (row.caption_text or "")[:200],
                "width": row.width,
                "height": row.height,
                "is_compound": "",  # <-- fill in 1 or 0 by hand after viewing thumbnail
                "notes": "",
            }
        )

    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(REVIEW_CSV_PATH, index=False)

    print(f"\nSaved {len(review_df)} thumbnails to: {SAMPLE_DIR}")
    print(f"Wrote review sheet: {REVIEW_CSV_PATH}")
    print(
        "\nNext: open the CSV (e.g. in Google Sheets or Colab's file editor), "
        "open each thumbnail in the file browser, and fill 'is_compound' with "
        "1 (genuinely multi-panel) or 0 (false positive) for all rows. "
        "Pay special attention to 'caption_word_only' -- that's the bucket most "
        "likely to be inflated by laterality language ('left ovary', 'right "
        "lobe') rather than actual panel layout. Once filled in, group by "
        "review_bucket and compute mean(is_compound) per bucket -- that's your "
        "precision per signal, and it tells us which buckets are safe to "
        "auto-exclude vs which need the word regex tightened."
    )


if __name__ == "__main__":
    main()