"""
Source: original Colab notebook, cell index [19]
Auto-extracted -- review before treating as final.
"""

"""
Colab: apply tiered compound-image policy using validated precision results.

Manual review of 200 stratified samples (40/bucket) gave:

    bucket                     precision   n
    caption_word_only          57.5%       40   <- laterality confound, confirmed
    aspect_only                82.5%       40
    caption_letter_only        92.5%       40
    both_signals                97.5%      40
    caption_letter_and_word   100.0%       40
    overall                     86.0%      200

All 17 false positives in caption_word_only were anatomical laterality
("right ovary", "left ventricle", "right adnexal") -- not panel structure.
That confirms the fix: standalone left/right/upper/lower must NOT count as
a compound signal on their own. They only count if paired with a panel
LETTER marker (e.g. "(A)", "figure 2b") in the same caption, which is
what caption_letter_and_word (100% precision) already captures.

This script:
  1. Recomputes compound_flag using the fixed rule set (drops
     word-only-without-letter as a standalone trigger).
  2. Assigns each row a confidence tier based on which signal(s) fired,
     using the *measured* precision as the tier's expected reliability
     (not a guess).
  3. Applies a tiered policy:
       - high_confidence  (both_signals, caption_letter_only,
         caption_letter_and_word; measured 92.5-100%)
             -> auto-mark compound=True, route to panel-split/exclude
                from single-image tasks
       - medium_confidence (aspect_only; measured 82.5%)
             -> auto-mark compound=True but flagged 'lower_confidence'
                for a lighter manual spot-check later, since aspect
                ratio alone still errs meaningfully
       - low_confidence / rejected (bare word-only, no letter marker)
             -> auto-mark compound=False (folded back into normal
                single-image pool) -- these are mostly laterality
                language, matching what we measured
  4. Writes the updated manifest + a short markdown report with the
     precision table, suitable for pasting into your methods section
     or supervisor update.

Nothing is cropped/split here -- this only sets the compound_flag /
compound_tier columns used downstream by the alignment step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

IN_PATH = Path(
    "/content/pipeline_data/compound_v1/relevant_manifest_with_compound_flags_v1.parquet"
)
OUT_PATH = Path(
    "/content/pipeline_data/compound_v2/relevant_manifest_with_compound_flags_v2.parquet"
)
REPORT_PATH = Path("/content/pipeline_data/compound_v2/compound_policy_report.md")

# Measured precision from the 200-image manual review (fill in if you
# re-run the review later with a different/larger sample).
MEASURED_PRECISION = {
    "caption_word_only": 0.575,
    "aspect_only": 0.825,
    "caption_letter_only": 0.925,
    "both_signals": 0.975,
    "caption_letter_and_word": 1.000,
}
N_REVIEWED = {k: 40 for k in MEASURED_PRECISION}

PANEL_LETTER_RE = re.compile(r"panel_letters_x\d+|panel_letter_list")
PANEL_WORD_ONLY = "panel_word"
ASPECT_PREFIXES = ("wide_aspect_", "tall_aspect_")


def signal_bucket(reasons: list[str]) -> str:
    reasons = list(reasons)
    has_letter = any(PANEL_LETTER_RE.fullmatch(r) for r in reasons)
    has_word = PANEL_WORD_ONLY in reasons
    has_aspect = any(r.startswith(ASPECT_PREFIXES) for r in reasons)

    if has_letter and has_word and has_aspect:
        return "both_signals"  # letter/word caption cue + aspect
    if has_letter and has_aspect:
        return "both_signals"
    if has_word and has_aspect and not has_letter:
        return "both_signals"  # aspect + word co-occur -> treat as both_signals tier
    if has_letter and has_word:
        return "caption_letter_and_word"
    if has_letter:
        return "caption_letter_only"
    if has_word:
        return "caption_word_only"
    if has_aspect:
        return "aspect_only"
    return "unflagged"


TIER_MAP = {
    "both_signals": "high_confidence",
    "caption_letter_and_word": "high_confidence",
    "caption_letter_only": "high_confidence",
    "aspect_only": "medium_confidence",
    "caption_word_only": "rejected",  # <-- the fix
    "unflagged": "not_flagged",
}

FINAL_COMPOUND = {
    "high_confidence": True,
    "medium_confidence": True,
    "rejected": False,  # folded back into single-image pool
    "not_flagged": False,
}


def main() -> None:
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df)} rows")

    df["signal_bucket"] = df["compound_reasons"].apply(
        lambda r: signal_bucket(list(r)) if r is not None and len(r) > 0 else "unflagged"
    )
    df["compound_tier"] = df["signal_bucket"].map(TIER_MAP)
    df["compound_flag_v2"] = df["compound_tier"].map(FINAL_COMPOUND)
    df["compound_tier_precision_est"] = df["signal_bucket"].map(
        lambda b: MEASURED_PRECISION.get(b)
    )

    before = int(df["compound_flag"].sum())
    after = int(df["compound_flag_v2"].sum())
    reclassified_to_false = int(
        ((df["compound_flag"]) & (~df["compound_flag_v2"])).sum()
    )

    print(f"\ncompound_flag (v1, old heuristic):  {before} / {len(df)} flagged")
    print(f"compound_flag_v2 (tiered, fixed):    {after} / {len(df)} flagged")
    print(f"Rows moved from flagged -> unflagged: {reclassified_to_false}")
    print("\nTier breakdown:")
    print(df["compound_tier"].value_counts())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    # ---- markdown report for methods section / supervisor update ----
    lines = [
        "# Compound-image detection: precision validation & tiered policy",
        "",
        "## Manual precision check (200 stratified samples, 40/bucket)",
        "",
        "| Signal bucket | Precision | N reviewed |",
        "|---|---|---|",
    ]
    for b, p in sorted(MEASURED_PRECISION.items(), key=lambda x: x[1]):
        lines.append(f"| {b} | {p*100:.1f}% | {N_REVIEWED[b]} |")
    lines += [
        "",
        "**Finding:** standalone panel-words (left/right/upper/lower) without an "
        "accompanying panel-letter marker had only 57.5% precision. Manual "
        "inspection showed all false positives were anatomical laterality "
        "language (e.g. 'right ovary', 'left ventricle', 'right adnexal') rather "
        "than references to multi-panel layout -- an expected confound in a "
        "medical-imaging caption corpus.",
        "",
        "**Fix:** `caption_word_only` (bare word, no letter marker) is no longer "
        "treated as sufficient evidence of a compound figure on its own and is "
        "excluded from the flagged set. All other signal combinations are kept, "
        "graded into confidence tiers matching their measured precision.",
        "",
        "## Policy applied",
        "",
        "| Tier | Buckets | Action |",
        "|---|---|---|",
        "| high_confidence | both_signals, caption_letter_only, "
        "caption_letter_and_word | flag compound=True, route to panel "
        "split/alignment |",
        "| medium_confidence | aspect_only | flag compound=True, but tagged for "
        "lighter spot-check before use in VQA/captioning ground truth |",
        "| rejected | caption_word_only | compound=False, returned to "
        "single-image pool |",
        "",
        f"## Result: {before} -> {after} rows flagged as compound "
        f"({reclassified_to_false} reclassified back to single-image)",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()