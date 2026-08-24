"""Stage 11.2: generate categorical cross-panel VQA questions."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", "./pipeline_data")).resolve()
OUT_DIR = PIPELINE_ROOT / "compositional_vqa_v1"

MODALITY_KEYWORDS = {
    "ultrasound": ["ultrasound", "sonograph", "doppler"],
    "MRI": ["mri", "magnetic resonance"],
    "CT": [" ct scan", "computed tomography", "ct imag"],
    "X-ray": ["x-ray", "radiograph"],
    "histopathology": ["histopatholog", "biopsy", "h&e stain", "microscop"],
}
ANATOMY_KEYWORDS = {
    "ovary": ["ovary", "ovarian", "adnexal"],
    "breast": ["breast", "mammary"],
    "uterus": ["uterus", "uterine", "endometri"],
    "cervix": ["cervix", "cervical"],
}
QUESTION_TYPES = ("modality_same_different", "anatomy_odd_one_out", "modality_count")
OUTPUT_COLUMNS = ["figure_id", "panel_ids", "question", "question_type", "answer", "answer_space", "split"]


def keyword_lookup(text: str, table: dict[str, list[str]]) -> str | None:
    if not isinstance(text, str):
        return None
    low = text.lower()
    for label, keywords in table.items():
        if any(keyword in low for keyword in keywords):
            return label
    return None


def panel_letter(panel_id: str) -> str:
    """Return the final reading-order label from ``{figure_id}_{letter}``."""
    match = re.search(r"_([A-Za-z]+)$", str(panel_id))
    return match.group(1).upper() if match else str(panel_id)


def _answer_spaces(panels: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "modality_same_different": ["same", "different"],
        "anatomy_odd_one_out": [panel_letter(p) for p in panels.panel_id] + ["none"],
    }


def generate_questions_for_figure(panels: pd.DataFrame) -> list[dict]:
    """Generate deterministic questions for one already-joined figure."""
    panels = panels.copy()
    panels["panel_id"] = panels["panel_id"].astype(str)
    panels["_letter"] = panels["panel_id"].map(panel_letter)
    panels = panels.sort_values("_letter", kind="stable").reset_index(drop=True)
    if len(panels) < 3:
        return []
    tagged = panels["modality_tag"].notna() | panels["anatomy_tag"].notna()
    if int(tagged.sum()) < 2:
        return []
    splits = set(panels["split"].astype(str))
    assert len(splits) == 1, f"split leakage in figure {panels.figure_id.iloc[0]}: {splits}"

    figure_id = str(panels.figure_id.iloc[0])
    panel_ids = panels.panel_id.tolist()
    split = next(iter(splits))
    spaces = _answer_spaces(panels)
    rows: list[dict] = []

    modality_panels = panels[panels.modality_tag.notna()]
    same_pair = different_pair = None
    for (_, a), (_, b) in combinations(modality_panels.iterrows(), 2):
        pair = (a, b)
        if a.modality_tag == b.modality_tag and same_pair is None:
            same_pair = pair
        elif a.modality_tag != b.modality_tag and different_pair is None:
            different_pair = pair
    for answer, pair in (("same", same_pair), ("different", different_pair)):
        if pair is None:
            continue
        a, b = pair
        rows.append({
            "figure_id": figure_id, "panel_ids": panel_ids,
            "question": f"Do panels {a._letter} and {b._letter} show the same imaging modality?",
            "question_type": "modality_same_different", "answer": answer,
            "answer_space": spaces["modality_same_different"], "split": split,
        })

    anatomy = panels[panels.anatomy_tag.notna()]
    counts = Counter(anatomy.anatomy_tag)
    # Exactly two tags, one singleton, and one majority of at least two.
    if len(counts) == 2 and sorted(counts.values())[0] == 1 and max(counts.values()) >= 2:
        odd_tag = next(tag for tag, count in counts.items() if count == 1)
        odd = anatomy[anatomy.anatomy_tag == odd_tag].iloc[0]
        rows.append({
            "figure_id": figure_id, "panel_ids": panel_ids,
            "question": "Which panel shows a different anatomical structure than the others?",
            "question_type": "anatomy_odd_one_out", "answer": odd._letter,
            "answer_space": spaces["anatomy_odd_one_out"], "split": split,
        })

    modalities = sorted(set(modality_panels.modality_tag))
    for modality in modalities:
        count = int((panels.modality_tag == modality).sum())
        rows.append({
            "figure_id": figure_id, "panel_ids": panel_ids,
            "question": f"How many panels show {modality}?",
            "question_type": "modality_count", "answer": str(count),
            "answer_space": [str(i) for i in range(len(panels) + 1)], "split": split,
        })
    return rows


def build_dataset(accepted: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    required = {"figure_id", "panel_id", "split"}
    missing = required - set(accepted.columns)
    if missing:
        raise ValueError(f"accepted dataset missing columns: {sorted(missing)}")
    accepted = accepted.copy(); alignment = alignment.copy()
    accepted["panel_id"] = accepted.panel_id.astype(str); alignment["panel_id"] = alignment.panel_id.astype(str)
    joined = accepted.merge(
        alignment[["panel_id", "assigned_segment_text", "alignment_mode"]],
        on="panel_id", how="left", validate="one_to_one",
    )
    # Shared figure captions do not provide defensible panel-specific tags.
    valid = ~joined["alignment_mode"].isin(["shared_caption", "unmatched", "no_candidate_text"])
    joined["modality_tag"] = joined["assigned_segment_text"].where(valid).map(
        lambda x: keyword_lookup(x, MODALITY_KEYWORDS))
    joined["anatomy_tag"] = joined["assigned_segment_text"].where(valid).map(
        lambda x: keyword_lookup(x, ANATOMY_KEYWORDS))
    rows = []
    for _, group in joined.groupby("figure_id", sort=True):
        rows.extend(generate_questions_for_figure(group))
    dataset = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if len(dataset):
        # Classification heads require one global candidate set per task,
        # not a subtly different space for every figure size.
        all_letters = sorted({panel_letter(pid) for ids in dataset.panel_ids for pid in ids})
        max_panels = max(len(ids) for ids in dataset.panel_ids)
        global_spaces = {
            "modality_same_different": ["same", "different"],
            "anatomy_odd_one_out": all_letters + ["none"],
            "modality_count": [str(i) for i in range(max_panels + 1)],
        }
        dataset["answer_space"] = dataset.question_type.map(global_spaces)
    return dataset


def process_track(track: str) -> pd.DataFrame | None:
    quality_dir = "quality_v1" if track == "clip" else "quality_biomedclip_v1"
    accepted_path = PIPELINE_ROOT / quality_dir / "accepted_dataset.parquet"
    alignment_path = OUT_DIR / f"alignment_constrained_{track}.parquet"
    out_path = OUT_DIR / f"compositional_vqa_dataset_{track}.parquet"
    if out_path.exists():
        print(f"[SKIP] {track}: {out_path}")
        return pd.read_parquet(out_path)
    if not accepted_path.exists() or not alignment_path.exists():
        print(f"[MISSING DATA] {track}: expected {accepted_path} and {alignment_path}")
        return None
    accepted = pd.read_parquet(accepted_path)
    if "split" not in accepted.columns:
        split_dir = PIPELINE_ROOT / ("splits_final_v1" if track == "clip" else "splits_biomedclip_v1")
        split_rows = []
        for split in ("train", "val", "test"):
            path = split_dir / f"{split}.parquet"
            if path.exists():
                part = pd.read_parquet(path)[["panel_id"]].copy(); part["split"] = split; split_rows.append(part)
        if not split_rows:
            raise FileNotFoundError(f"no split parquets found under {split_dir}")
        split_map = pd.concat(split_rows, ignore_index=True)
        if split_map.panel_id.astype(str).duplicated().any():
            raise ValueError(f"duplicate panel IDs across split files in {split_dir}")
        accepted["panel_id"] = accepted.panel_id.astype(str); split_map["panel_id"] = split_map.panel_id.astype(str)
        accepted = accepted.merge(split_map, on="panel_id", how="left", validate="one_to_one")
        if accepted.split.isna().any():
            raise ValueError(f"{int(accepted.split.isna().sum())} accepted panels have no inherited split")
    dataset = build_dataset(accepted, pd.read_parquet(alignment_path))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)
    summary = {
        "track": track, "n_questions": len(dataset),
        "n_figures": int(dataset.figure_id.nunique()) if len(dataset) else 0,
        "question_type_counts": dataset.question_type.value_counts().to_dict() if len(dataset) else {},
        "split_counts": dataset.split.value_counts().to_dict() if len(dataset) else {},
    }
    (OUT_DIR / f"question_generation_{track}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["clip", "biomedclip", "both"], default="both")
    args = parser.parse_args()
    for track in (["clip", "biomedclip"] if args.track == "both" else [args.track]):
        process_track(track)


if __name__ == "__main__":
    main()
