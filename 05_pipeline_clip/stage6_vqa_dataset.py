"""
Source: original Colab notebook, cell index [47]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 6 -- practical VQA dataset, grounded in existing text/metadata.

Every answer traces back to either the panel-specific matched text
(Stage 2) or simple, defensible derived facts (modality keyword lookup,
panel count) -- nothing is hallucinated from pixels. Each row records
its `source` (evidence) so you can audit/cite it later.

Keyword lists are intentionally small and conservative (precision over
coverage) -- extend MODALITY_KEYWORDS / ANATOMY_KEYWORDS as you validate
more, rather than guessing broadly now.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

ACCEPTED_PATH = Path("/content/pipeline_data/quality_v1/accepted_dataset.parquet")
SPLITS_DIR = Path("/content/pipeline_data/splits_final_v1")

OUT_DIR = Path("/content/pipeline_data/vqa_v1")
VQA_PATH = OUT_DIR / "vqa_dataset.parquet"
VALIDATION_SAMPLE_PATH = OUT_DIR / "manual_validation_sample.csv"

SEED = 42
N_VALIDATION_SAMPLE = 60

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


def keyword_lookup(text: str, table: dict[str, list[str]]) -> str | None:
    if not isinstance(text, str):
        return None
    low = text.lower()
    for label, kws in table.items():
        if any(kw in low for kw in kws):
            return label
    return None


def build_questions_for_row(r: pd.Series) -> list[dict]:
    qs = []
    caption = r.get("best_match_text") or ""

    modality = keyword_lookup(caption, MODALITY_KEYWORDS)
    if modality:
        qs.append({
            "question": "What imaging modality is shown in this panel?",
            "answer": modality,
            "question_type": "modality",
            "source": "keyword match on Stage-2 best_match_text",
        })

    anatomy = keyword_lookup(caption, ANATOMY_KEYWORDS)
    if anatomy:
        qs.append({
            "question": "What anatomical structure is depicted?",
            "answer": anatomy,
            "question_type": "anatomy",
            "source": "keyword match on Stage-2 best_match_text",
        })

    if isinstance(caption, str) and caption.strip():
        qs.append({
            "question": "What does the caption say about this panel?",
            "answer": caption.strip(),
            "question_type": "panel_caption",
            "source": "Stage-2 best_match_text (verbatim panel-specific text)",
        })

    if pd.notna(r.get("panels_in_figure")):
        qs.append({
            "question": "How many panels does this figure contain?",
            "answer": str(int(r["panels_in_figure"])),
            "question_type": "panel_count",
            "source": "panel_manifest (detector output count)",
        })

    return qs


def main() -> None:
    if VQA_PATH.exists():
        print(f"[SKIP] Stage 6 already done -- {VQA_PATH}")
        return

    accepted = pd.read_parquet(ACCEPTED_PATH)
    split_map = {}
    for split_name in ("train", "val", "test"):
        p = SPLITS_DIR / f"{split_name}.parquet"
        if p.exists():
            for pid in pd.read_parquet(p)["panel_id"].astype(str):
                split_map[pid] = split_name

    rows = []
    for _, r in accepted.iterrows():
        for q in build_questions_for_row(r):
            rows.append({
                "figure_id": r["figure_id"],
                "panel_id": r["panel_id"],
                "image_path": r["crop_path"],
                "split": split_map.get(str(r["panel_id"]), "unassigned"),
                **q,
            })

    vqa_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vqa_df.to_parquet(VQA_PATH, index=False)

    random.seed(SEED)
    sample_n = min(N_VALIDATION_SAMPLE, len(vqa_df))
    sample = vqa_df.sample(n=sample_n, random_state=SEED)
    sample.to_csv(VALIDATION_SAMPLE_PATH, index=False)

    print("\n[STAGE 6 COMPLETE]")
    print(f"VQA pairs: {len(vqa_df)}")
    print("By question_type:")
    print(vqa_df["question_type"].value_counts())
    print(f"Manual validation sample ({sample_n} rows): {VALIDATION_SAMPLE_PATH}")
    print(f"Output: {VQA_PATH}")


if __name__ == "__main__":
    main()