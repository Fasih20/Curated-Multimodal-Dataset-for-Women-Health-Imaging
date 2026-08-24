from __future__ import annotations

import pandas as pd
from _load_stage_module import load_module

mod = load_module("11_compositional_panel_vqa/02_generate_compositional_questions.py", "stage11_questions")


def frame(fig, modalities, anatomies, split="train"):
    return pd.DataFrame({
        "figure_id": [fig] * len(modalities),
        "panel_id": [f"{fig}_{chr(97+i)}" for i in range(len(modalities))],
        "modality_tag": modalities, "anatomy_tag": anatomies, "split": [split] * len(modalities),
    })


def test_eligible_figure_produces_all_three_types():
    rows = mod.generate_questions_for_figure(frame("f", ["MRI", "MRI", "CT"], ["breast", "breast", "ovary"]))
    assert {r["question_type"] for r in rows} == set(mod.QUESTION_TYPES)
    assert sum(r["question_type"] == "modality_same_different" for r in rows) == 2
    assert sum(r["question_type"] == "modality_count" for r in rows) == 2
    assert next(r for r in rows if r["question_type"] == "anatomy_odd_one_out")["answer"] == "C"


def test_below_eligibility_bar_is_empty():
    assert mod.generate_questions_for_figure(frame("f", ["MRI", None], [None, None])) == []
    assert mod.generate_questions_for_figure(frame("g", ["MRI", None, None], [None, None, None])) == []


def test_three_anatomy_tags_skips_ambiguous_odd_one_out():
    rows = mod.generate_questions_for_figure(frame("f", ["MRI"] * 4, ["breast", "breast", "ovary", "uterus"]))
    assert not any(r["question_type"] == "anatomy_odd_one_out" for r in rows)


def test_only_available_pairing_type_is_emitted():
    same = mod.generate_questions_for_figure(frame("s", ["MRI"] * 3, ["breast"] * 3))
    pairs = [r for r in same if r["question_type"] == "modality_same_different"]
    assert [r["answer"] for r in pairs] == ["same"]
    different = mod.generate_questions_for_figure(frame("d", ["MRI", "CT", "X-ray"], ["breast"] * 3))
    pairs = [r for r in different if r["question_type"] == "modality_same_different"]
    assert [r["answer"] for r in pairs] == ["different"]


def test_split_disagreement_is_rejected():
    data = frame("f", ["MRI"] * 3, ["breast"] * 3)
    data.loc[2, "split"] = "test"
    try:
        mod.generate_questions_for_figure(data)
    except AssertionError as exc:
        assert "split leakage" in str(exc)
    else:
        raise AssertionError("expected split assertion")
