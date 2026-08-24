"""Synthetic-data tests for 11_compositional_panel_vqa/01_fix_panel_caption_alignment.py.

No real pipeline data is needed or used here -- these construct small
hand-built embedding matrices to check the assignment logic itself is
correct, independent of whether real pipeline_data exists on this machine.
"""
from __future__ import annotations

import numpy as np
import pytest

from _load_stage_module import load_module

mod = load_module(
    "11_compositional_panel_vqa/01_fix_panel_caption_alignment.py",
    "stage11_align_fix",
)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_enough_segments_gives_fully_distinct_assignment():
    # 3 panels, 3 segments, each panel most similar to a DIFFERENT segment
    # -- Hungarian assignment should recover the obvious 1:1 matching.
    panel_vecs = np.stack([unit(np.array([1.0, 0.0, 0.0])),
                            unit(np.array([0.0, 1.0, 0.0])),
                            unit(np.array([0.0, 0.0, 1.0]))])
    seg_vecs = np.stack([unit(np.array([1.0, 0.1, 0.0])),
                          unit(np.array([0.0, 1.0, 0.1])),
                          unit(np.array([0.1, 0.0, 1.0]))])
    seg_texts = ["seg_A", "seg_B", "seg_C"]
    panel_ids = ["p0", "p1", "p2"]

    results = mod.solve_figure_assignment(panel_ids, panel_vecs, seg_texts, seg_vecs)

    assert {r["alignment_mode"] for r in results} == {"assigned_distinct"}
    assigned_texts = {r["assigned_segment_text"] for r in results}
    assert assigned_texts == {"seg_A", "seg_B", "seg_C"}, (
        "all three panels should get DISTINCT segments -- this is exactly the "
        "collapse bug being fixed (see figures/5588608__4246_demo.png)"
    )
    by_panel = {r["panel_id"]: r["assigned_segment_text"] for r in results}
    assert by_panel["p0"] == "seg_A"
    assert by_panel["p1"] == "seg_B"
    assert by_panel["p2"] == "seg_C"


def test_independent_argmax_would_have_collapsed_but_assignment_does_not():
    # Two panels that are BOTH individually most similar to the same
    # segment -- naive per-panel argmax collapses them onto it (the bug).
    # A correct one-to-one assignment must not.
    panel_vecs = np.stack([unit(np.array([1.0, 0.0])),
                            unit(np.array([0.9, 0.1]))])
    seg_vecs = np.stack([unit(np.array([1.0, 0.0])),
                          unit(np.array([0.3, 0.95]))])
    seg_texts = ["popular_seg", "other_seg"]
    panel_ids = ["p0", "p1"]

    # Confirm the premise: independent argmax really would collapse both
    # panels onto "popular_seg".
    sims = panel_vecs @ seg_vecs.T
    naive_choices = [seg_texts[i] for i in np.argmax(sims, axis=1)]
    assert naive_choices == ["popular_seg", "popular_seg"]

    results = mod.solve_figure_assignment(panel_ids, panel_vecs, seg_texts, seg_vecs)
    assigned_texts = [r["assigned_segment_text"] for r in results]
    assert set(assigned_texts) == {"popular_seg", "other_seg"}, (
        "constrained assignment must break the tie, unlike independent argmax"
    )


def test_single_shared_segment_is_flagged_not_hidden():
    panel_vecs = np.stack([unit(np.array([1.0, 0.0])), unit(np.array([0.9, 0.1])),
                            unit(np.array([0.8, 0.2]))])
    seg_vecs = np.stack([unit(np.array([1.0, 0.0]))])
    seg_texts = ["only_caption"]
    panel_ids = ["p0", "p1", "p2"]

    results = mod.solve_figure_assignment(panel_ids, panel_vecs, seg_texts, seg_vecs)

    assert len(results) == 3
    assert all(r["alignment_mode"] == "shared_caption" for r in results)
    assert all(r["assigned_segment_text"] == "only_caption" for r in results)


def test_no_candidate_text_at_all():
    panel_vecs = np.stack([unit(np.array([1.0, 0.0]))])
    seg_vecs = np.zeros((0, 2))
    seg_texts: list[str] = []
    panel_ids = ["p0"]

    results = mod.solve_figure_assignment(panel_ids, panel_vecs, seg_texts, seg_vecs)

    assert results[0]["alignment_mode"] == "no_candidate_text"
    assert results[0]["assigned_segment_text"] is None
    assert results[0]["assigned_similarity"] is None


def test_insufficient_segments_flagged_as_shared_not_silently_repeated():
    # 3 panels, only 2 segments -- one panel must share by necessity.
    panel_vecs = np.stack([unit(np.array([1.0, 0.0, 0.0])),
                            unit(np.array([0.0, 1.0, 0.0])),
                            unit(np.array([0.0, 0.9, 0.1]))])  # most similar to seg_B too
    seg_vecs = np.stack([unit(np.array([1.0, 0.0, 0.0])),
                          unit(np.array([0.0, 1.0, 0.0]))])
    seg_texts = ["seg_A", "seg_B"]
    panel_ids = ["p0", "p1", "p2"]

    results = mod.solve_figure_assignment(panel_ids, panel_vecs, seg_texts, seg_vecs)
    modes = [r["alignment_mode"] for r in results]

    assert len(results) == 3
    assert modes.count("assigned_distinct") == 2
    assert modes.count("shared_segment_insufficient") == 1
    # the sharing panel is p2 (its embedding is closest to an already-taken segment)
    sharing = [r for r in results if r["alignment_mode"] == "shared_segment_insufficient"]
    assert sharing[0]["panel_id"] == "p2"


def test_low_similarity_is_marked_unmatched_not_forced():
    # Similarity below MIN_SIMILARITY must be rejected even if it's the
    # "best available" match -- accepting it would silently manufacture a
    # confident-looking label out of noise.
    orthogonal = unit(np.array([1.0, 0.0]))
    almost_orthogonal_seg = unit(np.array([0.05, 0.999]))  # tiny positive similarity
    panel_vecs = np.stack([orthogonal])
    seg_vecs = np.stack([almost_orthogonal_seg])
    seg_texts = ["weak_match"]
    panel_ids = ["p0"]

    assert float(panel_vecs[0] @ seg_vecs[0]) < mod.MIN_SIMILARITY

    results = mod.solve_figure_assignment(panel_ids, panel_vecs, seg_texts, seg_vecs)
    assert results[0]["alignment_mode"] == "unmatched"
    assert results[0]["assigned_segment_text"] is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
