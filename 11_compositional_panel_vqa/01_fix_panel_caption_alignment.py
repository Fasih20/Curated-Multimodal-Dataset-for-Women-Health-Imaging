"""
Stage 11.1 -- constrained panel<->caption assignment (additive fix on top of
Stage 2 / 2b).

Problem this fixes (see docs/superpowers/specs/2026-08-24-compositional-panel-vqa-design.md
section 2): Stage 2/2b assigns each panel to its best-matching caption
segment via INDEPENDENT per-panel argmax. Nothing stops two different
panels in the same figure from being assigned the identical segment --
when a caption has fewer than 2 panel-letter markers, every panel in that
figure gets the same whole-caption text as its "best match" (see
figures/5588608__4246_demo.png for a real example: all 6 panels get
identical text).

Fix: within each figure, solve a one-to-one assignment (Hungarian
algorithm) between panels and candidate caption segments, maximizing
total similarity, instead of letting panels compete via independent
argmax. This is purely additive -- it writes NEW columns
(assigned_segment_text, assigned_similarity, alignment_mode) into a copy
of the alignment manifest. Stage 2/2b's own best_match_text/best_similarity
columns are untouched, so the existing captioning/VQA/retrieval pipeline
keeps working exactly as before.

Restartable: skips a track if its output file already exists.

Usage:
    python 01_fix_panel_caption_alignment.py --track clip
    python 01_fix_panel_caption_alignment.py --track biomedclip
    python 01_fix_panel_caption_alignment.py --track both   # default
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# -----------------------------------------------------------------------
# Config -- point PIPELINE_ROOT at wherever your pipeline_data lives.
# Override with the PIPELINE_ROOT environment variable if it's not at the
# default relative location.
# -----------------------------------------------------------------------
import os

PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", "./pipeline_data")).resolve()

OUT_DIR = PIPELINE_ROOT / "compositional_vqa_v1"

# Same threshold as Stage 3's quality analysis (MIN_SIMILARITY) -- kept
# consistent rather than inventing a new one.
MIN_SIMILARITY = 0.18

TRACK_PATHS = {
    "clip": {
        "panel_manifest": PIPELINE_ROOT / "panels_v1/panel_manifest.csv",
        "cache_dir": PIPELINE_ROOT / "alignment_v1/embed_cache",
        "out_path": OUT_DIR / "alignment_constrained_clip.parquet",
    },
    "biomedclip": {
        "panel_manifest": PIPELINE_ROOT / "panels_v1/panel_manifest.csv",
        "cache_dir": PIPELINE_ROOT / "alignment_biomedclip_v1/embed_cache",
        "out_path": OUT_DIR / "alignment_constrained_biomedclip.parquet",
    },
}


def load_embeddings(cache_dir: Path):
    img_embeds = np.load(cache_dir / "panel_image_embeds.npy")
    panel_ids = json.loads((cache_dir / "panel_image_ids.json").read_text())
    txt_embeds = np.load(cache_dir / "text_segment_embeds.npy")
    seg_records = json.loads((cache_dir / "text_segment_ids.json").read_text())
    return img_embeds, panel_ids, txt_embeds, seg_records


def group_segments_by_figure(seg_records: list[dict]) -> dict[str, list[int]]:
    """figure_id -> ordered list of indices into txt_embeds, in the order
    they were written by Stage 2/2b (same order split_caption_into_segments
    produced them, i.e. reading order when letter markers exist)."""
    fig_to_seg_idx: dict[str, list[int]] = {}
    for i, rec in enumerate(seg_records):
        fig_to_seg_idx.setdefault(str(rec["figure_id"]), []).append(i)
    return fig_to_seg_idx


def solve_figure_assignment(
    panel_ids: list[str],
    panel_vecs: np.ndarray,
    seg_texts: list[str],
    seg_vecs: np.ndarray,
) -> list[dict]:
    """Solve one figure's panel<->segment assignment.

    Returns one dict per panel: {panel_id, assigned_segment_text,
    assigned_similarity, alignment_mode}.
    """
    n_panels = len(panel_ids)
    n_segs = len(seg_texts)

    if n_segs < 2:
        # Only one candidate segment for the whole figure -- there is no
        # assignment problem to solve, every panel necessarily gets the
        # same text. Not a bug, just a figure whose caption never had
        # per-panel structure. Flag explicitly rather than pretend it's a
        # real match. Falls through to the shared similarity-floor check
        # below like every other branch -- a weak single-segment match
        # must still be rejectable, not silently exempted from the floor.
        if n_segs == 1:
            sim = panel_vecs @ seg_vecs[0]
            results = [
                {
                    "panel_id": pid,
                    "assigned_segment_text": seg_texts[0],
                    "assigned_similarity": float(sim[i]),
                    "alignment_mode": "shared_caption",
                }
                for i, pid in enumerate(panel_ids)
            ]
        else:
            results = [
                {
                    "panel_id": pid,
                    "assigned_segment_text": None,
                    "assigned_similarity": None,
                    "alignment_mode": "no_candidate_text",
                }
                for pid in panel_ids
            ]
        return _apply_similarity_floor(results)

    sim_matrix = panel_vecs @ seg_vecs.T  # [n_panels, n_segs], cosine (already normalized)

    results = [None] * n_panels

    if n_segs >= n_panels:
        # Enough distinct segments for every panel -- solve directly.
        row_idx, col_idx = linear_sum_assignment(-sim_matrix)
        for r, c in zip(row_idx, col_idx):
            results[r] = {
                "panel_id": panel_ids[r],
                "assigned_segment_text": seg_texts[c],
                "assigned_similarity": float(sim_matrix[r, c]),
                "alignment_mode": "assigned_distinct",
            }
    else:
        # Fewer segments than panels -- assign the n_segs segments to their
        # best distinct panels first (transpose so segments, the smaller
        # side, get a clean one-to-one match), then remaining panels must
        # share a segment by necessity. Logged as such, not hidden.
        col_idx, row_idx = linear_sum_assignment(-sim_matrix.T)  # segments -> panels
        assigned_panel_rows = set()
        for seg_i, panel_i in zip(col_idx, row_idx):
            results[panel_i] = {
                "panel_id": panel_ids[panel_i],
                "assigned_segment_text": seg_texts[seg_i],
                "assigned_similarity": float(sim_matrix[panel_i, seg_i]),
                "alignment_mode": "assigned_distinct",
            }
            assigned_panel_rows.add(panel_i)

        for panel_i in range(n_panels):
            if panel_i in assigned_panel_rows:
                continue
            best_seg = int(np.argmax(sim_matrix[panel_i]))
            results[panel_i] = {
                "panel_id": panel_ids[panel_i],
                "assigned_segment_text": seg_texts[best_seg],
                "assigned_similarity": float(sim_matrix[panel_i, best_seg]),
                "alignment_mode": "shared_segment_insufficient",
            }

    return _apply_similarity_floor(results)


def _apply_similarity_floor(results: list[dict]) -> list[dict]:
    """Reject any assignment below MIN_SIMILARITY, regardless of which
    branch of solve_figure_assignment produced it -- a weak match must not
    be accepted just because it was the only/best candidate available."""
    for r in results:
        if r["assigned_similarity"] is not None and r["assigned_similarity"] < MIN_SIMILARITY:
            r["alignment_mode"] = "unmatched"
            r["assigned_segment_text"] = None
    return results


def process_track(track: str) -> pd.DataFrame | None:
    paths = TRACK_PATHS[track]
    if paths["out_path"].exists():
        print(f"[SKIP] {track}: already done -- {paths['out_path']}")
        return pd.read_parquet(paths["out_path"])

    if not paths["panel_manifest"].exists() or not paths["cache_dir"].exists():
        print(f"[MISSING DATA] {track}: expected {paths['panel_manifest']} and "
              f"{paths['cache_dir']} -- run the main pipeline (Stage 1 + Stage 2"
              f"{'b' if track == 'biomedclip' else ''}) first. Skipping.")
        return None

    panel_df = pd.read_csv(paths["panel_manifest"])
    img_embeds, panel_ids_cached, txt_embeds, seg_records = load_embeddings(paths["cache_dir"])
    panel_id_to_row = {pid: i for i, pid in enumerate(panel_ids_cached)}
    fig_to_seg_idx = group_segments_by_figure(seg_records)

    panel_to_fig = dict(zip(panel_df["panel_id"].astype(str), panel_df["figure_id"].astype(str)))
    fig_to_panels: dict[str, list[str]] = {}
    for pid, fig_id in panel_to_fig.items():
        if pid in panel_id_to_row:  # only panels that actually got embedded
            fig_to_panels.setdefault(fig_id, []).append(pid)

    all_results = []
    n_figs_distinct = n_figs_shared_caption = n_figs_partial_share = n_figs_no_text = 0

    for fig_id, panel_ids in fig_to_panels.items():
        panel_vecs = np.stack([img_embeds[panel_id_to_row[pid]] for pid in panel_ids])
        seg_idx = fig_to_seg_idx.get(fig_id, [])
        seg_texts = [seg_records[i]["text"] for i in seg_idx]
        seg_vecs = txt_embeds[seg_idx] if seg_idx else np.zeros((0, txt_embeds.shape[1]))

        fig_results = solve_figure_assignment(panel_ids, panel_vecs, seg_texts, seg_vecs)
        for r in fig_results:
            r["figure_id"] = fig_id
        all_results.extend(fig_results)

        modes = {r["alignment_mode"] for r in fig_results}
        if modes == {"assigned_distinct"}:
            n_figs_distinct += 1
        elif "shared_caption" in modes:
            n_figs_shared_caption += 1
        elif "shared_segment_insufficient" in modes:
            n_figs_partial_share += 1
        elif "no_candidate_text" in modes:
            n_figs_no_text += 1

    out_df = pd.DataFrame(all_results)[
        ["figure_id", "panel_id", "assigned_segment_text", "assigned_similarity", "alignment_mode"]
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(paths["out_path"], index=False)

    summary = {
        "track": track,
        "n_panels": int(len(out_df)),
        "n_figures": len(fig_to_panels),
        "figures_fully_distinct_assignment": n_figs_distinct,
        "figures_shared_caption_no_split_possible": n_figs_shared_caption,
        "figures_partial_share_insufficient_segments": n_figs_partial_share,
        "figures_no_candidate_text": n_figs_no_text,
        "alignment_mode_counts": out_df["alignment_mode"].value_counts().to_dict(),
        "min_similarity_floor": MIN_SIMILARITY,
    }
    with open(OUT_DIR / f"alignment_constrained_{track}.summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[{track}] constrained alignment complete")
    print(json.dumps(summary, indent=2, default=str))
    print(f"Output: {paths['out_path']}")
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["clip", "biomedclip", "both"], default="both")
    args = parser.parse_args()

    tracks = ["clip", "biomedclip"] if args.track == "both" else [args.track]
    for track in tracks:
        process_track(track)


if __name__ == "__main__":
    main()
