"""
Source: original Colab notebook, cell index [45]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 5 -- CLIP retrieval baseline (text->panel and panel->panel).

Ground truth caveat (do not skip this in your writeup): the only "ground
truth" available is the automatically-generated Stage-2 alignment
(panel <-> best_match_text). Recall@k here measures whether retrieval
recovers that automatic pairing -- it is NOT validated against human
judgments. Report it as such.

Reuses the Stage-2 embedding cache (no CLIP recompute). Builds FAISS
indices on the TEST split only (standard retrieval-eval practice) and
saves them so retrieval is reproducible without rebuilding.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

TEST_SPLIT_PATH = Path("/content/pipeline_data/splits_final_v1/test.parquet")
IMG_EMB_PATH = Path("/content/pipeline_data/alignment_v1/embed_cache/panel_image_embeds.npy")
IMG_IDS_PATH = Path("/content/pipeline_data/alignment_v1/embed_cache/panel_image_ids.json")
TXT_EMB_PATH = Path("/content/pipeline_data/alignment_v1/embed_cache/text_segment_embeds.npy")
TXT_IDS_PATH = Path("/content/pipeline_data/alignment_v1/embed_cache/text_segment_ids.json")

OUT_DIR = Path("/content/pipeline_data/retrieval_v1")
IMAGE_INDEX_PATH = OUT_DIR / "panel_image.faiss"
RESULTS_PATH = OUT_DIR / "retrieval_results.json"

K_VALUES = (1, 5, 10)


def recall_at_k(ranked_indices: np.ndarray, correct_idx: np.ndarray, k: int) -> float:
    hits = (ranked_indices[:, :k] == correct_idx[:, None]).any(axis=1)
    return float(hits.mean())


def main() -> None:
    if RESULTS_PATH.exists():
        print(f"[SKIP] Stage 5 already done -- {RESULTS_PATH}")
        return

    test_panels = set(pd.read_parquet(TEST_SPLIT_PATH)["panel_id"].astype(str))
    all_panel_ids = json.loads(IMG_IDS_PATH.read_text())
    all_img_embeds = np.load(IMG_EMB_PATH).astype("float32")
    all_seg_records = json.loads(TXT_IDS_PATH.read_text())
    all_txt_embeds = np.load(TXT_EMB_PATH).astype("float32")

    test_mask = np.array([pid in test_panels for pid in all_panel_ids])
    test_panel_ids = [pid for pid, m in zip(all_panel_ids, test_mask) if m]
    test_img_embeds = all_img_embeds[test_mask]
    print(f"Test panels with cached embeddings: {len(test_panel_ids)}")

    # image -> image retrieval index
    dim = test_img_embeds.shape[1]
    img_index = faiss.IndexFlatIP(dim)
    img_index.add(test_img_embeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(img_index, str(IMAGE_INDEX_PATH))

    results = {}

    # --- image -> image: nearest panel from a DIFFERENT figure sharing the
    # same best_match_text is not well-defined as ground truth, so we only
    # report this as an unsupervised sanity check (top-1 self-match check
    # excluded), not a Recall@k metric requiring invented ground truth.
    results["image_to_image"] = {
        "note": "No reliable ground truth for panel-to-panel relevance exists "
                "in the metadata, so Recall@k is not computed for this direction "
                "(would require inventing ground truth, which the brief prohibits). "
                "Index is saved for qualitative nearest-neighbor inspection.",
        "index_path": str(IMAGE_INDEX_PATH),
        "n_indexed": len(test_panel_ids),
    }

    # --- text -> panel retrieval, ground truth = Stage 2's own best_match ---
    align_df = pd.read_parquet("/content/pipeline_data/alignment_v1/alignment_manifest.parquet")
    align_df = align_df[align_df["panel_id"].astype(str).isin(test_panel_ids)]

    # Map each test panel's aligned text back to its embedding row.
    seg_lookup = {}
    for i, rec in enumerate(all_seg_records):
        seg_lookup.setdefault(rec["figure_id"], []).append((i, rec["text"]))

    panel_id_to_row = {pid: i for i, pid in enumerate(test_panel_ids)}
    query_embeds, correct_panel_idx = [], []
    for _, r in align_df.iterrows():
        pid = str(r["panel_id"])
        if pid not in panel_id_to_row:
            continue
        cands = seg_lookup.get(str(r["figure_id"]), [])
        match = next((i for i, t in cands if t == r["best_match_text"]), None)
        if match is None:
            continue
        query_embeds.append(all_txt_embeds[match])
        correct_panel_idx.append(panel_id_to_row[pid])

    if len(query_embeds) == 0:
        results["text_to_image"] = {"note": "No evaluable text->panel pairs on test split."}
    else:
        query_embeds = np.stack(query_embeds).astype("float32")
        correct_panel_idx = np.array(correct_panel_idx)

        sims = query_embeds @ test_img_embeds.T
        ranked = np.argsort(-sims, axis=1)

        recalls = {f"recall@{k}": recall_at_k(ranked, correct_panel_idx, k) for k in K_VALUES}
        results["text_to_image"] = {
            "n_queries": int(len(query_embeds)),
            "ground_truth_source": "Stage 2 automatic CLIP alignment (best_match_text) "
                                    "-- not human-validated, treat as a proxy metric.",
            **recalls,
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n[STAGE 5 COMPLETE]")
    print(json.dumps(results, indent=2))
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()