"""
Source: original Colab notebook, cell index [61]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 5b (ABLATION) -- retrieval baseline using BiomedCLIP embeddings
from Stage 2b, mirroring Stage 5 exactly so the two are directly comparable.

Reads Stage 2b's cache (alignment_biomedclip_v1/), NOT Stage 2's. Writes to
its own retrieval_biomedclip_v1/ dir. Stage 5's outputs are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

TEST_SPLIT_PATH = Path("/content/pipeline_data/splits_biomedclip_v1/test.parquet")  # Stage 4b -- BiomedCLIP's own accepted/test population, not CLIP's
IMG_EMB_PATH = Path("/content/pipeline_data/alignment_biomedclip_v1/embed_cache/panel_image_embeds.npy")
IMG_IDS_PATH = Path("/content/pipeline_data/alignment_biomedclip_v1/embed_cache/panel_image_ids.json")
TXT_EMB_PATH = Path("/content/pipeline_data/alignment_biomedclip_v1/embed_cache/text_segment_embeds.npy")
TXT_IDS_PATH = Path("/content/pipeline_data/alignment_biomedclip_v1/embed_cache/text_segment_ids.json")
ALIGNMENT_MANIFEST_PATH = Path("/content/pipeline_data/alignment_biomedclip_v1/alignment_manifest.parquet")

OUT_DIR = Path("/content/pipeline_data/retrieval_biomedclip_v1")
IMAGE_INDEX_PATH = OUT_DIR / "panel_image.faiss"
RESULTS_PATH = OUT_DIR / "retrieval_results.json"

K_VALUES = (1, 5, 10)


def recall_at_k(ranked_indices: np.ndarray, correct_idx: np.ndarray, k: int) -> float:
    hits = (ranked_indices[:, :k] == correct_idx[:, None]).any(axis=1)
    return float(hits.mean())


def main() -> None:
    if RESULTS_PATH.exists():
        print(f"[SKIP] Stage 5b already done -- {RESULTS_PATH}")
        return

    test_panels = set(pd.read_parquet(TEST_SPLIT_PATH)["panel_id"].astype(str))
    all_panel_ids = json.loads(IMG_IDS_PATH.read_text())
    all_img_embeds = np.load(IMG_EMB_PATH).astype("float32")
    all_seg_records = json.loads(TXT_IDS_PATH.read_text())
    all_txt_embeds = np.load(TXT_EMB_PATH).astype("float32")

    test_mask = np.array([pid in test_panels for pid in all_panel_ids])
    test_panel_ids = [pid for pid, m in zip(all_panel_ids, test_mask) if m]
    test_img_embeds = all_img_embeds[test_mask]
    print(f"Test panels with cached BiomedCLIP embeddings: {len(test_panel_ids)}")

    dim = test_img_embeds.shape[1]
    img_index = faiss.IndexFlatIP(dim)
    img_index.add(test_img_embeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(img_index, str(IMAGE_INDEX_PATH))

    results = {}
    results["image_to_image"] = {
        "note": "Same as Stage 5 -- no reliable panel-to-panel ground truth exists, "
                "index saved for qualitative inspection only.",
        "index_path": str(IMAGE_INDEX_PATH),
        "n_indexed": len(test_panel_ids),
    }

    align_df = pd.read_parquet(ALIGNMENT_MANIFEST_PATH)
    align_df = align_df[align_df["panel_id"].astype(str).isin(test_panel_ids)]

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
            "ground_truth_source": "Stage 2b BiomedCLIP alignment (best_match_text) -- "
                                    "same proxy-metric caveat as Stage 5.",
            **recalls,
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n[STAGE 5b COMPLETE] (BiomedCLIP ablation)")
    print(json.dumps(results, indent=2))

    # Direct comparison against Stage 5, if it's already run.
    baseline_path = Path("/content/pipeline_data/retrieval_v1/retrieval_results.json")
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        print("\n--- Comparison: openai-CLIP (Stage 5) vs BiomedCLIP (Stage 5b) ---")
        for k in K_VALUES:
            b = baseline.get("text_to_image", {}).get(f"recall@{k}")
            n = results.get("text_to_image", {}).get(f"recall@{k}")
            if b is not None and n is not None:
                print(f"  recall@{k}: CLIP={b:.4f}  BiomedCLIP={n:.4f}  "
                      f"({'+' if n >= b else ''}{(n - b):.4f})")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()