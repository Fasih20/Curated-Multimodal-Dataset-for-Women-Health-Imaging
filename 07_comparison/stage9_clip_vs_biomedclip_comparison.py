"""
Source: original Colab notebook, cell index [69]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 9 -- head-to-head comparison, CLIP track (v1) vs BiomedCLIP
track (biomedclip_v1). Run after both Stage 8 and Stage 8b.

Writes a plain comparison table + a short markdown report. Does NOT pick
a winner automatically -- prints the numbers so you decide, since "better"
depends on what you weight (retrieval recall vs. accepted-rate vs. how
much of the corpus survives quality filtering).
"""
from __future__ import annotations
import json
from pathlib import Path

PIPELINE_ROOT = Path("/content/pipeline_data")
CLIP_SUMMARY = PIPELINE_ROOT / "final_summary_v1/dataset_summary.json"
BIOMED_SUMMARY = PIPELINE_ROOT / "final_summary_biomedclip_v1/dataset_summary.json"

OUT_DIR = PIPELINE_ROOT / "comparison_v1"
REPORT_PATH = OUT_DIR / "clip_vs_biomedclip.md"
JSON_PATH = OUT_DIR / "clip_vs_biomedclip.json"


def g(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default


def main() -> None:
    clip = json.loads(CLIP_SUMMARY.read_text())
    biomed = json.loads(BIOMED_SUMMARY.read_text())

    rows = [
        ("total panels", g(clip, "quality", "total_panels"), g(biomed, "quality", "total_panels")),
        ("accepted", g(clip, "quality", "accepted"), g(biomed, "quality", "accepted")),
        ("flagged", g(clip, "quality", "flagged"), g(biomed, "quality", "flagged")),
        ("accept rate", f"{g(clip,'quality','accepted',default=0)/max(g(clip,'quality','total_panels',default=1),1):.1%}",
                         f"{g(biomed,'quality','accepted',default=0)/max(g(biomed,'quality','total_panels',default=1),1):.1%}"),
        ("mean best_similarity", round(g(biomed, "quality", "mean_best_similarity", default=0), 4)
                                  if g(biomed, "quality", "mean_best_similarity") else None, None),
        ("test panels (retrieval)", g(clip, "retrieval", "text_to_image", "n_queries"),
                                     g(biomed, "retrieval", "text_to_image", "n_queries")),
        ("recall@1", g(clip, "retrieval", "text_to_image", "recall@1"),
                      g(biomed, "retrieval", "text_to_image", "recall@1")),
        ("recall@5", g(clip, "retrieval", "text_to_image", "recall@5"),
                      g(biomed, "retrieval", "text_to_image", "recall@5")),
        ("recall@10", g(clip, "retrieval", "text_to_image", "recall@10"),
                       g(biomed, "retrieval", "text_to_image", "recall@10")),
        ("VQA pairs", g(clip, "vqa_pairs"), g(biomed, "vqa_pairs")),
        ("caption pairs", g(clip, "captioning_pairs"), g(biomed, "captioning_pairs")),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump({"clip": clip, "biomedclip": biomed, "rows": rows}, f, indent=2, default=str)

    lines = ["# CLIP vs BiomedCLIP -- comparison", "",
             "IMPORTANT: recall@k in both tracks is measured against each model's own",
             "automatic alignment as ground truth (a proxy metric, not human-validated) --",
             "see each track's Stage 5 notes. Compare the *relative* gap, not the absolute numbers.",
             "", "| Metric | CLIP (openai) | BiomedCLIP |", "|---|---|---|"]
    for name, a, b in rows:
        lines.append(f"| {name} | {a} | {b} |")
    REPORT_PATH.write_text("\n".join(lines))

    print("\n[STAGE 9 COMPLETE]")
    print("\n".join(lines))
    print(f"\nOutput: {REPORT_PATH}")
    print("\nTo proceed: pick a track, then point Stage 10 (VLM benchmark) at "
          "either captioning_v1/vqa_v1 (CLIP) or captioning_biomedclip_v1/vqa_biomedclip_v1 (BiomedCLIP).")


if __name__ == "__main__":
    main()