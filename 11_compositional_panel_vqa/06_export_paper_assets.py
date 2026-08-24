"""Stage 11.6: export generated results/stats as booktabs LaTeX tables."""
from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd
PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", "./pipeline_data")).resolve(); OUT_DIR = PIPELINE_ROOT / "compositional_vqa_v1"
REPO_ROOT = Path(__file__).resolve().parent.parent; TABLE_DIR = REPO_ROOT / "paper/tables"
MEDVLM_RESULTS_DIR = Path(os.environ.get("MEDVLM_RESULTS_DIR", REPO_ROOT / "womens_health_medvlm_results")).resolve()
def pct(value): return "--" if pd.isna(value) else f"{100*float(value):.1f}"
def export_medical_vlms():
    pred_dir = MEDVLM_RESULTS_DIR / "medical_vlm_predictions"
    if not pred_dir.exists(): return
    names = {"medgemma-4b-it": "MedGemma 4B", "lingshu-7b": "Lingshu 7B", "medvlm-r1": "MedVLM-R1"}
    rows = []
    for path in sorted(pred_dir.glob("*.parquet")):
        stem = path.stem
        track = "biomedclip" if stem.endswith("_biomedclip") else "clip"
        key = stem[:-(len(track) + 1)]
        frame = pd.read_parquet(path)
        strict = frame.correct.fillna(False).astype(bool)
        parsed = frame.prediction.notna()
        by_type = frame.assign(_correct=strict).groupby("question_type")._correct.mean().to_dict()
        rows.append({"model": names[key], "track": track, "n": len(frame), "overall": strict.mean(),
                     "same": by_type.get("modality_same_different"), "count": by_type.get("modality_count"),
                     "parse": parsed.mean(), "normalized": False})
        if key == "lingshu-7b":
            prediction = frame.prediction.copy()
            raw = frame.raw_prediction.astype(str).str.strip().str.lower().str.replace(r"[^a-z]+", "", regex=True)
            binary = frame.question_type.eq("modality_same_different") & prediction.isna()
            prediction.loc[binary & raw.eq("yes")] = "same"
            prediction.loc[binary & raw.eq("no")] = "different"
            correct = prediction.astype("string").eq(frame.answer.astype("string")).fillna(False)
            by_type = frame.assign(_correct=correct).groupby("question_type")._correct.mean().to_dict()
            rows.append({"model": "Lingshu 7B (Y/N norm.)", "track": track, "n": len(frame),
                         "overall": correct.mean(), "same": by_type.get("modality_same_different"),
                         "count": by_type.get("modality_count"), "parse": prediction.notna().mean(),
                         "normalized": True})
    frame = pd.DataFrame(rows).sort_values(["track", "model"])
    frame.to_csv(MEDVLM_RESULTS_DIR / "medical_vlm_summary_verified.csv", index=False)
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Model & Track & $n$ & Overall & Same/diff. & Count & Parsed \\", r"\midrule"]
    for row in frame.itertuples():
        lines.append(f"{row.model} & {row.track} & {row.n} & {pct(row.overall)} & {pct(row.same)} & {pct(row.count)} & {pct(row.parse)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "medical_vlm_results_table.tex").write_text("\n".join(lines))
def export():
    frame = pd.read_csv(OUT_DIR / "final_ablation_table.csv"); TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Method & Backbone & $n$ & Overall & Same/diff. & Odd-one-out & Count \\", r"\midrule"]
    for r in frame.itertuples(): lines.append(f"{r.method} & {r.track} & {r.n_evaluated} & {pct(r.overall)} & {pct(r.modality_same_different)} & {pct(r.anatomy_odd_one_out)} & {pct(r.modality_count)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]; (TABLE_DIR / "results_table.tex").write_text("\n".join(lines))
    summaries = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("question_generation_*.summary.json"))]
    stats = [r"\begin{tabular}{lrr}", r"\toprule", r"Backbone & Figures & Questions \\", r"\midrule"]
    stats += [f"{s['track']} & {s['n_figures']} & {s['n_questions']} " + r"\\" for s in summaries]
    stats += [r"\bottomrule", r"\end{tabular}"]; (TABLE_DIR / "dataset_stats_table.tex").write_text("\n".join(stats))
    export_medical_vlms()
if __name__ == "__main__": export()
