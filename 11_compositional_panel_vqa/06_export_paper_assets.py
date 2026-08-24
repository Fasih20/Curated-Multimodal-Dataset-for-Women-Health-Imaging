"""Stage 11.6: export generated results/stats as booktabs LaTeX tables."""
from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd
PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", "./pipeline_data")).resolve(); OUT_DIR = PIPELINE_ROOT / "compositional_vqa_v1"
REPO_ROOT = Path(__file__).resolve().parent.parent; TABLE_DIR = REPO_ROOT / "paper/tables"
def pct(value): return "--" if pd.isna(value) else f"{100*float(value):.1f}"
def export():
    frame = pd.read_csv(OUT_DIR / "final_ablation_table.csv"); TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Method & Backbone & $n$ & Overall & Same/diff. & Odd-one-out & Count \\", r"\midrule"]
    for r in frame.itertuples(): lines.append(f"{r.method} & {r.track} & {r.n_evaluated} & {pct(r.overall)} & {pct(r.modality_same_different)} & {pct(r.anatomy_odd_one_out)} & {pct(r.modality_count)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]; (TABLE_DIR / "results_table.tex").write_text("\n".join(lines))
    summaries = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("question_generation_*.summary.json"))]
    stats = [r"\begin{tabular}{lrr}", r"\toprule", r"Backbone & Figures & Questions \\", r"\midrule"]
    stats += [f"{s['track']} & {s['n_figures']} & {s['n_questions']} " + r"\\" for s in summaries]
    stats += [r"\bottomrule", r"\end{tabular}"]; (TABLE_DIR / "dataset_stats_table.tex").write_text("\n".join(stats))
if __name__ == "__main__": export()
