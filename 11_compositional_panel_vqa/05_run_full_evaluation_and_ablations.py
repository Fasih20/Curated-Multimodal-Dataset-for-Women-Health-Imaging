"""Stage 11.5: combine real metric JSON files into tables and a plot."""
from __future__ import annotations
import json, os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", "./pipeline_data")).resolve(); OUT_DIR = PIPELINE_ROOT / "compositional_vqa_v1"
QUESTION_TYPES = ["modality_same_different", "anatomy_odd_one_out", "modality_count"]

def aggregate(out_dir=OUT_DIR):
    rows = []
    for path in sorted(out_dir.glob("baseline_vlm_metrics_*.json")):
        data = json.loads(path.read_text()); track = data["track"]
        for model, metrics in data["models"].items():
            row = {"method": model, "track": track, "overall": metrics["overall"]["accuracy"],
                   "n_evaluated": metrics["overall"]["n_evaluated"], "n_excluded": metrics["overall"]["n_excluded"]}
            row.update({q: metrics["per_question_type"].get(q, {}).get("accuracy") for q in QUESTION_TYPES}); rows.append(row)
    for path in sorted(out_dir.glob("metrics_*.json")):
        data = json.loads(path.read_text()); test = data["test"]
        row = {"method": "set-attention (proposed)" if data["mode"] == "set_attention" else "mean-pool ablation",
               "track": data["track"], "overall": test["overall"], "n_evaluated": test["n"], "n_excluded": 0}
        row.update({q: test["per_question_type"].get(q, {}).get("accuracy") for q in QUESTION_TYPES}); rows.append(row)
    frame = pd.DataFrame(rows, columns=["method", "track", "n_evaluated", "n_excluded", "overall", *QUESTION_TYPES])
    out_dir.mkdir(parents=True, exist_ok=True); frame.to_csv(out_dir / "final_ablation_table.csv", index=False)
    (out_dir / "final_ablation_table.json").write_text(frame.to_json(orient="records", indent=2))
    if len(frame):
        plot_dir = out_dir / "plots"; plot_dir.mkdir(exist_ok=True)
        labels = [f"{r.method}\n({r.track})" for r in frame.itertuples()]
        fig, ax = plt.subplots(figsize=(max(8, len(frame) * 1.25), 5)); ax.bar(labels, frame.overall, color="#4C78A8")
        ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1); ax.set_title("Compositional cross-panel VQA")
        ax.grid(axis="y", alpha=.25); plt.xticks(rotation=25, ha="right"); fig.tight_layout(); fig.savefig(plot_dir / "ablation_comparison.png", dpi=200); plt.close(fig)
    print(frame.to_string(index=False)); return frame
if __name__ == "__main__": aggregate()
