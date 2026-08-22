"""
Source: original Colab notebook, cell index [73]
Auto-extracted -- review before treating as final.
"""

import json, re
from pathlib import Path
import pandas as pd

OUT_DIR = Path("/content/pipeline_data/vlm_benchmark_clip_v1")
cap = pd.read_parquet(OUT_DIR / "captioning_predictions/qwen2-vl-2b.parquet")
vqa = pd.read_parquet(OUT_DIR / "vqa_predictions/qwen2-vl-2b.parquet")

def token_f1(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r: return 0.0
    common = set(p) & set(r)
    if not common: return 0.0
    ov = sum(min(p.count(w), r.count(w)) for w in common)
    pr, rc = ov/len(p), ov/len(r)
    return 2*pr*rc/(pr+rc) if pr+rc else 0.0

def clean(pred, ref):
    cp = re.sub(r'[^\w\s]', '', pred.lower()).strip()
    cr = re.sub(r'[^\w\s]', '', ref.lower()).strip()
    return cr if cr in cp else cp

cap = cap[~cap.prediction.str.startswith("[ERROR", na=False)]
cap["f1"] = [token_f1(p, r) for p, r in zip(cap.prediction, cap.reference)]

vqa = vqa[~vqa.prediction.str.startswith("[ERROR", na=False)]
vqa = vqa[vqa.question_type != "panel_count"].copy()   # structurally unanswerable from a crop
vqa["cp"] = [clean(p, r) for p, r in zip(vqa.prediction, vqa.reference)]
vqa["cr"] = [re.sub(r'[^\w\s]', '', str(r).lower()).strip() for r in vqa.reference]
vqa["em"] = (vqa.cp == vqa.cr).astype(float)
vqa["f1"] = [token_f1(p, r) for p, r in zip(vqa.cp, vqa.cr)]

qwen_metrics = {
    "captioning_token_f1_mean": round(float(cap.f1.mean()), 4),
    "vqa_exact_match_mean": round(float(vqa.em.mean()), 4),
    "vqa_token_f1_mean": round(float(vqa.f1.mean()), 4),
    "vqa_exact_match_by_type": {k: round(v, 4) for k, v in vqa.groupby("question_type").em.mean().items()},
    "vqa_token_f1_by_type": {k: round(v, 4) for k, v in vqa.groupby("question_type").f1.mean().items()},
    "n_captions": int(len(cap)), "n_vqa": int(len(vqa)),
    "note": "rescored offline from saved predictions (verbose answers + substring matching)",
}
(OUT_DIR / "qwen_rescored_metrics.json").write_text(json.dumps(qwen_metrics, indent=2))
print(json.dumps(qwen_metrics, indent=2))