"""
Source: original Colab notebook, cell index [75]
Auto-extracted -- review before treating as final.
"""

import re
import pandas as pd
from pathlib import Path

BENCH = Path("/content/pipeline_data/vlm_benchmark_biomedclip_v1")

def norm(s): return re.sub(r'[^\w\s]', '', str(s).lower()).strip()
def clean(pred, ref):
    cp, cr = norm(pred), norm(ref)
    return cr if cr and cr in cp else cp
def token_f1(p, r):
    p, r = p.lower().split(), r.lower().split()
    if not p or not r: return 0.0
    common = set(p) & set(r)
    if not common: return 0.0
    ov = sum(min(p.count(w), r.count(w)) for w in common)
    pr, rc = ov/len(p), ov/len(r)
    return 2*pr*rc/(pr+rc) if pr+rc else 0.0

rows = []
for model in ["qwen2-vl-2b", "blip2-opt-2.7b"]:
    cap = pd.read_parquet(BENCH / f"captioning_predictions/{model}.parquet")
    cap = cap[~cap.prediction.str.startswith("[ERROR", na=False)]
    cap_f1 = cap.apply(lambda r: token_f1(r.prediction, r.reference), axis=1).mean()

    vqa = pd.read_parquet(BENCH / f"vqa_predictions/{model}.parquet")
    vqa = vqa[~vqa.prediction.str.startswith("[ERROR", na=False)]
    vqa = vqa[vqa.question_type != "panel_count"].copy()   # unanswerable from a single crop
    vqa["cp"] = [clean(p, r) for p, r in zip(vqa.prediction, vqa.reference)]
    vqa["cr"] = [norm(r) for r in vqa.reference]
    vqa["em"] = (vqa.cp == vqa.cr).astype(float)
    vqa["f1"] = [token_f1(p, r) for p, r in zip(vqa.cp, vqa.cr)]

    rows.append({"model": model, "caption_F1": round(cap_f1, 3),
                 "VQA_EM": round(vqa.em.mean(), 3), "VQA_F1": round(vqa.f1.mean(), 3),
                 "modality_EM": round(vqa[vqa.question_type == "modality"].em.mean(), 3),
                 "anatomy_EM": round(vqa[vqa.question_type == "anatomy"].em.mean(), 3),
                 "n_vqa": int(len(vqa))})

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv(BENCH / "model_comparison_table.csv", index=False)