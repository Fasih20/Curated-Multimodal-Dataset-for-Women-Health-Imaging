"""
Source: original Colab notebook, cell index [72]
Auto-extracted -- review before treating as final.
"""

import re
import pandas as pd

P = "/content/pipeline_data/vlm_benchmark_clip_v1/vqa_predictions/qwen2-vl-2b.parquet"
df = pd.read_parquet(P)

# --- 1) Eyeball raw predictions per type (the other AI's diagnostic) ---
for t in ["modality", "anatomy", "panel_count", "panel_caption"]:
    print(f"\n===== {t} =====")
    print(df[df.question_type == t][["reference", "prediction"]].head(5).to_string())

# --- 2) Rescore with normalization ---
def norm(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
df["np"], df["nr"] = df["prediction"].map(norm), df["reference"].map(norm)
df["em_strict"] = df["np"] == df["nr"]
df["em_lenient"] = [(r == p) or (len(r) >= 2 and r in p) for p, r in zip(df["np"], df["nr"])]

def tok_f1(p, r):
    pt, rt = p.split(), r.split()
    if not pt or not rt: return 0.0
    common = set(pt) & set(rt)
    if not common: return 0.0
    ov = sum(min(pt.count(w), rt.count(w)) for w in common)
    prec, rec = ov / len(pt), ov / len(rt)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0
df["f1_norm"] = [tok_f1(p, r) for p, r in zip(df["np"], df["nr"])]

print("\n=== rescored by type ===")
print(df.groupby("question_type")[["em_strict", "em_lenient", "f1_norm"]].mean().round(3))