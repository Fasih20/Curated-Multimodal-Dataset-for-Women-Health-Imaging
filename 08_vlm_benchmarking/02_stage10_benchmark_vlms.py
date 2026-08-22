"""
Source: original Colab notebook, cell index [84]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 10 -- benchmark multiple VLMs on the curated captioning + VQA
datasets from whichever track you chose after Stage 9.

Set DATASET_TRACK below to "clip" or "biomedclip". Everything else reads
from that track's already-built captioning_v1/vqa_v1 (or *_biomedclip_v1)
outputs -- no upstream stage is touched.

Restartable: each model's predictions are written incrementally to their
own parquet; a model already fully predicted on a task is skipped.

Add/remove models in MODEL_REGISTRY. Each entry is a (loader, caption_fn,
vqa_fn) triple so different model families (BLIP-2, Qwen2-VL, LLaVA, ...)
can have different call conventions without branching logic scattered
through the run loop.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

# -----------------------------------------------------------------------
# Config -- change this one line to switch which curated dataset you use.
# -----------------------------------------------------------------------
DATASET_TRACK = "clip"  # "clip" or "biomedclip"

_SUFFIX = "" if DATASET_TRACK == "clip" else "_biomedclip"
PIPELINE_ROOT = Path("/content/pipeline_data")
CAPTIONING_PATH = PIPELINE_ROOT / f"captioning{_SUFFIX}_v1/captioning_dataset.parquet"
VQA_PATH = PIPELINE_ROOT / f"vqa{_SUFFIX}_v1/vqa_dataset.parquet"

OUT_DIR = PIPELINE_ROOT / f"vlm_benchmark_{DATASET_TRACK}_v1"
CAPTION_PRED_DIR = OUT_DIR / "captioning_predictions"
VQA_PRED_DIR = OUT_DIR / "vqa_predictions"
METRICS_PATH = OUT_DIR / "benchmark_metrics.json"

EVAL_SPLIT = "test"
MAX_CAPTION_SAMPLES = 300   # keep the benchmark affordable on remaining Colab time
MAX_VQA_SAMPLES = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------
# Model adapters -- add new models here without touching the run loop.
# -----------------------------------------------------------------------
def load_blip2():
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to(DEVICE).eval()
    return {"processor": processor, "model": model}


def blip2_caption(ctx, image: Image.Image) -> str:
    inputs = ctx["processor"](images=image, return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=40)
    return ctx["processor"].decode(out[0], skip_special_tokens=True).strip()


def blip2_vqa(ctx, image: Image.Image, question: str) -> str:
    prompt = f"Question: {question} Answer:"
    inputs = ctx["processor"](images=image, text=prompt, return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=20)
    return ctx["processor"].decode(out[0], skip_special_tokens=True).strip()


def load_qwen2_vl():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float16
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return {"processor": processor, "model": model}


def _qwen2_vl_chat(ctx, image: Image.Image, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = ctx["processor"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ctx["processor"](text=[text], images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=40)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return ctx["processor"].batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def clean_vqa_answer(pred: str, ref: str) -> str:
    """Strips conversational filler and checks if the reference answer is hiding in the output."""
    # Lowercase and strip punctuation
    clean_pred = re.sub(r'[^\w\s]', '', pred.lower()).strip()
    clean_ref = re.sub(r'[^\w\s]', '', ref.lower()).strip()

    # If the model said "The modality is mri", catch it and force the exact match
    if clean_ref in clean_pred:
        return clean_ref
    return clean_pred


def qwen2_vl_caption(ctx, image: Image.Image) -> str:
    # Force a clinical tone, ban introductory filler
    prompt = "Provide a concise, one-sentence clinical description of the findings in this medical image panel. Do not use introductory phrases like 'This image shows'."
    return _qwen2_vl_chat(ctx, image, prompt)


def qwen2_vl_vqa(ctx, image: Image.Image, question: str) -> str:
    # FORCE short answers. This is the magic fix for Exact Match.
    strict_prompt = (
        f"Answer the following question with a single word or short phrase. "
        f"Do not write full sentences or conversational filler.\n"
        f"Question: {question}\nAnswer:"
    )
    return _qwen2_vl_chat(ctx, image, strict_prompt)


# MODEL_REGISTRY = {
#     "blip2-opt-2.7b": {"loader": load_blip2, "caption_fn": blip2_caption, "vqa_fn": blip2_vqa},
#     "qwen2-vl-2b": {"loader": load_qwen2_vl, "caption_fn": qwen2_vl_caption, "vqa_fn": qwen2_vl_vqa},
#     # Add more here, e.g. "llava-1.5-7b": {...}
# }
# MODEL_REGISTRY = {
#     "blip2-opt-2.7b": {"loader": load_blip2, "caption_fn": blip2_caption, "vqa_fn": blip2_vqa},
#     "qwen2-vl-2b": {"loader": load_qwen2_vl, "caption_fn": qwen2_vl_caption, "vqa_fn": qwen2_vl_vqa},
#     # Add more here, e.g. "llava-1.5-7b": {...}
# }
# MODELS_TO_RUN = list(MODEL_REGISTRY.keys())  # trim this list to control runtime
MODEL_REGISTRY = {
    "blip2-opt-2.7b": {"loader": load_blip2, "caption_fn": blip2_caption, "vqa_fn": blip2_vqa},
    "qwen2-vl-2b": {"loader": load_qwen2_vl, "caption_fn": qwen2_vl_caption, "vqa_fn": qwen2_vl_vqa},
    "paligemma-3b": {"loader": load_paligemma_3b, "caption_fn": paligemma_caption, "vqa_fn": paligemma_vqa},
}
MODELS_TO_RUN = ["paligemma-3b"]  # just this run -- BLIP-2/Qwen already have saved predictions, no need to redo


# -----------------------------------------------------------------------
# Metrics -- simple, dependency-light, documented as a baseline (not a
# substitute for human eval or CIDEr/SPICE if you have time to add them).
# -----------------------------------------------------------------------
def token_f1(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return 0.0
    common = set(p) & set(r)
    if not common:
        return 0.0
    overlap = sum(min(p.count(w), r.count(w)) for w in common)
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return float(pred.strip().lower() == ref.strip().lower())


def run_captioning(model_name: str, ctx: dict, df: pd.DataFrame) -> pd.DataFrame:
    out_path = CAPTION_PRED_DIR / f"{model_name}.parquet"
    done_ids = set()
    rows = []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        rows = prev.to_dict("records")
        done_ids = set(prev["panel_id"].astype(str))
        print(f"  [resume] {len(done_ids)} captions already done for {model_name}")

    todo = df[~df["panel_id"].astype(str).isin(done_ids)]
    for i, (_, r) in enumerate(todo.iterrows()):
        try:
            image = Image.open(r["image_path"]).convert("RGB")
            pred = ctx["caption_fn"](ctx["ctx"], image)
        except Exception as e:  # noqa: BLE001
            pred = f"[ERROR: {e}]"
        rows.append({"panel_id": r["panel_id"], "figure_id": r["figure_id"],
                      "reference": r["caption"], "prediction": pred})
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            print(f"    {model_name} captioning: {i + 1}/{len(todo)}")

    out_df = pd.DataFrame(rows)
    CAPTION_PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_df


def run_vqa(model_name: str, ctx: dict, df: pd.DataFrame) -> pd.DataFrame:
    out_path = VQA_PRED_DIR / f"{model_name}.parquet"
    done_keys = set()
    rows = []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        rows = prev.to_dict("records")
        done_keys = set(prev["panel_id"].astype(str) + "||" + prev["question"])
        print(f"  [resume] {len(done_keys)} VQA answers already done for {model_name}")

    todo = df[~(df["panel_id"].astype(str) + "||" + df["question"]).isin(done_keys)]
    for i, (_, r) in enumerate(todo.iterrows()):
        try:
            image = Image.open(r["image_path"]).convert("RGB")
            pred = ctx["vqa_fn"](ctx["ctx"], image, r["question"])
        except Exception as e:  # noqa: BLE001
            pred = f"[ERROR: {e}]"
        rows.append({"panel_id": r["panel_id"], "figure_id": r["figure_id"], "question": r["question"],
                      "question_type": r["question_type"], "reference": r["answer"], "prediction": pred})
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            print(f"    {model_name} VQA: {i + 1}/{len(todo)}")

    out_df = pd.DataFrame(rows)
    VQA_PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CAPTION_PRED_DIR.mkdir(parents=True, exist_ok=True)
    VQA_PRED_DIR.mkdir(parents=True, exist_ok=True)

    cap_df = pd.read_parquet(CAPTIONING_PATH)
    cap_df = cap_df[cap_df["split"] == EVAL_SPLIT]
    cap_df = cap_df.sample(n=min(len(cap_df), MAX_CAPTION_SAMPLES), random_state=42).reset_index(drop=True)
    vqa_df = pd.read_parquet(VQA_PATH)
    vqa_df = vqa_df[vqa_df["split"] == EVAL_SPLIT]
    vqa_df = vqa_df.sample(n=min(len(vqa_df), MAX_VQA_SAMPLES), random_state=42).reset_index(drop=True)
    print(f"Benchmarking on track={DATASET_TRACK}: {len(cap_df)} captions, {len(vqa_df)} VQA pairs")

    metrics = {}
    for model_name in MODELS_TO_RUN:
        print(f"\n=== {model_name} ===")
        entry = MODEL_REGISTRY[model_name]
        model_ctx = entry["loader"]()
        ctx = {"ctx": model_ctx, "caption_fn": entry["caption_fn"], "vqa_fn": entry["vqa_fn"]}

        cap_preds = run_captioning(model_name, ctx, cap_df)
        vqa_preds = run_vqa(model_name, ctx, vqa_df)

        n_cap_errors = int(cap_preds["prediction"].str.startswith("[ERROR", na=False).sum())
        n_vqa_errors = int(vqa_preds["prediction"].str.startswith("[ERROR", na=False).sum())
        if n_cap_errors or n_vqa_errors:
            print(f"  [WARN] {model_name}: {n_cap_errors} captioning errors, "
                  f"{n_vqa_errors} VQA errors -- excluded from metrics below, not just scored as wrong.")
        cap_preds = cap_preds[~cap_preds["prediction"].str.startswith("[ERROR", na=False)]
        vqa_preds = vqa_preds[~vqa_preds["prediction"].str.startswith("[ERROR", na=False)]

        # Captioning metrics (keep as is, but expect low F1 due to visual vs clinical gap)
        cap_preds["token_f1"] = cap_preds.apply(lambda r: token_f1(r["prediction"], r["reference"]), axis=1)

        # VQA metrics: Clean the answers first, and DROP panel_count (the VLM can't see the whole figure)
        vqa_eval = vqa_preds[vqa_preds["question_type"] != "panel_count"].copy()
        vqa_eval["clean_prediction"] = vqa_eval.apply(lambda r: clean_vqa_answer(r["prediction"], r["reference"]), axis=1)
        vqa_eval["clean_reference"] = vqa_eval["reference"].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()).strip())

        vqa_eval["exact_match"] = vqa_eval.apply(lambda r: exact_match(r["clean_prediction"], r["clean_reference"]), axis=1)
        vqa_eval["token_f1"] = vqa_eval.apply(lambda r: token_f1(r["clean_prediction"], r["clean_reference"]), axis=1)

        # Overwrite for the summary stats
        vqa_preds = vqa_eval

        metrics[model_name] = {
            "captioning_token_f1_mean": float(cap_preds["token_f1"].mean()) if len(cap_preds) else None,
            "vqa_exact_match_mean": float(vqa_preds["exact_match"].mean()) if len(vqa_preds) else None,
            "vqa_token_f1_mean": float(vqa_preds["token_f1"].mean()) if len(vqa_preds) else None,
            # exact_match is only meaningful for short-answer types (modality/anatomy/
            # panel_count); panel_caption answers are free text -- read token_f1 for
            # that type instead of the EM figure, which will look artificially low.
            "vqa_exact_match_by_type": vqa_preds.groupby("question_type")["exact_match"].mean().to_dict() if len(vqa_preds) else {},
            "vqa_token_f1_by_type": vqa_preds.groupby("question_type")["token_f1"].mean().to_dict() if len(vqa_preds) else {},
            "n_captions": int(len(cap_preds)), "n_vqa": int(len(vqa_preds)),
            "n_captioning_errors_excluded": n_cap_errors, "n_vqa_errors_excluded": n_vqa_errors,
        }

        del model_ctx, ctx
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n[STAGE 10 COMPLETE]")
    print(json.dumps(metrics, indent=2))
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()