"""
Source: pasted directly by user (Kaggle, 2x T4, 32GB combined) -- not extracted
from the Colab notebook.

Runs the 6 additional VLMs that weren't in the Colab Stage 10 benchmark
(qwen2-vl-2b and blip already covered there -- see
08_vlm_benchmarking/02_stage10_benchmark_vlms.py):
  1. llava-v1.6-34b (4-bit quantized)
  2. qwen2-vl-7b
  3. llava-1.5-7b
  4. instructblip-vicuna-7b
  5. idefics2-8b
  6. llama-3.2-11b-vision
  7. llava-v1.6-13b

(Note: docstring below says "llava-v1.6-34b" in the model lineup comment
but MODEL_REGISTRY as pasted doesn't actually include it -- see note in
README under "Kaggle model list mismatch". Worth double-checking against
your actual Kaggle run before calling this final.)

Uses the same token_f1 / exact_match scoring as the Colab Stage 10 script
so results are directly comparable across all 8 models total.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import BitsAndBytesConfig

# -----------------------------------------------------------------------
# 0. Setup -- run this cell first, alone, before anything else.
# -----------------------------------------------------------------------
# !pip install -q -U transformers accelerate bitsandbytes sentencepiece einops
#
# Point this at wherever Kaggle mounted your pipeline_data dataset, e.g.
# /kaggle/input/pipeline-data/pipeline_data -- adjust to your actual mount path.
PIPELINE_ROOT = Path("/kaggle/working/pipeline_data")
# Kaggle input datasets are READ-ONLY -- all outputs go to /kaggle/working/
OUT_ROOT = Path("/kaggle/working/vlm_benchmark_output")

TRACKS = ["biomedclip", "clip"]
EVAL_SPLIT = "test"
MAX_CAPTION_SAMPLES = 300
MAX_VQA_SAMPLES = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"GPUs visible: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} -- {torch.cuda.get_device_name(i)}, "
          f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


# -----------------------------------------------------------------------
# 1. Model loaders + call adapters -- one block per model family.
# -----------------------------------------------------------------------

# --- Qwen2-VL-7B -- unquantized, sharded across both T4s ---
def load_qwen2_vl_7b():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "10GiB", 1: "10GiB"},  # below model size -> forces real sharding across both GPUs
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return {"processor": processor, "model": model}


def _qwen2_vl_chat(ctx, image, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = ctx["processor"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ctx["processor"](text=[text], images=[image], return_tensors="pt").to(ctx["model"].device)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=40)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return ctx["processor"].batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def qwen2_vl_7b_caption(ctx, image) -> str:
    return _qwen2_vl_chat(ctx, image, "Provide a concise, one-sentence clinical description of the "
                                       "findings in this medical image panel. Do not use introductory "
                                       "phrases like 'This image shows'.")


def qwen2_vl_7b_vqa(ctx, image, question: str) -> str:
    return _qwen2_vl_chat(ctx, image, f"Answer the following question with a single word or short "
                                       f"phrase. Do not write full sentences.\nQuestion: {question}\nAnswer:")


# --- LLaVA-1.5-7B -- unquantized, sharded ---
def load_llava_15_7b():
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"},
    ).eval()
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    return {"processor": processor, "model": model}


def _llava_15_chat(ctx, image, prompt: str) -> str:
    full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
    inputs = ctx["processor"](text=full_prompt, images=image, return_tensors="pt").to(ctx["model"].device)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=40)
    text = ctx["processor"].decode(out[0], skip_special_tokens=True)
    return text.split("ASSISTANT:")[-1].strip()


def llava_15_7b_caption(ctx, image) -> str:
    return _llava_15_chat(ctx, image, "Describe this medical image panel in one sentence.")


def llava_15_7b_vqa(ctx, image, question: str) -> str:
    return _llava_15_chat(ctx, image, f"Answer with a single word or short phrase. {question}")


# --- InstructBLIP-Vicuna-7B -- unquantized, sharded ---
def load_instructblip_7b():
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-vicuna-7b", torch_dtype=torch.float16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"},
    ).eval()
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    return {"processor": processor, "model": model}


def instructblip_7b_caption(ctx, image) -> str:
    prompt = "Describe this medical image panel in one sentence."
    inputs = ctx["processor"](images=image, text=prompt, return_tensors="pt").to(ctx["model"].device, torch.float16)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=40)
    return ctx["processor"].batch_decode(out, skip_special_tokens=True)[0].strip()


def instructblip_7b_vqa(ctx, image, question: str) -> str:
    prompt = f"Answer the following question with a single word or short phrase. Question: {question} Answer:"
    inputs = ctx["processor"](images=image, text=prompt, return_tensors="pt").to(ctx["model"].device, torch.float16)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=20)
    return ctx["processor"].batch_decode(out, skip_special_tokens=True)[0].strip()


# --- Idefics2-8B -- unquantized, sharded ---
def load_idefics2_8b():
    from transformers import Idefics2ForConditionalGeneration, AutoProcessor
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b", torch_dtype=torch.float16, device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
    return {"processor": processor, "model": model}


def _idefics2_chat(ctx, image, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = ctx["processor"].apply_chat_template(messages, add_generation_prompt=True)
    inputs = ctx["processor"](text=text, images=[image], return_tensors="pt").to(ctx["model"].device)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=40)
    decoded = ctx["processor"].batch_decode(out, skip_special_tokens=True)[0]
    return decoded.split("Assistant:")[-1].strip()


def idefics2_8b_caption(ctx, image) -> str:
    return _idefics2_chat(ctx, image, "Describe this medical image panel in one sentence.")


def idefics2_8b_vqa(ctx, image, question: str) -> str:
    return _idefics2_chat(ctx, image, f"Answer with a single word or short phrase. {question}")


# --- Llama-3.2-11B-Vision -- unquantized, sharded ---
def load_llama3_2_11b_vision():
    from transformers import MllamaForConditionalGeneration, AutoProcessor
    model = MllamaForConditionalGeneration.from_pretrained(
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "12GiB", 1: "12GiB"}
    ).eval()
    processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    return {"processor": processor, "model": model}


def _llama3_chat(ctx, image, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = ctx["processor"].apply_chat_template(messages, add_generation_prompt=True)
    inputs = ctx["processor"](image, text, return_tensors="pt").to(ctx["model"].device)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=40)
    decoded = ctx["processor"].decode(out[0], skip_special_tokens=True)
    return decoded.split("assistant\n")[-1].strip()


def llama3_11b_caption(ctx, image) -> str:
    return _llama3_chat(ctx, image, "Provide a concise, one-sentence clinical description of the findings in this medical image panel.")


def llama3_11b_vqa(ctx, image, question: str) -> str:
    return _llama3_chat(ctx, image, f"Answer with a single word or short phrase. {question}")


# --- LLaVA-NeXT 13B -- unquantized fp16, sharded ---
def load_llava_next_13b():
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-vicuna-13b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: "14GiB", 1: "14GiB"}
    ).eval()
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
    return {"processor": processor, "model": model}


def _llava_next_13_chat(ctx, image, prompt: str) -> str:
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = ctx["processor"].apply_chat_template(conversation, add_generation_prompt=True)
    inputs = ctx["processor"](images=image, text=text, return_tensors="pt").to(ctx["model"].device)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=40)
    decoded = ctx["processor"].decode(out[0], skip_special_tokens=True)
    return decoded.split("assistant")[-1].strip(" :\n")


def llava_next_13b_caption(ctx, image) -> str:
    return _llava_next_13_chat(ctx, image, "Describe this medical image panel in one sentence.")


def llava_next_13b_vqa(ctx, image, question: str) -> str:
    return _llava_next_13_chat(ctx, image, f"Answer with a single word or short phrase. {question}")


# -----------------------------------------------------------------------
# Registry -- ordered biggest to smallest, as requested.
# -----------------------------------------------------------------------
MODEL_REGISTRY = {
    "qwen2-vl-7b": {
        "loader": load_qwen2_vl_7b, "caption_fn": qwen2_vl_7b_caption, "vqa_fn": qwen2_vl_7b_vqa,
        "note": "7B, unquantized fp16, sharded via device_map=auto",
    },
    "llava-1.5-7b": {
        "loader": load_llava_15_7b, "caption_fn": llava_15_7b_caption, "vqa_fn": llava_15_7b_vqa,
        "note": "7B, unquantized fp16, sharded",
    },
    "instructblip-vicuna-7b": {
        "loader": load_instructblip_7b, "caption_fn": instructblip_7b_caption, "vqa_fn": instructblip_7b_vqa,
        "note": "7B, unquantized fp16, sharded",
    },
    "idefics2-8b": {
        "loader": load_idefics2_8b, "caption_fn": idefics2_8b_caption, "vqa_fn": idefics2_8b_vqa,
        "note": "8B, unquantized fp16, sharded",
    },
    "llama-3.2-11b": {
        "loader": load_llama3_2_11b_vision, "caption_fn": llama3_11b_caption, "vqa_fn": llama3_11b_vqa,
        "note": "11B, unquantized fp16, sharded",
    },
    "llava-v1.6-13b": {
        "loader": load_llava_next_13b, "caption_fn": llava_next_13b_caption, "vqa_fn": llava_next_13b_vqa,
        "note": "13B, unquantized fp16, sharded",
    },
}

MODELS_TO_RUN = list(MODEL_REGISTRY.keys())  # trim this list if a run times out


# -----------------------------------------------------------------------
# Metrics (identical to Colab Stage 10 -- kept consistent for comparability)
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


def clean_vqa_answer(pred: str, ref: str) -> str:
    clean_pred = re.sub(r'[^\w\s]', '', pred.lower()).strip()
    clean_ref = re.sub(r'[^\w\s]', '', ref.lower()).strip()
    return clean_ref if clean_ref in clean_pred else clean_pred


def exact_match(pred: str, ref: str) -> float:
    return float(pred.strip().lower() == ref.strip().lower())


def run_captioning(caption_pred_dir: Path, model_name: str, ctx: dict, df: pd.DataFrame) -> pd.DataFrame:
    caption_pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = caption_pred_dir / f"{model_name}.parquet"
    done_ids, rows = set(), []
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
    out_df.to_parquet(out_path, index=False)
    return out_df


def run_vqa(vqa_pred_dir: Path, model_name: str, ctx: dict, df: pd.DataFrame) -> pd.DataFrame:
    vqa_pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = vqa_pred_dir / f"{model_name}.parquet"
    done_keys, rows = set(), []
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
    out_df.to_parquet(out_path, index=False)
    return out_df


def score_model(cap_df: pd.DataFrame, vqa_df: pd.DataFrame) -> dict:
    n_cap_errors = int(cap_df["prediction"].str.startswith("[ERROR", na=False).sum())
    n_vqa_errors = int(vqa_df["prediction"].str.startswith("[ERROR", na=False).sum())
    cap_df = cap_df[~cap_df["prediction"].str.startswith("[ERROR", na=False)].copy()
    vqa_df = vqa_df[~vqa_df["prediction"].str.startswith("[ERROR", na=False)].copy()

    cap_df["token_f1"] = cap_df.apply(lambda r: token_f1(r["prediction"], r["reference"]), axis=1)

    vqa_eval = vqa_df[vqa_df["question_type"] != "panel_count"].copy()
    vqa_eval["clean_prediction"] = vqa_eval.apply(lambda r: clean_vqa_answer(r["prediction"], r["reference"]), axis=1)
    vqa_eval["clean_reference"] = vqa_eval["reference"].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()).strip())
    vqa_eval["exact_match"] = vqa_eval.apply(lambda r: exact_match(r["clean_prediction"], r["clean_reference"]), axis=1)
    vqa_eval["token_f1"] = vqa_eval.apply(lambda r: token_f1(r["clean_prediction"], r["clean_reference"]), axis=1)

    return {
        "captioning_token_f1_mean": float(cap_df["token_f1"].mean()) if len(cap_df) else None,
        "vqa_exact_match_mean": float(vqa_eval["exact_match"].mean()) if len(vqa_eval) else None,
        "vqa_token_f1_mean": float(vqa_eval["token_f1"].mean()) if len(vqa_eval) else None,
        "vqa_exact_match_by_type": vqa_eval.groupby("question_type")["exact_match"].mean().to_dict() if len(vqa_eval) else {},
        "vqa_token_f1_by_type": vqa_eval.groupby("question_type")["token_f1"].mean().to_dict() if len(vqa_eval) else {},
        "n_captions": int(len(cap_df)), "n_vqa": int(len(vqa_eval)),
        "n_captioning_errors_excluded": n_cap_errors, "n_vqa_errors_excluded": n_vqa_errors,
    }


# -----------------------------------------------------------------------
# Main -- loops over BOTH tracks for EVERY model.
# -----------------------------------------------------------------------
def main() -> None:
    all_metrics = {}

    for model_name in MODELS_TO_RUN:
        print(f"\n{'=' * 60}\n=== {model_name} ({MODEL_REGISTRY[model_name]['note']}) ===\n{'=' * 60}")
        entry = MODEL_REGISTRY[model_name]
        model_ctx = entry["loader"]()
        ctx = {"ctx": model_ctx, "caption_fn": entry["caption_fn"], "vqa_fn": entry["vqa_fn"]}

        for track in TRACKS:
            suffix = "" if track == "clip" else "_biomedclip"
            captioning_path = PIPELINE_ROOT / f"captioning{suffix}_v1/captioning_dataset.parquet"
            vqa_path = PIPELINE_ROOT / f"vqa{suffix}_v1/vqa_dataset.parquet"
            out_dir = OUT_ROOT / f"vlm_benchmark_{track}_v1"
            caption_pred_dir = out_dir / "captioning_predictions"
            vqa_pred_dir = out_dir / "vqa_predictions"
            metrics_path = out_dir / "benchmark_metrics.json"

            print(f"\n--- track={track} ---")
            cap_df = pd.read_parquet(captioning_path)
            cap_df["image_path"] = cap_df["image_path"].str.replace(
                "/content/pipeline_data", str(PIPELINE_ROOT), regex=False
            )
            cap_df = cap_df[cap_df["split"] == EVAL_SPLIT]
            cap_df = cap_df.sample(n=min(len(cap_df), MAX_CAPTION_SAMPLES), random_state=42).reset_index(drop=True)

            vqa_df = pd.read_parquet(vqa_path)
            vqa_df["image_path"] = vqa_df["image_path"].str.replace(
                "/content/pipeline_data", str(PIPELINE_ROOT), regex=False
            )
            vqa_df = vqa_df[vqa_df["split"] == EVAL_SPLIT]
            vqa_df = vqa_df.sample(n=min(len(vqa_df), MAX_VQA_SAMPLES), random_state=42).reset_index(drop=True)
            print(f"  {len(cap_df)} captions, {len(vqa_df)} VQA pairs")

            cap_preds = run_captioning(caption_pred_dir, model_name, ctx, cap_df)
            vqa_preds = run_vqa(vqa_pred_dir, model_name, ctx, vqa_df)
            model_track_metrics = score_model(cap_preds, vqa_preds)

            out_dir.mkdir(parents=True, exist_ok=True)
            existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            existing[model_name] = model_track_metrics
            metrics_path.write_text(json.dumps(existing, indent=2, default=str))

            all_metrics.setdefault(track, {})[model_name] = model_track_metrics
            print(json.dumps(model_track_metrics, indent=2))

        del model_ctx, ctx
        gc.collect()
        torch.cuda.empty_cache()

    combined_path = OUT_ROOT / "combined_all_tracks_all_models.json"
    combined_path.write_text(json.dumps(all_metrics, indent=2, default=str))
    print(f"\n[ALL MODELS x ALL TRACKS COMPLETE]\nCombined summary: {combined_path}")


if __name__ == "__main__":
    main()
