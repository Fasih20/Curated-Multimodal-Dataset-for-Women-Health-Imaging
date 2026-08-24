"""Direct multi-panel evaluation of the three requested medical VLMs.

Each test question is shown with every panel crop in reading order. Predictions
are written incrementally; interrupted Colab runs resume by question key.
"""
from __future__ import annotations
import argparse, gc, json, os, re
from pathlib import Path
import pandas as pd
import torch
from PIL import Image

PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", "./pipeline_data")).resolve()
OUT_DIR = PIPELINE_ROOT / "compositional_vqa_v1"
MODELS = {
    "medgemma-4b-it": "google/medgemma-4b-it",
    "lingshu-7b": "lingshu-medical-mllm/Lingshu-7B",
    "medvlm-r1": "JZPeterPan/MedVLM-R1",
}

def parse_candidate(text: str, candidates: list[str]) -> str | None:
    text = str(text).strip(); low = text.lower()
    for marker in ("final answer:", "answer:"):
        if marker in low: low = low.rsplit(marker, 1)[-1].strip()
    normalized = re.sub(r"[^a-z0-9-]+", " ", low).strip()
    exact = {str(c).lower(): str(c) for c in candidates}
    if normalized in exact: return exact[normalized]
    hits = []
    for candidate in sorted(candidates, key=lambda x: -len(str(x))):
        pattern = rf"(?<![a-z0-9]){re.escape(str(candidate).lower())}(?![a-z0-9])"
        found = list(re.finditer(pattern, low))
        if found: hits.append((found[-1].start(), str(candidate)))
    return max(hits)[1] if hits else None

def resolve_crop(stored: str, panel_id: str) -> Path:
    candidates = [Path(stored), PIPELINE_ROOT / "panels_v1/crops" / Path(stored).name,
                  PIPELINE_ROOT / "panels_v1/crops" / f"{panel_id}.png"]
    for path in candidates:
        if path.exists(): return path
    raise FileNotFoundError(f"crop missing for {panel_id}: {candidates}")

def load_model(key: str, token: str | None):
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              BitsAndBytesConfig, Qwen2VLForConditionalGeneration,
                              Qwen2_5_VLForConditionalGeneration)
    model_id = MODELS[key]; processor = AutoProcessor.from_pretrained(model_id, token=token)
    common = {"device_map": "auto", "token": token, "torch_dtype": torch.bfloat16}
    if key == "lingshu-7b":
        common["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **common)
    elif key == "medvlm-r1":
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **common)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **common)
    return model.eval(), processor

def evaluate(track: str, key: str, token: str | None):
    dataset = pd.read_parquet(OUT_DIR / f"compositional_vqa_dataset_{track}.parquet")
    dataset = dataset[dataset.split == "test"].reset_index(drop=True)
    manifest = pd.read_csv(PIPELINE_ROOT / "panels_v1/panel_manifest.csv")
    manifest["panel_id"] = manifest.panel_id.astype(str)
    paths = manifest.set_index("panel_id").crop_path.to_dict()
    pred_dir = OUT_DIR / "medical_vlm_predictions"; pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{key}_{track}.parquet"
    previous = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    done = set(previous.question_key.astype(str)) if len(previous) else set(); rows = previous.to_dict("records") if len(previous) else []
    model, processor = load_model(key, token)
    for index, row in dataset.iterrows():
        qkey = f"{row.figure_id}||{row.question_type}||{row.question}"
        if qkey in done: continue
        images = [Image.open(resolve_crop(paths[str(pid)], str(pid))).convert("RGB") for pid in row.panel_ids]
        choices = [str(x) for x in row.answer_space]
        prompt = (f"Panels are supplied in reading order A, B, C, and so on. {row.question} "
                  f"Answer using exactly one choice from: {', '.join(choices)}. Give only the answer.")
        content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[-1]:]
        raw = processor.decode(generated, skip_special_tokens=True).strip()
        parsed = parse_candidate(raw, choices)
        rows.append({"question_key": qkey, "figure_id": str(row.figure_id), "question_type": row.question_type,
                     "question": row.question, "answer": str(row.answer), "raw_prediction": raw,
                     "prediction": parsed, "correct": parsed == str(row.answer)})
        if len(rows) % 10 == 0: pd.DataFrame(rows).to_parquet(pred_path, index=False)
    predictions = pd.DataFrame(rows); predictions.to_parquet(pred_path, index=False)
    by_type = {q: {"accuracy": float(g.correct.mean()), "n": len(g)} for q, g in predictions.groupby("question_type")}
    return {"model": key, "model_id": MODELS[key], "track": track, "n": len(predictions),
            "accuracy": float(predictions.correct.mean()), "per_question_type": by_type}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--track", choices=["clip", "biomedclip", "both"], default="both")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all"); args = parser.parse_args()
    token = os.environ.get("HF_TOKEN"); tracks = ["clip", "biomedclip"] if args.track == "both" else [args.track]
    models = list(MODELS) if args.model == "all" else [args.model]; metrics = []
    for key in models:
        for track in tracks: metrics.append(evaluate(track, key, token))
        gc.collect(); torch.cuda.empty_cache()
    path = OUT_DIR / "medical_vlm_metrics.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    merged = {(x["model"], x["track"]): x for x in [*existing, *metrics]}
    path.write_text(json.dumps(list(merged.values()), indent=2)); print(json.dumps(metrics, indent=2))
if __name__ == "__main__": main()
