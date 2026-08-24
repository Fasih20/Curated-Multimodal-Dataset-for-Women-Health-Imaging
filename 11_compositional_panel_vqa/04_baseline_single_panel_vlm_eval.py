"""Stage 11.4: derive cross-panel answers from saved single-panel VLM predictions."""
from __future__ import annotations
import argparse, importlib.util, json, os, re
from collections import Counter
from pathlib import Path
import pandas as pd

PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", "./pipeline_data")).resolve()
OUT_DIR = PIPELINE_ROOT / "compositional_vqa_v1"

# Kept identical to Stage 6/11.2; this script is standalone when executed by run_pipeline.
MODALITY_KEYWORDS = {"ultrasound": ["ultrasound", "sonograph", "doppler"], "MRI": ["mri", "magnetic resonance"],
 "CT": [" ct scan", "computed tomography", "ct imag"], "X-ray": ["x-ray", "radiograph"],
 "histopathology": ["histopatholog", "biopsy", "h&e stain", "microscop"]}
ANATOMY_KEYWORDS = {"ovary": ["ovary", "ovarian", "adnexal"], "breast": ["breast", "mammary"],
 "uterus": ["uterus", "uterine", "endometri"], "cervix": ["cervix", "cervical"]}

def keyword_lookup(text, table):
    if not isinstance(text, str): return None
    low = text.lower()
    for label, keywords in table.items():
        if any(k in low for k in keywords): return label
    return None

def letter(pid):
    match = re.search(r"_([A-Za-z]+)$", str(pid)); return match.group(1).upper() if match else str(pid)

def build_tag_maps(predictions):
    maps = {"modality": {}, "anatomy": {}}
    for qtype, table in (("modality", MODALITY_KEYWORDS), ("anatomy", ANATOMY_KEYWORDS)):
        for _, row in predictions[predictions.question_type == qtype].iterrows():
            tag = keyword_lookup(row.prediction, table)
            if tag is not None: maps[qtype][str(row.panel_id)] = tag
    return maps

def derive_answer(row, maps):
    panel_ids = [str(p) for p in row.panel_ids]; qtype = row.question_type
    if qtype == "modality_same_different":
        match = re.search(r"panels\s+(\w+)\s+and\s+(\w+)", row.question, re.I)
        if not match: return None
        by_letter = {letter(p): p for p in panel_ids}; needed = [by_letter.get(x.upper()) for x in match.groups()]
        if None in needed or any(p not in maps["modality"] for p in needed): return None
        return "same" if maps["modality"][needed[0]] == maps["modality"][needed[1]] else "different"
    tag_type = "anatomy" if qtype == "anatomy_odd_one_out" else "modality"
    if any(p not in maps[tag_type] for p in panel_ids): return None
    tags = {p: maps[tag_type][p] for p in panel_ids}
    if qtype == "modality_count":
        match = re.search(r"show\s+(.+?)\?*$", row.question, re.I)
        return str(sum(tag == match.group(1) for tag in tags.values())) if match else None
    counts = Counter(tags.values())
    if len(counts) == 1: return "none"
    if len(counts) == 2 and sorted(counts.values())[0] == 1 and max(counts.values()) >= 2:
        odd_tag = next(k for k, v in counts.items() if v == 1)
        return letter(next(p for p, tag in tags.items() if tag == odd_tag))
    return None

def prediction_dirs(track):
    names = [f"vlm_benchmark_{track}_v1/vqa_predictions"]
    if track == "clip": names.append("vlm_benchmark_v1/vqa_predictions")
    return [PIPELINE_ROOT / name for name in names if (PIPELINE_ROOT / name).exists()]

def evaluate(track):
    dataset = pd.read_parquet(OUT_DIR / f"compositional_vqa_dataset_{track}.parquet")
    files = {}; [files.setdefault(p.stem, p) for directory in prediction_dirs(track) for p in directory.glob("*.parquet")]
    results = {}
    for model, path in sorted(files.items()):
        maps = build_tag_maps(pd.read_parquet(path)); records = []
        for _, row in dataset.iterrows(): records.append((row.question_type, str(row.answer), derive_answer(row, maps)))
        per_type = {}; evaluated = [(t, truth, pred) for t, truth, pred in records if pred is not None]
        for qtype in sorted(dataset.question_type.unique()):
            subset = [(truth, pred) for t, truth, pred in records if t == qtype and pred is not None]
            total = int((dataset.question_type == qtype).sum()); correct = sum(a == b for a, b in subset)
            guesses = Counter(pred for _, pred in subset)
            per_type[qtype] = {"accuracy": correct / len(subset) if subset else None, "n_evaluated": len(subset),
                               "n_excluded": total - len(subset), "prediction_distribution": dict(guesses)}
        results[model] = {"overall": {"accuracy": sum(a == b for _, a, b in evaluated) / len(evaluated) if evaluated else None,
                                      "n_evaluated": len(evaluated), "n_excluded": len(records) - len(evaluated)},
                          "per_question_type": per_type}
    output = {"track": track, "models": results}
    (OUT_DIR / f"baseline_vlm_metrics_{track}.json").write_text(json.dumps(output, indent=2)); print(json.dumps(output, indent=2))
    return output

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--track", choices=["clip", "biomedclip", "both"], default="both"); args = parser.parse_args()
    for track in (["clip", "biomedclip"] if args.track == "both" else [args.track]): evaluate(track)
if __name__ == "__main__": main()
