"""
Source: original Colab notebook, cell index [55]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 2b (ABLATION) -- same alignment logic as Stage 2, but using
BiomedCLIP instead of generic openai/clip-vit-base-patch32.

BiomedCLIP is NOT a standard HF CLIPModel -- it ships in open_clip format
(PubMedBERT text tower + ViT-B/16 image tower), so it needs open_clip's
loader, not transformers.CLIPModel. Embedding dim also differs from the
Stage 2 cache (512-d openai CLIP vs BiomedCLIP's own dim), so this writes
to a completely separate cache -- nothing from Stage 2 is touched or reused.

Run this AFTER Stage 1 (needs panel_manifest.csv) and independently of
Stage 2 -- they don't depend on each other.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

WORKLIST_PATH = Path(
    "/content/pipeline_data/compound_worklist_v1/compound_figures_manifest.parquet"
)
PANEL_MANIFEST_PATH = Path("/content/pipeline_data/panels_v1/panel_manifest.csv")

# Separate output tree -- Stage 2's alignment_v1/ is never read or written here.
OUT_DIR = Path("/content/pipeline_data/alignment_biomedclip_v1")
CACHE_DIR = OUT_DIR / "embed_cache"
IMG_EMB_PATH = CACHE_DIR / "panel_image_embeds.npy"
IMG_IDS_PATH = CACHE_DIR / "panel_image_ids.json"
TXT_EMB_PATH = CACHE_DIR / "text_segment_embeds.npy"
TXT_IDS_PATH = CACHE_DIR / "text_segment_ids.json"
ALIGNMENT_MANIFEST_PATH = OUT_DIR / "alignment_manifest.parquet"

BIOMEDCLIP_HUB_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LETTER_MARKER_RE = re.compile(r"\(([A-Za-z])\)|\b(?:fig(?:ure)?\.?\s*\d*)([a-hA-H])\b")


def split_caption_into_segments(caption: str) -> list[str]:
    if not isinstance(caption, str) or not caption.strip():
        return [""]
    matches = list(LETTER_MARKER_RE.finditer(caption))
    if len(matches) < 2:
        return [caption.strip()]
    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(caption)
        seg = caption[start:end].strip()
        if seg:
            segments.append(seg)
    return segments if segments else [caption.strip()]


def load_biomedclip():
    """open_clip loader -- BiomedCLIP is not a transformers.CLIPModel."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_HUB_ID)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_HUB_ID)
    model = model.to(DEVICE).eval()
    return model, preprocess, tokenizer


def embed_images(model, preprocess, panel_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if IMG_EMB_PATH.exists() and IMG_IDS_PATH.exists():
        print("[CACHE HIT] BiomedCLIP panel image embeddings")
        return np.load(IMG_EMB_PATH), json.loads(IMG_IDS_PATH.read_text())

    ids, embeds = [], []
    paths = panel_df["crop_path"].tolist()
    panel_ids = panel_df["panel_id"].tolist()
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i + BATCH_SIZE]
        tensors = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu().numpy())
        ids.extend(panel_ids[i:i + BATCH_SIZE])
        if i % (BATCH_SIZE * 20) == 0:
            print(f"  embedded {i + len(batch_paths)}/{len(paths)} panel images (BiomedCLIP)")

    all_embeds = np.concatenate(embeds, axis=0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(IMG_EMB_PATH, all_embeds)
    IMG_IDS_PATH.write_text(json.dumps(ids))
    return all_embeds, ids


def embed_texts(model, tokenizer, seg_records: list[dict]) -> tuple[np.ndarray, list[dict]]:
    if TXT_EMB_PATH.exists() and TXT_IDS_PATH.exists():
        print("[CACHE HIT] BiomedCLIP text segment embeddings")
        return np.load(TXT_EMB_PATH), json.loads(TXT_IDS_PATH.read_text())

    embeds = []
    texts = [r["text"] if r["text"].strip() else " " for r in seg_records]
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        tokens = tokenizer(batch, context_length=256).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu().numpy())
        if i % (BATCH_SIZE * 20) == 0:
            print(f"  embedded {i + len(batch)}/{len(texts)} text segments (BiomedCLIP)")

    all_embeds = np.concatenate(embeds, axis=0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TXT_EMB_PATH, all_embeds)
    TXT_IDS_PATH.write_text(json.dumps(seg_records))
    return all_embeds, seg_records


def main() -> None:
    if ALIGNMENT_MANIFEST_PATH.exists():
        df = pd.read_parquet(ALIGNMENT_MANIFEST_PATH)
        print(f"[SKIP] Stage 2b already done -- {len(df)} aligned panels at {ALIGNMENT_MANIFEST_PATH}")
        return

    panel_df = pd.read_csv(PANEL_MANIFEST_PATH)
    worklist = pd.read_parquet(WORKLIST_PATH)[["figure_id", "caption_text"]]
    print(f"Panels: {len(panel_df)} | Figures with captions: {len(worklist)}")

    seg_records = []
    fig_to_seg_range: dict[str, tuple[int, int]] = {}
    for _, row in worklist.iterrows():
        fig_id = str(row["figure_id"])
        segs = split_caption_into_segments(row["caption_text"])
        start = len(seg_records)
        for s in segs:
            seg_records.append({"figure_id": fig_id, "text": s})
        fig_to_seg_range[fig_id] = (start, len(seg_records))

    model, preprocess, tokenizer = load_biomedclip()

    img_embeds, panel_ids = embed_images(model, preprocess, panel_df)
    txt_embeds, seg_records = embed_texts(model, tokenizer, seg_records)

    panel_id_to_row = panel_df.set_index("panel_id")

    results = []
    for idx, panel_id in enumerate(panel_ids):
        fig_id = str(panel_id_to_row.loc[panel_id, "figure_id"])
        seg_range = fig_to_seg_range.get(fig_id)
        if seg_range is None:
            continue
        start, end = seg_range
        if start == end:
            continue

        cand_embeds = txt_embeds[start:end]
        sims = cand_embeds @ img_embeds[idx]
        order = np.argsort(-sims)
        best_i = order[0]
        second_sim = float(sims[order[1]]) if len(order) > 1 else None

        results.append({
            "figure_id": fig_id,
            "panel_id": panel_id,
            "best_match_text": seg_records[start + best_i]["text"],
            "best_similarity": float(sims[best_i]),
            "second_best_similarity": second_sim,
            "similarity_margin": (float(sims[best_i]) - second_sim) if second_sim is not None else None,
            "n_candidate_segments": end - start,
            "match_type": "panel_letter_segment" if (end - start) > 1 else "figure_level_caption",
        })

    out_df = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(ALIGNMENT_MANIFEST_PATH, index=False)

    print("\n[STAGE 2b COMPLETE] (BiomedCLIP ablation)")
    print(f"Aligned panels: {len(out_df)}")
    print(f"Output: {ALIGNMENT_MANIFEST_PATH}")
    print(f"(Compare best_similarity distributions against alignment_v1/alignment_manifest.parquet "
          f"from Stage 2 to judge whether BiomedCLIP is worth adopting.)")


if __name__ == "__main__":
    main()