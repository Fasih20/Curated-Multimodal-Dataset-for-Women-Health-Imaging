"""Stage 11.3: train lightweight heads over frozen panel/question embeddings."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", "./pipeline_data")).resolve()
OUT_DIR = PIPELINE_ROOT / "compositional_vqa_v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = {
    "clip": "openai/clip-vit-base-patch32",
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class PanelSetAttention(nn.Module):
    def __init__(self, embed_dim: int, answer_sizes: dict[str, int], max_panels: int = 32,
                 num_heads: int = 2, hidden_dim: int = 128):
        super().__init__()
        if embed_dim % num_heads:
            num_heads = 1
        self.position = nn.Embedding(max_panels, embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.question_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.heads = nn.ModuleDict({k: nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n)
        ) for k, n in answer_sizes.items()})

    def representation(self, panels: torch.Tensor, question: torch.Tensor,
                       padding_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(panels.shape[1], device=panels.device)
        tokens = panels + self.position(positions)[None, :, :]
        attended, _ = self.self_attention(tokens, tokens, tokens, key_padding_mask=padding_mask)
        scores = (attended * self.question_projection(question)[:, None, :]).sum(-1)
        scores = scores.masked_fill(padding_mask, float("-inf"))
        pooled = (torch.softmax(scores, dim=1).unsqueeze(-1) * attended).sum(1)
        return torch.cat([pooled, question], dim=-1)

    def forward(self, panels, question, padding_mask, question_types):
        rep = self.representation(panels, question, padding_mask)
        return [self.heads[qtype](rep[i:i + 1]).squeeze(0) for i, qtype in enumerate(question_types)]


class MeanPoolAblation(nn.Module):
    def __init__(self, embed_dim: int, answer_sizes: dict[str, int], hidden_dim: int = 128, **_):
        super().__init__()
        self.heads = nn.ModuleDict({k: nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n)
        ) for k, n in answer_sizes.items()})

    def representation(self, panels: torch.Tensor, question: torch.Tensor,
                       padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (panels * valid).sum(1) / valid.sum(1).clamp_min(1)
        return torch.cat([pooled, question], dim=-1)

    def forward(self, panels, question, padding_mask, question_types):
        rep = self.representation(panels, question, padding_mask)
        return [self.heads[qtype](rep[i:i + 1]).squeeze(0) for i, qtype in enumerate(question_types)]


class CompositionalDataset(Dataset):
    def __init__(self, frame, panel_map, question_map, answer_spaces):
        self.rows = frame.to_dict("records"); self.panel_map = panel_map
        self.question_map = question_map; self.answer_spaces = answer_spaces
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]; space = self.answer_spaces[row["question_type"]]
        return (torch.tensor(np.stack([self.panel_map[str(p)] for p in row["panel_ids"]]), dtype=torch.float32),
                torch.tensor(self.question_map[row["question"]], dtype=torch.float32), row["question_type"],
                space.index(str(row["answer"])))


def collate(batch):
    max_n = max(x[0].shape[0] for x in batch); dim = batch[0][0].shape[1]
    panels = torch.zeros(len(batch), max_n, dim); mask = torch.ones(len(batch), max_n, dtype=torch.bool)
    for i, (vectors, _, _, _) in enumerate(batch):
        panels[i, :len(vectors)] = vectors; mask[i, :len(vectors)] = False
    return panels, torch.stack([x[1] for x in batch]), mask, [x[2] for x in batch], torch.tensor([x[3] for x in batch])


def answer_spaces_from_frame(frame: pd.DataFrame) -> dict[str, list[str]]:
    spaces = {}
    for qtype, group in frame.groupby("question_type"):
        values = set()
        for space in group.answer_space:
            values.update(str(x) for x in list(space))
        spaces[str(qtype)] = sorted(values, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    return spaces


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None; model.train(training)
    correct = total = 0; by_type = {}
    for panels, questions, mask, qtypes, labels in loader:
        panels, questions, mask, labels = panels.to(DEVICE), questions.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
        logits = model(panels, questions, mask, qtypes)
        losses = [nn.functional.cross_entropy(logit[None], labels[i:i+1]) for i, logit in enumerate(logits)]
        loss = torch.stack(losses).mean()
        if training:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        for i, (logit, qtype) in enumerate(zip(logits, qtypes)):
            hit = int(logit.argmax().item() == labels[i].item()); correct += hit; total += 1
            stats = by_type.setdefault(qtype, [0, 0]); stats[0] += hit; stats[1] += 1
    return {"overall": correct / total if total else None,
            "per_question_type": {k: {"accuracy": c/n, "n": n} for k, (c, n) in by_type.items()}, "n": total}


def embed_questions(track: str, texts: list[str]) -> np.ndarray:
    cache_dir = OUT_DIR / "embed_cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    embeds_path = cache_dir / f"question_embeds_{track}.npy"; texts_path = cache_dir / f"question_texts_{track}.json"
    if embeds_path.exists() and texts_path.exists() and json.loads(texts_path.read_text()) == texts:
        return np.load(embeds_path)
    if track == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(MODEL_NAMES[track]).to(DEVICE).eval(); proc = CLIPProcessor.from_pretrained(MODEL_NAMES[track])
        inputs = proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
        with torch.no_grad(): vectors = model.get_text_features(**inputs)
    else:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(MODEL_NAMES[track]); model = model.to(DEVICE).eval()
        tokenizer = open_clip.get_tokenizer(MODEL_NAMES[track]); tokens = tokenizer(texts, context_length=256).to(DEVICE)
        with torch.no_grad(): vectors = model.encode_text(tokens)
    if not torch.is_tensor(vectors):
        if hasattr(vectors, "text_embeds"):
            vectors = vectors.text_embeds
        elif hasattr(vectors, "pooler_output"):
            vectors = vectors.pooler_output
        else:
            raise TypeError(f"unexpected text-feature output: {type(vectors)}")
    vectors = nn.functional.normalize(vectors, dim=-1).cpu().numpy()
    np.save(embeds_path, vectors); texts_path.write_text(json.dumps(texts, indent=2))
    return vectors


def train(track: str, mode: str, epochs: int = 30, batch_size: int = 32):
    dataset_path = OUT_DIR / f"compositional_vqa_dataset_{track}.parquet"
    align_dir = "alignment_v1" if track == "clip" else "alignment_biomedclip_v1"
    cache = PIPELINE_ROOT / align_dir / "embed_cache"
    frame = pd.read_parquet(dataset_path)
    panel_vectors = np.load(cache / "panel_image_embeds.npy")
    panel_ids = json.loads((cache / "panel_image_ids.json").read_text())
    panel_map = dict(zip(map(str, panel_ids), panel_vectors))
    texts = sorted(frame.question.unique().tolist()); qvectors = embed_questions(track, texts)
    question_map = dict(zip(texts, qvectors)); spaces = answer_spaces_from_frame(frame)
    model_cls = PanelSetAttention if mode == "set_attention" else MeanPoolAblation
    model = model_cls(panel_vectors.shape[1], {k: len(v) for k, v in spaces.items()}).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loaders = {split: DataLoader(CompositionalDataset(frame[frame.split == split], panel_map, question_map, spaces),
                                 batch_size=batch_size, shuffle=split == "train", collate_fn=collate)
               for split in ("train", "val", "test")}
    for _ in range(epochs): run_epoch(model, loaders["train"], optimizer)
    metrics = {"track": track, "mode": mode, "seed": SEED, "epochs": epochs,
               "hyperparameters": {"batch_size": batch_size, "learning_rate": 1e-3, "hidden_dim": 128},
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "val": run_epoch(model, loaders["val"]), "test": run_epoch(model, loaders["test"])}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "answer_spaces": spaces, "metrics": metrics}, OUT_DIR / f"model_{mode}_{track}.pt")
    (OUT_DIR / f"metrics_{mode}_{track}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2)); return metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--track", choices=["clip", "biomedclip", "both"], default="both")
    parser.add_argument("--mode", choices=["set_attention", "mean_pool", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30); args = parser.parse_args(); seed_everything()
    tracks = ["clip", "biomedclip"] if args.track == "both" else [args.track]
    modes = ["set_attention", "mean_pool"] if args.mode == "both" else [args.mode]
    for track in tracks:
        for mode in modes: train(track, mode, args.epochs)


if __name__ == "__main__": main()
