"""
Source: original Colab notebook, cell index [82]
Auto-extracted -- review before treating as final.
"""

def load_paligemma_3b():
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-mix-224", torch_dtype=torch.float16
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
    return {"processor": processor, "model": model}


def paligemma_caption(ctx, image: Image.Image) -> str:
    inputs = ctx["processor"](text="caption en", images=image, return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=40)
    return ctx["processor"].decode(out[0], skip_special_tokens=True).strip()


def paligemma_vqa(ctx, image: Image.Image, question: str) -> str:
    inputs = ctx["processor"](text=f"answer en {question}", images=image, return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        out = ctx["model"].generate(**inputs, max_new_tokens=20)
    return ctx["processor"].decode(out[0], skip_special_tokens=True).strip()