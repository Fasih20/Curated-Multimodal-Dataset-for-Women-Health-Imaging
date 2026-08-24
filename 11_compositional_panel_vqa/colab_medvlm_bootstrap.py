"""Install medical-VLM runtime dependencies and validate gated access."""
from pathlib import Path
import subprocess, sys
root = Path("/content/womens_health_stage11")
(root / "code").mkdir(parents=True, exist_ok=True)
(root / "drive_archive/compositional_vqa_v1").mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers", "accelerate", "bitsandbytes", "qwen-vl-utils"], check=True)
token_path = root / ".hf_token"
if not token_path.exists():
    raise RuntimeError("Materialize HF_TOKEN from the Colab UI into the runtime token file")
token = token_path.read_text().strip()
from huggingface_hub import HfApi
HfApi(token=token).model_info("google/medgemma-4b-it")
print("Medical-VLM dependencies and gated MedGemma access verified.")
