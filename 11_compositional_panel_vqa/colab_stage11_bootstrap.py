"""Bootstrap the minimal directory/dependency layout on a Colab runtime.

This helper contains no credentials or data. Execute it with ``colab exec``
before uploading the cached Stage 11 inputs.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path("/content/womens_health_stage11")
for relative in (
    "code", "pipeline_data/panels_v1",
    "pipeline_data/alignment_v1/embed_cache",
    "pipeline_data/alignment_biomedclip_v1/embed_cache",
    "pipeline_data/quality_v1", "pipeline_data/quality_biomedclip_v1",
    "pipeline_data/splits_final_v1", "pipeline_data/splits_biomedclip_v1",
    "pipeline_data/vlm_benchmark_clip_v1/vqa_predictions",
    "pipeline_data/vlm_benchmark_biomedclip_v1/vqa_predictions",
):
    (ROOT / relative).mkdir(parents=True, exist_ok=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pyarrow", "scipy", "transformers", "open_clip_torch"],
    check=True,
)
print(f"Stage 11 Colab root ready: {ROOT}")
