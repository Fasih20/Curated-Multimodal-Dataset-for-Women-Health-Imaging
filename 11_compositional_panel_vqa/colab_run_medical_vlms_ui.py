"""Colab-UI launcher for gated medical-VLM evaluation.

Run with ``%run 11_compositional_panel_vqa/colab_run_medical_vlms_ui.py``
from a Colab UI cell so ``userdata`` and Drive mounting are available.
"""
from pathlib import Path
import json, os, runpy, shutil, subprocess, sys

from google.colab import drive, files, userdata

REPO = Path.cwd()
ROOT = Path("/content/womens_health_medvlm")
PIPELINE = ROOT / "drive_archive"
RESULTS = Path("/content/drive/MyDrive/womens_health_medvlm_results")

token = userdata.get("HF_TOKEN")
if not token:
    raise RuntimeError("Create the HF_TOKEN Colab secret and enable notebook access")
os.environ["HF_TOKEN"] = token
drive.mount("/content/drive")
RESULTS.mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "gdown", "pyarrow", "scipy",
                "transformers", "accelerate", "bitsandbytes", "qwen-vl-utils"], check=True)

# Reuse the safe Drive downloader after redirecting its fixed runtime root.
fetch_source = (REPO / "11_compositional_panel_vqa/colab_fetch_drive_data.py").read_text()
fetch_source = fetch_source.replace('/content/womens_health_stage11', str(ROOT))
exec(compile(fetch_source, "colab_fetch_drive_data.py", "exec"), {"__name__": "__main__"})

env = os.environ.copy(); env["PIPELINE_ROOT"] = str(PIPELINE)
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
for script in ("01_fix_panel_caption_alignment.py", "02_generate_compositional_questions.py"):
    subprocess.run([sys.executable, str(REPO / "11_compositional_panel_vqa" / script), "--track", "both"],
                   check=True, env=env)

out = PIPELINE / "compositional_vqa_v1"
drive_predictions = RESULTS / "medical_vlm_predictions"; drive_predictions.mkdir(exist_ok=True)
local_predictions = out / "medical_vlm_predictions"
if local_predictions.exists() and not local_predictions.is_symlink():
    shutil.copytree(local_predictions, drive_predictions, dirs_exist_ok=True); shutil.rmtree(local_predictions)
if not local_predictions.exists(): local_predictions.symlink_to(drive_predictions, target_is_directory=True)
metrics_link = out / "medical_vlm_metrics.json"; drive_metrics = RESULTS / "medical_vlm_metrics.json"
if not metrics_link.exists() and not metrics_link.is_symlink(): metrics_link.symlink_to(drive_metrics)

subprocess.run([sys.executable, str(REPO / "11_compositional_panel_vqa/07_benchmark_medical_vlms.py"),
                "--track", "both", "--model", "all"], check=True, env=env)

for artifact in out.glob("*.summary.json"): shutil.copy2(artifact, RESULTS / artifact.name)
for artifact in out.glob("compositional_vqa_dataset_*.parquet"): shutil.copy2(artifact, RESULTS / artifact.name)
archive = shutil.make_archive("/content/womens_health_medvlm_results", "zip", RESULTS)
print("Durable results:", RESULTS)
files.download(archive)
