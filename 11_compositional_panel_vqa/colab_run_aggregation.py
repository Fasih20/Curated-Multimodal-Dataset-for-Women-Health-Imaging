"""Regenerate Stage 11 aggregate outputs on Colab."""
from pathlib import Path
import os
import runpy

ROOT = Path("/content/womens_health_stage11")
os.environ["PIPELINE_ROOT"] = str(ROOT / "drive_archive")
runpy.run_path(str(ROOT / "code/05_run_full_evaluation_and_ablations.py"), run_name="__main__")
