"""Run saved-prediction baselines and aggregate Stage 11 results on Colab."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path("/content/womens_health_stage11")
env = os.environ.copy()
env["PIPELINE_ROOT"] = str(ROOT / "drive_archive")
commands = [
    [sys.executable, str(ROOT / "code/04_baseline_single_panel_vlm_eval.py"), "--track", "both"],
    [sys.executable, str(ROOT / "code/05_run_full_evaluation_and_ablations.py")],
]
for command in commands:
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    result.check_returncode()
