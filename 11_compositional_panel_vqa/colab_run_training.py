"""Run both fixed Stage 11 training modes/tracks on Colab."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path("/content/womens_health_stage11")
env = os.environ.copy()
env["PIPELINE_ROOT"] = str(ROOT / "drive_archive")
env["TOKENIZERS_PARALLELISM"] = "false"
result = subprocess.run(
    [sys.executable, str(ROOT / "code/03_train_panel_set_attention.py"),
     "--track", "both", "--mode", "both", "--epochs", "30"],
    env=env,
    text=True,
    capture_output=True,
)
print(result.stdout)
print(result.stderr, file=sys.stderr)
result.check_returncode()
