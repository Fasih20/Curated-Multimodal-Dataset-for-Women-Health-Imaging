"""Run Stage 11.1--11.2 inside the prepared Colab runtime."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path("/content/womens_health_stage11")
env = os.environ.copy()
env["PIPELINE_ROOT"] = str(ROOT / "drive_archive")
for script in ("01_fix_panel_caption_alignment.py", "02_generate_compositional_questions.py"):
    subprocess.run(
        [sys.executable, str(ROOT / "code" / script), "--track", "both"],
        check=True,
        env=env,
    )
