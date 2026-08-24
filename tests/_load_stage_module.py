"""Shared helper: load a numbered pipeline script as an importable module.

The pipeline scripts are named like "01_fix_panel_caption_alignment.py" --
not valid Python identifiers -- because the numbering is how the repo
documents stage order (see README.md and run_pipeline.py, which already
runs them as subprocesses for the same reason). Tests import the function
bodies directly via importlib rather than subprocess, so this helper
loads a script by path and returns it as a module object.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def load_module(relative_path: str, module_name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
