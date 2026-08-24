"""Fetch and safely extract the user-provided pipeline archive on Colab."""
from pathlib import Path
import subprocess
import sys
import zipfile

FILE_ID = "1hKgZ3jNz6NRrlPEuNg8LBTXiX3kMtphX"
ROOT = Path("/content/womens_health_stage11")
ARCHIVE = ROOT / "pipeline_data.zip"
EXTRACT = ROOT / "drive_archive"

# gdown creates the output file, but it does not create its parent directory.
ROOT.mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
import gdown

gdown.download(id=FILE_ID, output=str(ARCHIVE), quiet=False)
if not ARCHIVE.exists() or not zipfile.is_zipfile(ARCHIVE):
    raise RuntimeError(f"Drive download is not a valid ZIP: {ARCHIVE}")

EXTRACT.mkdir(parents=True, exist_ok=True)
root_resolved = EXTRACT.resolve()
with zipfile.ZipFile(ARCHIVE) as zf:
    members = zf.infolist()
    for member in members:
        target = (EXTRACT / member.filename).resolve()
        if target != root_resolved and root_resolved not in target.parents:
            raise RuntimeError(f"unsafe ZIP path: {member.filename!r}")
    print(f"ZIP entries: {len(members)}; uncompressed bytes: {sum(m.file_size for m in members):,}")
    print("Top-level entries:", sorted({Path(m.filename).parts[0] for m in members if Path(m.filename).parts})[:30])
    zf.extractall(EXTRACT)

candidates = sorted({p.parent for p in EXTRACT.rglob("panels_v1/panel_manifest.csv")})
if not candidates:
    raise RuntimeError("Could not locate a pipeline_data root containing panels_v1/panel_manifest.csv")
for candidate in candidates:
    print("PIPELINE_ROOT candidate:", candidate)
