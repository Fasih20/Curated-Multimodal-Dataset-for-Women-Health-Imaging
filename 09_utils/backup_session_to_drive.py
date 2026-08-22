"""
Source: original Colab notebook, cell index [79]
Auto-extracted -- review before treating as final.
"""

"""
Colab: run at the END of your session to back up everything you've built
locally (/content/pipeline_data, /content/local_download manifests, etc.)
to a separate Drive folder -- so it survives runtime disconnects.

Only copies the small stuff (parquet/json manifests, logs) by default --
NOT the 16K raw image files, which would eat Drive space/time for no
reason since they already exist on Drive under the original download
folder. Flip COPY_LOCAL_IMAGES to True if you specifically want a backup
copy of the local image folder too.
"""

from pathlib import Path
import shutil
import os

DRIVE_ROOT = Path("/content/drive/MyDrive")
BACKUP_ROOT = DRIVE_ROOT / "colab_session_backups"

# Timestamped subfolder so repeated runs don't clobber each other.
from datetime import datetime, timezone
STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = BACKUP_ROOT / f"backup_{STAMP}"

SOURCES = [
    Path("/content/pipeline_data"),   # all manifests, splits, dedup/cleaning outputs, summaries
]

COPY_LOCAL_IMAGES = False  # set True to also back up /content/local_download (large, slow)
if COPY_LOCAL_IMAGES:
    SOURCES.append(Path("/content/local_download"))


def main() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        if not src.exists():
            print(f"Skip (not found): {src}")
            continue

        # Check if the current source is pipeline_data to apply zipping logic
        if src.name == "pipeline_data":
            zip_base_path = f"/content/{src.name}"
            zip_file_path = Path(f"{zip_base_path}.zip")

            print(f"Zipping {src} ...")
            # Creates /content/pipeline_data.zip
            shutil.make_archive(zip_base_path, 'zip', src)

            dest = BACKUP_DIR / zip_file_path.name
            print(f"Copying {zip_file_path} -> {dest} ...")
            shutil.copy2(zip_file_path, dest)

            print(f"Deleting local zip file {zip_file_path} ...")
            zip_file_path.unlink() # removes the zip from the colab environment

        else:
            dest = BACKUP_DIR / src.name
            print(f"Copying {src} -> {dest} ...")
            shutil.copytree(src, dest, dirs_exist_ok=True)

    print(f"\nDone. Backup written to: {BACKUP_DIR}")


if __name__ == "__main__":
    main()