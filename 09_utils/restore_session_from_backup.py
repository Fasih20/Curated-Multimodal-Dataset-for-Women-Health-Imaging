"""
Source: original Colab notebook, cell index [10]
Auto-extracted -- review before treating as final.
"""

"""
Colab: run at the START of your session to restore the latest pipeline_data
from your Drive backups.

Finds the most recent backup folder in /content/drive/MyDrive/colab_session_backups,
locates pipeline_data.zip, and extracts it directly into /content/pipeline_data.
"""

from pathlib import Path
import shutil

DRIVE_ROOT = Path("/content/drive/MyDrive")
BACKUP_ROOT = DRIVE_ROOT / "colab_session_backups"
LOCAL_EXTRACT_DIR = Path("/content/pipeline_data")

def main() -> None:
    if not BACKUP_ROOT.exists():
        print(f"Backup directory not found: {BACKUP_ROOT}")
        print("Have you mounted Google Drive yet?")
        return

    # Find all backup folders starting with "backup_"
    backup_folders = [d for d in BACKUP_ROOT.iterdir() if d.is_dir() and d.name.startswith("backup_")]

    if not backup_folders:
        print("No backup folders found in Drive.")
        return

    # Sort by name (the YYYYMMDD_HHMMSS timestamp format ensures alphabetical = chronological)
    backup_folders.sort(key=lambda x: x.name)
    latest_backup = backup_folders[-1]

    print(f"Found latest backup: {latest_backup.name}")

    zip_path = latest_backup / "pipeline_data.zip"

    if not zip_path.exists():
        print(f"Error: pipeline_data.zip not found in {latest_backup.name}")
        return

    print(f"Extracting {zip_path.name} from Drive to {LOCAL_EXTRACT_DIR} ...")

    # Clear out any existing data in the local folder to avoid messy overlaps
    if LOCAL_EXTRACT_DIR.exists():
        print("Cleaning up existing local /content/pipeline_data directory...")
        shutil.rmtree(LOCAL_EXTRACT_DIR)

    LOCAL_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract the zip directly from Drive into the local Colab storage
    shutil.unpack_archive(zip_path, extract_dir=LOCAL_EXTRACT_DIR, format="zip")

    print("Done! Restore complete.")

if __name__ == "__main__":
    main()