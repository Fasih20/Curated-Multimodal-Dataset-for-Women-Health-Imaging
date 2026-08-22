"""
Source: original Colab notebook, cell index [27]
Auto-extracted -- review before treating as final.
"""

import json
import tarfile
import urllib.request
from pathlib import Path
from collections import Counter

ARCHIVE_URL = (
    "https://ai2-s2-medicat.s3.us-west-2.amazonaws.com/2020-10-05/"
    "medicat_release.tar.gz"
)

YOLO_ROOT = Path("/content/pipeline_data/medicat_yolo")
NEEDED_IMAGES_PATH = YOLO_ROOT / "needed_images.json"
IMAGES_ROOT = YOLO_ROOT / "images"

with open(NEEDED_IMAGES_PATH, "r", encoding="utf-8") as f:
    needed = json.load(f)

# filename -> metadata
wanted = {
    item["out_name"]: item
    for item in needed
}

for split in ("train", "valid", "test"):
    (IMAGES_ROOT / split).mkdir(parents=True, exist_ok=True)

print(f"Target images: {len(wanted)}")
print(f"Archive: {ARCHIVE_URL}")
print("\nStarting MedICaT archive stream...\n")

req = urllib.request.Request(
    ARCHIVE_URL,
    headers={"User-Agent": "Mozilla/5.0"}
)

resp = urllib.request.urlopen(req, timeout=120)

found = 0
scanned = 0
found_by_split = Counter()

try:
    with tarfile.open(fileobj=resp, mode="r|gz") as tar:

        for member in tar:

            scanned += 1

            # We only care about files
            if not member.isfile():
                continue

            # Example:
            # release/figures/hash_4-Figure1-1.png
            basename = Path(member.name).name

            if basename not in wanted:
                # Progress every 50k entries
                if scanned % 50000 == 0:
                    print(
                        f"Scanned {scanned:,} entries | "
                        f"Found {found:,}/{len(needed):,}"
                    )
                continue

            item = wanted[basename]
            split = item["split"]

            out_path = IMAGES_ROOT / split / basename

            # Safety: don't overwrite an existing extracted file
            if out_path.exists() and out_path.stat().st_size > 0:
                found += 1
                found_by_split[split] += 1
                del wanted[basename]
                continue

            fobj = tar.extractfile(member)

            if fobj is None:
                print(f"WARNING: Could not extract {member.name}")
                continue

            with open(out_path, "wb") as out:
                while True:
                    chunk = fobj.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

            # Verify that something was actually written
            if out_path.exists() and out_path.stat().st_size > 0:
                found += 1
                found_by_split[split] += 1
                del wanted[basename]

            if found % 50 == 0:
                print(
                    f"Found {found:,}/{len(needed):,} images | "
                    f"Scanned {scanned:,} entries"
                )

            # IMPORTANT: stop as soon as every target is found
            if not wanted:
                print("\nAll target images found!")
                break

finally:
    resp.close()

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(f"Scanned archive entries: {scanned:,}")
print(f"Images extracted:        {found:,}")
print(f"Images still missing:    {len(wanted):,}")

print("\nExtracted by split:")
for split in ("train", "valid", "test"):
    print(f"  {split}: {found_by_split[split]:,}")

if wanted:
    missing_path = YOLO_ROOT / "missing_images.json"

    with open(missing_path, "w", encoding="utf-8") as f:
        json.dump(
            list(wanted.values()),
            f,
            indent=2
        )

    print(f"\nMissing-image manifest:")
    print(missing_path)