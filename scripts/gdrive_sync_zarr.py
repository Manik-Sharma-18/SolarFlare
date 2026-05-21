#!/usr/bin/env python3
"""Download new zarr.zip files from Google Drive, unzip to data/, skip existing.

Usage:
    python3 scripts/gdrive_sync_zarr.py [--dry-run]
"""
import argparse
import os
import sys
import zipfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import gdown
except ImportError:
    print("ERROR: pip install gdown")
    sys.exit(1)

FOLDER_URL = "https://drive.google.com/drive/folders/1uyuN-2RKB_tW4VdDa1ydxz3JT9ulevD7"
DATA_DIR = Path(__file__).parent.parent / "data"
TMP_DIR = DATA_DIR / "_gdrive_tmp"


def zarr_name(drive_path: str) -> str:
    """'harp_86.zarr.zip' -> 'harp_86.zarr'"""
    return drive_path.replace(".zip", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    existing = {p.name for p in DATA_DIR.iterdir() if p.suffix == ".zarr"}
    print(f"Existing zarr ({len(existing)}): {sorted(existing)}")

    print("\nListing Drive folder...")
    entries = gdown.download_folder(
        url=FOLDER_URL,
        skip_download=True,
        quiet=True,
    )

    if not entries:
        print("No files found (check folder sharing permissions).")
        sys.exit(1)

    zarr_entries = [e for e in entries if e.path.endswith(".zarr.zip")]
    extra_entries = [e for e in entries if e.path == "hek_cache.zip"]
    new_entries = [e for e in zarr_entries if zarr_name(e.path) not in existing]
    skip_entries = [e for e in zarr_entries if zarr_name(e.path) in existing]
    all_new = new_entries + extra_entries

    print(f"\nDrive zarr total : {len(zarr_entries)}")
    print(f"Already local    : {[zarr_name(e.path) for e in skip_entries]}")
    print(f"New to download  : {[zarr_name(e.path) for e in new_entries]}")
    print(f"Extras           : {[e.path for e in extra_entries]}")

    if not all_new:
        print("\nNothing new. Done.")
        return

    if args.dry_run:
        print("\n[dry-run] Would download the above. Exiting.")
        return

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(all_new, 1):
        zip_path = TMP_DIR / entry.path
        zarr_out = DATA_DIR / zarr_name(entry.path)

        print(f"\n[{i}/{len(new_entries)}] Downloading {entry.path} ...")
        gdown.download(
            id=entry.id,
            output=str(zip_path),
            quiet=False,
            resume=True,
        )

        print(f"  Extracting -> {zarr_out} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)

        zip_path.unlink()
        print(f"  Done: {zarr_out}")

    TMP_DIR.rmdir()
    print(f"\nDone. {len(all_new)} file(s) added to {DATA_DIR}.")


if __name__ == "__main__":
    main()
