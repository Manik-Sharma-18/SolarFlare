#!/usr/bin/env python3
"""Download SDO/HMI SHARP magnetograms from JSOC for pretraining.

Setup:
    pip install drms astropy

    No JSOC registration is required for small queries (<~50 exports).
    For bulk exports (>50 HARPs), register at:
        http://jsoc.stanford.edu/ajax/register_email.html

Usage:
    # Download pilot set (10 well-known flare-active ARs)
    python scripts/download_magnetograms.py

    # Download a specific HARP
    python scripts/download_magnetograms.py --harpnum 7115

    # Full-scale download (200+ HARPs from 2012-2024)
    python scripts/download_magnetograms.py --scale full

    # Custom date range
    python scripts/download_magnetograms.py --start 2017-09-01 --end 2017-09-15

Output:
    data_magnetogram/magnetogram_HARP{num}.npy — (T, H, W) float32 cubes
    data_magnetogram/manifest.json — metadata for all downloaded HARPs
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import drms
except ImportError:
    print("ERROR: drms is required. Install with: pip install drms")
    sys.exit(1)

try:
    from astropy.io import fits as astropy_fits
except ImportError:
    print("ERROR: astropy is required. Install with: pip install astropy")
    sys.exit(1)

# Well-known flare-active regions for pilot validation
PILOT_HARPS = [
    (7115, "AR 12673, Sep 2017 — X9.3 + X8.2 flares"),
    (6063, "AR 12192, Oct 2014 — X3.1, largest AR of Cycle 24"),
    (6555, "AR 12297, Mar 2015 — X2.1 flare"),
    (377,  "AR 11158, Feb 2011 — X2.2 flare"),
    (4315, "AR 11875, Oct 2013 — X2.1 flare"),
    (1654, "AR 11429, Mar 2012 — X5.4 flare"),
    (2673, "AR 11520, Jul 2012 — X1.4 flare"),
    (5765, "AR 12158, Sep 2014 — X1.6 flare"),
    (8088, "AR 12887, Oct 2021 — X1.0 flare"),
    (8948, "AR 13664, May 2024 — X8.7, strongest of Cycle 25"),
]

DEFAULT_OUTPUT_DIR = Path("data_magnetogram")
DEFAULT_CADENCE = 12  # minutes
DEFAULT_TARGET_SIZE = (256, 256)  # (H, W) for spatial standardization


def query_harps_by_date(start_date: str, end_date: str, min_area: float = 100.0,
                        email: str = None) -> list:
    """Query JSOC for SHARP HARPNUMs in a date range.

    Args:
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        min_area: Minimum active region area in micro-hemispheres.
        email: JSOC registered email (optional, for bulk exports).

    Returns:
        List of (harpnum, description) tuples.
    """
    client = drms.Client(email=email) if email else drms.Client()

    # Query SHARP keywords to find active regions
    query = f"hmi.sharp_cea_720s[{start_date}_00:00:00_TAI-{end_date}_00:00:00_TAI]"
    print(f"Querying JSOC: {query}")

    try:
        keys = client.query(query, key=['HARPNUM', 'NOAA_AR', 'T_REC', 'AREA_ACR'])
    except Exception as e:
        print(f"JSOC query failed: {e}")
        print("If this is a connection error, check your internet or try again later.")
        return []

    if keys is None or len(keys) == 0:
        print("No results found for this date range.")
        return []

    # Filter for significant regions
    keys = keys.dropna(subset=['HARPNUM', 'AREA_ACR'])
    keys = keys[keys['AREA_ACR'] > min_area]

    # Get unique HARPs with NOAA AR numbers
    harps = keys.groupby('HARPNUM').agg({
        'NOAA_AR': 'first',
        'AREA_ACR': 'max',
    }).reset_index()
    harps = harps.sort_values('AREA_ACR', ascending=False)

    results = []
    for _, row in harps.iterrows():
        harpnum = int(row['HARPNUM'])
        noaa = int(row['NOAA_AR']) if row['NOAA_AR'] > 0 else None
        area = row['AREA_ACR']
        desc = f"NOAA AR {noaa}, area={area:.0f}" if noaa else f"area={area:.0f}"
        results.append((harpnum, desc))

    print(f"Found {len(results)} HARPs with area > {min_area}")
    return results


def download_harp_fits(harpnum: int, cadence_minutes: int = 12,
                       output_dir: Path = DEFAULT_OUTPUT_DIR,
                       email: str = None, max_retries: int = 3) -> list:
    """Download magnetogram FITS files for one HARP.

    Args:
        harpnum: SHARP HARPNUM identifier.
        cadence_minutes: Temporal cadence (12 = native SHARP cadence).
        output_dir: Directory to save FITS files.
        email: JSOC registered email (required for export).
        max_retries: Number of retry attempts for failed exports.

    Returns:
        List of downloaded FITS file paths.
    """
    client = drms.Client(email=email) if email else drms.Client()
    fits_dir = output_dir / f"fits_harp_{harpnum}"
    fits_dir.mkdir(parents=True, exist_ok=True)

    # Check for already-downloaded files
    existing = sorted(fits_dir.glob("*.fits"))
    if existing:
        print(f"  Found {len(existing)} existing FITS files, skipping download")
        return [str(f) for f in existing]

    ds = f"hmi.sharp_cea_720s[{harpnum}][@{cadence_minutes}m]{{magnetogram}}"
    print(f"  Exporting: {ds}")

    for attempt in range(max_retries):
        try:
            export = client.export(ds, protocol='fits')
            print(f"  Export request submitted (attempt {attempt + 1}). Waiting...")

            # Wait for export to complete
            export.wait(sleep=10)

            if export.status != 0:
                print(f"  Export failed with status {export.status}")
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1)
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return []

            # Download files
            downloaded = export.download(str(fits_dir))
            if downloaded is not None and len(downloaded) > 0:
                fits_files = sorted(fits_dir.glob("*.fits"))
                print(f"  Downloaded {len(fits_files)} FITS files")
                return [str(f) for f in fits_files]
            else:
                print("  Download returned no files")
                return []

        except Exception as e:
            print(f"  Export error: {e}")
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts")
                return []

    return []


def fits_to_cube(fits_paths: list, target_size: tuple = DEFAULT_TARGET_SIZE) -> np.ndarray:
    """Stack FITS magnetogram files into a (T, H, W) numpy cube.

    Args:
        fits_paths: Sorted list of FITS file paths.
        target_size: (H, W) target spatial dimensions for resizing.

    Returns:
        Float32 numpy array of shape (T, target_H, target_W).
    """
    from scipy.ndimage import zoom

    frames = []
    for fpath in fits_paths:
        try:
            with astropy_fits.open(fpath, memmap=True) as hdul:
                data = hdul[1].data if len(hdul) > 1 else hdul[0].data
                if data is None:
                    continue
                data = data.astype(np.float32)
                # Replace NaN with 0 (NaN padding in SHARP patches)
                data = np.nan_to_num(data, nan=0.0)

                # Resize to target dimensions
                if data.shape != target_size:
                    zoom_factors = (target_size[0] / data.shape[0],
                                    target_size[1] / data.shape[1])
                    data = zoom(data, zoom_factors, order=1)

                frames.append(data)
        except Exception as e:
            print(f"  Warning: skipping {Path(fpath).name}: {e}")
            continue

    if not frames:
        return None

    cube = np.stack(frames, axis=0)  # (T, H, W)
    return cube


def download_and_convert(harpnum: int, description: str,
                         output_dir: Path, cadence: int,
                         target_size: tuple, email: str = None) -> dict:
    """Download one HARP and convert to numpy cube.

    Returns:
        Metadata dict for this HARP, or None on failure.
    """
    npy_path = output_dir / f"magnetogram_HARP{harpnum}.npy"
    if npy_path.exists():
        cube = np.load(str(npy_path))
        print(f"  HARP {harpnum}: already exists ({cube.shape}), skipping")
        return {
            'harpnum': harpnum,
            'description': description,
            'shape': list(cube.shape),
            'file': str(npy_path.name),
        }

    print(f"\nHARP {harpnum}: {description}")

    # Download FITS
    fits_paths = download_harp_fits(harpnum, cadence, output_dir, email)
    if not fits_paths:
        print(f"  HARP {harpnum}: no FITS files downloaded, skipping")
        return None

    # Convert to cube
    cube = fits_to_cube(fits_paths, target_size)
    if cube is None or len(cube) < 14:  # Need at least t_in + t_out frames
        print(f"  HARP {harpnum}: too few valid frames ({len(cube) if cube is not None else 0}), skipping")
        return None

    # Save as .npy
    np.save(str(npy_path), cube)
    print(f"  Saved: {npy_path.name} — shape {cube.shape}")

    return {
        'harpnum': harpnum,
        'description': description,
        'shape': list(cube.shape),
        'file': str(npy_path.name),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download SDO/HMI SHARP magnetograms for pretraining"
    )
    parser.add_argument('--scale', choices=['pilot', 'full'], default='pilot',
                        help='Download scale: pilot (10 HARPs) or full (200+)')
    parser.add_argument('--harpnum', type=int, default=None,
                        help='Download a single specific HARPNUM')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD) for custom range')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD) for custom range')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Output directory for .npy cubes')
    parser.add_argument('--cadence', type=int, default=DEFAULT_CADENCE,
                        help='Temporal cadence in minutes (default: 12)')
    parser.add_argument('--target-size', type=int, nargs=2, default=list(DEFAULT_TARGET_SIZE),
                        help='Target spatial size H W (default: 256 256)')
    parser.add_argument('--email', type=str, default=None,
                        help='JSOC registered email (required for bulk exports)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size = tuple(args.target_size)

    print("=" * 60)
    print("SDO/HMI SHARP Magnetogram Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Target spatial size: {target_size}")
    print(f"Cadence: {args.cadence} minutes")

    # Validate email for JSOC exports
    if not args.email:
        print("\nERROR: JSOC exports require a registered email address.")
        print("  1. Register (free, 30s): http://jsoc.stanford.edu/ajax/register_email.html")
        print("  2. Re-run with: python scripts/download_magnetograms.py --email your@email.com")
        sys.exit(1)
    print(f"JSOC email: {args.email}")

    # Determine which HARPs to download
    if args.harpnum:
        harps = [(args.harpnum, f"User-specified HARP {args.harpnum}")]
    elif args.start and args.end:
        harps = query_harps_by_date(args.start, args.end, email=args.email)
    elif args.scale == 'pilot':
        harps = PILOT_HARPS
        print(f"\nPilot mode: {len(harps)} well-known flare-active regions")
    else:
        # Full scale: query major periods of Solar Cycles 24 & 25
        print("\nFull-scale mode: querying major active periods...")
        harps = []
        date_ranges = [
            ("2011-01-01", "2012-12-31"),
            ("2013-01-01", "2014-12-31"),
            ("2015-01-01", "2017-12-31"),
            ("2021-01-01", "2024-12-31"),
        ]
        for start, end in date_ranges:
            found = query_harps_by_date(start, end, min_area=200, email=args.email)
            harps.extend(found)
        # Deduplicate
        seen = set()
        unique_harps = []
        for h in harps:
            if h[0] not in seen:
                seen.add(h[0])
                unique_harps.append(h)
        harps = unique_harps
        print(f"Total unique HARPs to download: {len(harps)}")

    if not harps:
        print("No HARPs to download. Exiting.")
        return

    # Download and convert each HARP
    manifest = []
    for i, (harpnum, desc) in enumerate(harps):
        print(f"\n[{i+1}/{len(harps)}] ", end="")
        result = download_and_convert(
            harpnum, desc, output_dir, args.cadence, target_size, args.email
        )
        if result:
            manifest.append(result)

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            'total_harps': len(manifest),
            'target_size': list(target_size),
            'cadence_minutes': args.cadence,
            'harps': manifest,
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Downloaded {len(manifest)}/{len(harps)} HARPs")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext step: python scripts/preprocess_magnetograms.py")


if __name__ == '__main__':
    main()
