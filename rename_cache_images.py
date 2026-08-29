#!/usr/bin/env python3
"""Rename stale doc_image_cache files to their new pytest-pyvista test-id names.

This uses a hard-coded mapping (derived once via visual-similarity matching
against the docs build output) rather than re-deriving it every time, since
the old -> new name mapping is now known and fixed.

Usage:
    python rename_cache_images_fixed.py <cache_dir> [--apply]

Defaults to a DRY RUN -- pass --apply to actually rename files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# old stem (without extension) -> new stem (without extension)
RENAME_MAP: dict[str, str] = {
    "pyvista-examples-gltf-download_gearbox-da15a6d3f963a451_00_00":
        "pyvista-examples-downloads-download_gearbox-efeb49a68583182a_00_00",
    "pyvista-examples-gltf-download_gearbox-da15a6d3f963a451_00_00_vtksz":
        "pyvista-examples-downloads-download_gearbox-efeb49a68583182a_00_00_vtksz",
    "pyvista-ThreeDSReader-b9d4460da7a7417d_00_01":
        "pyvista-ThreeDSReader-b08023bed9b90835_00_01",
    "pyvista-ThreeDSReader-b9d4460da7a7417d_00_01_vtksz":
        "pyvista-ThreeDSReader-b08023bed9b90835_00_01_vtksz",
    "pyvista-VRMLReader-16fac38b2acbb0ec_00_01":
        "pyvista-VRMLReader-4a1831f2a1d6a67b_00_01",
    "pyvista-VRMLReader-16fac38b2acbb0ec_00_01_vtksz":
        "pyvista-VRMLReader-4a1831f2a1d6a67b_00_01_vtksz",
    "pyvista-examples-gltf-download_milk_truck-62593202625dad31_00_00":
        "pyvista-examples-downloads-download_milk_truck-57620f1cefec8d32_00_00",
    "pyvista-examples-gltf-download_milk_truck-62593202625dad31_00_00_vtksz":
        "pyvista-examples-downloads-download_milk_truck-57620f1cefec8d32_00_00_vtksz",
    "pyvista-examples-gltf-download_damaged_helmet-f59dfcee157b5af4_00_00":
        "pyvista-examples-downloads-download_damaged_helmet-1a217a664a94cfd1_00_00",
    "pyvista-examples-gltf-download_damaged_helmet-f59dfcee157b5af4_00_00_vtksz":
        "pyvista-examples-downloads-download_damaged_helmet-1a217a664a94cfd1_00_00_vtksz",
    "pyvista-examples-vrml-download_teapot-f8fa9e287feab173_00_00":
        "pyvista-examples-downloads-download_teapot_vrml-969eb1c5a2edf939_00_00",
    "pyvista-examples-vrml-download_teapot-f8fa9e287feab173_00_00_vtksz":
        "pyvista-examples-downloads-download_teapot_vrml-969eb1c5a2edf939_00_00_vtksz",
    "pyvista-examples-vrml-download_grasshopper-eb03c6e5d0e1effd_00_00":
        "pyvista-examples-downloads-download_grasshopper-9ecf7cd85019c6e3_00_00",
    "pyvista-examples-vrml-download_grasshopper-eb03c6e5d0e1effd_00_00_vtksz":
        "pyvista-examples-downloads-download_grasshopper-9ecf7cd85019c6e3_00_00_vtksz",
    "pyvista-Plotter-import_3ds-74de5b60b89d5738_00_00":
        "pyvista-Plotter-import_3ds-b009693feee38c55_00_00",
    "pyvista-Plotter-import_3ds-74de5b60b89d5738_00_00_vtksz":
        "pyvista-Plotter-import_3ds-b009693feee38c55_00_00_vtksz",
    "pyvista-examples-download_3ds-download_iflamigm-74de5b60b89d5738_00_00":
        "pyvista-examples-downloads-download_flamingo-b009693feee38c55_00_00",
    "pyvista-examples-download_3ds-download_iflamigm-74de5b60b89d5738_00_00_vtksz":
        "pyvista-examples-downloads-download_flamingo-b009693feee38c55_00_00_vtksz",
    "pyvista-examples-vrml-download_sextant-bc79a1b83bbbbcc9_00_00":
        "pyvista-examples-downloads-download_sextant-fa96c180b69465ce_00_00",
    "pyvista-examples-vrml-download_sextant-bc79a1b83bbbbcc9_00_00_vtksz":
        "pyvista-examples-downloads-download_sextant-fa96c180b69465ce_00_00_vtksz",
    "pyvista-examples-gltf-download_avocado-670d49097e1835f2_00_00":
        "pyvista-examples-downloads-download_avocado-a281271fb2906f6d_00_00",
    "pyvista-examples-gltf-download_avocado-670d49097e1835f2_00_00_vtksz":
        "pyvista-examples-downloads-download_avocado-a281271fb2906f6d_00_00_vtksz",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path, help="doc_image_cache dir containing the stale-named files")
    parser.add_argument("--apply", action="store_true", help="Actually perform renames (default: dry run)")
    args = parser.parse_args()

    if not args.cache_dir.is_dir():
        print(f"ERROR: not a directory: {args.cache_dir}", file=sys.stderr)
        sys.exit(1)

    # Index actual files in cache_dir by stem, so we can pick up whatever
    # extension they actually have (.jpg in the failures we saw).
    files_by_stem = {p.stem: p for p in args.cache_dir.iterdir() if p.is_file()}

    print(f"Renaming files in: {args.cache_dir}\n")

    n_renamed = 0
    n_missing = 0
    n_skipped = 0

    for old_stem, new_stem in RENAME_MAP.items():
        src = files_by_stem.get(old_stem)
        if src is None:
            print(f"  MISSING (not found in cache_dir): {old_stem}.*")
            n_missing += 1
            continue

        dst = src.with_name(new_stem + src.suffix)

        if dst.exists():
            print(f"  SKIP (target already exists): {dst.name}")
            n_skipped += 1
            continue

        if args.apply:
            src.rename(dst)
            print(f"  renamed: {src.name} -> {dst.name}")
        else:
            print(f"  would rename: {src.name} -> {dst.name}")
        n_renamed += 1

    verb = "Renamed" if args.apply else "Would rename"
    print(f"\n{verb} {n_renamed} file(s). {n_missing} missing, {n_skipped} skipped.")

    if not args.apply:
        print("\nDry run only -- pass --apply to actually rename files.")


if __name__ == "__main__":
    main()
