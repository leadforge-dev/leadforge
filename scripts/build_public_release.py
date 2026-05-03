#!/usr/bin/env python3
"""Build the public release bundles for Kaggle/HuggingFace.

Usage:
    python scripts/build_public_release.py [OUTPUT_DIR]

Generates four bundles:
- intro/          (student_public, intro difficulty)
- intermediate/   (student_public, intermediate difficulty)
- advanced/       (student_public, advanced difficulty)
- intermediate_instructor/  (research_instructor, intermediate difficulty)

Each student_public bundle also gets a flat CSV convenience export
(lead_scoring.csv) merging train/valid/test with a ``split`` column.

All bundles are validated with ``leadforge validate`` after generation.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from leadforge.api.generator import Generator
from leadforge.validation.bundle_checks import validate_bundle

SEED = 42
RECIPE = "b2b_saas_procurement_v1"

# (directory_name, exposure_mode, difficulty)
BUNDLES = [
    ("intro", "student_public", "intro"),
    ("intermediate", "student_public", "intermediate"),
    ("advanced", "student_public", "advanced"),
    ("intermediate_instructor", "research_instructor", "intermediate"),
]


def generate_and_save(
    out_dir: Path,
    exposure_mode: str,
    difficulty: str,
    seed: int = SEED,
) -> None:
    """Generate a bundle and write it to *out_dir*."""
    gen = Generator.from_recipe(
        RECIPE,
        seed=seed,
        exposure_mode=exposure_mode,
        difficulty=difficulty,
    )
    bundle = gen.generate()
    bundle.save(str(out_dir))


# Columns to drop from the flat CSV convenience export.
# current_stage at the 90-day horizon contains terminal stages (closed_won /
# closed_lost) that perfectly encode the label — it is leakage.  The column
# remains in the Parquet task splits for completeness but must be excluded
# from modeling.  The flat CSV drops it to prevent accidental misuse.
_FLAT_CSV_DROP_COLS = {"current_stage"}


def write_flat_csv(bundle_dir: Path) -> Path:
    """Merge task splits into a single CSV with a ``split`` column.

    Drops columns listed in ``_FLAT_CSV_DROP_COLS`` to prevent accidental
    leakage in the convenience export.
    """
    task_dir = bundle_dir / "tasks" / "converted_within_90_days"
    frames = []
    for split_name in ("train", "valid", "test"):
        path = task_dir / f"{split_name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df.insert(0, "split", split_name)
            frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    drop = [c for c in _FLAT_CSV_DROP_COLS if c in merged.columns]
    if drop:
        merged = merged.drop(columns=drop)
    csv_path = bundle_dir / "lead_scoring.csv"
    merged.to_csv(csv_path, index=False)
    return csv_path


def print_summary(bundle_dir: Path, name: str) -> None:
    """Print row counts and conversion rate for a bundle."""
    manifest_path = bundle_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    table_summary = ", ".join(f"{t}={info['row_count']}" for t, info in manifest["tables"].items())
    task_info = manifest["tasks"].get("converted_within_90_days", {})
    total_task_rows = sum(task_info.get(f"{s}_rows", 0) for s in ("train", "valid", "test"))

    # Compute conversion rate from the train split Parquet (avoid re-reading CSV).
    conv_str = ""
    train_path = bundle_dir / "tasks" / "converted_within_90_days" / "train.parquet"
    if train_path.exists():
        train_df = pd.read_parquet(train_path, columns=["converted_within_90_days"])
        rate = train_df["converted_within_90_days"].mean()
        conv_str = f", train_conversion={rate:.1%}"

    print(f"  {name}: {table_summary}")
    print(f"    task rows={total_task_rows}{conv_str}")


def main() -> None:
    output_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("release")
    output_root.mkdir(parents=True, exist_ok=True)

    # Copy LICENSE
    license_src = Path(__file__).resolve().parent.parent / "LICENSE"
    if license_src.exists():
        shutil.copy2(license_src, output_root / "LICENSE")

    for dir_name, exposure_mode, difficulty in BUNDLES:
        bundle_dir = output_root / dir_name
        print(f"Generating {dir_name} ({exposure_mode}, {difficulty})...", file=sys.stderr)
        generate_and_save(bundle_dir, exposure_mode, difficulty)

        # Flat CSV for student_public bundles
        if exposure_mode == "student_public":
            csv_path = write_flat_csv(bundle_dir)
            print(f"  Flat CSV: {csv_path}", file=sys.stderr)

        # Validate
        print(f"  Validating {dir_name}...", file=sys.stderr)
        errors = validate_bundle(bundle_dir)
        if errors:
            print(f"  FAIL: {len(errors)} error(s):", file=sys.stderr)
            for e in errors:
                print(f"    - {e}", file=sys.stderr)
            sys.exit(1)
        print(f"  OK: {dir_name} passed validation.", file=sys.stderr)

    # Summary
    print("\n=== Release Summary ===")
    for dir_name, _, _ in BUNDLES:
        bundle_dir = output_root / dir_name
        print_summary(bundle_dir, dir_name)

    print(f"\nRelease directory: {output_root.resolve()}")


if __name__ == "__main__":
    main()
