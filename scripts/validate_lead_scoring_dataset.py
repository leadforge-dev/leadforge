#!/usr/bin/env python3
"""CLI entrypoint for lead scoring dataset validation.

Usage:
    python scripts/validate_lead_scoring_dataset.py --csv lead_scoring_intro_v5.csv
    python scripts/validate_lead_scoring_dataset.py --csv data.csv --out-json report.json
    python scripts/validate_lead_scoring_dataset.py --csv data.csv --emit-release-snippet

Exit code 0 = all checks pass.
Exit code 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import json
import sys

from leadforge.validation.lead_scoring import ValidationConfig, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a lead scoring intro CSV dataset.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV file to validate.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Write JSON report to this path.",
    )
    parser.add_argument(
        "--emit-release-snippet",
        action="store_true",
        help="Print a markdown snippet suitable for RELEASE docs.",
    )
    parser.add_argument(
        "--enforce-1000",
        action="store_true",
        help="Fail (instead of warn) if row count != 1000.",
    )
    parser.add_argument(
        "--release",
        default=None,
        help="Path to RELEASE markdown file (currently unused, reserved).",
    )
    args = parser.parse_args()

    cfg = ValidationConfig(enforce_row_count=args.enforce_1000)
    report = validate_dataset(args.csv, cfg)

    print(report.summary())

    if args.emit_release_snippet:
        print("\n--- RELEASE SNIPPET ---\n")
        print(report.emit_release_snippet())

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nJSON report written to {args.out_json}", file=sys.stderr)

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
