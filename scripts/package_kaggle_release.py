#!/usr/bin/env python3
"""Build the Kaggle-shaped upload directory for the v1 public dataset.

PR 5.1 packaging is intentionally local and read-only with respect to
platforms: this script copies the committed public release artifacts into
``release/kaggle/``, generates ``dataset-metadata.json``, generates the
deterministic cover image, and validates the resulting shape.  Actual
Kaggle publishing belongs to Phase 7.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

DEFAULT_RELEASE_DIR: Final[Path] = Path("release")
DEFAULT_OUT_DIR: Final[Path] = Path("release/kaggle")
DEFAULT_COVER_IMAGE: Final[Path] = Path("release/dataset-cover-image.png")
DEFAULT_TIERS: Final[tuple[str, ...]] = ("intro", "intermediate", "advanced")
DEFAULT_TASK: Final[str] = "converted_within_90_days"

DATASET_ID: Final[str] = "leadforge-lead-scoring-v1"
TITLE: Final[str] = "LeadForge Lead Scoring V1"
SUBTITLE: Final[str] = "Synthetic B2B CRM funnel data for lead scoring"
EXPECTED_UPDATE_FREQUENCY: Final[str] = "never"
LICENSE_NAME: Final[str] = "MIT"
IMAGE_FILENAME: Final[str] = "dataset-cover-image.png"

KEYWORDS: Final[tuple[str, ...]] = (
    "synthetic-data",
    "lead-scoring",
    "crm",
    "b2b",
    "tabular-classification",
    "funnel-analytics",
    "marketing-analytics",
    "sales-analytics",
)
EXPECTED_UPDATE_FREQUENCIES: Final[tuple[str, ...]] = (
    "never",
    "annually",
    "quarterly",
    "monthly",
    "weekly",
    "daily",
    "hourly",
)
SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}[a-z0-9]$")


@dataclass(frozen=True)
class SchemaField:
    name: str
    type: str
    description: str | None = None


@dataclass(frozen=True)
class KaggleResource:
    path: str
    description: str
    schema_fields: tuple[SchemaField, ...]


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_feature_dictionary(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["name"]] = row
    return rows


def _kaggle_type_from_feature_dtype(dtype: str) -> str:
    lowered = dtype.lower()
    if lowered in {"bool", "boolean"}:
        return "boolean"
    if lowered in {"int", "int64", "integer"}:
        return "integer"
    if lowered in {"float", "float64", "double", "number"}:
        return "number"
    if "date" in lowered or "time" in lowered:
        return "datetime"
    return "string"


def _kaggle_type_from_arrow(dtype: pa.DataType) -> str:
    if pa.types.is_boolean(dtype):
        return "boolean"
    if pa.types.is_integer(dtype):
        return "integer"
    if pa.types.is_floating(dtype) or pa.types.is_decimal(dtype):
        return "number"
    if pa.types.is_date(dtype) or pa.types.is_timestamp(dtype) or pa.types.is_time(dtype):
        return "datetime"
    return "string"


def fields_from_feature_dictionary(
    feature_dictionary: dict[str, dict[str, str]], column_order: list[str]
) -> tuple[SchemaField, ...]:
    fields: list[SchemaField] = []
    for column in column_order:
        row = feature_dictionary.get(column)
        if row is None:
            fields.append(SchemaField(name=column, type="string"))
            continue
        description = row.get("description") or None
        fields.append(
            SchemaField(
                name=column,
                type=_kaggle_type_from_feature_dtype(row.get("dtype", "string")),
                description=description,
            )
        )
    return tuple(fields)


def fields_from_parquet(path: Path) -> tuple[SchemaField, ...]:
    schema = pq.read_schema(path)
    return tuple(
        SchemaField(name=field.name, type=_kaggle_type_from_arrow(field.type)) for field in schema
    )


def fields_from_csv(path: Path) -> tuple[SchemaField, ...]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    return tuple(SchemaField(name=str(column), type="string") for column in columns)


def _resource(path: Path, release_dir: Path, description: str) -> KaggleResource:
    relative = path.relative_to(release_dir).as_posix()
    if path.suffix == ".parquet":
        fields = fields_from_parquet(path)
    elif path.name == "lead_scoring.csv":
        tier_dir = path.parent
        feature_dictionary = _read_feature_dictionary(tier_dir / "feature_dictionary.csv")
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        fields = fields_from_feature_dictionary(feature_dictionary, [str(c) for c in columns])
    elif path.suffix == ".csv":
        fields = fields_from_csv(path)
    else:
        fields = ()
    return KaggleResource(path=relative, description=description, schema_fields=fields)


def discover_resources(
    release_dir: Path,
    tiers: tuple[str, ...] = DEFAULT_TIERS,
    task: str = DEFAULT_TASK,
) -> tuple[KaggleResource, ...]:
    resources: list[KaggleResource] = []
    for tier in tiers:
        tier_dir = release_dir / tier
        manifest_path = tier_dir / "manifest.json"
        feature_dictionary_path = tier_dir / "feature_dictionary.csv"
        dataset_card_path = tier_dir / "dataset_card.md"
        for required in (manifest_path, feature_dictionary_path, dataset_card_path):
            if not required.exists():
                raise FileNotFoundError(f"missing required tier artifact: {required}")

        manifest = _read_json(manifest_path)
        resources.append(
            _resource(
                tier_dir / "lead_scoring.csv",
                release_dir,
                f"{tier} flat lead-scoring CSV with train/valid/test split column.",
            )
        )
        resources.append(
            _resource(
                feature_dictionary_path,
                release_dir,
                f"{tier} feature dictionary for the flat lead-scoring task.",
            )
        )
        for table_name, table_info in manifest["tables"].items():
            resources.append(
                _resource(
                    tier_dir / table_info["file"],
                    release_dir,
                    f"{tier} snapshot-safe relational table: {table_name}.",
                )
            )
        task_dir = tier_dir / "tasks" / task
        for split in ("train", "valid", "test"):
            resources.append(
                _resource(
                    task_dir / f"{split}.parquet",
                    release_dir,
                    f"{tier} {split} parquet split for {task}.",
                )
            )
    return tuple(resources)


def _read_release_description(release_dir: Path) -> str:
    readme = (release_dir / "README.md").read_text(encoding="utf-8")
    return readme.replace("../", "https://github.com/leadforge-dev/leadforge/blob/main/")


def _resource_to_json(resource: KaggleResource) -> dict[str, Any]:
    fields = [
        {
            key: value
            for key, value in {
                "name": field.name,
                "type": field.type,
                "description": field.description,
            }.items()
            if value is not None
        }
        for field in resource.schema_fields
    ]
    return {
        "path": resource.path,
        "description": resource.description,
        "schema": {"fields": fields},
    }


def build_metadata(
    release_dir: Path,
    resources: tuple[KaggleResource, ...],
) -> dict[str, Any]:
    manifests = {tier: _read_json(release_dir / tier / "manifest.json") for tier in DEFAULT_TIERS}
    return {
        "title": TITLE,
        "subtitle": SUBTITLE,
        "id": DATASET_ID,
        "licenses": [{"name": LICENSE_NAME}],
        "expectedUpdateFrequency": EXPECTED_UPDATE_FREQUENCY,
        "keywords": list(KEYWORDS),
        "description": _read_release_description(release_dir),
        "userSpecifiedSources": [
            {
                "title": "LeadForge source repository",
                "url": "https://github.com/leadforge-dev/leadforge",
            },
            {
                "title": "Release validation report",
                "url": "https://github.com/leadforge-dev/leadforge/tree/main/release/validation",
            },
        ],
        "image": IMAGE_FILENAME,
        "resources": [_resource_to_json(resource) for resource in resources],
        "leadforge": {
            "datasetSlug": DATASET_ID,
            "recipe": manifests["intermediate"]["recipe_id"],
            "packageVersion": manifests["intermediate"]["package_version"],
            "bundleSchemaVersion": manifests["intermediate"]["bundle_schema_version"],
            "tiers": {
                tier: {
                    "difficulty": manifest["difficulty"],
                    "seed": manifest["seed"],
                    "nLeads": manifest["n_leads"],
                    "snapshotDay": manifest["snapshot_day"],
                    "primaryTask": manifest["primary_task"],
                }
                for tier, manifest in manifests.items()
            },
        },
    }


def generate_cover_image(path: Path) -> None:
    width, height = 1120, 560
    image = Image.new("RGB", (width, height), "#f7f3ea")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, 112), fill="#19324a")
    draw.text((64, 34), "LeadForge Lead Scoring V1", fill="#ffffff")
    draw.text((64, 72), "Synthetic CRM / AP automation funnel", fill="#d9efe7")

    band_colors = ("#45a29e", "#f2b84b", "#d45d4c")
    labels = ("Intro", "Intermediate", "Advanced")
    for i, (label, color) in enumerate(zip(labels, band_colors, strict=True)):
        x0 = 720 + i * 118
        draw.rounded_rectangle((x0, 42, x0 + 92, 84), radius=12, fill=color)
        draw.text((x0 + 16, 55), label, fill="#102027")

    funnel = [
        ((160, 165), (960, 165), (875, 245), (245, 245), "#45a29e", "5,000 leads"),
        ((260, 270), (860, 270), (780, 350), (340, 350), "#f2b84b", "Snapshot-safe CRM"),
        ((370, 375), (750, 375), (690, 455), (430, 455), "#d45d4c", "90-day conversion"),
    ]
    for left_top, right_top, right_bottom, left_bottom, color, label in funnel:
        draw.polygon((left_top, right_top, right_bottom, left_bottom), fill=color)
        cx = int((left_top[0] + right_top[0] + right_bottom[0] + left_bottom[0]) / 4)
        cy = int((left_top[1] + right_top[1] + right_bottom[1] + left_bottom[1]) / 4)
        draw.text((cx - 64, cy - 7), label, fill="#102027")

    draw.line((560, 165, 560, 455), fill="#102027", width=3)
    draw.text(
        (64, 500), "Procurement SaaS scenario | relational tables | flat ML splits", fill="#19324a"
    )
    draw.text(
        (722, 500),
        "Designed deterministically for Kaggle header and thumbnail crops",
        fill="#19324a",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _normalize_text_eofs(root: Path) -> None:
    text_suffixes = {".csv", ".json", ".md"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in text_suffixes and path.name != "LICENSE":
            continue
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            path.write_bytes(data + b"\n")


def build_upload_dir(
    release_dir: Path,
    out_dir: Path,
    cover_image: Path,
    *,
    tiers: tuple[str, ...] = DEFAULT_TIERS,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for filename in ("README.md", "LICENSE"):
        _copy_path(release_dir / filename, out_dir / filename)
    _copy_path(cover_image, out_dir / IMAGE_FILENAME)

    for tier in tiers:
        tier_src = release_dir / tier
        tier_dst = out_dir / tier
        shutil.copytree(tier_src, tier_dst)
    _normalize_text_eofs(out_dir)


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    title = str(metadata.get("title", ""))
    subtitle = str(metadata.get("subtitle", ""))
    dataset_id = str(metadata.get("id", ""))
    licenses = metadata.get("licenses")
    frequency = metadata.get("expectedUpdateFrequency")
    if not 6 <= len(title) <= 50:
        errors.append("title must be 6-50 characters")
    if not 20 <= len(subtitle) <= 80:
        errors.append("subtitle must be 20-80 characters")
    if not 3 <= len(dataset_id) <= 50 or not SLUG_PATTERN.fullmatch(dataset_id):
        errors.append("id must be a 3-50 character lowercase slug")
    if licenses != [{"name": LICENSE_NAME}]:
        errors.append("licenses must contain exactly one MIT entry")
    if frequency not in EXPECTED_UPDATE_FREQUENCIES:
        errors.append("expectedUpdateFrequency is not approved")
    resources = metadata.get("resources")
    if not isinstance(resources, list) or not resources:
        errors.append("resources must be a non-empty list")
    else:
        for index, resource in enumerate(resources):
            fields = resource.get("schema", {}).get("fields", [])
            if not fields:
                errors.append(f"resources[{index}] must include schema.fields")
            for field in fields:
                if not field.get("name") or not field.get("type"):
                    errors.append(f"resources[{index}] has a field without name/type")
    if not metadata.get("userSpecifiedSources"):
        errors.append("userSpecifiedSources must not be empty")
    if metadata.get("image") != IMAGE_FILENAME:
        errors.append(f"image must be {IMAGE_FILENAME}")
    return errors


def validate_cover_image(path: Path) -> list[str]:
    if not path.exists():
        return [f"cover image missing: {path}"]
    with Image.open(path) as image:
        width, height = image.size
    if width < 560 or height < 280:
        return [f"cover image must be at least 560x280; got {width}x{height}"]
    return []


def validate_upload_dir(out_dir: Path, metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for relative in ("dataset-metadata.json", "README.md", "LICENSE", IMAGE_FILENAME):
        if not (out_dir / relative).exists():
            errors.append(f"missing Kaggle upload file: {relative}")
    for resource in metadata.get("resources", []):
        path = out_dir / resource.get("path", "")
        if not path.exists():
            errors.append(f"resource path missing from upload dir: {resource.get('path')}")
    return errors


def package_release(
    release_dir: Path = DEFAULT_RELEASE_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    cover_image: Path = DEFAULT_COVER_IMAGE,
) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    out_dir = out_dir.resolve()
    cover_image = cover_image.resolve()
    if not release_dir.exists():
        raise FileNotFoundError(f"release directory not found: {release_dir}")

    generate_cover_image(cover_image)
    resources = discover_resources(release_dir)
    metadata = build_metadata(release_dir, resources)
    errors = [
        *validate_metadata(metadata),
        *validate_cover_image(cover_image),
    ]
    if errors:
        raise ValueError("; ".join(errors))

    build_upload_dir(release_dir, out_dir, cover_image)
    _write_json(out_dir / "dataset-metadata.json", metadata)
    errors = validate_upload_dir(out_dir, metadata)
    if errors:
        raise ValueError("; ".join(errors))
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cover-image", type=Path, default=DEFAULT_COVER_IMAGE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the local Kaggle package without uploading.",
    )
    parser.add_argument("--print", action="store_true", help="Print generated metadata JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metadata = package_release(args.release_dir, args.out_dir, args.cover_image)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.print:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    mode = "dry-run" if args.dry_run else "package"
    print(f"Kaggle {mode} validated: {args.out_dir / 'dataset-metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
