#!/usr/bin/env python3
"""Lint canonical Kaggle / Hugging Face metadata before preview or publish.

This is the explicit diff gate between the two platform artifacts that
drive publication:

* ``release/kaggle/dataset-metadata.json``
* ``release/huggingface/README.md``

The preview renderers intentionally read those files directly. This
script catches the cases where the canonical files themselves drift:
private Kaggle metadata, license / task / tag mismatches, HF splits
that are absent from the Kaggle resource list, task-split schema drift,
and missing per-tier inputs needed by offline reviewers.

Exit codes: 0 pass / 1 lint failure / 2 pre-flight error.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

# Make ``scripts/`` importable regardless of whether this file is run
# as ``python scripts/lint_platform_metadata.py`` or imported from tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _release_common import (  # noqa: E402
    AGENT_REVIEWABLE_DOC_FILES,
    AGENT_REVIEWABLE_DOCS_DIR,
    AGENT_REVIEWABLE_ROOT_FILES,
)
from package_hf_release import DEFAULT_TAGS as DEFAULT_HF_TAGS  # noqa: E402
from package_kaggle_release import (  # noqa: E402
    BUNDLE_TABLES,
    DEFAULT_TASK,
    DEFAULT_TIERS,
    fields_from_parquet,
)
from package_kaggle_release import (
    DEFAULT_KEYWORDS as DEFAULT_KAGGLE_KEYWORDS,
)

DEFAULT_RELEASE_DIR: Final[Path] = Path("release")
HF_SPLIT_TO_FILE_SPLIT: Final[dict[str, str]] = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}
REQUIRED_COMMON_TAGS: Final[frozenset[str]] = frozenset(
    {"b2b", "crm", "lead-scoring", "synthetic-data", "tabular"}
)
EXPECTED_KAGGLE_KEYWORDS: Final[frozenset[str]] = frozenset(DEFAULT_KAGGLE_KEYWORDS)
EXPECTED_HF_TAGS: Final[frozenset[str]] = frozenset(DEFAULT_HF_TAGS)
REQUIRED_HF_TASK: Final[str] = "tabular-classification"
REQUIRED_LICENSES: Final[dict[str, str]] = {
    "kaggle": "MIT",
    "hf": "mit",
}
REQUIRED_TIER_RESOURCES: Final[tuple[str, ...]] = (
    "lead_scoring.csv",
    "feature_dictionary.csv",
    "dataset_card.md",
    "metrics.json",
    "manifest.json",
    f"tasks/{DEFAULT_TASK}/train.parquet",
    f"tasks/{DEFAULT_TASK}/valid.parquet",
    f"tasks/{DEFAULT_TASK}/test.parquet",
    *(f"tables/{table}.parquet" for table in BUNDLE_TABLES),
)
REQUIRED_ROOT_RESOURCES: Final[tuple[str, ...]] = (
    *(rel for rel, required in AGENT_REVIEWABLE_ROOT_FILES if required),
    *(f"{AGENT_REVIEWABLE_DOCS_DIR}/{filename}" for filename in AGENT_REVIEWABLE_DOC_FILES),
)

_FRONTMATTER_RE: Final[re.Pattern[str]] = re.compile(
    r"\A---\n(?P<yaml>.*?)\n---\n(?P<body>.*)\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class LintFinding:
    """One platform-metadata mismatch."""

    field: str
    message: str


@dataclass(frozen=True)
class LintOutcome:
    """Return value from :func:`run_lint`."""

    findings: tuple[LintFinding, ...]

    @property
    def ok(self) -> bool:
        return not self.findings


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_hf_frontmatter(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing HF README artifact: {path}")
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"{path} is missing YAML frontmatter")
    value = yaml.safe_load(match.group("yaml")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} frontmatter is not a YAML mapping")
    return value


def _resource_map(kaggle_metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    resources = kaggle_metadata.get("resources", [])
    if not isinstance(resources, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for resource in resources:
        if isinstance(resource, dict) and isinstance(resource.get("path"), str):
            out[resource["path"]] = resource
    return out


def _field_signature(resource: dict[str, Any]) -> tuple[tuple[str, str], ...] | None:
    schema = resource.get("schema")
    if not isinstance(schema, dict):
        return None
    fields = schema.get("fields")
    if not isinstance(fields, list):
        return None
    signature: list[tuple[str, str]] = []
    for field in fields:
        if not isinstance(field, dict):
            return None
        name = field.get("name")
        field_type = field.get("type")
        if not isinstance(name, str) or not isinstance(field_type, str):
            return None
        signature.append((name, field_type))
    return tuple(signature)


def _field_names(resource: dict[str, Any]) -> tuple[str, ...] | None:
    signature = _field_signature(resource)
    if signature is None:
        return None
    return tuple(name for name, _ in signature)


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str)]


def _hf_configs(frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    configs = frontmatter.get("configs")
    if not isinstance(configs, list):
        return []
    return [c for c in configs if isinstance(c, dict)]


def _iter_hf_data_files(configs: Iterable[dict[str, Any]]) -> Iterable[tuple[str, str, str]]:
    for config in configs:
        config_name = config.get("config_name")
        data_files = config.get("data_files")
        if not isinstance(config_name, str) or not isinstance(data_files, list):
            continue
        for data_file in data_files:
            if not isinstance(data_file, dict):
                continue
            split = data_file.get("split")
            path = data_file.get("path")
            if isinstance(split, str) and isinstance(path, str):
                yield config_name, split, path


def _expected_hf_data_files(tier: str, *, task: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (hf_split, f"{tier}/tasks/{task}/{file_split}.parquet")
        for hf_split, file_split in HF_SPLIT_TO_FILE_SPLIT.items()
    )


def _lint_privacy_license_task_tags(
    kaggle_metadata: dict[str, Any],
    hf_frontmatter: dict[str, Any],
) -> list[LintFinding]:
    findings: list[LintFinding] = []

    if kaggle_metadata.get("isPrivate") is not False:
        findings.append(
            LintFinding(
                "kaggle.isPrivate",
                "expected false so the preview catches the private-publish blocker",
            )
        )

    kaggle_licenses = kaggle_metadata.get("licenses")
    kaggle_license = None
    if (
        isinstance(kaggle_licenses, list)
        and kaggle_licenses
        and isinstance(kaggle_licenses[0], dict)
    ):
        kaggle_license = kaggle_licenses[0].get("name")
    if kaggle_license != REQUIRED_LICENSES["kaggle"]:
        findings.append(
            LintFinding(
                "kaggle.licenses[0].name",
                f"expected {REQUIRED_LICENSES['kaggle']!r}, got {kaggle_license!r}",
            )
        )

    hf_license = hf_frontmatter.get("license")
    if hf_license != REQUIRED_LICENSES["hf"]:
        findings.append(
            LintFinding(
                "hf.license",
                f"expected {REQUIRED_LICENSES['hf']!r}, got {hf_license!r}",
            )
        )

    hf_tasks = set(_as_str_list(hf_frontmatter.get("task_categories")))
    if REQUIRED_HF_TASK not in hf_tasks:
        findings.append(
            LintFinding(
                "hf.task_categories",
                f"must contain {REQUIRED_HF_TASK!r}",
            )
        )

    kaggle_keywords = set(_as_str_list(kaggle_metadata.get("keywords")))
    hf_tags = set(_as_str_list(hf_frontmatter.get("tags")))
    missing_kaggle_tags = sorted(REQUIRED_COMMON_TAGS - kaggle_keywords)
    missing_hf_tags = sorted(REQUIRED_COMMON_TAGS - hf_tags)
    if missing_kaggle_tags:
        findings.append(
            LintFinding(
                "kaggle.keywords",
                f"missing common topical tag(s): {missing_kaggle_tags}",
            )
        )
    if missing_hf_tags:
        findings.append(
            LintFinding(
                "hf.tags",
                f"missing common topical tag(s): {missing_hf_tags}",
            )
        )

    if kaggle_keywords != EXPECTED_KAGGLE_KEYWORDS:
        findings.append(
            LintFinding(
                "kaggle.keywords",
                (
                    "expected exact keyword set "
                    f"{sorted(EXPECTED_KAGGLE_KEYWORDS)!r}, got {sorted(kaggle_keywords)!r}"
                ),
            )
        )
    if hf_tags != EXPECTED_HF_TAGS:
        findings.append(
            LintFinding(
                "hf.tags",
                f"expected exact tag set {sorted(EXPECTED_HF_TAGS)!r}, got {sorted(hf_tags)!r}",
            )
        )

    # The HF task category should be echoed on Kaggle through the two
    # searchable keywords Kaggle actually exposes for this release.
    for keyword in ("classification", "tabular"):
        if keyword not in kaggle_keywords:
            findings.append(
                LintFinding(
                    "kaggle.keywords",
                    f"missing task-discovery keyword {keyword!r}",
                )
            )

    return findings


def _lint_hf_configs(
    configs: list[dict[str, Any]],
    *,
    tiers: Sequence[str],
    task: str,
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    names = [c.get("config_name") for c in configs if isinstance(c.get("config_name"), str)]
    if names != list(tiers):
        findings.append(
            LintFinding(
                "hf.configs",
                f"expected config order {list(tiers)!r}, got {names!r}",
            )
        )

    defaults = [c.get("config_name") for c in configs if c.get("default") is True]
    if len(defaults) != 1:
        findings.append(
            LintFinding(
                "hf.configs",
                f"expected exactly one default config, got {defaults!r}",
            )
        )

    by_name = {
        str(config.get("config_name")): config
        for config in configs
        if isinstance(config.get("config_name"), str)
    }
    for tier in tiers:
        config = by_name.get(tier)
        if config is None:
            continue
        data_files = config.get("data_files")
        if not isinstance(data_files, list):
            findings.append(
                LintFinding(
                    "hf.configs.data_files",
                    f"{tier} must declare data_files as a list",
                )
            )
            continue
        actual: list[tuple[str, str]] = []
        malformed = 0
        for data_file in data_files:
            if not isinstance(data_file, dict):
                malformed += 1
                continue
            split = data_file.get("split")
            path = data_file.get("path")
            if not isinstance(split, str) or not isinstance(path, str):
                malformed += 1
                continue
            actual.append((split, path))
        if malformed:
            findings.append(
                LintFinding(
                    "hf.configs.data_files",
                    f"{tier} has {malformed} malformed data_files entrie(s)",
                )
            )
        expected = list(_expected_hf_data_files(tier, task=task))
        if actual != expected:
            findings.append(
                LintFinding(
                    "hf.configs.data_files",
                    f"{tier} data_files expected {expected!r}, got {actual!r}",
                )
            )

    return findings


def _lint_resource_coverage(
    resources: dict[str, dict[str, Any]],
    *,
    tiers: Sequence[str],
    task: str,
) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for path in REQUIRED_ROOT_RESOURCES:
        if path not in resources:
            findings.append(
                LintFinding(
                    "kaggle.resources",
                    f"missing agent-reviewable root artifact {path!r}",
                )
            )
    for tier in tiers:
        for suffix in REQUIRED_TIER_RESOURCES:
            path = f"{tier}/{suffix}"
            if task != DEFAULT_TASK:
                path = path.replace(DEFAULT_TASK, task)
            if path not in resources:
                findings.append(
                    LintFinding(
                        "kaggle.resources",
                        f"missing per-tier review artifact {path!r}",
                    )
                )
    return findings


def _lint_split_and_schema_consistency(
    resources: dict[str, dict[str, Any]],
    hf_configs: list[dict[str, Any]],
    *,
    tiers: Sequence[str],
    task: str,
) -> list[LintFinding]:
    findings: list[LintFinding] = []

    for _tier, _split, path in _iter_hf_data_files(hf_configs):
        if path not in resources:
            findings.append(
                LintFinding(
                    "hf.configs.data_files",
                    f"HF data file {path!r} is absent from Kaggle resources",
                )
            )

    for tier in tiers:
        flat_path = f"{tier}/lead_scoring.csv"
        flat_fields = _field_names(resources.get(flat_path, {}))
        if flat_fields is None:
            findings.append(
                LintFinding(
                    "kaggle.resources.schema",
                    f"{flat_path!r} must declare schema.fields",
                )
            )
            continue
        if not flat_fields or flat_fields[0] != "split":
            findings.append(
                LintFinding(
                    "kaggle.resources.schema",
                    f"{flat_path!r} must expose the split column first",
                )
            )
        task_expected_fields = tuple(name for name in flat_fields if name != "split")

        split_signatures: dict[str, tuple[tuple[str, str], ...]] = {}
        for file_split in HF_SPLIT_TO_FILE_SPLIT.values():
            split_path = f"{tier}/tasks/{task}/{file_split}.parquet"
            resource = resources.get(split_path)
            if resource is None:
                continue
            signature = _field_signature(resource)
            if signature is None:
                findings.append(
                    LintFinding(
                        "kaggle.resources.schema",
                        f"{split_path!r} must declare schema.fields",
                    )
                )
                continue
            split_signatures[file_split] = signature
            split_names = tuple(name for name, _ in signature)
            if split_names != task_expected_fields:
                findings.append(
                    LintFinding(
                        "kaggle.resources.schema",
                        f"{split_path!r} schema differs from {flat_path!r} minus split",
                    )
                )

        if split_signatures:
            first_split, first_signature = next(iter(split_signatures.items()))
            for split_name, signature in split_signatures.items():
                if signature != first_signature:
                    findings.append(
                        LintFinding(
                            "kaggle.resources.schema",
                            (f"{tier} {split_name!r} schema differs from {first_split!r} schema"),
                        )
                    )

    return findings


def _flat_csv_actual_fields(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return ()
    return tuple(header)


def _parquet_actual_signature(path: Path) -> tuple[tuple[str, str], ...]:
    return tuple((field.name, field.type) for field in fields_from_parquet(path))


def _lint_actual_file_schemas(
    resources: dict[str, dict[str, Any]],
    *,
    release_dir: Path,
    tiers: Sequence[str],
    task: str,
    strict_files: bool,
) -> list[LintFinding]:
    """Compare metadata schemas to files on disk when bundle files exist.

    Fresh checkouts intentionally do not materialise the heavy tier
    bundle directories, so missing files are soft-skipped by default.
    Release-readiness jobs can pass ``--strict-files`` to turn those
    skips into failures.
    """

    findings: list[LintFinding] = []
    for tier in tiers:
        flat_rel = f"{tier}/lead_scoring.csv"
        flat_path = release_dir / flat_rel
        if not flat_path.is_file():
            if strict_files:
                findings.append(
                    LintFinding(
                        "release.files",
                        f"missing release file required for strict schema lint: {flat_rel!r}",
                    )
                )
            continue

        actual_flat_fields = _flat_csv_actual_fields(flat_path)
        declared_flat_fields = _field_names(resources.get(flat_rel, {}))
        if declared_flat_fields != actual_flat_fields:
            findings.append(
                LintFinding(
                    "kaggle.resources.schema",
                    (
                        f"{flat_rel!r} metadata fields differ from actual CSV header: "
                        f"declared={declared_flat_fields!r}, actual={actual_flat_fields!r}"
                    ),
                )
            )

        for file_split in HF_SPLIT_TO_FILE_SPLIT.values():
            split_rel = f"{tier}/tasks/{task}/{file_split}.parquet"
            split_path = release_dir / split_rel
            if not split_path.is_file():
                if strict_files:
                    findings.append(
                        LintFinding(
                            "release.files",
                            (
                                "missing release file required for strict schema lint: "
                                f"{split_rel!r}"
                            ),
                        )
                    )
                continue
            declared = _field_signature(resources.get(split_rel, {}))
            actual = _parquet_actual_signature(split_path)
            if declared != actual:
                findings.append(
                    LintFinding(
                        "kaggle.resources.schema",
                        (
                            f"{split_rel!r} metadata schema differs from actual parquet "
                            f"schema: declared={declared!r}, actual={actual!r}"
                        ),
                    )
                )

    return findings


def lint_metadata(
    kaggle_metadata: dict[str, Any],
    hf_frontmatter: dict[str, Any],
    *,
    tiers: Sequence[str] = DEFAULT_TIERS,
    task: str = DEFAULT_TASK,
    release_dir: Path | None = None,
    strict_files: bool = False,
) -> LintOutcome:
    """Run all platform-metadata lint checks against parsed artifacts."""

    findings: list[LintFinding] = []
    resources = _resource_map(kaggle_metadata)
    configs = _hf_configs(hf_frontmatter)

    if not resources:
        findings.append(LintFinding("kaggle.resources", "must contain resource objects"))
    if not configs:
        findings.append(LintFinding("hf.configs", "must contain config objects"))

    findings.extend(_lint_privacy_license_task_tags(kaggle_metadata, hf_frontmatter))
    findings.extend(_lint_hf_configs(configs, tiers=tiers, task=task))
    findings.extend(_lint_resource_coverage(resources, tiers=tiers, task=task))
    findings.extend(_lint_split_and_schema_consistency(resources, configs, tiers=tiers, task=task))
    if release_dir is not None:
        findings.extend(
            _lint_actual_file_schemas(
                resources,
                release_dir=release_dir,
                tiers=tiers,
                task=task,
                strict_files=strict_files,
            )
        )
    return LintOutcome(findings=tuple(findings))


def run_lint(
    release_dir: Path,
    *,
    tiers: Sequence[str] = DEFAULT_TIERS,
    task: str = DEFAULT_TASK,
    strict_files: bool = False,
) -> LintOutcome:
    """Load canonical artifacts from ``release_dir`` and lint them."""

    kaggle_metadata = _load_json_object(release_dir / "kaggle" / "dataset-metadata.json")
    hf_frontmatter = _load_hf_frontmatter(release_dir / "huggingface" / "README.md")
    return lint_metadata(
        kaggle_metadata,
        hf_frontmatter,
        tiers=tiers,
        task=task,
        release_dir=release_dir,
        strict_files=strict_files,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lint_platform_metadata",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help="release tree containing kaggle/ and huggingface/ artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        default=None,
        help="tier/config to validate (repeatable; default: intro/intermediate/advanced)",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="task directory under each tier (default: %(default)s)",
    )
    parser.add_argument(
        "--strict-files",
        action="store_true",
        help=(
            "fail if tier CSV/parquet files are missing instead of soft-skipping "
            "file-backed schema checks"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tiers = tuple(args.tiers) if args.tiers else DEFAULT_TIERS
    try:
        outcome = run_lint(
            args.release_dir,
            tiers=tiers,
            task=args.task,
            strict_files=args.strict_files,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not outcome.ok:
        for finding in outcome.findings:
            print(f"{finding.field}: {finding.message}", file=sys.stderr)
        return 1
    print("platform metadata lint passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
