#!/usr/bin/env python3
"""Build a ShmuggingFace review minisite from leadforge release artifacts.

Reads the three public release tiers (intro / intermediate / advanced),
renders the release README to HTML, and generates a static site via
ShmuggingFaceCore that mirrors how the dataset will look on Kaggle and
Hugging Face.  The site can then be deployed to Cloudflare Pages.

Usage::

    python scripts/build_shmuggingface_site.py [OPTIONS]

Options
-------
--release-dir PATH
    Root of the release directory.  Default: ``release/``.
--out-dir PATH
    Output directory for the generated static site.
    Default: ``release/_shmuggingface/dist``.
--smf-core PATH
    Path to a local ShmuggingFaceCore checkout.  If absent the repo is
    cloned to ``/tmp/shmuggingface-core`` (and pulled on subsequent runs).
--deploy
    Deploy the built site to Cloudflare Pages after building.
--cf-env PATH
    Cloudflare env file to source before wrangler.
    Default: ``~/.config/adanim/cloudflare_api_token.env``.
--project-name NAME
    Cloudflare Pages project name.
    Default: ``leadforge-lead-scoring-v1-preview``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from markdown_it import MarkdownIt
except ImportError:
    sys.exit("markdown-it-py is required: pip install -e '.[publish]'")

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIERS = ["intro", "intermediate", "advanced"]
TASK = "converted_within_90_days"

GITHUB_BLOB_BASE = "https://github.com/leadforge-dev/leadforge/blob/main"
SMF_CORE_REPO = "https://github.com/ShmuggingFace/ShmuggingFaceCore.git"
SMF_CORE_CACHE = Path("/tmp/shmuggingface-core")
DEFAULT_CF_ENV = Path.home() / ".config/adanim/cloudflare_api_token.env"
DEFAULT_PROJECT = "leadforge-lead-scoring-v1-preview"

TIER_LABEL = {"intro": "Intro", "intermediate": "Intermediate", "advanced": "Advanced"}
TIER_USABILITY = {"intro": "9.4", "intermediate": "9.1", "advanced": "8.9"}
TIER_MEDAL = {"intro": "Gold", "intermediate": "Silver", "advanced": "Bronze"}

DISCUSSIONS = [
    "What is `snapshot_day = 30` and how does it affect which features are valid at inference time?",
    "Is `total_touches_all` a safe feature or a time-window leakage trap?",
    "LR and GBM AUCs are very close across tiers — does relational feature engineering help?",
    "How would you set a probability threshold for a team that can only work 50 leads per week?",
    "What happens to AUC when you evaluate on a chronological hold-out instead of a random split?",
]

# ---------------------------------------------------------------------------
# README rendering
# ---------------------------------------------------------------------------

_PARENT_LINK_RE = re.compile(r"\]\(\.\./([^)]+)\)")
_VALIDATION_LINK_RE = re.compile(r"\]\(validation/validation_report\.md\)")


def _rewrite_links(text: str) -> str:
    """Rewrite relative markdown links to GitHub blob URLs."""
    text = _PARENT_LINK_RE.sub(rf"]({GITHUB_BLOB_BASE}/\1)", text)
    text = _VALIDATION_LINK_RE.sub(
        f"]({GITHUB_BLOB_BASE}/release/validation/validation_report.md)", text
    )
    return text


def render_readme_html(release_dir: Path) -> str:
    """Render release/README.md to HTML with link rewriting."""
    readme_text = (release_dir / "README.md").read_text(encoding="utf-8")
    readme_text = _rewrite_links(readme_text)
    md = MarkdownIt("gfm-like").disable("linkify")
    return md.render(readme_text)


# ---------------------------------------------------------------------------
# Tier metadata loading
# ---------------------------------------------------------------------------


def load_tier(release_dir: Path, tier: str) -> dict:
    """Load manifest, metrics, feature dictionary, and sample rows for one tier."""
    tier_dir = release_dir / tier
    manifest = json.loads((tier_dir / "manifest.json").read_text())
    metrics = json.loads((tier_dir / "metrics.json").read_text())

    fd = pd.read_csv(tier_dir / "feature_dictionary.csv")
    columns = list(fd["name"])

    df = pd.read_csv(tier_dir / "lead_scoring.csv")
    # Stringify every cell so JSON serialization is clean
    sample_rows = [
        {k: ("" if str(v) in ("nan", "None") else str(v)) for k, v in row.items()}
        for row in df.head(8).to_dict("records")
    ]

    return {
        "tier": tier,
        "tier_dir": tier_dir,
        "task_dir": tier_dir / "tasks" / TASK,
        "manifest": manifest,
        "metrics": metrics,
        "columns": columns,
        "sample_rows": sample_rows,
        "n_rows": int(df.shape[0]),
    }


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------


def _rel(path: Path, from_dir: Path) -> str:
    """Relative POSIX path from from_dir to path."""
    return os.path.relpath(path, from_dir).replace(os.sep, "/")


def make_dataset_config(tier_data: dict, config_dir: Path, readme_html: str) -> dict:
    """Build a ShmuggingFace dataset config dict for one tier."""
    tier = tier_data["tier"]
    tier_dir = tier_data["tier_dir"]
    task_dir = tier_data["task_dir"]
    manifest = tier_data["manifest"]
    metrics = tier_data["metrics"]
    label = TIER_LABEL[tier]
    medians = metrics.get("medians", {})

    cr = medians.get("conversion_rate_test", 0.0)
    lr_auc = medians.get("lr_auc", 0.0)
    n_leads = manifest.get("n_leads", 5000)
    snapshot_day = manifest.get("snapshot_day", 30)

    task_info = manifest.get("tasks", {}).get(TASK, {})
    train_rows = task_info.get("train_rows", 0)
    valid_rows = task_info.get("valid_rows", 0)
    test_rows = task_info.get("test_rows", 0)

    def kb(path: Path) -> str:
        return f"{max(1, path.stat().st_size // 1024)} KB"

    files = [
        {
            "path": "lead_scoring.csv",
            "size": kb(tier_dir / "lead_scoring.csv"),
            "kind": "CSV",
            "sourcePath": _rel(tier_dir / "lead_scoring.csv", config_dir),
            "about": (
                f"Flat ML-ready snapshot CSV: {n_leads:,} leads × "
                f"{len(tier_data['columns'])} features, "
                f"snapshot day {snapshot_day}.  Includes a 'split' column "
                f"(train / valid / test) for conventional ML workflows."
            ),
        },
        {
            "path": "feature_dictionary.csv",
            "size": kb(tier_dir / "feature_dictionary.csv"),
            "kind": "CSV",
            "sourcePath": _rel(tier_dir / "feature_dictionary.csv", config_dir),
            "about": (
                "Per-column documentation: dtype, analytical category, "
                "leakage-risk flag, and plain-language description."
            ),
        },
        {
            "path": "tasks/converted_within_90_days/train.parquet",
            "size": kb(task_dir / "train.parquet"),
            "kind": "Parquet",
            "sourcePath": _rel(task_dir / "train.parquet", config_dir),
            "about": (
                f"Training split — {train_rows:,} leads, "
                f"stratified by conversion rate.  Target column: "
                f"`converted_within_90_days` (bool)."
            ),
        },
        {
            "path": "tasks/converted_within_90_days/valid.parquet",
            "size": kb(task_dir / "valid.parquet"),
            "kind": "Parquet",
            "sourcePath": _rel(task_dir / "valid.parquet", config_dir),
            "about": f"Validation split — {valid_rows:,} leads.",
        },
        {
            "path": "tasks/converted_within_90_days/test.parquet",
            "size": kb(task_dir / "test.parquet"),
            "kind": "Parquet",
            "sourcePath": _rel(task_dir / "test.parquet", config_dir),
            "about": (f"Test split — {test_rows:,} leads, held out for final evaluation only."),
        },
        {
            "path": "dataset_card.md",
            "size": kb(tier_dir / "dataset_card.md"),
            "kind": "Dataset card",
            "sourcePath": _rel(tier_dir / "dataset_card.md", config_dir),
            "about": "Auto-generated tier-specific dataset card.",
        },
    ]

    cover_rel = _rel(tier_dir.parent / "dataset-cover-image.png", config_dir)

    return {
        "slug": f"leadforge-lead-scoring-v1-{tier}",
        "title": f"LeadForge Lead Scoring v1 — {label}",
        "owner": "leadforge-dev",
        "subtitle": (
            f"{label} difficulty · {n_leads:,} leads · ~{cr:.0%} conversion rate · "
            f"LR AUC {lr_auc:.3f} (5-seed median)"
        ),
        "license": "MIT",
        "task": "tabular-classification",
        "language": "English",
        "rowCount": n_leads,
        "splits": ["train", "valid", "test"],
        "subsets": ["leadforge-lead-scoring-v1"],
        "coverImage": cover_rel,
        "descriptionHtml": readme_html,
        "tags": [
            "tabular",
            "lead-scoring",
            "synthetic-data",
            "crm",
            "b2b",
            "datasets",
            "pandas",
            tier,
        ],
        "columns": tier_data["columns"],
        "rows": tier_data["sample_rows"],
        "files": files,
        "discussions": DISCUSSIONS,
        "downloads": "0",
        "likes": "0",
        "kaggleUsability": TIER_USABILITY[tier],
        "kaggleMedals": TIER_MEDAL[tier],
    }


# ---------------------------------------------------------------------------
# Config file writing
# ---------------------------------------------------------------------------


def write_config(site_config: dict, datasets: list[dict], config_path: Path) -> None:
    """Write shmuggingface.config.mjs."""
    full_config = {"site": site_config, "datasets": datasets}
    config_json = json.dumps(full_config, indent=2, ensure_ascii=False)
    config_path.write_text(f"export default {config_json};\n", encoding="utf-8")
    print(f"  Config → {config_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# ShmuggingFaceCore management
# ---------------------------------------------------------------------------


def ensure_smf_core(smf_core: Path | None) -> Path:
    """Return path to a working ShmuggingFaceCore checkout, cloning if needed."""
    if smf_core is not None:
        entry = smf_core / "bin/shmuggingface.mjs"
        if not entry.exists():
            sys.exit(f"ShmuggingFaceCore entry point not found at {entry}")
        return smf_core

    entry = SMF_CORE_CACHE / "bin/shmuggingface.mjs"
    if SMF_CORE_CACHE.exists() and entry.exists():
        print(f"  Updating ShmuggingFaceCore cache at {SMF_CORE_CACHE}", file=sys.stderr)
        subprocess.run(
            ["git", "-C", str(SMF_CORE_CACHE), "pull", "--quiet"],
            check=False,
        )
    else:
        print(f"  Cloning ShmuggingFaceCore → {SMF_CORE_CACHE}", file=sys.stderr)
        subprocess.run(
            ["git", "clone", "--depth=1", SMF_CORE_REPO, str(SMF_CORE_CACHE)],
            check=True,
        )
    return SMF_CORE_CACHE


# ---------------------------------------------------------------------------
# Build and deploy
# ---------------------------------------------------------------------------


def build_site(config_path: Path, out_dir: Path, smf_core: Path) -> None:
    """Run the ShmuggingFaceCore generator."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Building static site → {out_dir}", file=sys.stderr)
    subprocess.run(  # noqa: S603, S607
        [
            "node",
            str(smf_core / "bin/shmuggingface.mjs"),
            "build",
            "--config",
            str(config_path),
            "--out",
            str(out_dir),
        ],
        check=True,
    )


def _load_cf_env(cf_env_path: Path) -> dict:
    """Parse a shell env file and return a dict of variable overrides."""
    env = os.environ.copy()
    for raw_line in cf_env_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" in line:
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip("'\"")
    return env


def deploy_site(out_dir: Path, project_name: str, cf_env_path: Path) -> None:
    """Deploy the built site to Cloudflare Pages via wrangler."""
    if not cf_env_path.exists():
        sys.exit(
            f"Cloudflare env file not found: {cf_env_path}\n"
            f"Expected format:\n"
            f"  export CLOUDFLARE_ACCOUNT_ID='...'\n"
            f"  export CLOUDFLARE_API_TOKEN='...'"
        )

    env = _load_cf_env(cf_env_path)
    account_id = env.get("CLOUDFLARE_ACCOUNT_ID", "(not set)")
    print(
        f"  Deploying to Cloudflare Pages\n"
        f"    project : {project_name}\n"
        f"    account : {account_id}\n"
        f"    source  : {out_dir}",
        file=sys.stderr,
    )
    result = subprocess.run(
        [
            "wrangler",
            "pages",
            "deploy",
            str(out_dir),
            "--project-name",
            project_name,
            "--branch",
            "main",  # deploy to production slot, not a branch preview
            "--commit-dirty=true",  # suppress the "uncommitted changes" warning
        ],
        env=env,
    )
    if result.returncode != 0:
        sys.exit(f"Deployment failed (wrangler exit code {result.returncode})")

    print(
        f"\n  Live at: https://{project_name}.pages.dev",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build (and optionally deploy) the ShmuggingFace review minisite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--release-dir",
        default="release",
        type=Path,
        metavar="PATH",
        help="Root of the release directory (default: release/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        metavar="PATH",
        help="Output directory for the static site (default: release/_shmuggingface/dist)",
    )
    parser.add_argument(
        "--smf-core",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a local ShmuggingFaceCore checkout (auto-cloned if absent)",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy to Cloudflare Pages after building",
    )
    parser.add_argument(
        "--cf-env",
        type=Path,
        default=DEFAULT_CF_ENV,
        metavar="PATH",
        help=f"Cloudflare env file (default: {DEFAULT_CF_ENV})",
    )
    parser.add_argument(
        "--project-name",
        default=DEFAULT_PROJECT,
        metavar="NAME",
        help=f"Cloudflare Pages project name (default: {DEFAULT_PROJECT})",
    )
    args = parser.parse_args()

    release_dir = args.release_dir.resolve()
    if not release_dir.is_dir():
        sys.exit(f"Release directory not found: {release_dir}")

    config_dir = release_dir / "_shmuggingface"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "shmuggingface.config.mjs"
    out_dir = args.out_dir.resolve() if args.out_dir else (config_dir / "dist")

    # --- Render README -------------------------------------------------------
    print("Rendering README.md → HTML …", file=sys.stderr)
    readme_html = render_readme_html(release_dir)
    print(f"  {len(readme_html):,} bytes of HTML", file=sys.stderr)

    # --- Load tiers ----------------------------------------------------------
    print("Loading release tiers …", file=sys.stderr)
    datasets = []
    for tier in TIERS:
        print(f"  {tier}", file=sys.stderr)
        tier_data = load_tier(release_dir, tier)
        ds = make_dataset_config(tier_data, config_dir, readme_html)
        datasets.append(ds)

    # --- Write config --------------------------------------------------------
    print("Writing shmuggingface.config.mjs …", file=sys.stderr)
    site_config = {
        "title": "LeadForge Lead Scoring v1 — Pre-Publication Review",
        "owner": "leadforge-dev",
        "visibility": "Pre-publication review mock — not yet live on Kaggle or Hugging Face",
        "reviewerHint": (
            "Review the dataset card copy, metadata accuracy, file listings, column "
            "preview, and download behaviour across all three difficulty tiers. "
            "The Shmaggle tab mirrors the Kaggle page; the ShmuggingFace tab mirrors "
            "the Hugging Face page.  Flag anything that looks wrong before the real publish."
        ),
    }
    write_config(site_config, datasets, config_path)

    # --- Ensure ShmuggingFaceCore --------------------------------------------
    smf_core = ensure_smf_core(args.smf_core)

    # --- Build ---------------------------------------------------------------
    print("Building static site …", file=sys.stderr)
    build_site(config_path, out_dir, smf_core)
    print(f"Done.  Site at: {out_dir}", file=sys.stderr)

    # --- Deploy --------------------------------------------------------------
    if args.deploy:
        print("Deploying to Cloudflare Pages …", file=sys.stderr)
        deploy_site(out_dir, args.project_name, args.cf_env)


if __name__ == "__main__":
    main()
