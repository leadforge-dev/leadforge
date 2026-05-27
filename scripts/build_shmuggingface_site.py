#!/usr/bin/env python3
"""Build a ShmuggingFace review minisite from leadforge release artifacts.

Reads the three public release tiers (intro / intermediate / advanced),
renders each tier's ``dataset_card.md`` to HTML, and generates a static
site via ShmuggingFaceCore that mirrors how the dataset will look on
Kaggle and Hugging Face.  The site can then be deployed to Cloudflare
Pages.

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
    Path to a local ShmuggingFaceCore checkout.  Overrides the default,
    which is the npm-installed package at ``node_modules/@shmuggingface/core``
    (pinned to v1.0.2 via ``package.json``).  Run ``npm install`` first.
--config-only
    Write the ``shmuggingface.config.mjs`` file and stop — do not invoke
    the Node build.  Useful for inspecting generated config without a
    full Node environment.
--deploy
    Deploy the built site to Cloudflare Pages after building.
--production
    With ``--deploy``: push to the production slot (``--branch main``).
    Default (without this flag) is a branch preview
    (``--branch preview``).  Using ``--production`` intentionally
    requires a separate flag so a local run never clobbers the live
    site by accident.
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
from typing import Any, TypedDict

try:
    from markdown_it import MarkdownIt
except ImportError:
    sys.exit("markdown-it-py is required: pip install -e '.[dev]'")

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIERS = ["intro", "intermediate", "advanced"]
TASK = "converted_within_90_days"

GITHUB_BLOB_BASE = "https://github.com/leadforge-dev/leadforge/blob/main"
GITHUB_BLOB_RELEASE = f"{GITHUB_BLOB_BASE}/release"
# Pinned via package.json → package-lock.json; `npm install` resolves it.
SMF_CORE_NPM = Path(__file__).parent.parent / "node_modules/@shmuggingface/core"
DEFAULT_CF_ENV = Path.home() / ".config/adanim/cloudflare_api_token.env"
DEFAULT_PROJECT = "leadforge-lead-scoring-v1-preview"

TIER_LABEL = {"intro": "Intro", "intermediate": "Intermediate", "advanced": "Advanced"}

DISCUSSIONS = [
    (
        "What is `snapshot_day = 30` and how does it affect which features are valid"
        " at inference time?"
    ),
    "Is `total_touches_all` a safe feature or a time-window leakage trap?",
    "LR and GBM AUCs are very close across tiers — does relational feature engineering help?",
    "How would you set a probability threshold for a team that can only work 50 leads per week?",
    "What happens to AUC when you evaluate on a chronological hold-out instead of a random split?",
]

# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

# Rewrite patterns for markdown links that would 404 on the static host.
# Relative links of the form ``[text](../foo)`` stay relative to the
# release tree root on GitHub.  Plain ``[LICENSE](LICENSE)`` and other
# bare-name links need an explicit GitHub blob URL.
_PARENT_LINK_RE = re.compile(r"\]\(\.\./([^)]+)\)")
# Rewrites ``validation/validation_report.md`` links that appear in the
# global ``release/README.md``.  This is a no-op for per-tier
# ``dataset_card.md`` files (which don't contain that path), but kept so
# ``_rewrite_links`` remains safe to call on the global README too.
_VALIDATION_LINK_RE = re.compile(r"\]\(validation/validation_report\.md\)")
# Bare relative links: match the full ``[text](word.ext)`` or
# ``![alt](image.png)`` construct so we can distinguish images from links.
# Group 1 captures the optional leading ``!``; group 2 the link text/alt;
# group 3 the path.  In ``_rewrite_links`` we skip the rewrite when
# group 1 is ``!`` so inline image sources are never mangled.
# Note: _PARENT_LINK_RE uses the module-level GITHUB_BLOB_BASE constant
# (repo root); _BARE_RELATIVE_LINK_RE uses the caller-supplied
# ``github_base`` parameter (tier-specific subdirectory).  The asymmetry
# is intentional: ``../`` links always resolve from the repo root, while
# bare names are relative to the document's own directory.
_BARE_RELATIVE_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\((?!https?://|#)([^/][^)]*)\)")


def _rewrite_links(text: str, github_base: str) -> str:
    """Rewrite relative markdown links to absolute GitHub blob URLs.

    Three classes handled:
    1. ``[text](../foo)`` → ``[text](<github_base>/foo)``  (parent-dir links)
    2. ``[text](validation/validation_report.md)`` → absolute blob URL
    3. ``[text](bare-name)`` → ``[text](<github_base>/bare-name)``
       (bare relative links like ``[LICENSE](LICENSE)`` that would 404)
    """
    text = _PARENT_LINK_RE.sub(rf"]({GITHUB_BLOB_BASE}/\1)", text)
    text = _VALIDATION_LINK_RE.sub(
        f"]({GITHUB_BLOB_BASE}/release/validation/validation_report.md)", text
    )
    text = _BARE_RELATIVE_LINK_RE.sub(
        lambda m: (
            m.group(0)  # image syntax (![alt](src)) — preserve unchanged
            if m.group(1) == "!"
            else f"[{m.group(2)}]({github_base}/{m.group(3)})"
        ),
        text,
    )
    return text


def _render_md(text: str) -> str:
    """Render ``text`` (markdown) to HTML."""
    return MarkdownIt("gfm-like").disable("linkify").render(text)


def render_tier_html(tier_dir: Path) -> str:
    """Render a tier's ``dataset_card.md`` to HTML with link rewriting.

    Each tier ships its own ``dataset_card.md`` (auto-generated by
    ``leadforge/narrative/dataset_card.py``).  Using the per-tier card
    rather than the global ``release/README.md`` means the description
    shown for each tier is specific to that tier — not a copy of the
    global README embedded three times.

    Relative links inside the card are prefixed with the GitHub blob
    URL for the tier's directory so they resolve correctly on the
    static preview host.
    """
    card_path = tier_dir / "dataset_card.md"
    text = card_path.read_text(encoding="utf-8")
    tier_name = tier_dir.name  # "intro" / "intermediate" / "advanced"
    github_tier_base = f"{GITHUB_BLOB_RELEASE}/{tier_name}"
    text = _rewrite_links(text, github_tier_base)
    return _render_md(text)


# ---------------------------------------------------------------------------
# Tier metadata loading
# ---------------------------------------------------------------------------


def _require(d: dict, key: str, context: str) -> Any:
    """Return ``d[key]``, raising ``KeyError`` with context on miss.

    Silent ``dict.get()`` defaults produce plausible-but-false preview
    pages when a required manifest / metrics field is absent or
    renamed.  Raising here catches schema drift at build time rather
    than silently misrepresenting the dataset.
    """
    if key not in d:
        raise KeyError(
            f"Required field {key!r} missing from {context}. "
            "Was the bundle regenerated with a different schema version?"
        )
    return d[key]


def _file_size_kb(path: Path) -> str:
    """Return a human-readable file size string, e.g. ``'42 KB'``."""
    return f"{max(1, path.stat().st_size // 1024)} KB"


class TierData(TypedDict):
    """Typed container returned by :func:`load_tier`."""

    tier: str
    tier_dir: Path
    task_dir: Path
    manifest: dict[str, Any]
    ctx_manifest: str
    metrics: dict[str, Any]
    columns: list[str]
    sample_rows: list[dict[str, str]]
    n_rows: int


def load_tier(release_dir: Path, tier: str) -> TierData:
    """Load manifest, metrics, feature dictionary, and sample rows for one tier."""
    tier_dir = release_dir / tier

    manifest_raw = (tier_dir / "manifest.json").read_text()
    manifest = json.loads(manifest_raw)
    ctx_manifest = f"{tier}/manifest.json"

    metrics_raw = (tier_dir / "metrics.json").read_text()
    metrics = json.loads(metrics_raw)

    fd = pd.read_csv(tier_dir / "feature_dictionary.csv")
    # ``split`` is the first column in ``lead_scoring.csv`` (added by
    # ``build_public_release.py``).  Older bundles built before PR 8.4
    # won't have it in their feature_dictionary.csv; newer ones will.
    # Normalise here: always put ``split`` exactly once at the front.
    fd_names = list(fd["name"])
    other_cols = [c for c in fd_names if c != "split"]
    columns = ["split"] + other_cols

    df = pd.read_csv(tier_dir / "lead_scoring.csv")
    # Stringify every cell so JSON serialisation is clean.
    sample_rows = [
        {k: ("" if str(v) in ("nan", "None") else str(v)) for k, v in row.items()}
        for row in df.head(8).to_dict("records")
    ]

    return {
        "tier": tier,
        "tier_dir": tier_dir,
        "task_dir": tier_dir / "tasks" / TASK,
        "manifest": manifest,
        "ctx_manifest": ctx_manifest,
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


def make_dataset_config(tier_data: TierData, config_dir: Path) -> dict:
    """Build a ShmuggingFace dataset config dict for one tier.

    Each tier page shows its own ``dataset_card.md`` as the description
    body (rendered to HTML here), keeping the per-tier copy in sync with
    the published card without duplicating the global README three times.
    """
    tier = tier_data["tier"]
    tier_dir = tier_data["tier_dir"]
    task_dir = tier_data["task_dir"]
    manifest = tier_data["manifest"]
    ctx_manifest = tier_data["ctx_manifest"]
    metrics = tier_data["metrics"]
    label = TIER_LABEL[tier]
    ctx_metrics = f"{tier}/metrics.json"
    # _require raises on schema drift rather than silently defaulting to
    # plausible-but-false values — including for metrics, not just manifest.
    medians = _require(metrics, "medians", ctx_metrics)
    cr = float(_require(medians, "conversion_rate_test", ctx_metrics))
    lr_auc = float(_require(medians, "lr_auc", ctx_metrics))
    n_leads = int(_require(manifest, "n_leads", ctx_manifest))
    snapshot_day = int(_require(manifest, "snapshot_day", ctx_manifest))

    task_info_all = _require(manifest, "tasks", ctx_manifest)
    if not isinstance(task_info_all, dict) or TASK not in task_info_all:
        raise KeyError(
            f"Task {TASK!r} not found in {ctx_manifest}['tasks']. "
            "Bundle may have been generated with a different task name."
        )
    task_info = task_info_all[TASK]
    train_rows = int(_require(task_info, "train_rows", f"{ctx_manifest}[tasks][{TASK}]"))
    valid_rows = int(_require(task_info, "valid_rows", f"{ctx_manifest}[tasks][{TASK}]"))
    test_rows = int(_require(task_info, "test_rows", f"{ctx_manifest}[tasks][{TASK}]"))

    files = [
        {
            "path": "lead_scoring.csv",
            "size": _file_size_kb(tier_dir / "lead_scoring.csv"),
            "kind": "CSV",
            "sourcePath": _rel(tier_dir / "lead_scoring.csv", config_dir),
            "about": (
                f"Flat ML-ready snapshot CSV: {n_leads:,} leads × "
                f"{len(tier_data['columns'])} columns (including 'split'), "
                f"snapshot day {snapshot_day}.  The 'split' column "
                f"(train / valid / test) lets conventional ML workflows load "
                f"a single file."
            ),
        },
        {
            "path": "feature_dictionary.csv",
            "size": _file_size_kb(tier_dir / "feature_dictionary.csv"),
            "kind": "CSV",
            "sourcePath": _rel(tier_dir / "feature_dictionary.csv", config_dir),
            "about": (
                "Per-column documentation: dtype, analytical category, "
                "leakage-risk flag, and plain-language description."
            ),
        },
        {
            "path": "tasks/converted_within_90_days/train.parquet",
            "size": _file_size_kb(task_dir / "train.parquet"),
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
            "size": _file_size_kb(task_dir / "valid.parquet"),
            "kind": "Parquet",
            "sourcePath": _rel(task_dir / "valid.parquet", config_dir),
            "about": f"Validation split — {valid_rows:,} leads.",
        },
        {
            "path": "tasks/converted_within_90_days/test.parquet",
            "size": _file_size_kb(task_dir / "test.parquet"),
            "kind": "Parquet",
            "sourcePath": _rel(task_dir / "test.parquet", config_dir),
            "about": (f"Test split — {test_rows:,} leads, held out for final evaluation only."),
        },
        {
            "path": "dataset_card.md",
            "size": _file_size_kb(tier_dir / "dataset_card.md"),
            "kind": "Dataset card",
            "sourcePath": _rel(tier_dir / "dataset_card.md", config_dir),
            "about": "Auto-generated tier-specific dataset card.",
        },
    ]

    cover_rel = _rel(tier_dir.parent / "dataset-cover-image.png", config_dir)

    # Per-tier description: use the tier's own dataset_card.md so each
    # tier page is self-contained and matches what's published per-tier.
    description_html = render_tier_html(tier_dir)

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
        "subsets": [f"leadforge-lead-scoring-v1-{tier}"],
        "coverImage": cover_rel,
        "descriptionHtml": description_html,
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
        # ShmuggingFaceCore v1.0.2 accepts these as strings and renders
        # them verbatim in the stats bar.  They are intentionally zero for
        # the pre-publication review; the real platform will populate them
        # after publish.
        "downloads": "0",
        "likes": "0",
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
    """Return path to a working ShmuggingFaceCore installation.

    Resolution order:
    1. ``--smf-core PATH`` override (for local dev / CI with a custom checkout).
    2. npm-installed package at ``node_modules/@shmuggingface/core`` — the
       canonical path when ``npm install`` has been run from the repo root
       (pinned to v1.0.2 via ``package.json`` / ``package-lock.json``).

    Exits with an informative error if neither source is available.
    """
    if smf_core is not None:
        entry = smf_core / "bin/shmuggingface.mjs"
        if not entry.exists():
            sys.exit(f"ShmuggingFaceCore entry point not found at {entry}")
        return smf_core

    entry = SMF_CORE_NPM / "bin/shmuggingface.mjs"
    if entry.exists():
        pkg = SMF_CORE_NPM / "package.json"
        version = json.loads(pkg.read_text()).get("version", "unknown")
        print(f"  Using npm-installed @shmuggingface/core v{version}", file=sys.stderr)
        return SMF_CORE_NPM

    sys.exit(
        "ShmuggingFaceCore not found.\n"
        f"  Expected npm installation at: {SMF_CORE_NPM}\n"
        "  Run `npm install` from the repo root to install the pinned v1.0.2 release,\n"
        "  or pass --smf-core PATH to a local checkout."
    )


# ---------------------------------------------------------------------------
# Build and deploy
# ---------------------------------------------------------------------------


def build_site(config_path: Path, out_dir: Path, smf_core: Path) -> None:
    """Run the ShmuggingFaceCore generator."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Building static site → {out_dir}", file=sys.stderr)
    subprocess.run(  # noqa: S603
        [  # noqa: S607 — trusted local tool, path from npm install
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


def deploy_site(
    out_dir: Path,
    project_name: str,
    cf_env_path: Path,
    *,
    production: bool = False,
) -> None:
    """Deploy the built site to Cloudflare Pages via wrangler.

    By default deploys to the ``preview`` branch slot so a routine
    local run never clobbers the live production URL.  Pass
    ``production=True`` (``--production`` on the CLI) to push to the
    ``main`` branch (the Cloudflare Pages production slot).
    """
    if not cf_env_path.exists():
        sys.exit(
            f"Cloudflare env file not found: {cf_env_path}\n"
            f"Expected format:\n"
            f"  export CLOUDFLARE_ACCOUNT_ID='...'\n"
            f"  export CLOUDFLARE_API_TOKEN='...'"
        )

    env = _load_cf_env(cf_env_path)
    account_id = env.get("CLOUDFLARE_ACCOUNT_ID", "(not set)")
    branch = "main" if production else "preview"
    print(
        f"  Deploying to Cloudflare Pages\n"
        f"    project  : {project_name}\n"
        f"    account  : {account_id}\n"
        f"    branch   : {branch} ({'production slot' if production else 'branch preview'})\n"
        f"    source   : {out_dir}",
        file=sys.stderr,
    )
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607 — wrangler is a pinned devDependency, not user input
            "wrangler",
            "pages",
            "deploy",
            str(out_dir),
            "--project-name",
            project_name,
            "--branch",
            branch,
            "--commit-dirty=true",  # suppress the "uncommitted changes" warning
        ],
        env=env,
    )
    if result.returncode != 0:
        sys.exit(f"Deployment failed (wrangler exit code {result.returncode})")

    suffix = "" if production else "/preview"
    print(
        f"\n  Live at: https://{project_name}.pages.dev{suffix}",
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
        help=(
            "Path to a local ShmuggingFaceCore checkout "
            "(default: node_modules/@shmuggingface/core from `npm install`)"
        ),
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy to Cloudflare Pages after building",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help=(
            "With --deploy: push to the production slot (--branch main). "
            "Default without this flag is a branch preview (--branch preview). "
            "Requires explicit opt-in to prevent accidental production deploys."
        ),
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
    parser.add_argument(
        "--config-only",
        action="store_true",
        help=(
            "Write shmuggingface.config.mjs and stop — skip the Node build and deploy. "
            "Useful for verifying generated config without a full Node environment."
        ),
    )
    args = parser.parse_args()

    release_dir = args.release_dir.resolve()
    if not release_dir.is_dir():
        sys.exit(f"Release directory not found: {release_dir}")

    config_dir = release_dir / "_shmuggingface"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "shmuggingface.config.mjs"
    out_dir = args.out_dir.resolve() if args.out_dir else (config_dir / "dist")

    # --- Load tiers ----------------------------------------------------------
    print("Loading release tiers …", file=sys.stderr)
    datasets = []
    for tier in TIERS:
        print(f"  {tier}", file=sys.stderr)
        tier_data = load_tier(release_dir, tier)
        ds = make_dataset_config(tier_data, config_dir)
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

    if args.config_only:
        print(f"--config-only: stopping after config write.  Config at: {config_path}")
        return

    # --- Ensure ShmuggingFaceCore --------------------------------------------
    smf_core = ensure_smf_core(args.smf_core)

    # --- Build ---------------------------------------------------------------
    print("Building static site …", file=sys.stderr)
    build_site(config_path, out_dir, smf_core)
    print(f"Done.  Site at: {out_dir}", file=sys.stderr)

    # --- Deploy --------------------------------------------------------------
    if args.deploy:
        print("Deploying to Cloudflare Pages …", file=sys.stderr)
        deploy_site(out_dir, args.project_name, args.cf_env, production=args.production)


if __name__ == "__main__":
    main()
