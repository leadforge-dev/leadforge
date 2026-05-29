#!/usr/bin/env python3
"""Publish ``leadforge-lead-scoring-v1`` to Hugging Face.

This is the final publish gate for the Hugging Face side of the v1
release.  It wraps the two pre-publish steps that must already be
complete (packaging and linting), runs a local ``load_dataset()``
smoke test (G12.3 / G12.4), and then uploads via
``huggingface_hub.HfApi``.

Three-stage runbook
-------------------
1. **Dry-run** (safe, no credentials needed for the smoke test)::

       python scripts/publish_hf.py --dry-run
       python scripts/publish_hf.py --dry-run --variant=instructor

   Re-packages ``release/huggingface/`` (or ``release/huggingface-instructor/``),
   lints the metadata, and runs ``load_dataset()`` locally to verify
   that the HuggingFace ``datasets`` library can read every config
   (G12.3 / G12.4).  Exits ``0`` only if every check passes.

2. **Upload as private** (first publish)::

       python scripts/publish_hf.py
       python scripts/publish_hf.py --variant=instructor

   Requires a HuggingFace token with write access to the target org.
   Set ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` in the environment,
   or run ``huggingface-cli login`` first.  Creates the repo if it
   doesn't exist (private), then uploads the assembled directory.

3. **Flip to public**::

       python scripts/publish_hf.py --go-public
       python scripts/publish_hf.py --go-public --variant=instructor

   Updates the repo visibility to public via ``HfApi.update_repo_visibility``.
   Can be run separately after reviewing the private upload.

Options
-------
--release-dir PATH      Root of the release directory (default: release/).
--variant {public,instructor}
                        Which dataset to publish (default: public).
--dry-run               Package + lint + load_dataset; no upload.
--private               Force private upload even when the repo already exists
                        (default: private on first create, unchanged on update).
--go-public             Flip the repo to public visibility and exit.
--token TOKEN           Hugging Face API token (default: HF_TOKEN env var or
                        the token stored by ``huggingface-cli login``).
--commit-message MSG    Commit message for the upload (default: auto).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

_CREDENTIALS_FILE: Final[Path] = Path.home() / ".config" / "huggingface" / "credentials"


def _read_token_from_credentials_file() -> str | None:
    """Read HF_TOKEN from ``~/.config/huggingface/credentials`` if present.

    The file uses ``KEY=VALUE`` lines; blank lines and lines starting with
    ``#`` are ignored.  Returns the first value for ``HF_TOKEN`` found, or
    ``None`` if the file doesn't exist or the key isn't set.
    """
    if not _CREDENTIALS_FILE.exists():
        return None
    try:
        for raw_line in _CREDENTIALS_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                if key.strip() == "HF_TOKEN":
                    token = value.strip()
                    if token and not token.startswith("REPLACE_WITH"):
                        return token
    except OSError:
        pass
    return None


def _resolve_token(cli_token: str | None) -> str | None:
    """Return the best available HF token.

    Priority order (highest first):
    1. ``--token`` CLI argument
    2. ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env var
    3. ``~/.config/leadforge/credentials`` file
    4. ``None`` — falls through to ``huggingface_hub``'s own credential cache
    """
    if cli_token:
        return cli_token
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(env_key)
        if value:
            return value
    file_token = _read_token_from_credentials_file()
    if file_token:
        print(
            f"  token: read from {_CREDENTIALS_FILE}",
            file=sys.stderr,
        )
        return file_token
    return None


# Make ``scripts/`` importable regardless of invocation style.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lint_platform_metadata import LintOutcome, run_lint  # noqa: E402
from package_hf_release import (  # noqa: E402
    DEFAULT_HUGGINGFACE_DIR,
    DEFAULT_HUGGINGFACE_INSTRUCTOR_DIR,
    DEFAULT_RELEASE_DIR,
    run_packager,
)

# ---------------------------------------------------------------------------
# Repo identity
# ---------------------------------------------------------------------------

HF_ORG: Final[str] = "shaypal5"
REPO_IDS: Final[dict[str, str]] = {
    "public": f"{HF_ORG}/leadforge-lead-scoring-v1",
    "instructor": f"{HF_ORG}/leadforge-lead-scoring-v1-instructor",
}
HF_DATASET_URLS: Final[dict[str, str]] = {
    "public": f"https://huggingface.co/datasets/{REPO_IDS['public']}",
    "instructor": f"https://huggingface.co/datasets/{REPO_IDS['instructor']}",
}
PUBLIC_CONFIGS: Final[tuple[str, ...]] = ("intro", "intermediate", "advanced")
INSTRUCTOR_CONFIGS: Final[tuple[str, ...]] = ("intermediate",)
UPLOAD_DIRS: Final[dict[str, Path]] = {
    "public": DEFAULT_HUGGINGFACE_DIR,
    "instructor": DEFAULT_HUGGINGFACE_INSTRUCTOR_DIR,
}


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def _repackage(release_dir: Path, variant: str) -> bool:
    """Re-run the HF packager to ensure the upload tree is current.

    Returns ``True`` on success, ``False`` on validation failure.
    """
    upload_dir = UPLOAD_DIRS[variant]
    print(f"[ 1/4 ] Packaging {upload_dir} …", file=sys.stderr)
    try:
        outcome = run_packager(
            release_dir,
            huggingface_dir=upload_dir,
            variant=variant,
            dry_run=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"  error: {exc}", file=sys.stderr)
        sys.exit(2)

    if outcome.errors:
        print("  FAIL: packaging validation errors:", file=sys.stderr)
        for err in outcome.errors:
            print(f"    - {err.field}: {err.message}", file=sys.stderr)
        return False

    print(f"  OK   README → {outcome.readme_path}", file=sys.stderr)
    if outcome.assembled:
        print(f"  OK   upload tree → {upload_dir}", file=sys.stderr)
    return True


def _lint(release_dir: Path) -> bool:
    """Run ``lint_platform_metadata`` against the packaged artifacts.

    Returns ``True`` on clean, ``False`` on lint failure.
    """
    print("[ 2/4 ] Linting platform metadata …", file=sys.stderr)
    try:
        outcome: LintOutcome = run_lint(release_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  error: {exc}", file=sys.stderr)
        sys.exit(2)

    if not outcome.ok:
        print("  FAIL: lint errors:", file=sys.stderr)
        for f in outcome.findings:
            print(f"    - {f.field}: {f.message}", file=sys.stderr)
        return False

    print("  OK   metadata passes all lint checks", file=sys.stderr)
    return True


def _smoke_test(upload_dir: Path, variant: str) -> bool:
    """Run ``load_dataset()`` locally against the assembled upload tree.

    Verifies G12.3 (public) / G12.4 (instructor) — that the HuggingFace
    ``datasets`` library can load every config without error.  Requires
    ``pip install -e '.[publish]'``.
    """
    print("[ 3/4 ] Running load_dataset() smoke tests …", file=sys.stderr)
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        print(
            "  SKIP load_dataset() not available — install with: pip install -e '.[publish]'",
            file=sys.stderr,
        )
        return True  # Not a hard failure; skip gracefully

    configs = PUBLIC_CONFIGS if variant == "public" else INSTRUCTOR_CONFIGS
    all_ok = True
    for config in configs:
        try:
            ds = load_dataset(str(upload_dir), config, trust_remote_code=False)
            n_splits = len(ds)
            total_rows = sum(len(split) for split in ds.values())
            print(
                f"  OK   config={config!r}: {n_splits} splits, {total_rows:,} rows total",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL config={config!r}: {exc}", file=sys.stderr)
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _upload(
    upload_dir: Path,
    variant: str,
    *,
    token: str | None,
    private: bool,
    commit_message: str,
) -> None:
    """Create or update the HF dataset repo and upload the folder.

    Raises on error; caller handles exit code.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
    except ImportError:
        print(
            "error: huggingface_hub is required — install with: pip install -e '.[publish]'",
            file=sys.stderr,
        )
        sys.exit(2)

    api = HfApi(token=token)
    repo_id = REPO_IDS[variant]

    print(f"[ 4/4 ] Uploading {upload_dir} → {repo_id} …", file=sys.stderr)

    # Create repo if it doesn't exist.
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    print(f"  repo  : {repo_id} (private={private})", file=sys.stderr)

    # Upload the assembled directory.
    url = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(upload_dir),
        commit_message=commit_message,
    )
    print(f"  commit: {url}", file=sys.stderr)


def _go_public(variant: str, *, token: str | None) -> None:
    """Flip a private HF dataset repo to public visibility."""
    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
    except ImportError:
        print("error: huggingface_hub not installed", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=token)
    repo_id = REPO_IDS[variant]
    print(f"Making {repo_id} public …", file=sys.stderr)
    api.update_repo_visibility(repo_id=repo_id, repo_type="dataset", private=False)
    print(f"  Done. {HF_DATASET_URLS[variant]}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", maxsplit=1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        metavar="PATH",
        help="Root of the release directory (default: release/)",
    )
    parser.add_argument(
        "--variant",
        choices=["public", "instructor"],
        default="public",
        help="Which dataset to publish: public (3 tiers) or instructor (default: public)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Package + lint + load_dataset smoke; no upload",
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload as private (default). Pass --no-private to upload public directly.",
    )
    parser.add_argument(
        "--go-public",
        action="store_true",
        help="Flip the repo to public visibility and exit",
    )
    parser.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help="HuggingFace API token (default: HF_TOKEN env var or stored login)",
    )
    parser.add_argument(
        "--commit-message",
        default="feat: v1 release — leadforge-lead-scoring-v1",
        metavar="MSG",
        help="Commit message for the upload",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    release_dir: Path = args.release_dir.resolve()
    variant: str = args.variant
    token: str | None = _resolve_token(args.token)

    # --- Go-public shortcut -------------------------------------------------
    if args.go_public:
        _go_public(variant, token=token)
        return 0

    # --- Pre-flight ---------------------------------------------------------
    ok = _repackage(release_dir, variant)
    ok = _lint(release_dir) and ok
    upload_dir = UPLOAD_DIRS[variant].resolve()
    ok = _smoke_test(upload_dir, variant) and ok

    if not ok:
        print("\nPre-flight FAILED — fix errors above before uploading.", file=sys.stderr)
        return 1

    if args.dry_run:
        print(
            "\nDry-run complete — all pre-flight checks passed.",
            file=sys.stderr,
        )
        print(f"Upload tree is ready at: {upload_dir}", file=sys.stderr)
        print(
            f"\nTo upload (private):\n"
            f"  python scripts/publish_hf.py --variant={variant}\n"
            f"\nTo flip to public after reviewing:\n"
            f"  python scripts/publish_hf.py --go-public --variant={variant}",
            file=sys.stderr,
        )
        return 0

    # --- Upload -------------------------------------------------------------
    try:
        _upload(
            upload_dir,
            variant,
            token=token,
            private=args.private,
            commit_message=args.commit_message,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\nUpload failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nUpload succeeded (private={args.private}).", file=sys.stderr)
    print(f"Dataset URL: {HF_DATASET_URLS[variant]}", file=sys.stderr)
    if args.private:
        print(
            "\nNext: review the private dataset at the URL above, then make it\n"
            "public with:\n"
            f"  python scripts/publish_hf.py --go-public --variant={variant}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
