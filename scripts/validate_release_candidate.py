#!/usr/bin/env python3
"""Release-candidate validator for ``leadforge-lead-scoring-v1``.

PR 3.3's driver. Orchestrates a cross-seed × cross-tier release-quality
sweep, runs split-level leakage probes against the canonical seed, and
gates the release on the YAML-declared acceptance bands.

Relationship to ``leadforge validate``
--------------------------------------

``leadforge validate <bundle_dir>`` checks one bundle's structural+FK+
leakage contract — it answers "is this single bundle internally
consistent and free of structural leakage?" and runs in seconds.  This
script is complementary: it answers "does the *family* of three tier
bundles, each rebuilt across N seeds, fall within the v1 acceptance
bands declared in ``v1_acceptance_gates.md``?"  The two are not merged
because their inputs (one bundle vs. a tier directory tree), runtimes
(seconds vs. minutes), and audiences (the bundle-validation contract
vs. the release-readiness contract) differ.

Output contract (pinned in ``docs/release/v1_release_design.md``
§"Output contract")::

    release/validation/
      validation_report.json
      validation_report.md
      figures/
        lift_curve_intro.png
        lift_curve_intermediate.png
        lift_curve_advanced.png
        calibration_intermediate.png
        leakage_delta.png
        cohort_shift.png
        value_capture.png

Exit codes
----------

* ``0`` — all gates pass.
* ``1`` — at least one gate failed; per-failure detail is printed to
  stderr.
* ``2`` — pre-flight failure (missing release dir, missing tier under
  ``--no-rebuild``, malformed bands YAML).

Usage examples::

    # Full release run — N=5 sweep against release/{intro,intermediate,advanced}/
    python scripts/validate_release_candidate.py

    # Smoke run — N=2 with tiny populations, completes in under a minute
    python scripts/validate_release_candidate.py --quick

    # Reuse already-regenerated bundles (bands tweak, no resimulation)
    python scripts/validate_release_candidate.py --no-rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from leadforge.validation.difficulty import (
    AcceptanceBands,
    GateFailure,
    check_release_bands,
    load_bands,
)
from leadforge.validation.leakage_probes import (
    LeakageReport,
    run_split_probes,
)
from leadforge.validation.release_quality import (
    DEFAULT_MODEL_RANDOM_STATE,
    LABEL_COLUMN,
    ReleaseQualityReport,
    TierBuildSpec,
    measure_release_quality,
    regenerate_tier_for_seeds,
)
from leadforge.validation.reporting import render_report

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Tier directory names under ``--release-dir``.
TIERS: tuple[str, ...] = ("intro", "intermediate", "advanced")

#: Default cross-seed sweep — five seeds is the smallest N that yields a
#: stable median ± spread under HistGBM tree-split tie-break drift.
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

#: Canonical seed for cohort-shift evaluation and leakage probes.  Held
#: at the bundle's own generation seed so the probes inherit the same
#: data ChatGPT v2 audited against.
DEFAULT_COHORT_CANONICAL_SEED: int = 42

#: ``--quick`` mode: smaller seed list and tiny populations.  Larger
#: than the round-trip test's ``_SMALL`` because the advanced tier's
#: ~8% base rate × 15% test split needs at least a few hundred leads to
#: produce both classes in the test split (see PR 3.2 release_quality
#: degenerate-split guard).  ~10s per seed per tier on commodity
#: hardware → full --quick sweep completes well under a minute.
QUICK_SEEDS: tuple[int, ...] = (42, 43)
QUICK_POPULATION: dict[str, int] = {"n_leads": 500, "n_accounts": 250, "n_contacts": 750}

DEFAULT_RELEASE_DIR: Path = Path("release")
DEFAULT_WORKDIR: Path = Path("release/_release_quality")
DEFAULT_OUT_DIR: Path = Path("release/validation")
DEFAULT_BANDS: Path = Path("docs/release/v1_acceptance_gates_bands.yaml")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse driver CLI arguments.

    Kept as a free function so the integration tests can build a
    ``Namespace`` directly without exec'ing the script.
    """
    parser = argparse.ArgumentParser(
        prog="validate_release_candidate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help=(
            "Directory containing the per-tier bundle subdirectories "
            f"({', '.join(TIERS)}). Default: {DEFAULT_RELEASE_DIR}"
        ),
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help=(
            "Where to materialise the cross-seed bundle sweep. Idempotent "
            f"— existing per-seed bundles are reused. Default: {DEFAULT_WORKDIR}"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Where to write validation_report.{{json,md}} + figures/. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--bands",
        type=Path,
        default=DEFAULT_BANDS,
        help=f"YAML acceptance bands file. Default: {DEFAULT_BANDS}",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Generation seeds for the cross-seed sweep. Default: {list(DEFAULT_SEEDS)}",
    )
    parser.add_argument(
        "--cohort-canonical-seed",
        type=int,
        default=DEFAULT_COHORT_CANONICAL_SEED,
        help=(
            "Seed at which to run cohort-shift evaluation and leakage probes. "
            f"Default: {DEFAULT_COHORT_CANONICAL_SEED}"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke mode: N=2 seeds with tiny populations. Completes in under "
            "a minute. Override seed list / population sizes are ignored."
        ),
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help=(
            "Use bundles already on disk under --workdir. Fails fast if any "
            "tier × seed bundle is missing. Use for fast band-tweak iteration."
        ),
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(TIERS),
        choices=list(TIERS),
        help=f"Subset of tiers to validate. Default: {list(TIERS)}",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Per-tier orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriverConfig:
    """Resolved driver settings — produced from CLI args, consumed by run().

    Carrying this as an explicit dataclass makes the integration tests
    cleaner: they build one of these directly rather than constructing an
    ``argparse.Namespace`` via private constructor.
    """

    release_dir: Path
    workdir: Path
    out_dir: Path
    bands_path: Path
    seeds: tuple[int, ...]
    cohort_canonical_seed: int
    tiers: tuple[str, ...]
    quick: bool
    no_rebuild: bool


def _config_from_args(args: argparse.Namespace) -> DriverConfig:
    # Sort + dedup so the seed list is independent of user input order; the
    # cohort_canonical_seed fallback below has to be deterministic across
    # equivalent invocations (e.g. ``--seeds 11 10`` vs ``--seeds 10 11``).
    seeds_input = QUICK_SEEDS if args.quick else args.seeds
    seeds = tuple(sorted(set(seeds_input)))
    canonical = args.cohort_canonical_seed
    if canonical not in seeds:
        # Fall back to the smallest seed in the sweep; PR 3.2 already does
        # this internally, but surfacing the substitution at config-time
        # keeps the CLI deterministic and the JSON ``seeds`` field
        # consistent with the cohort_shift result.
        canonical = min(seeds)
    return DriverConfig(
        release_dir=args.release_dir,
        workdir=args.workdir,
        out_dir=args.out_dir,
        bands_path=args.bands,
        seeds=seeds,
        cohort_canonical_seed=canonical,
        tiers=tuple(args.tiers),
        quick=args.quick,
        no_rebuild=args.no_rebuild,
    )


def build_tier_spec(release_dir: Path, tier: str, *, quick: bool) -> TierBuildSpec:
    """Build a :class:`TierBuildSpec` for one tier.

    The spec is read from the canonical bundle's manifest under
    ``<release_dir>/<tier>/``; ``--quick`` overrides the population sizes
    so the smoke sweep completes in under a minute regardless of the
    canonical bundle's row counts.
    """
    bundle_dir = release_dir / tier
    if not (bundle_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"missing manifest at {bundle_dir / 'manifest.json'}; "
            f"is {release_dir} a leadforge release directory?"
        )
    spec = TierBuildSpec.from_bundle(bundle_dir, name=tier)
    if quick:
        spec = TierBuildSpec(
            name=spec.name,
            recipe_id=spec.recipe_id,
            difficulty=spec.difficulty,
            n_leads=QUICK_POPULATION["n_leads"],
            n_accounts=QUICK_POPULATION["n_accounts"],
            n_contacts=QUICK_POPULATION["n_contacts"],
            snapshot_day=spec.snapshot_day,
            primary_task=spec.primary_task,
            label_window_days=spec.label_window_days,
            exposure_mode=spec.exposure_mode,
        )
    return spec


def regenerate_or_load(
    spec: TierBuildSpec,
    seeds: Sequence[int],
    workdir: Path,
    *,
    no_rebuild: bool,
) -> dict[int, Path]:
    """Materialise (or look up) the per-seed bundles for one tier.

    With ``no_rebuild=True``, refuses to call the generator and instead
    asserts that every ``<workdir>/<tier>__seed{seed}/manifest.json``
    already exists.  This is the fast band-tweak iteration mode.
    """
    if not no_rebuild:
        return regenerate_tier_for_seeds(spec, seeds, workdir)
    out: dict[int, Path] = {}
    missing: list[Path] = []
    for seed in seeds:
        target = workdir / f"{spec.name}__seed{seed}"
        if (target / "manifest.json").exists():
            out[seed] = target
        else:
            missing.append(target)
    if missing:
        raise FileNotFoundError(
            "--no-rebuild was set but the following tier × seed bundles are "
            f"missing under {workdir}:\n  - " + "\n  - ".join(str(p) for p in missing)
        )
    return out


def run_tier_leakage_probes(
    bundle_dir: Path,
    *,
    bands: AcceptanceBands,
) -> LeakageReport:
    """Run :func:`run_split_probes` on the canonical seed's task splits.

    Reads ``train``/``valid``/``test`` parquet files under
    ``<bundle_dir>/tasks/<primary_task>/`` and applies the calibrated
    thresholds from ``bands.leakage_probes``.

    Returns an empty :class:`LeakageReport` (i.e. "no findings") when the
    primary task split files are missing — the structural validator
    catches that case; this driver intentionally degrades to "skip the
    leakage panel" rather than double-reporting the same defect.
    """
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return LeakageReport(findings=())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary_task = str(manifest.get("primary_task", "converted_within_90_days"))
    task_dir = bundle_dir / "tasks" / primary_task
    splits: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "valid", "test"):
        path = task_dir / f"{split_name}.parquet"
        if path.exists():
            splits[split_name] = pd.read_parquet(path)
    if not splits:
        return LeakageReport(findings=())
    probes = bands.leakage_probes
    feature_subsets = {
        name: (max_auc, list(cols)) for name, (max_auc, cols) in probes.feature_subsets.items()
    }
    return run_split_probes(
        splits,
        label_col=LABEL_COLUMN,
        label_drift_max=probes.label_drift_max,
        id_only_max_auc=probes.id_only_max_auc,
        feature_subsets=feature_subsets or None,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriverResult:
    """Materialised outputs returned from :func:`run_validation`.

    Includes the report itself, the per-tier leakage findings, and the
    list of acceptance-band failures.  Tests assert against the result
    directly; the CLI prints from it and translates to an exit code.
    """

    report: ReleaseQualityReport
    leakage_reports: dict[str, LeakageReport]
    failures: list[GateFailure]


def run_validation(config: DriverConfig) -> DriverResult:
    """Execute the full validate-release-candidate pipeline.

    Steps:

    1. Pre-flight: confirm release dir exists, parse bands.
    2. For each requested tier, build a :class:`TierBuildSpec` and either
       regenerate the cross-seed bundles or assert they already exist.
    3. Aggregate per-(tier, seed) measurements via
       :func:`measure_release_quality`.
    4. Run :func:`run_split_probes` against each tier's canonical-seed
       bundle.
    5. Render the JSON / markdown / figures output.
    6. Evaluate :func:`check_release_bands` against the report and the
       leakage findings.

    Returns the materialised :class:`DriverResult`.  The CLI translates
    its ``failures`` into stderr lines and an exit code; tests assert
    against the structured fields.
    """
    bands = load_bands(config.bands_path)

    if not config.release_dir.exists():
        raise FileNotFoundError(
            f"--release-dir {config.release_dir} does not exist; expected per-tier "
            f"bundles under {config.release_dir}/{{intro,intermediate,advanced}}/"
        )

    tier_bundles: dict[str, dict[int, Path]] = {}
    for tier in config.tiers:
        spec = build_tier_spec(config.release_dir, tier, quick=config.quick)
        tier_bundles[tier] = regenerate_or_load(
            spec, config.seeds, config.workdir, no_rebuild=config.no_rebuild
        )

    report = measure_release_quality(
        tier_bundles,
        cohort_canonical_seed=config.cohort_canonical_seed,
        model_random_state=DEFAULT_MODEL_RANDOM_STATE,
    )

    leakage_reports: dict[str, LeakageReport] = {}
    for tier, by_seed in tier_bundles.items():
        canonical = config.cohort_canonical_seed
        if canonical not in by_seed:
            canonical = sorted(by_seed.keys())[0]
        leakage_reports[tier] = run_tier_leakage_probes(by_seed[canonical], bands=bands)

    render_report(report, config.out_dir)

    failures = check_release_bands(report, bands, leakage_reports=leakage_reports)
    return DriverResult(report=report, leakage_reports=leakage_reports, failures=failures)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_failures(failures: Sequence[GateFailure]) -> str:
    """Render a list of :class:`GateFailure` for stderr.

    Groups by gate id, then sorts within each gate by ``(tier, message)``
    so the output is stable across runs regardless of the order in which
    individual band checks emit their failures (per-tier checks emit
    in YAML iteration order; cross-tier checks emit in code order).
    """
    if not failures:
        return ""
    by_gate: dict[str, list[GateFailure]] = {}
    for f in failures:
        by_gate.setdefault(f.gate, []).append(f)
    lines: list[str] = ["Acceptance-band failures:"]
    for gate in sorted(by_gate):
        lines.append(f"  [{gate}]")
        # ``tier`` is ``None`` for cross-tier gates; bucket those last by
        # using the empty string as the sort key for "no tier".
        for f in sorted(by_gate[gate], key=lambda x: (x.tier or "", x.message)):
            scope = f.tier or "(all tiers)"
            lines.append(f"    - {scope}: {f.message}")
    return "\n".join(lines) + "\n"


def format_summary(result: DriverResult) -> str:
    """Single-line summary suitable for stdout."""
    n_failures = len(result.failures)
    n_tiers = len(result.report.tiers)
    n_seeds = len(result.report.seeds)
    n_findings = sum(len(lr.findings) for lr in result.leakage_reports.values())
    status = "PASS" if n_failures == 0 else f"FAIL ({n_failures} gate(s) failed)"
    return (
        f"validate_release_candidate: {status} — {n_tiers} tier(s), {n_seeds} seed(s); "
        f"leakage findings: {n_findings}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _config_from_args(args)
    try:
        result = run_validation(config)
    except FileNotFoundError as exc:
        print(f"validate_release_candidate: pre-flight error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as exc:
        print(f"validate_release_candidate: malformed input: {exc}", file=sys.stderr)
        return 2

    print(format_summary(result))
    if result.failures:
        print(format_failures(result.failures), file=sys.stderr, end="")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
