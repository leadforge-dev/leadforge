"""Render :class:`ReleaseQualityReport` to JSON + markdown + figures.

PR 3.2 ships the renderer; PR 3.3 wires it into
``scripts/validate_release_candidate.py``.

Output contract (pinned in ``docs/release/v1_release_design.md``
§"Output contract")::

    <output_dir>/
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

Filenames are *exact* — they are referenced from the dataset card and
the markdown report; renaming them is a contract change.

Matplotlib is the only figure dependency; we force the Agg backend
before importing :mod:`matplotlib.pyplot` so this module is safe in
headless CI.  Figures are deterministic byte-for-byte under the same
:class:`ReleaseQualityReport` input — the renderer does no sampling and
pins every text-source font option.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless / deterministic; must precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402

from leadforge.validation.release_quality import (  # noqa: E402
    PRECISION_KS,
    CohortShiftMetrics,
    CrossSeedTierMetrics,
    ReleaseQualityReport,
    report_to_json,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

REPORT_JSON: str = "validation_report.json"
REPORT_MD: str = "validation_report.md"
FIGURES_DIRNAME: str = "figures"

#: Pinned figure filenames — must match
#: ``docs/release/v1_release_design.md`` §"Output contract" exactly.
LIFT_CURVE_FIGURE_TEMPLATE: str = "lift_curve_{tier}.png"
CALIBRATION_FIGURE: str = "calibration_intermediate.png"
LEAKAGE_DELTA_FIGURE: str = "leakage_delta.png"
COHORT_SHIFT_FIGURE: str = "cohort_shift.png"
VALUE_CAPTURE_FIGURE: str = "value_capture.png"

#: Tiers for which a lift curve is rendered.  The design doc names
#: these three explicitly; rendering for unknown tiers would diverge
#: from the contract filenames.
_LIFT_CURVE_TIERS: tuple[str, ...] = ("intro", "intermediate", "advanced")

#: Tier whose calibration curve is the canonical figure.  Per the
#: design doc; PR 3.3 may grow per-tier reliability later.
_CALIBRATION_TIER: str = "intermediate"


# ---------------------------------------------------------------------------
# Public renderer
# ---------------------------------------------------------------------------


def render_report(report: ReleaseQualityReport, output_dir: Path) -> dict[str, Path]:
    """Write JSON, markdown and figures under *output_dir*.

    Returns a mapping of logical name → path written, for callers that
    want to assert presence (the round-trip integration test) or list
    the artefacts in a higher-level manifest (PR 3.3's driver).

    The output directory is created if missing; existing files are
    overwritten.  No file is *deleted* — a stale figure from a previous
    run will still be present unless the caller pre-cleans the
    directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    json_path = output_dir / REPORT_JSON
    json_path.write_text(report_to_json(report))
    written["json"] = json_path

    md_path = output_dir / REPORT_MD
    md_path.write_text(_render_markdown(report))
    written["md"] = md_path

    for tier_name in _LIFT_CURVE_TIERS:
        if tier_name not in report.tiers:
            continue
        path = figures_dir / LIFT_CURVE_FIGURE_TEMPLATE.format(tier=tier_name)
        _write_lift_curve(report.tiers[tier_name], path)
        written[f"lift_curve_{tier_name}"] = path

    if _CALIBRATION_TIER in report.tiers:
        cal_path = figures_dir / CALIBRATION_FIGURE
        _write_calibration_curve(report.tiers[_CALIBRATION_TIER], cal_path)
        written["calibration"] = cal_path

    if report.tiers:
        leak_path = figures_dir / LEAKAGE_DELTA_FIGURE
        _write_leakage_delta(report.tiers, leak_path)
        written["leakage_delta"] = leak_path

        value_path = figures_dir / VALUE_CAPTURE_FIGURE
        _write_value_capture(report.tiers, value_path)
        written["value_capture"] = value_path

    if report.cohort_shift:
        cohort_path = figures_dir / COHORT_SHIFT_FIGURE
        _write_cohort_shift(report.cohort_shift, cohort_path)
        written["cohort_shift"] = cohort_path

    return written


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _render_markdown(report: ReleaseQualityReport) -> str:
    """Human-readable report; every claim cites the JSON path that backs it.

    Format invariant: every metric value is followed by a parenthesised
    ``(<json-path>)`` reference matching ``$.tiers.<tier>....`` so a
    reader can grep the markdown and find the exact field in
    ``validation_report.json``.  G10.6 in ``v1_acceptance_gates.md``
    requires this.
    """
    out: list[str] = []
    out.append(f"# {report.release_id} — release quality report")
    out.append("")
    out.append(f"**Package version:** `{report.package_version}`  ")
    out.append(f"**Generated:** `{report.generation_timestamp}`  ")
    out.append(f"**Seeds:** {report.seeds}  ")
    out.append(
        "Every value below cites the JSON field that backs it; see "
        f"`{REPORT_JSON}` for the machine-readable form."
    )
    out.append("")

    out.append("## Per-tier headline metrics")
    out.append("")
    out.append(
        "| Tier | Conv. rate (test) | LR AUC | GBM AUC | GBM−LR | LR AP | Brier | "
        "Cal. max-bin err | Top-decile rate |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for tier_name, csm in sorted(report.tiers.items()):
        m = csm.medians
        path = f"$.tiers.{tier_name}.medians"
        out.append(
            "| {tier} | {rate} | {lr} | {gbm} | {delta} | {ap} | {br} | {cal} | {td} |".format(
                tier=tier_name,
                rate=_fmt(m.get("conversion_rate_test"), f"{path}.conversion_rate_test"),
                lr=_fmt(m.get("lr_auc"), f"{path}.lr_auc"),
                gbm=_fmt(m.get("gbm_auc"), f"{path}.gbm_auc"),
                delta=_fmt(m.get("gbm_minus_lr_auc"), f"{path}.gbm_minus_lr_auc"),
                ap=_fmt(m.get("lr_average_precision"), f"{path}.lr_average_precision"),
                br=_fmt(m.get("brier_score"), f"{path}.brier_score"),
                cal=_fmt(
                    m.get("calibration_max_bin_error"),
                    f"{path}.calibration_max_bin_error",
                ),
                td=_fmt(m.get("top_decile_rate"), f"{path}.top_decile_rate"),
            )
        )
    out.append("")

    out.append("## Cross-seed stability (G8.1)")
    out.append("")
    out.append("| Tier | Seeds | LR AUC spread | GBM AUC spread | AP spread | Brier spread |")
    out.append("|---|---|---|---|---|---|")
    for tier_name, csm in sorted(report.tiers.items()):
        sp = csm.spreads
        path = f"$.tiers.{tier_name}.spreads"
        out.append(
            "| {tier} | {seeds} | {lr} | {gbm} | {ap} | {br} |".format(
                tier=tier_name,
                seeds=csm.seeds,
                lr=_fmt(sp.get("lr_auc"), f"{path}.lr_auc"),
                gbm=_fmt(sp.get("gbm_auc"), f"{path}.gbm_auc"),
                ap=_fmt(sp.get("lr_average_precision"), f"{path}.lr_average_precision"),
                br=_fmt(sp.get("brier_score"), f"{path}.brier_score"),
            )
        )
    out.append("")

    out.append("## Cross-tier ordering (G7.4)")
    out.append("")
    ord_path = "$.cross_tier_ordering"
    o = report.cross_tier_ordering
    out.append(
        f"- AP ranking (descending): {o.by_average_precision} (`{ord_path}.by_average_precision`)"
    )
    out.append(
        f"- P@100 ranking (descending): {o.by_precision_at_100} (`{ord_path}.by_precision_at_100`)"
    )
    out.append(f"- GBM−LR ranking (descending): {o.by_gbm_minus_lr} (`{ord_path}.by_gbm_minus_lr`)")
    out.append(
        f"- Conversion-rate ranking (descending): {o.by_conversion_rate} "
        f"(`{ord_path}.by_conversion_rate`)"
    )
    out.append(
        f"- AP intro > intermediate: **{o.average_precision_intro_gt_intermediate}** "
        f"(`{ord_path}.average_precision_intro_gt_intermediate`)"
    )
    out.append(
        f"- AP intermediate > advanced: **{o.average_precision_intermediate_gt_advanced}** "
        f"(`{ord_path}.average_precision_intermediate_gt_advanced`)"
    )
    out.append(
        f"- GBM−LR positive in every tier: **{o.gbm_minus_lr_positive_in_every_tier}** "
        f"(`{ord_path}.gbm_minus_lr_positive_in_every_tier`)"
    )
    out.append("")

    out.append("## Cohort-shift evaluation (G6.4)")
    out.append("")
    out.append("| Tier | Random-split AUC | Cohort-split AUC | Degradation (random − cohort) |")
    out.append("|---|---|---|---|")
    for tier_name, cs in sorted(report.cohort_shift.items()):
        cs_path = f"$.cohort_shift.{tier_name}"
        out.append(
            "| {tier} | {r} | {c} | {d} |".format(
                tier=tier_name,
                r=_fmt(cs.random_split_auc, f"{cs_path}.random_split_auc"),
                c=_fmt(cs.cohort_split_auc, f"{cs_path}.cohort_split_auc"),
                d=_fmt(cs.auc_degradation, f"{cs_path}.auc_degradation"),
            )
        )
    out.append("")

    out.append("## Baseline AUCs (G5.* / leakage probes)")
    out.append("")
    out.append("Each cell is HistGBM AUC trained on the named feature subset only.")
    out.append("")
    baseline_names = sorted(
        {name for csm in report.tiers.values() for tm in csm.per_seed for name in tm.baselines}
    )
    if baseline_names:
        header = "| Tier | seed | " + " | ".join(baseline_names) + " |"
        sep = "|---|---|" + "|".join(["---"] * len(baseline_names)) + "|"
        out.append(header)
        out.append(sep)
        for tier_name, csm in sorted(report.tiers.items()):
            for tm in csm.per_seed:
                cells = [tier_name, str(tm.seed)]
                for bn in baseline_names:
                    cell_path = f"$.tiers.{tier_name}.per_seed[seed={tm.seed}].baselines.{bn}"
                    cells.append(_fmt(tm.baselines.get(bn), cell_path))
                out.append("| " + " | ".join(cells) + " |")
        out.append("")
    else:
        out.append("_No baseline AUCs were computed._")
        out.append("")

    out.append("## Figures")
    out.append("")
    figures_paths = [
        f"`{FIGURES_DIRNAME}/{LIFT_CURVE_FIGURE_TEMPLATE.format(tier=t)}`"
        for t in _LIFT_CURVE_TIERS
        if t in report.tiers
    ]
    if figures_paths:
        out.append("- Lift curves: " + ", ".join(figures_paths))
    if _CALIBRATION_TIER in report.tiers:
        out.append(f"- Calibration ({_CALIBRATION_TIER}): `{FIGURES_DIRNAME}/{CALIBRATION_FIGURE}`")
    if report.tiers:
        out.append(f"- Leakage / baseline deltas: `{FIGURES_DIRNAME}/{LEAKAGE_DELTA_FIGURE}`")
        out.append(f"- Value capture: `{FIGURES_DIRNAME}/{VALUE_CAPTURE_FIGURE}`")
    if report.cohort_shift:
        out.append(f"- Cohort shift: `{FIGURES_DIRNAME}/{COHORT_SHIFT_FIGURE}`")
    out.append("")

    out.append("---")
    out.append("")
    out.append("**Gate references** (see `docs/release/v1_acceptance_gates.md`):")
    out.append("")
    out.append("- **G6.4** — Cohort/time-shift AUC degradation band.")
    out.append("- **G7.\\*** — Per-tier ROC-AUC, AP, P@K, lift, calibration bands.")
    out.append("- **G7.4** — Cross-tier ordering (AP / P@K / GBM−LR / conversion-rate).")
    out.append("- **G8.1** — Cross-seed stability (per-metric spread within tolerance).")
    out.append("")
    out.append(f"_Renderer: `leadforge.validation.reporting`. JSON sibling: `{REPORT_JSON}`._")
    return "\n".join(out) + "\n"


def _fmt(value: Any, json_path: str) -> str:
    """Format a numeric metric with its JSON path citation appended."""
    if value is None:
        return f"_n/a_ (`{json_path}`)"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return f"_n/a_ (`{json_path}`)"
        return f"{value:.4f} (`{json_path}`)"
    if isinstance(value, int):
        return f"{value} (`{json_path}`)"
    return f"{value} (`{json_path}`)"


# ---------------------------------------------------------------------------
# Figures.  Determinism notes:
#  - We never call ``plt.show`` or any interactive backend.
#  - All bar / line orderings are sorted alphabetically by tier so the
#    PNG bytes are stable across runs.
#  - We ``close`` every figure after writing, otherwise long-running
#    drivers (PR 3.3) accumulate matplotlib state.
# ---------------------------------------------------------------------------


def _figure(figsize: tuple[float, float] = (6.0, 4.0)) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120, format="png")
    plt.close(fig)


def _write_lift_curve(csm: CrossSeedTierMetrics, path: Path) -> None:
    """Cumulative-gains chart at the median seed for one tier.

    Plots the actual ``cumulative_gains`` curve sampled by
    :func:`leadforge.validation.release_quality._cumulative_gains_curve`.
    Earlier versions of this function fabricated the curve by
    interpolating between the three measured ``lift_at_pct`` points
    (1% / 5% / 10%) and then jumping straight to (100%, 1.0); that lied
    about model quality between the data points and saturated at 1.0
    for high-lift models.  The fix is to plot the precomputed curve
    directly — no interpolation tricks.
    """
    if not csm.per_seed:
        empty_fig, _ = _figure()
        _save(empty_fig, path)
        return
    metrics = csm.per_seed[len(csm.per_seed) // 2]
    fig, ax = _figure(figsize=(6.0, 5.0))

    points: list[tuple[float, float]] = []
    for key, v in metrics.cumulative_gains.items():
        try:
            pct = float(key)
        except ValueError:
            continue
        if v is None or math.isnan(v):
            continue
        points.append((pct, v))
    points.sort()
    if points:
        xs = [p for p, _ in points]
        ys = [v for _, v in points]
        ax.plot(xs, ys, marker="o", label=f"{csm.tier} (median seed)")
    ax.plot([0, 100], [0, 1], linestyle="--", color="grey", label="random")
    ax.set_xlabel("Top-K% of leads (sorted by predicted P(convert))")
    ax.set_ylabel("Fraction of positives captured")
    ax.set_title(f"Cumulative gains — {csm.tier}")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":")
    _save(fig, path)


def _write_calibration_curve(csm: CrossSeedTierMetrics, path: Path) -> None:
    """Reliability diagram for the canonical (median) seed of a tier."""
    fig, ax = _figure(figsize=(5.0, 5.0))
    if not csm.per_seed:
        _save(fig, path)
        return
    metrics = csm.per_seed[len(csm.per_seed) // 2]
    bins = list(metrics.calibration_bins)
    if bins:
        xs = [b.mean_predicted for b in bins]
        ys = [b.mean_actual for b in bins]
        ax.plot(xs, ys, marker="o", label=csm.tier)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted P(convert)")
    ax.set_ylabel("Empirical conversion rate")
    ax.set_title(f"Reliability diagram — {csm.tier}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":")
    _save(fig, path)


def _write_leakage_delta(tiers: Mapping[str, CrossSeedTierMetrics], path: Path) -> None:
    """Bar chart of baseline AUCs per tier — the leakage-delta panel.

    ``id_only`` should hover near 0.5 (G5.3); ``post_snapshot_aggregates``
    well above LR is the trap signal; ``stage_only`` is typically absent
    from public bundles.
    """
    fig, ax = _figure(figsize=(8.0, 5.0))
    baseline_names = sorted(
        {name for csm in tiers.values() for tm in csm.per_seed for name in tm.baselines}
    )
    if not baseline_names:
        _save(fig, path)
        return
    tier_names = sorted(tiers.keys())
    n_groups = len(tier_names)
    n_bars = len(baseline_names)
    bar_w = 0.8 / max(1, n_bars)
    xs = np.arange(n_groups)
    for i, bn in enumerate(baseline_names):
        ys: list[float] = []
        for tier_name in tier_names:
            csm = tiers[tier_name]
            seed_aucs: list[float] = [tm.baselines[bn] for tm in csm.per_seed if bn in tm.baselines]
            ys.append(float(np.median(seed_aucs)) if seed_aucs else 0.0)
        ax.bar(xs + i * bar_w, ys, width=bar_w, label=bn)
    ax.set_xticks(xs + bar_w * (n_bars - 1) / 2)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("AUC (median across seeds)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="grey", linestyle="--", label="random (0.5)")
    ax.set_title("Baseline AUCs per tier")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", axis="y")
    _save(fig, path)


def _write_cohort_shift(cohort: Mapping[str, CohortShiftMetrics], path: Path) -> None:
    """Side-by-side bars: random vs chronological-cohort split AUC per tier."""
    fig, ax = _figure(figsize=(7.0, 4.5))
    tier_names = sorted(cohort.keys())
    xs = np.arange(len(tier_names))
    rand = [cohort[t].random_split_auc for t in tier_names]
    coh = [
        cohort[t].cohort_split_auc if not math.isnan(cohort[t].cohort_split_auc) else 0.0
        for t in tier_names
    ]
    width = 0.35
    ax.bar(xs - width / 2, rand, width=width, label="random split AUC")
    ax.bar(xs + width / 2, coh, width=width, label="cohort-shift AUC")
    ax.set_xticks(xs)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.set_title("Cohort-shift evaluation")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", axis="y")
    _save(fig, path)


def _write_value_capture(tiers: Mapping[str, CrossSeedTierMetrics], path: Path) -> None:
    """ACV captured at K (across the K values in :data:`PRECISION_KS`)."""
    fig, ax = _figure(figsize=(7.0, 4.5))
    has_any = False
    for tier_name in sorted(tiers.keys()):
        csm = tiers[tier_name]
        # Median across seeds for each K.
        ys = []
        xs: list[int] = []
        for k in PRECISION_KS:
            vals = [
                m.expected_acv_capture_at_k.get(str(k))
                for m in csm.per_seed
                if str(k) in m.expected_acv_capture_at_k
            ]
            vals_clean = [v for v in vals if v is not None and not math.isnan(v)]
            if vals_clean:
                xs.append(int(k))
                ys.append(float(np.median(vals_clean)))
        if xs:
            has_any = True
            ax.plot(xs, ys, marker="o", label=tier_name)
    if not has_any:
        _save(fig, path)
        return
    ax.set_xlabel("Top-K leads ranked by P(convert)")
    ax.set_ylabel("Fraction of total converted-ACV captured")
    ax.set_title("Value capture at top-K")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    ax.grid(True, linestyle=":")
    _save(fig, path)


# ---------------------------------------------------------------------------
# Re-export for convenience — callers commonly import the top-level names
# from this module rather than ``release_quality``.
# ---------------------------------------------------------------------------

__all__ = [
    "CALIBRATION_FIGURE",
    "COHORT_SHIFT_FIGURE",
    "FIGURES_DIRNAME",
    "LEAKAGE_DELTA_FIGURE",
    "LIFT_CURVE_FIGURE_TEMPLATE",
    "REPORT_JSON",
    "REPORT_MD",
    "VALUE_CAPTURE_FIGURE",
    "render_report",
]
