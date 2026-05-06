#!/usr/bin/env python3
"""Audit how strongly the lead-source channel signals conversion.

Companion analysis for PR 4.1 (recommendation #8 v1 scope from
``docs/external_review/summaries/recommendations_pass.md``).  For every
tier in a release bundle family we compute:

* conversion rate by channel (``lead_source`` and ``first_touch_channel``)
* the univariate AUC of channel against ``converted_within_90_days``,
  scored as the empirical positive rate per channel (a 1-D Bayes
  classifier; equivalent to a saturated logistic regression on one-hot
  channel features)

and compare those to the G2 / Gemini v2 industry MQL→SQL benchmarks.

Outputs (defaults are pinned via the v1 acceptance gates):

* ``docs/release/channel_signal_audit.md`` — human-readable audit
* ``docs/release/channel_signal_audit.json`` — machine-readable sibling

The script is deterministic given a fixed bundle: it reads
``train.parquet`` only, derives empirical rates, and uses
``sklearn.metrics.roc_auc_score`` with no fit-time randomness.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL_COLUMNS: Final[tuple[str, ...]] = ("lead_source", "first_touch_channel")
LABEL_COLUMN: Final[str] = "converted_within_90_days"
DEFAULT_TIERS: Final[tuple[str, ...]] = ("intro", "intermediate", "advanced")
DEFAULT_TASK: Final[str] = "converted_within_90_days"

#: G2 industry MQL→SQL conversion rates surfaced in
#: ``gemini_v2_summary.md`` (recommendation #8).  These are not directly
#: comparable to v1's 90-day closed-won label, but they are the closest
#: public anchor for "how much should channel matter" and the audit
#: reports the comparison band rather than asserting a hard match.
INDUSTRY_MQL_TO_SQL_BENCHMARKS: Final[Mapping[str, float]] = {
    "SEO": 0.51,
    "PPC": 0.26,
    "Email": 0.005,
}

DEFAULT_RELEASE_DIR: Final[Path] = Path("release")
DEFAULT_OUT_MD: Final[Path] = Path("docs/release/channel_signal_audit.md")
DEFAULT_OUT_JSON: Final[Path] = Path("docs/release/channel_signal_audit.json")

#: Bands used to label the verdict for each channel column.  Tuned to
#: surface "weak / moderate / strong" against G2-style benchmarks where
#: SEO vs Email differs by ~50 percentage points.  Bands operate on the
#: per-channel max-min conversion-rate spread.
SIGNAL_BAND_WEAK_MAX: Final[float] = 0.05
SIGNAL_BAND_MODERATE_MAX: Final[float] = 0.15
AUC_NEAR_CHANCE_MAX: Final[float] = 0.55


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel rollup for one channel column in one tier."""

    name: str
    n: int
    share: float
    n_converted: int
    conversion_rate: float


@dataclass(frozen=True)
class ChannelAudit:
    """Audit results for one channel column in one tier."""

    column: str
    n_total: int
    overall_conversion_rate: float
    channels: tuple[ChannelStats, ...]
    rate_spread: float
    univariate_auc: float


@dataclass(frozen=True)
class TierAudit:
    """Audit results for one tier across every channel column."""

    tier: str
    n_leads: int
    conversion_rate_overall: float
    columns: tuple[ChannelAudit, ...]


@dataclass(frozen=True)
class AuditReport:
    """Full audit: every requested tier × channel column."""

    release_dir: str
    task: str
    label_column: str
    channel_columns: tuple[str, ...]
    tiers: tuple[TierAudit, ...]
    industry_mql_to_sql_benchmarks: Mapping[str, float]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _label_to_int(series: pd.Series) -> pd.Series:
    """Coerce a (possibly nullable boolean) label to ``int``."""

    if series.dtype == "bool":
        return series.astype(int)
    return pd.to_numeric(series, errors="raise").astype(int)


def audit_channel(
    df: pd.DataFrame,
    channel_col: str,
    label_col: str = LABEL_COLUMN,
) -> ChannelAudit:
    """Per-channel stats + univariate AUC for a single channel column.

    ``univariate_auc`` is the AUC obtained by replacing each row's
    channel value with that channel's empirical positive rate.  This is
    a 1-D Bayes classifier, equivalent (up to ties) to a saturated
    logistic regression on one-hot channel features and stable across
    sklearn versions.  Returns ``0.5`` when the label has only one
    class, since AUC is undefined.
    """

    if channel_col not in df.columns:
        raise KeyError(f"channel column {channel_col!r} not present")
    if label_col not in df.columns:
        raise KeyError(f"label column {label_col!r} not present")

    y = _label_to_int(df[label_col])
    n_total = len(df)
    n_converted_total = int(y.sum())
    overall_rate = float(n_converted_total / n_total) if n_total else 0.0

    # Per-channel rollup, sorted by name for determinism.
    grouped = df.assign(_y=y).groupby(channel_col, dropna=False)
    rows: list[ChannelStats] = []
    for name, sub in sorted(grouped, key=lambda kv: str(kv[0])):
        n = len(sub)
        n_conv = int(sub["_y"].sum())
        rows.append(
            ChannelStats(
                name=str(name),
                n=n,
                share=float(n / n_total) if n_total else 0.0,
                n_converted=n_conv,
                conversion_rate=float(n_conv / n) if n else 0.0,
            )
        )

    rate_spread = (
        max(c.conversion_rate for c in rows) - min(c.conversion_rate for c in rows) if rows else 0.0
    )

    if y.nunique() < 2 or len(rows) < 2:
        univariate_auc = 0.5
    else:
        rate_lookup = {c.name: c.conversion_rate for c in rows}
        scores = df[channel_col].astype(str).map(rate_lookup).astype(float)
        univariate_auc = float(roc_auc_score(y.to_numpy(), scores.to_numpy()))

    return ChannelAudit(
        column=channel_col,
        n_total=n_total,
        overall_conversion_rate=overall_rate,
        channels=tuple(rows),
        rate_spread=float(rate_spread),
        univariate_auc=univariate_auc,
    )


def audit_tier(
    df: pd.DataFrame,
    tier: str,
    *,
    channel_columns: Sequence[str] = CHANNEL_COLUMNS,
    label_col: str = LABEL_COLUMN,
) -> TierAudit:
    """Run :func:`audit_channel` for every channel column on one tier."""

    y = _label_to_int(df[label_col])
    n = len(df)
    overall_rate = float(int(y.sum()) / n) if n else 0.0

    columns = tuple(audit_channel(df, col, label_col=label_col) for col in channel_columns)
    return TierAudit(
        tier=tier,
        n_leads=n,
        conversion_rate_overall=overall_rate,
        columns=columns,
    )


def load_train_df(release_dir: Path, tier: str, task: str = DEFAULT_TASK) -> pd.DataFrame:
    """Load ``release_dir/<tier>/tasks/<task>/train.parquet``."""

    path = release_dir / tier / "tasks" / task / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing train split for tier {tier!r}: {path}")
    return pd.read_parquet(path)


def build_report(
    release_dir: Path,
    tiers: Sequence[str] = DEFAULT_TIERS,
    *,
    task: str = DEFAULT_TASK,
    channel_columns: Sequence[str] = CHANNEL_COLUMNS,
    label_col: str = LABEL_COLUMN,
) -> AuditReport:
    """Run the audit across every requested tier."""

    tier_audits: list[TierAudit] = []
    for tier in tiers:
        df = load_train_df(release_dir, tier, task=task)
        tier_audits.append(
            audit_tier(
                df,
                tier=tier,
                channel_columns=channel_columns,
                label_col=label_col,
            )
        )

    return AuditReport(
        release_dir=str(release_dir),
        task=task,
        label_column=label_col,
        channel_columns=tuple(channel_columns),
        tiers=tuple(tier_audits),
        industry_mql_to_sql_benchmarks=dict(INDUSTRY_MQL_TO_SQL_BENCHMARKS),
    )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def _classify_signal(audit: ChannelAudit) -> str:
    """Map (rate spread, univariate AUC) to one of weak/moderate/strong."""

    if audit.univariate_auc < AUC_NEAR_CHANCE_MAX and audit.rate_spread < SIGNAL_BAND_WEAK_MAX:
        return "weak"
    if audit.rate_spread < SIGNAL_BAND_MODERATE_MAX:
        return "moderate"
    return "strong"


def _verdict_paragraph(report: AuditReport) -> str:
    """One-paragraph human-readable verdict."""

    rows = [
        (tier.tier, col.column, col.rate_spread, col.univariate_auc, _classify_signal(col))
        for tier in report.tiers
        for col in tier.columns
    ]
    strengths = {row[4] for row in rows}
    max_spread = max((row[2] for row in rows), default=0.0)
    max_auc = max((row[3] for row in rows), default=0.5)

    seo_minus_email = (
        INDUSTRY_MQL_TO_SQL_BENCHMARKS["SEO"] - INDUSTRY_MQL_TO_SQL_BENCHMARKS["Email"]
    )

    if strengths <= {"weak"}:
        verdict = "weak"
        intent = (
            "well below the G2 / Gemini v2 industry MQL→SQL benchmark band, where SEO leads "
            f"convert {seo_minus_email * 100:.0f} percentage points more than Email leads."
        )
    elif "strong" in strengths:
        verdict = "strong"
        intent = (
            "comparable to or stronger than the G2 / Gemini v2 industry benchmark band — "
            "channel-conditional encoding may already be implicit in v1."
        )
    else:
        verdict = "moderate"
        intent = (
            "below the G2 / Gemini v2 industry benchmark band — channel signal is present but "
            "weaker than published MQL→SQL spreads."
        )

    return (
        f"v1's channel signal is **{verdict}**: across all tiers and both channel columns the "
        f"largest per-channel conversion-rate spread is {max_spread:.3f} and the largest "
        f"univariate AUC is {max_auc:.3f}. That is {intent} v1 drives conversion through "
        "motif-family hazards keyed off latent traits, not channel-conditional probabilities, "
        "so this is the expected outcome; channel-conditional encoding is tracked as post-v1 "
        "work in `docs/release/post_v1_roadmap.md`."
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def report_to_dict(report: AuditReport) -> dict[str, Any]:
    """Convert the report to a JSON-primitive dict (deterministic)."""

    payload = asdict(report)
    payload["industry_mql_to_sql_benchmarks"] = dict(report.industry_mql_to_sql_benchmarks)
    return payload


def render_json(report: AuditReport) -> str:
    """Render the audit report as a deterministic JSON string."""

    return json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n"


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_markdown(report: AuditReport) -> str:
    """Render the audit report as Markdown."""

    lines: list[str] = []
    lines.append("# Channel-signal audit — leadforge-lead-scoring-v1")
    lines.append("")
    lines.append(
        "Audit produced by `scripts/audit_channel_signal.py`; see also "
        "`docs/release/channel_signal_audit.json` for the machine-readable form."
    )
    lines.append("")
    lines.append(
        "**Scope.** For every tier we compute per-channel conversion rates and the univariate "
        "AUC of channel against `converted_within_90_days`, scored as the empirical positive "
        "rate per channel (a 1-D Bayes classifier, equivalent to a saturated logistic "
        "regression on one-hot channel features). Compared against the G2 / Gemini v2 industry "
        "MQL→SQL benchmark band (SEO ~51%, PPC ~26%, Email <1%, surfaced in "
        "`docs/external_review/summaries/recommendations_pass.md` recommendation #8)."
    )
    lines.append("")
    lines.append(
        "**Caveat.** Industry benchmarks are MQL→SQL rates, not 90-day closed-won rates. They "
        "are the closest public anchor for *how much* channel ought to matter; use them as a "
        "band of reference, not a hard target."
    )
    lines.append("")

    lines.append("## Industry benchmark band")
    lines.append("")
    lines.append("| Channel | MQL→SQL conversion rate |")
    lines.append("|---|---|")
    for name, rate in sorted(report.industry_mql_to_sql_benchmarks.items()):
        lines.append(f"| {name} | {_format_pct(rate)} |")
    lines.append("")

    for tier in report.tiers:
        lines.append(f"## Tier: `{tier.tier}`")
        lines.append("")
        lines.append(
            f"`n_leads = {tier.n_leads}`, overall 90-day conversion rate "
            f"{_format_pct(tier.conversion_rate_overall)}."
        )
        lines.append("")

        for col in tier.columns:
            lines.append(f"### Column: `{col.column}`")
            lines.append("")
            lines.append(
                f"Univariate AUC: **{col.univariate_auc:.4f}**  ·  "
                f"Per-channel rate spread (max − min): **{col.rate_spread:.4f}**  ·  "
                f"Verdict: **{_classify_signal(col)} signal**"
            )
            lines.append("")
            lines.append("| Channel | n | Share | Converted | Conversion rate |")
            lines.append("|---|---:|---:|---:|---:|")
            for ch in col.channels:
                lines.append(
                    f"| `{ch.name}` | {ch.n} | {_format_pct(ch.share)} | "
                    f"{ch.n_converted} | {_format_pct(ch.conversion_rate)} |"
                )
            lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(_verdict_paragraph(report))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit how strongly source channel signals conversion in a release "
        "bundle family.",
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help="release bundle root containing one subdirectory per tier (default: %(default)s)",
    )
    parser.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        default=None,
        help="limit the audit to one tier (repeatable; default: intro/intermediate/advanced)",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="task subdirectory under each tier (default: %(default)s)",
    )
    parser.add_argument(
        "--channel-column",
        action="append",
        dest="channel_columns",
        default=None,
        help="channel column to audit (repeatable; default: lead_source + first_touch_channel)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_OUT_MD,
        help="markdown output path (default: %(default)s)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="JSON output path (default: %(default)s)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="print the markdown report to stdout in addition to writing it",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    release_dir: Path = args.release_dir
    tiers: tuple[str, ...] = tuple(args.tiers) if args.tiers else DEFAULT_TIERS
    channel_columns: tuple[str, ...] = (
        tuple(args.channel_columns) if args.channel_columns else CHANNEL_COLUMNS
    )

    if not release_dir.exists():
        print(f"error: release directory not found: {release_dir}", file=sys.stderr)
        return 2

    try:
        report = build_report(
            release_dir,
            tiers,
            task=args.task,
            channel_columns=channel_columns,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: required column missing: {exc}", file=sys.stderr)
        return 2

    md = render_markdown(report)
    js = render_json(report)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    args.out_json.write_text(js)

    if args.print:
        sys.stdout.write(md)

    print(f"wrote {args.out_md}", file=sys.stderr)
    print(f"wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
