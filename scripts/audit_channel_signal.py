#!/usr/bin/env python3
"""Audit how strongly the lead-source channel signals conversion.

Companion analysis for PR 4.1 (recommendation #8 v1 scope from
``docs/external_review/summaries/recommendations_pass.md``).  For every
tier in a release bundle family we compute, for each channel column
(default: ``lead_source``):

* per-channel conversion rate, share, and counts on the **train** split
* the **in-sample** univariate AUC: per-channel rates derived on train
  and scored against train labels (a 1-D Bayes classifier; biased upward
  for small categorical alphabets)
* the **out-of-sample** univariate AUC: per-channel rates derived on
  train and scored against **test** labels — directly comparable to the
  ``source_only`` baselines in ``release/validation/validation_report.json``

The script does not assign a categorical "weak / moderate / strong"
verdict.  Industry MQL→SQL benchmarks are surfaced for context only;
they measure a different funnel transition (single MQL→SQL step, not
the 90-day closed-won label v1 reports), so a hard comparison would be
a category error.  The audit doc states the v1 numbers and an explicit
caveat; readers draw the comparison.

Outputs (defaults are pinned via the v1 acceptance gates):

* ``docs/release/channel_signal_audit.md`` — human-readable audit
* ``docs/release/channel_signal_audit.json`` — machine-readable sibling

The script is deterministic given a fixed bundle: it reads
``train.parquet`` and ``test.parquet`` only, derives empirical rates,
and uses ``sklearn.metrics.roc_auc_score`` with no fit-time randomness.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL_COLUMNS: Final[tuple[str, ...]] = ("lead_source",)
LABEL_COLUMN: Final[str] = "converted_within_90_days"
DEFAULT_TIERS: Final[tuple[str, ...]] = ("intro", "intermediate", "advanced")
DEFAULT_TASK: Final[str] = "converted_within_90_days"

#: G2 industry MQL→SQL conversion rates surfaced in
#: ``docs/external_review/summaries/gemini_v2_summary.md`` (recommendation #8).
#: They measure a single MQL→SQL transition, NOT v1's 90-day closed-won
#: label.  Stored as a tuple of pairs so the dataclass field is genuinely
#: immutable; converted to a plain dict at JSON-render time.
INDUSTRY_MQL_TO_SQL_BENCHMARKS: Final[tuple[tuple[str, float], ...]] = (
    ("Email", 0.005),
    ("PPC", 0.26),
    ("SEO", 0.51),
)

DEFAULT_RELEASE_DIR: Final[Path] = Path("release")
DEFAULT_OUT_MD: Final[Path] = Path("docs/release/channel_signal_audit.md")
DEFAULT_OUT_JSON: Final[Path] = Path("docs/release/channel_signal_audit.json")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel rollup for one channel column on the train split."""

    name: str
    n: int
    share: float
    n_converted: int
    conversion_rate: float


@dataclass(frozen=True)
class ChannelAudit:
    """Audit results for one channel column in one tier.

    Per-channel statistics come from the train split.
    ``univariate_auc_in_sample`` re-uses train labels (bias-prone but
    matches the historical 1-D Bayes-classifier interpretation);
    ``univariate_auc_out_of_sample`` scores the train-derived rates
    against the held-out test split.
    """

    column: str
    n_train: int
    n_test: int
    train_conversion_rate: float
    test_conversion_rate: float
    channels: tuple[ChannelStats, ...]
    rate_spread: float
    univariate_auc_in_sample: float
    univariate_auc_out_of_sample: float


@dataclass(frozen=True)
class ChannelGroup:
    """One or more channel columns with byte-identical audit values.

    Allows the markdown renderer to collapse deduplicate columns into one
    section without losing information.
    """

    columns: tuple[str, ...]
    audit: ChannelAudit


@dataclass(frozen=True)
class TierAudit:
    """Audit results for one tier across every channel column."""

    tier: str
    n_train: int
    n_test: int
    train_conversion_rate: float
    test_conversion_rate: float
    columns: tuple[ChannelAudit, ...]


@dataclass(frozen=True)
class AuditReport:
    """Full audit: every requested tier × channel column."""

    release_dir: str
    task: str
    label_column: str
    channel_columns: tuple[str, ...]
    tiers: tuple[TierAudit, ...]
    industry_mql_to_sql_benchmarks: tuple[tuple[str, float], ...]


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def _label_to_int(series: pd.Series) -> pd.Series:
    """Coerce a label column to ``int``.

    Handles three dtypes the v1 bundles actually carry: numpy ``bool``,
    pandas nullable ``BooleanDtype`` (used by the parquet schema), and
    plain numeric.  Other dtypes raise via ``pd.to_numeric``.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64").astype(int)
    return pd.to_numeric(series, errors="raise").astype(int)


def _conversion_rate(df: pd.DataFrame, label_col: str) -> float:
    if len(df) == 0:
        return 0.0
    return float(int(_label_to_int(df[label_col]).sum()) / len(df))


def _auc_or_chance(y: pd.Series, scores: pd.Series) -> float:
    """ROC AUC, falling back to ``0.5`` when undefined (single class)."""

    if y.nunique() < 2:
        return 0.5
    return float(roc_auc_score(y.to_numpy(), scores.to_numpy()))


def audit_channel(
    train: pd.DataFrame,
    channel_col: str,
    *,
    test: pd.DataFrame,
    label_col: str = LABEL_COLUMN,
) -> ChannelAudit:
    """Per-channel stats and univariate AUCs (in-sample + OOS).

    Both AUCs use the same scoring function: the per-channel positive
    rate derived from the train split.  The "in-sample" AUC scores
    that against train labels (biased upward by construction); the
    "out-of-sample" AUC scores it against held-out test labels and
    is directly comparable to the ``source_only`` baselines in
    ``release/validation/validation_report.json``.
    """

    for df_name, df in (("train", train), ("test", test)):
        if channel_col not in df.columns:
            raise KeyError(f"channel column {channel_col!r} not present in {df_name}")
        if label_col not in df.columns:
            raise KeyError(f"label column {label_col!r} not present in {df_name}")

    y_train = _label_to_int(train[label_col])
    n_train = len(train)
    n_test = len(test)
    train_rate = float(int(y_train.sum()) / n_train) if n_train else 0.0
    test_rate = _conversion_rate(test, label_col)

    grouped = train.assign(_y=y_train).groupby(channel_col, dropna=False)
    rows: list[ChannelStats] = []
    for name, sub in sorted(grouped, key=lambda kv: str(kv[0])):
        n = len(sub)
        n_conv = int(sub["_y"].sum())
        rows.append(
            ChannelStats(
                name=str(name),
                n=n,
                share=float(n / n_train) if n_train else 0.0,
                n_converted=n_conv,
                conversion_rate=float(n_conv / n) if n else 0.0,
            )
        )

    rate_spread = (
        max(c.conversion_rate for c in rows) - min(c.conversion_rate for c in rows) if rows else 0.0
    )

    if len(rows) < 2:
        in_sample_auc = 0.5
        oos_auc = 0.5
    else:
        rate_lookup = {c.name: c.conversion_rate for c in rows}
        train_scores = train[channel_col].astype(str).map(rate_lookup).astype(float)
        in_sample_auc = _auc_or_chance(y_train, train_scores)

        # Test-set channels are scored using the train-derived rates;
        # any channel value unseen on train falls back to the train
        # base rate so the AUC stays well-defined.
        test_scores = (
            test[channel_col].astype(str).map(rate_lookup).fillna(train_rate).astype(float)
        )
        y_test = _label_to_int(test[label_col])
        oos_auc = _auc_or_chance(y_test, test_scores)

    return ChannelAudit(
        column=channel_col,
        n_train=n_train,
        n_test=n_test,
        train_conversion_rate=train_rate,
        test_conversion_rate=test_rate,
        channels=tuple(rows),
        rate_spread=float(rate_spread),
        univariate_auc_in_sample=in_sample_auc,
        univariate_auc_out_of_sample=oos_auc,
    )


def audit_tier(
    train: pd.DataFrame,
    tier: str,
    *,
    test: pd.DataFrame,
    channel_columns: Sequence[str] = CHANNEL_COLUMNS,
    label_col: str = LABEL_COLUMN,
) -> TierAudit:
    """Run :func:`audit_channel` for every channel column on one tier."""

    train_rate = _conversion_rate(train, label_col)
    test_rate = _conversion_rate(test, label_col)
    columns = tuple(
        audit_channel(train, col, test=test, label_col=label_col) for col in channel_columns
    )
    return TierAudit(
        tier=tier,
        n_train=len(train),
        n_test=len(test),
        train_conversion_rate=train_rate,
        test_conversion_rate=test_rate,
        columns=columns,
    )


def load_split(release_dir: Path, tier: str, split: str, task: str = DEFAULT_TASK) -> pd.DataFrame:
    """Load ``release_dir/<tier>/tasks/<task>/<split>.parquet``."""

    path = release_dir / tier / "tasks" / task / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing {split} split for tier {tier!r}: {path}")
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
        train = load_split(release_dir, tier, "train", task=task)
        test = load_split(release_dir, tier, "test", task=task)
        tier_audits.append(
            audit_tier(
                train,
                tier=tier,
                test=test,
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
        industry_mql_to_sql_benchmarks=INDUSTRY_MQL_TO_SQL_BENCHMARKS,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def report_to_dict(report: AuditReport) -> dict[str, Any]:
    """Convert the report to a JSON-primitive dict.

    The dataclass stores ``industry_mql_to_sql_benchmarks`` as a tuple
    of pairs (immutability); this helper converts it back into a
    ``{name: rate}`` mapping for the JSON output, where a dict shape
    is more ergonomic for downstream tooling.
    """

    payload = asdict(report)
    payload["industry_mql_to_sql_benchmarks"] = dict(report.industry_mql_to_sql_benchmarks)
    return payload


def render_json(report: AuditReport) -> str:
    """Render the audit report as a deterministic JSON string."""

    return json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n"


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _audit_signature(audit: ChannelAudit) -> tuple[Any, ...]:
    """Hashable signature used to group columns whose audits are identical."""

    return (
        audit.n_train,
        audit.n_test,
        audit.train_conversion_rate,
        audit.test_conversion_rate,
        tuple(_stats_signature(c) for c in audit.channels),
        audit.rate_spread,
        audit.univariate_auc_in_sample,
        audit.univariate_auc_out_of_sample,
    )


def _stats_signature(stats: ChannelStats) -> tuple[Any, ...]:
    """Hashable tuple representing one ``ChannelStats``."""

    return (stats.name, stats.n, stats.share, stats.n_converted, stats.conversion_rate)


def _group_identical_columns(audits: Sequence[ChannelAudit]) -> list[ChannelGroup]:
    """Collapse columns whose audit values are byte-identical."""

    groups: list[ChannelGroup] = []
    seen_signatures: dict[tuple[Any, ...], int] = {}
    for audit in audits:
        sig = _audit_signature(audit)
        if sig in seen_signatures:
            idx = seen_signatures[sig]
            existing = groups[idx]
            groups[idx] = ChannelGroup(
                columns=existing.columns + (audit.column,),
                audit=existing.audit,
            )
        else:
            seen_signatures[sig] = len(groups)
            groups.append(ChannelGroup(columns=(audit.column,), audit=audit))
    return groups


def render_markdown(
    report: AuditReport,
    *,
    md_path: Path | None = None,
    json_path: Path | None = None,
) -> str:
    """Render the audit report as Markdown.

    The inline "see also" link to the machine-readable sibling adapts
    to the actual output paths: when ``md_path`` and ``json_path`` are
    given, the link is the JSON path expressed *relative to the
    markdown file's directory* so it works whether the artifacts are
    written to the canonical ``docs/release/`` location, a tmp
    directory, or anywhere a CI script overrides.  When neither is
    given, the link is the canonical ``channel_signal_audit.json``
    filename.
    """

    if md_path is not None and json_path is not None:
        try:
            json_link = str(Path(json_path).relative_to(Path(md_path).parent))
        except ValueError:
            # Different drive roots — keep the markdown readable by
            # falling back to the caller's path verbatim.
            json_link = str(json_path)
    else:
        json_link = DEFAULT_OUT_JSON.name

    lines: list[str] = []
    lines.append("# Channel-signal audit — leadforge-lead-scoring-v1")
    lines.append("")
    lines.append(
        "Audit produced by `scripts/audit_channel_signal.py`; see "
        f"`{json_link}` for the machine-readable form."
    )
    lines.append("")
    lines.append(
        "**Scope.** For every tier we compute per-channel conversion rates on the train "
        "split and the univariate AUC of channel against `converted_within_90_days`, "
        "scored as the empirical positive rate per channel (a 1-D Bayes classifier). Two "
        "AUCs are reported: an **in-sample** number (train rates → train labels — biased "
        "upward by construction) and an **out-of-sample** number (train rates → test labels "
        "— directly comparable to the `source_only` baselines in "
        "`release/validation/validation_report.json`)."
    )
    lines.append("")
    lines.append(
        "**Caveat on the industry benchmark.** The G2 / Gemini v2 numbers below are "
        "single-step **MQL→SQL** rates (recommendation #8 in "
        "`docs/external_review/summaries/recommendations_pass.md`). v1's label is "
        "**90-day closed-won**, the entire funnel resolved. The two metrics are not "
        "directly comparable; the table is reproduced for context only."
    )
    lines.append("")

    lines.append("## Industry benchmark (context, not target)")
    lines.append("")
    lines.append("| Channel | MQL→SQL conversion rate |")
    lines.append("|---|---|")
    for name, rate in report.industry_mql_to_sql_benchmarks:
        lines.append(f"| {name} | {_format_pct(rate)} |")
    lines.append("")

    for tier in report.tiers:
        lines.append(f"## Tier: `{tier.tier}`")
        lines.append("")
        lines.append(
            f"`n_train = {tier.n_train}` (90-day conversion rate "
            f"{_format_pct(tier.train_conversion_rate)}); "
            f"`n_test = {tier.n_test}` (rate "
            f"{_format_pct(tier.test_conversion_rate)})."
        )
        lines.append("")

        groups = _group_identical_columns(tier.columns)
        for group in groups:
            cols_label = ", ".join(f"`{c}`" for c in group.columns)
            if len(group.columns) > 1:
                heading = f"### Columns: {cols_label} (audit values identical)"
            else:
                heading = f"### Column: {cols_label}"
            lines.append(heading)
            lines.append("")
            lines.append(
                f"Per-channel rate spread (max − min): **{group.audit.rate_spread:.4f}**  ·  "
                f"In-sample univariate AUC: **{group.audit.univariate_auc_in_sample:.4f}**  ·  "
                f"Out-of-sample univariate AUC: **{group.audit.univariate_auc_out_of_sample:.4f}**"
            )
            lines.append("")
            lines.append("| Channel | n (train) | Share (train) | Converted (train) | Train rate |")
            lines.append("|---|---:|---:|---:|---:|")
            for ch in group.audit.channels:
                lines.append(
                    f"| `{ch.name}` | {ch.n} | {_format_pct(ch.share)} | "
                    f"{ch.n_converted} | {_format_pct(ch.conversion_rate)} |"
                )
            lines.append("")

    lines.append("## Discussion")
    lines.append("")
    lines.append(
        "The numbers above answer one question: *how strongly does channel alone signal "
        "90-day conversion in v1?* They do not answer *whether v1 matches industry channel "
        "performance*, since the benchmarks measure a different funnel transition (single "
        "MQL→SQL step) and v1 measures the entire funnel resolved over 90 days. Treat the "
        "v1 numbers as an internal description of the simulator's channel signal."
    )
    lines.append("")
    lines.append("Two empirical observations a reader can make from the numbers above:")
    lines.append("")
    lines.append(
        "1. **The out-of-sample univariate AUC is the comparable number** for any "
        "external baseline. It uses train-derived rates scored against held-out test "
        "labels — the same shape as the `source_only` HistGBM baseline reported in "
        "`release/validation/validation_report.json`, which is built on the same task "
        "splits with `lead_source` as the only feature. The "
        "in-sample number is biased upward by construction — small at v1's N but "
        "visible — and is reported here for transparency rather than comparison."
    )
    lines.append(
        "2. **The numerical conclusion is bundle-specific.** When the per-channel rate "
        "spread is small and the OOS univariate AUC is close to chance, channel alone "
        "is a weak feature for the bundle this audit was run against. v1's bundles "
        "currently produce that outcome (see the per-tier sections above) — consistent "
        "with the design: the simulator drives conversion through motif-family hazards "
        "keyed off latent traits, not channel-conditional probabilities. "
        "Channel-conditional encoding is tracked as post-v1 work in "
        "`docs/release/post_v1_roadmap.md`."
    )
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
        help="channel column to audit (repeatable; default: lead_source)",
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

    md = render_markdown(report, md_path=args.out_md, json_path=args.out_json)
    js = render_json(report)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    # Pin UTF-8 explicitly so the audit output is byte-identical across
    # operating systems and locale configurations.
    args.out_md.write_text(md, encoding="utf-8")
    args.out_json.write_text(js, encoding="utf-8")

    if args.print:
        sys.stdout.write(md)

    print(f"wrote {args.out_md}", file=sys.stderr)
    print(f"wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
