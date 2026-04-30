"""Single source of truth for lead scoring dataset validation and baseline evaluation.

This module validates ``lead_scoring_intro_v*.csv`` datasets and computes
reproducible baseline metrics.  All ML evaluation uses deterministic
hold-out splits with preprocessing fit on the training fold only.

Usage (programmatic)::

    from leadforge.validation.lead_scoring import validate_dataset
    report = validate_dataset("lead_scoring_intro_v5.csv")
    print(report.summary())

Usage (CLI)::

    python scripts/validate_lead_scoring_dataset.py --csv lead_scoring_intro_v5.csv
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "converted"

EXPECTED_CAT_FEATURES = [
    "industry",
    "region",
    "company_size",
    "company_revenue",
    "contact_role",
    "seniority",
    "lead_source",
]

EXPECTED_NUMERIC_FEATURES = [
    "expected_acv",
    "inbound_touches",
    "outbound_touches",
    "touches_week_1",
    "web_sessions",
    "sales_activities",
    "days_since_last_touch",
    "days_since_first_touch",
]

EXPECTED_BINARY_FEATURES = [
    "opportunity_created",
    "demo_completed",
]

LEAKAGE_PREFIX = "__leakage__"

BANNED_COLUMNS = {
    "current_stage",
    "funnel_stage",
    "conversion_timestamp",
    "is_sql",
    "is_mql",
    "lead_created_at",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationConfig:
    """Tunable thresholds for dataset validation."""

    expected_rows: int = 1000
    enforce_row_count: bool = False
    max_duplicate_rate: float = 0.01
    max_col_missing_rate: float = 0.10
    min_group_size: int = 50
    min_group_rate: float = 0.02
    max_group_rate: float = 0.98
    auc_lower: float = 0.62
    auc_upper: float = 0.90
    trap_mean_delta: float = 0.03
    trap_min_delta: float = 0.015
    trap_n_seeds: int = 10
    trap_seed_start: int = 42
    ks: tuple[int, ...] = (25, 50)
    test_size: float = 0.30
    default_seed: int = 42


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """One validation check."""

    name: str
    passed: bool
    details: str = ""
    data: dict | None = None


@dataclass
class BaselineMetrics:
    """Metrics from a single baseline evaluation."""

    seed: int
    auc: float
    pr_auc: float
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    lift_at_k: dict[int, float] = field(default_factory=dict)
    base_rate: float = 0.0


@dataclass
class ValueMetrics:
    """Value-aware ranking metrics."""

    k: int
    captured_acv_by_prob: float
    captured_acv_by_ev: float
    uplift_pct: float


@dataclass
class TrapMetrics:
    """Leakage trap evaluation across seeds."""

    column: str
    deltas_auc: list[float]
    deltas_pr_auc: list[float]
    seeds: list[int]

    @property
    def mean_delta_auc(self) -> float:
        return float(np.mean(self.deltas_auc))

    @property
    def min_delta_auc(self) -> float:
        return float(np.min(self.deltas_auc))

    @property
    def max_delta_auc(self) -> float:
        return float(np.max(self.deltas_auc))


@dataclass
class ValidationReport:
    """Full validation report."""

    csv_path: str
    checks: list[CheckResult] = field(default_factory=list)
    baseline: BaselineMetrics | None = None
    value_metrics: list[ValueMetrics] = field(default_factory=list)
    trap_metrics: list[TrapMetrics] = field(default_factory=list)
    missingness: dict[str, float] = field(default_factory=dict)
    n_rows: int = 0
    test_size: float = 0.30

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            line = f"  {status}  {c.name}"
            if c.details:
                line += f"  ({c.details})"
            lines.append(line)

        if self.baseline:
            b = self.baseline
            lines.append(f"\nBaseline (seed={b.seed}, hold-out):")
            lines.append(f"  AUC:    {b.auc:.3f}")
            lines.append(f"  PR-AUC: {b.pr_auc:.3f}")
            lines.append(f"  Base rate: {b.base_rate:.1%}")
            for k in sorted(b.precision_at_k):
                lines.append(
                    f"  P@{k}={b.precision_at_k[k]:.3f}  "
                    f"R@{k}={b.recall_at_k[k]:.3f}  "
                    f"Lift@{k}={b.lift_at_k[k]:.2f}x"
                )

        for vm in self.value_metrics:
            lines.append(
                f"\nValue@{vm.k}: "
                f"by_prob=${vm.captured_acv_by_prob:,.0f}  "
                f"by_ev=${vm.captured_acv_by_ev:,.0f}  "
                f"uplift={vm.uplift_pct:+.1f}%"
            )

        for tm in self.trap_metrics:
            lines.append(
                f"\nTrap '{tm.column}' ({len(tm.seeds)} seeds): "
                f"mean_delta={tm.mean_delta_auc:.4f}  "
                f"min_delta={tm.min_delta_auc:.4f}  "
                f"max_delta={tm.max_delta_auc:.4f}"
            )
            neg = [(s, d) for s, d in zip(tm.seeds, tm.deltas_auc, strict=True) if d < 0]
            if neg:
                for s, d in neg:
                    lines.append(f"  ⚠ seed {s}: delta={d:.4f} (negative)")

        if self.missingness:
            lines.append("\nMissingness:")
            for col, rate in sorted(self.missingness.items(), key=lambda x: -x[1]):
                if rate > 0:
                    lines.append(f"  {col}: {rate:.1%}")

        sep = "=" * 60
        if self.passed:
            lines.append(f"\n{sep}\nALL CHECKS PASSED\n{sep}")
        else:
            lines.append(f"\n{sep}\nFAILED — {self.n_errors} check(s)\n{sep}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisable dict for JSON output."""
        d: dict = {
            "csv_path": self.csv_path,
            "passed": self.passed,
            "n_errors": self.n_errors,
            "checks": [
                {"name": c.name, "passed": c.passed, "details": c.details} for c in self.checks
            ],
        }
        if self.baseline:
            b = self.baseline
            d["baseline"] = {
                "seed": b.seed,
                "auc": round(b.auc, 4),
                "pr_auc": round(b.pr_auc, 4),
                "base_rate": round(b.base_rate, 4),
                "precision_at_k": {str(k): round(v, 4) for k, v in b.precision_at_k.items()},
                "recall_at_k": {str(k): round(v, 4) for k, v in b.recall_at_k.items()},
                "lift_at_k": {str(k): round(v, 2) for k, v in b.lift_at_k.items()},
            }
        if self.value_metrics:
            d["value_metrics"] = [
                {
                    "k": vm.k,
                    "captured_acv_by_prob": round(vm.captured_acv_by_prob, 0),
                    "captured_acv_by_ev": round(vm.captured_acv_by_ev, 0),
                    "uplift_pct": round(vm.uplift_pct, 1),
                }
                for vm in self.value_metrics
            ]
        if self.trap_metrics:
            d["trap_metrics"] = [
                {
                    "column": tm.column,
                    "mean_delta_auc": round(tm.mean_delta_auc, 4),
                    "min_delta_auc": round(tm.min_delta_auc, 4),
                    "max_delta_auc": round(tm.max_delta_auc, 4),
                    "seeds": tm.seeds,
                    "deltas_auc": [round(d, 4) for d in tm.deltas_auc],
                    "mean_delta_pr_auc": round(float(np.mean(tm.deltas_pr_auc)), 4),
                    "deltas_pr_auc": [round(d, 4) for d in tm.deltas_pr_auc],
                }
                for tm in self.trap_metrics
            ]
        d["missingness"] = {
            col: round(rate, 4) for col, rate in self.missingness.items() if rate > 0
        }
        return d

    def emit_release_snippet(self) -> str:
        """Markdown snippet for pasting into RELEASE docs."""
        lines = ["<!-- BEGIN AUTO-METRICS -->"]

        if self.baseline:
            b = self.baseline
            lines.append("")
            lines.append("### Baseline performance")
            lines.append("")
            train_pct = int(round((1 - self.test_size) * 100))
            test_pct = int(round(self.test_size * 100))
            lines.append(
                f"Evaluated on a {train_pct}/{test_pct} stratified hold-out split (seed {b.seed})."
            )
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| ROC-AUC | {b.auc:.3f} |")
            lines.append(f"| PR-AUC (Average Precision) | {b.pr_auc:.3f} |")
            lines.append(f"| Base rate | {b.base_rate:.1%} |")
            for k in sorted(b.precision_at_k):
                lines.append(
                    f"| Precision@{k} | {b.precision_at_k[k]:.3f} (Lift: {b.lift_at_k[k]:.2f}x) |"
                )
                lines.append(f"| Recall@{k} | {b.recall_at_k[k]:.3f} |")

        if self.value_metrics:
            lines.append("")
            lines.append("### Value-aware ranking")
            lines.append("")
            lines.append("| K | By P(convert) | By expected value | Uplift |")
            lines.append("|---|---|---|---|")
            for vm in self.value_metrics:
                lines.append(
                    f"| {vm.k} | ${vm.captured_acv_by_prob:,.0f} "
                    f"| ${vm.captured_acv_by_ev:,.0f} "
                    f"| {vm.uplift_pct:+.1f}% |"
                )

        if self.trap_metrics:
            lines.append("")
            lines.append("### Leakage trap evaluation")
            lines.append("")
            for tm in self.trap_metrics:
                lines.append("| Metric | Value |")
                lines.append("|---|---|")
                lines.append(f"| Column | `{tm.column}` |")
                lines.append(f"| Seeds | {len(tm.seeds)} ({tm.seeds[0]}–{tm.seeds[-1]}) |")
                lines.append(f"| Mean AUC delta | {tm.mean_delta_auc:.4f} |")
                lines.append(f"| Min AUC delta | {tm.min_delta_auc:.4f} |")
                lines.append(f"| Max AUC delta | {tm.max_delta_auc:.4f} |")

        if self.missingness:
            lines.append("")
            lines.append("### Missingness")
            lines.append("")
            lines.append("| Column | Missing | Rate |")
            lines.append("|---|---|---|")
            total = 0
            for col, rate in sorted(self.missingness.items(), key=lambda x: -x[1]):
                if rate > 0:
                    n = int(round(rate * self.n_rows))
                    total += n
                    lines.append(f"| `{col}` | {n} | {rate:.1%} |")
            lines.append(f"| **Total** | **{total}** | |")

        lines.append("")
        lines.append("<!-- END AUTO-METRICS -->")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ML pipeline (single source of truth)
# ---------------------------------------------------------------------------


def _build_pipeline(
    cat_cols: list[str],
    num_cols: list[str],
) -> Pipeline:
    """Build the canonical sklearn baseline pipeline.

    - Numeric: median imputation + standard scaling
    - Categorical: most-frequent imputation + one-hot encoding
    - Model: L2-regularised logistic regression (lbfgs solver)
    """
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ]
    )


def _get_feature_cols(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Partition feature columns into (cat_cols, num_cols)."""
    exclude = (exclude or set()) | {TARGET}
    cat_cols = []
    num_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return cat_cols, num_cols


def _evaluate_split(
    df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
    seed: int = 42,
    test_size: float = 0.30,
    ks: tuple[int, ...] = (25, 50),
) -> BaselineMetrics:
    """Train on hold-out split, return metrics on test set."""
    y = df[TARGET].astype(int)
    cat_cols, num_cols = _get_feature_cols(df, exclude=exclude_cols)
    x = df[cat_cols + num_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    pipe = _build_pipeline(cat_cols, num_cols)
    pipe.fit(x_train, y_train)
    probs = pipe.predict_proba(x_test)[:, 1]

    auc = float(roc_auc_score(y_test, probs))
    pr_auc = float(average_precision_score(y_test, probs))
    base_rate = float(y_test.mean())

    # Precision@K, Recall@K, Lift@K  (stable sort by -prob; ties preserve array order)
    order = np.argsort(-probs, kind="stable")
    y_sorted = y_test.values[order]

    n_pos = int(y_test.sum())
    precision_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}
    lift_at_k: dict[int, float] = {}
    for k in ks:
        if k > len(y_test):
            continue
        top_k = y_sorted[:k]
        prec = float(top_k.mean())
        rec = float(top_k.sum() / n_pos) if n_pos > 0 else 0.0
        precision_at_k[k] = prec
        recall_at_k[k] = rec
        lift_at_k[k] = prec / base_rate if base_rate > 0 else 0.0

    return BaselineMetrics(
        seed=seed,
        auc=auc,
        pr_auc=pr_auc,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        lift_at_k=lift_at_k,
        base_rate=base_rate,
    )


def _evaluate_value_aware(
    df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
    seed: int = 42,
    test_size: float = 0.30,
    ks: tuple[int, ...] = (25, 50),
) -> list[ValueMetrics]:
    """Compute value-aware ranking metrics on hold-out."""
    if "expected_acv" not in df.columns:
        return []

    y = df[TARGET].astype(int)
    cat_cols, num_cols = _get_feature_cols(df, exclude=exclude_cols)
    x = df[cat_cols + num_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    pipe = _build_pipeline(cat_cols, num_cols)
    pipe.fit(x_train, y_train)
    probs = pipe.predict_proba(x_test)[:, 1]

    test_acv = pd.to_numeric(df.loc[x_test.index, "expected_acv"], errors="coerce").fillna(0).values
    test_converted = y_test.values
    expected_value = probs * test_acv

    results = []
    for k in ks:
        if k > len(y_test):
            continue

        # Rank by probability
        order_prob = np.argsort(-probs, kind="stable")
        top_k_prob = order_prob[:k]
        captured_prob = float(np.sum(test_acv[top_k_prob] * test_converted[top_k_prob]))

        # Rank by expected value
        order_ev = np.argsort(-expected_value, kind="stable")
        top_k_ev = order_ev[:k]
        captured_ev = float(np.sum(test_acv[top_k_ev] * test_converted[top_k_ev]))

        uplift = ((captured_ev - captured_prob) / captured_prob * 100) if captured_prob > 0 else 0.0

        results.append(
            ValueMetrics(
                k=k,
                captured_acv_by_prob=captured_prob,
                captured_acv_by_ev=captured_ev,
                uplift_pct=uplift,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_schema(df: pd.DataFrame, cfg: ValidationConfig) -> list[CheckResult]:
    """Schema and basic structural checks."""
    results = []

    # Target column
    if TARGET not in df.columns:
        results.append(CheckResult("target_exists", False, f"'{TARGET}' column missing"))
        return results
    results.append(CheckResult("target_exists", True))
    target_vals = set(df[TARGET].dropna().unique())
    if not target_vals <= {0, 1}:
        results.append(CheckResult("target_binary", False, f"target values: {target_vals}"))
    else:
        results.append(CheckResult("target_binary", True))
    if df[TARGET].isna().any():
        results.append(CheckResult("target_no_missing", False, "target has missing values"))
    else:
        results.append(CheckResult("target_no_missing", True))
    # Both classes must be present for stratified splitting
    if target_vals == {0, 1}:
        results.append(CheckResult("target_both_classes", True))
    else:
        results.append(
            CheckResult("target_both_classes", False, f"need both {{0, 1}}, got {target_vals}")
        )

    # Banned columns
    present = BANNED_COLUMNS & set(df.columns)
    if present:
        results.append(CheckResult("no_banned_columns", False, f"banned: {sorted(present)}"))
    else:
        results.append(CheckResult("no_banned_columns", True))

    # ID columns
    id_cols = [c for c in df.columns if c.endswith("_id")]
    if id_cols:
        results.append(CheckResult("no_id_columns", False, f"ID cols: {sorted(id_cols)}"))
    else:
        results.append(CheckResult("no_id_columns", True))

    # Row count
    n = len(df)
    if cfg.enforce_row_count and n != cfg.expected_rows:
        results.append(CheckResult("row_count", False, f"{n} rows (expected {cfg.expected_rows})"))
    elif n != cfg.expected_rows:
        results.append(CheckResult("row_count", True, f"{n} rows (expected {cfg.expected_rows})"))
    else:
        results.append(CheckResult("row_count", True, f"{n} rows"))

    # Duplicates
    n_dupes = df.duplicated().sum()
    dupe_rate = n_dupes / len(df) if len(df) > 0 else 0
    if dupe_rate > cfg.max_duplicate_rate:
        results.append(CheckResult("duplicates", False, f"{n_dupes} duplicates ({dupe_rate:.1%})"))
    else:
        results.append(CheckResult("duplicates", True, f"{n_dupes} duplicates"))

    # Expected features (warn, don't fail)
    missing_cat = [c for c in EXPECTED_CAT_FEATURES if c not in df.columns]
    missing_num = [c for c in EXPECTED_NUMERIC_FEATURES if c not in df.columns]
    missing_bin = [c for c in EXPECTED_BINARY_FEATURES if c not in df.columns]
    if missing_cat or missing_num or missing_bin:
        all_missing = missing_cat + missing_num + missing_bin
        results.append(CheckResult("expected_features", True, f"missing: {all_missing} (warning)"))
    else:
        results.append(CheckResult("expected_features", True))

    # Leakage column naming
    leakage_cols = [c for c in df.columns if c.startswith(LEAKAGE_PREFIX)]
    if "total_touches_all" in df.columns:
        results.append(CheckResult("leakage_naming", False, "old name 'total_touches_all' found"))
    elif len(leakage_cols) == 0:
        results.append(CheckResult("leakage_naming", True, "no leakage columns"))
    elif len(leakage_cols) == 1:
        results.append(CheckResult("leakage_naming", True, f"trap: {leakage_cols[0]}"))
    else:
        results.append(
            CheckResult("leakage_naming", True, f"multiple traps: {leakage_cols} (warning)")
        )

    return results


def _check_missingness(
    df: pd.DataFrame,
    cfg: ValidationConfig,
) -> tuple[list[CheckResult], dict[str, float]]:
    """Per-column missingness checks."""
    results = []
    miss_map: dict[str, float] = {}

    for col in df.columns:
        if col == TARGET:
            continue
        rate = float(df[col].isna().mean())
        if rate > 0:
            miss_map[col] = rate

    violations = {col: rate for col, rate in miss_map.items() if rate > cfg.max_col_missing_rate}
    if violations:
        detail = ", ".join(f"{c}={r:.1%}" for c, r in violations.items())
        results.append(
            CheckResult("missingness_bounds", False, f">{cfg.max_col_missing_rate:.0%}: {detail}")
        )
    else:
        results.append(CheckResult("missingness_bounds", True))

    return results, miss_map


def _check_group_determinism(
    df: pd.DataFrame,
    cfg: ValidationConfig,
) -> list[CheckResult]:
    """No categorical/binary group should be near-deterministic."""
    # Gather all categorical + binary columns present in the data
    check_cols = [c for c in EXPECTED_CAT_FEATURES + EXPECTED_BINARY_FEATURES if c in df.columns]
    # Also include any non-numeric or binary columns not in the expected list
    for col in df.columns:
        if col == TARGET or col in check_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]) or set(df[col].dropna().unique()) <= {0, 1}:
            check_cols.append(col)

    violations = []
    for col in check_cols:
        stats = df.groupby(col)[TARGET].agg(["mean", "count"])
        large = stats[stats["count"] >= cfg.min_group_size]
        for val, row in large.iterrows():
            if row["mean"] < cfg.min_group_rate:
                violations.append(f"{col}={val}: {row['mean']:.1%} (n={int(row['count'])})")
            if row["mean"] > cfg.max_group_rate:
                violations.append(f"{col}={val}: {row['mean']:.1%} (n={int(row['count'])})")

    if violations:
        return [
            CheckResult(
                "group_determinism",
                False,
                f"{len(violations)} violation(s): " + "; ".join(violations),
            )
        ]
    return [CheckResult("group_determinism", True)]


def _check_baseline_auc(
    metrics: BaselineMetrics,
    cfg: ValidationConfig,
) -> list[CheckResult]:
    """Baseline AUC within expected range."""
    results = []
    auc = metrics.auc
    if auc < cfg.auc_lower:
        results.append(CheckResult("baseline_auc", False, f"AUC={auc:.3f} < {cfg.auc_lower}"))
    elif auc > cfg.auc_upper:
        results.append(CheckResult("baseline_auc", False, f"AUC={auc:.3f} > {cfg.auc_upper}"))
    else:
        results.append(
            CheckResult("baseline_auc", True, f"AUC={auc:.3f}, PR-AUC={metrics.pr_auc:.3f}")
        )
    return results


def _check_conversion_rate(df: pd.DataFrame) -> list[CheckResult]:
    """Conversion rate in realistic B2B range [15%, 40%]."""
    rate = df[TARGET].mean()
    if rate < 0.15 or rate > 0.40:
        return [CheckResult("conversion_rate", False, f"{rate:.1%} outside [15%, 40%]")]
    return [CheckResult("conversion_rate", True, f"{rate:.1%}")]


def _check_acv_range(df: pd.DataFrame) -> list[CheckResult]:
    """expected_acv within narrative range."""
    if "expected_acv" not in df.columns:
        return [CheckResult("acv_range", True, "column not present (skip)")]
    acv = pd.to_numeric(df["expected_acv"], errors="coerce").dropna()
    if acv.empty:
        return [CheckResult("acv_range", False, "no usable values")]
    errors = []
    if acv.min() < 18_000 - 1:
        errors.append(f"min={acv.min():.0f} < 18,000")
    if acv.max() > 120_000 + 1:
        errors.append(f"max={acv.max():.0f} > 120,000")
    if errors:
        return [CheckResult("acv_range", False, "; ".join(errors))]
    return [CheckResult("acv_range", True, f"[{acv.min():.0f}, {acv.max():.0f}]")]


def _evaluate_trap(
    df: pd.DataFrame,
    cfg: ValidationConfig,
) -> tuple[list[CheckResult], list[TrapMetrics]]:
    """Leakage trap evaluation across multiple seeds."""
    leakage_cols = [c for c in df.columns if c.startswith(LEAKAGE_PREFIX)]
    if not leakage_cols:
        return [CheckResult("leakage_trap", True, "no trap columns (skip)")], []

    all_trap_metrics = []
    all_checks = []
    all_leakage = set(leakage_cols)

    for trap_col in leakage_cols:
        seeds = list(range(cfg.trap_seed_start, cfg.trap_seed_start + cfg.trap_n_seeds))
        deltas_auc = []
        deltas_pr_auc = []

        for seed in seeds:
            m_without = _evaluate_split(
                df,
                exclude_cols=all_leakage,
                seed=seed,
                test_size=cfg.test_size,
                ks=(),
            )
            m_with = _evaluate_split(
                df,
                exclude_cols=all_leakage - {trap_col},
                seed=seed,
                test_size=cfg.test_size,
                ks=(),
            )
            deltas_auc.append(m_with.auc - m_without.auc)
            deltas_pr_auc.append(m_with.pr_auc - m_without.pr_auc)

        tm = TrapMetrics(
            column=trap_col,
            deltas_auc=deltas_auc,
            deltas_pr_auc=deltas_pr_auc,
            seeds=seeds,
        )
        all_trap_metrics.append(tm)

        # Check thresholds
        errors = []
        if tm.mean_delta_auc < cfg.trap_mean_delta:
            errors.append(f"mean delta {tm.mean_delta_auc:.4f} < {cfg.trap_mean_delta}")
        if tm.min_delta_auc < cfg.trap_min_delta:
            bad_seeds = [
                f"seed {s}: {d:.4f}"
                for s, d in zip(seeds, deltas_auc, strict=True)
                if d < cfg.trap_min_delta
            ]
            errors.append(
                f"min delta {tm.min_delta_auc:.4f} < {cfg.trap_min_delta} [{', '.join(bad_seeds)}]"
            )

        if errors:
            all_checks.append(
                CheckResult(
                    f"leakage_trap:{trap_col}",
                    False,
                    "; ".join(errors),
                )
            )
        else:
            all_checks.append(
                CheckResult(
                    f"leakage_trap:{trap_col}",
                    True,
                    f"mean={tm.mean_delta_auc:.4f} min={tm.min_delta_auc:.4f}",
                )
            )

    return all_checks, all_trap_metrics


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def validate_dataset(
    csv_path: str | Path,
    cfg: ValidationConfig | None = None,
) -> ValidationReport:
    """Run full validation suite on a lead scoring CSV.

    Args:
        csv_path: Path to the CSV file.
        cfg: Validation thresholds.  Uses defaults if ``None``.

    Returns:
        A :class:`ValidationReport` with all check results and metrics.
    """
    cfg = cfg or ValidationConfig()
    df = pd.read_csv(csv_path)
    report = ValidationReport(csv_path=str(csv_path), n_rows=len(df), test_size=cfg.test_size)

    # Schema checks
    schema_checks = _check_schema(df, cfg)
    report.checks.extend(schema_checks)
    if TARGET not in df.columns:
        return report
    # Short-circuit if target is unusable (non-binary, has NaNs, or single class)
    if any(
        not c.passed
        for c in schema_checks
        if c.name in ("target_binary", "target_no_missing", "target_both_classes")
    ):
        return report

    # Conversion rate
    report.checks.extend(_check_conversion_rate(df))

    # Missingness
    miss_checks, miss_map = _check_missingness(df, cfg)
    report.checks.extend(miss_checks)
    report.missingness = miss_map

    # Group determinism
    report.checks.extend(_check_group_determinism(df, cfg))

    # ACV range
    report.checks.extend(_check_acv_range(df))

    # Baseline evaluation
    leakage_cols = {c for c in df.columns if c.startswith(LEAKAGE_PREFIX)}
    baseline = _evaluate_split(
        df,
        exclude_cols=leakage_cols,
        seed=cfg.default_seed,
        test_size=cfg.test_size,
        ks=cfg.ks,
    )
    report.baseline = baseline
    report.checks.extend(_check_baseline_auc(baseline, cfg))

    # Value-aware metrics
    report.value_metrics = _evaluate_value_aware(
        df,
        exclude_cols=leakage_cols,
        seed=cfg.default_seed,
        test_size=cfg.test_size,
        ks=cfg.ks,
    )

    # Leakage trap
    trap_checks, trap_metrics = _evaluate_trap(df, cfg)
    report.checks.extend(trap_checks)
    report.trap_metrics = trap_metrics

    return report
