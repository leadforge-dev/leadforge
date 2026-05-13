#!/usr/bin/env python3
"""Validate the mid-project lead scoring dataset against spec.

Usage:
    python scripts/validate_midproject_lead_scoring.py CSV_PATH [--out-json PATH]

Validates a single student-safe CSV (no instructor/trap variant for midproject).
Exit code 0 = all mandatory checks pass.

Canonical pipeline:
- Numeric: SimpleImputer(median) + StandardScaler
- Categorical: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
- Model: LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
- Split: 70/30 stratified hold-out, random_state=42
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from leadforge.pipelines.common import BINARY_FEATURES, CAT_FEATURES, TARGET
from leadforge.pipelines.common import FINAL_COLUMNS_STUDENT as EXPECTED_COLUMNS
from leadforge.pipelines.ml import (
    LEAKAGE_PREFIX,
    build_baseline_pipeline,
    fit_evaluate,
    get_feature_cols,
    sanitize_categoricals,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
AUC_LOWER = 0.62
AUC_UPPER = 0.80
PR_AUC_LOWER = 0.35
MAX_COL_MISSING_RATE = 0.10
MAX_DUPLICATE_RATE = 0.005
MIN_CONVERSION_RATE = 0.25
MAX_CONVERSION_RATE = 0.35
MIN_GROUP_SIZE = 50
RATE_LOWER = 0.02
RATE_UPPER = 0.98
ACV_MIN = 18_000.0
ACV_MAX = 120_000.0
ACV_PILE_UP_WARN = 0.05

BANNED_COLUMNS = {
    "current_stage",
    "funnel_stage",
    "conversion_timestamp",
    "is_sql",
    "is_mql",
    "lead_created_at",
    "close_outcome",
    "converted_within_90_days",
}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_basic(df: pd.DataFrame) -> list[str]:
    errors = []

    n = len(df)
    if not (1000 <= n <= 1500):
        errors.append(f"Row count {n} outside acceptable range [1000, 1500]")

    if TARGET not in df.columns:
        errors.append(f"Missing target column '{TARGET}'")
        return errors

    target_vals = set(df[TARGET].dropna().unique())
    if not target_vals <= {0, 1}:
        errors.append(f"Target values not binary: {target_vals}")
    if df[TARGET].isna().any():
        errors.append("Target has missing values")

    conv_rate = df[TARGET].mean()
    if conv_rate < MIN_CONVERSION_RATE:
        errors.append(f"Conversion rate {conv_rate:.2%} < {MIN_CONVERSION_RATE:.0%}")
    if conv_rate > MAX_CONVERSION_RATE:
        errors.append(f"Conversion rate {conv_rate:.2%} > {MAX_CONVERSION_RATE:.0%}")

    leakage = [c for c in df.columns if c.startswith(LEAKAGE_PREFIX)]
    if leakage:
        errors.append(f"Leakage columns must not appear in student CSV: {leakage}")

    banned = BANNED_COLUMNS & set(df.columns)
    if banned:
        errors.append(f"Banned columns present: {sorted(banned)}")

    id_cols = [c for c in df.columns if c.endswith("_id")]
    if id_cols:
        errors.append(f"ID columns found (should not appear): {sorted(id_cols)}")

    n_dupes = df.duplicated().sum()
    dupe_rate = n_dupes / n if n > 0 else 0.0
    if dupe_rate > MAX_DUPLICATE_RATE:
        errors.append(f"{n_dupes} duplicate rows ({dupe_rate:.1%}) > {MAX_DUPLICATE_RATE:.1%}")

    return errors


def check_schema(df: pd.DataFrame) -> list[str]:
    errors = []
    expected = list(EXPECTED_COLUMNS)
    actual = list(df.columns)
    if actual != expected:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        if missing:
            errors.append(f"Missing expected columns: {missing}")
        if extra:
            errors.append(f"Extra unexpected columns: {extra}")
        if actual != expected and not missing and not extra:
            errors.append("Column order differs from v7 student schema")
    return errors


def check_missingness(df: pd.DataFrame) -> tuple[list[str], dict]:
    errors = []
    report: dict[str, dict] = {}
    for col in df.columns:
        if col == TARGET:
            continue
        n_miss = int(df[col].isna().sum())
        if n_miss > 0:
            rate = n_miss / len(df)
            report[col] = {"count": n_miss, "rate": round(rate, 4)}
            if rate > MAX_COL_MISSING_RATE:
                errors.append(f"{col}: {rate:.1%} missing > {MAX_COL_MISSING_RATE:.0%} limit")
    if df[TARGET].isna().any():
        errors.append("Target column has missing values")
    # Check structural missingness exists in expected columns
    for col in ["web_sessions", "days_since_last_touch"]:
        if col in df.columns and df[col].isna().sum() == 0:
            errors.append(f"{col} has zero missing values (expected structured missingness)")
    return errors, report


def check_determinism(df: pd.DataFrame) -> list[str]:
    errors = []
    check_cols = [c for c in CAT_FEATURES + BINARY_FEATURES if c in df.columns]
    for col in check_cols:
        stats = df.groupby(col)[TARGET].agg(["mean", "count"])
        for val, row in stats[stats["count"] >= MIN_GROUP_SIZE].iterrows():
            if row["mean"] < RATE_LOWER:
                errors.append(
                    f"DETERMINISTIC: {col}={val}: {row['mean']:.1%} (n={int(row['count'])})"
                )
            elif row["mean"] > RATE_UPPER:
                errors.append(
                    f"DETERMINISTIC: {col}={val}: {row['mean']:.1%} (n={int(row['count'])})"
                )
    return errors


def check_acv(df: pd.DataFrame) -> tuple[list[str], dict]:
    errors = []
    if "expected_acv" not in df.columns:
        return ["expected_acv column missing"], {}
    acv = pd.to_numeric(df["expected_acv"], errors="coerce").dropna()
    if acv.empty:
        return ["expected_acv has no non-null values"], {}
    stats = {
        "min": float(acv.min()),
        "mean": float(acv.mean()),
        "median": float(acv.median()),
        "p95": float(acv.quantile(0.95)),
        "p99": float(acv.quantile(0.99)),
        "max": float(acv.max()),
        "at_max_frac": float((acv >= acv.max() - 1).sum() / len(acv)),
    }
    if acv.min() < ACV_MIN - 1:
        errors.append(f"expected_acv min {acv.min():.0f} < {ACV_MIN:.0f}")
    if acv.max() > ACV_MAX + 1:
        errors.append(f"expected_acv max {acv.max():.0f} > {ACV_MAX:.0f}")
    if stats["at_max_frac"] > ACV_PILE_UP_WARN:
        errors.append(f"{stats['at_max_frac']:.1%} of expected_acv at max — possible pile-up")
    return errors, stats


def check_baseline(df: pd.DataFrame) -> tuple[list[str], dict]:
    auc, pr_auc, probs, y_test = fit_evaluate(df)
    errors = []
    if auc < AUC_LOWER:
        errors.append(f"Baseline AUC {auc:.3f} < {AUC_LOWER}")
    if auc > AUC_UPPER:
        errors.append(f"Baseline AUC {auc:.3f} > {AUC_UPPER}")
    if pr_auc < PR_AUC_LOWER:
        errors.append(f"Baseline PR-AUC {pr_auc:.3f} < {PR_AUC_LOWER}")

    base_rate = float(y_test.mean())
    n_pos = int(y_test.sum())
    order = np.argsort(-probs, kind="stable")
    y_sorted = y_test.values[order]

    metrics: dict[str, float] = {"auc": auc, "pr_auc": pr_auc, "base_rate": base_rate}
    for k in [25, 50, 100]:
        if k > len(y_test):
            continue
        prec = float(y_sorted[:k].mean())
        rec = float(y_sorted[:k].sum() / n_pos) if n_pos > 0 else 0.0
        lift = prec / base_rate if base_rate > 0 else 0.0
        metrics[f"precision@{k}"] = prec
        metrics[f"recall@{k}"] = rec
        metrics[f"lift@{k}"] = lift
        metrics[f"conversions@{k}"] = int(y_sorted[:k].sum())
        metrics[f"random_conversions@{k}"] = round(k * base_rate, 1)

    return errors, metrics


def check_value_aware(df: pd.DataFrame) -> tuple[list[str], list[dict]]:
    if "expected_acv" not in df.columns:
        return ["expected_acv column missing"], []

    cat_cols, num_cols = get_feature_cols(df)
    df_clean = sanitize_categoricals(df, cat_cols)
    y = df[TARGET].astype(int)
    x = df_clean[cat_cols + num_cols]

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)
    pipe = build_baseline_pipeline(num_cols, cat_cols)
    pipe.fit(x_tr, y_tr)
    probs = pipe.predict_proba(x_te)[:, 1]

    test_acv = pd.to_numeric(df.loc[x_te.index, "expected_acv"], errors="coerce").fillna(0).values
    test_conv = y_te.values
    ev = probs * test_acv

    results = []
    for k in [25, 50]:
        if k > len(y_te):
            continue
        top_prob_idx = np.argsort(-probs)[:k]
        cap_prob = float(np.sum(test_acv[top_prob_idx] * test_conv[top_prob_idx]))
        conv_prob = int(test_conv[top_prob_idx].sum())

        top_ev_idx = np.argsort(-ev)[:k]
        cap_ev = float(np.sum(test_acv[top_ev_idx] * test_conv[top_ev_idx]))
        conv_ev = int(test_conv[top_ev_idx].sum())

        uplift = (cap_ev - cap_prob) / cap_prob * 100 if cap_prob > 0 else 0.0
        results.append(
            {
                "k": k,
                "captured_prob": cap_prob,
                "captured_ev": cap_ev,
                "conversions_prob": conv_prob,
                "conversions_ev": conv_ev,
                "uplift_pct": uplift,
            }
        )

    return [], results


def check_cohort(df: pd.DataFrame) -> dict | None:
    if "acquisition_wave" not in df.columns:
        return None
    cat_cols, num_cols = get_feature_cols(df, exclude={"acquisition_wave"})
    df_clean = sanitize_categoricals(df, cat_cols)
    y = df[TARGET].astype(int)
    x = df_clean[cat_cols + num_cols]

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)
    pipe_r = build_baseline_pipeline(num_cols, cat_cols)
    pipe_r.fit(x_tr, y_tr)
    random_auc = roc_auc_score(y_te, pipe_r.predict_proba(x_te)[:, 1])
    random_pr = average_precision_score(y_te, pipe_r.predict_proba(x_te)[:, 1])

    train_mask = df["acquisition_wave"].isin(["A", "B"])
    test_mask = df["acquisition_wave"] == "C"
    if test_mask.sum() < 30 or train_mask.sum() < 100:
        return None

    pipe_c = build_baseline_pipeline(num_cols, cat_cols)
    pipe_c.fit(x[train_mask], y[train_mask])
    cohort_auc = roc_auc_score(y[test_mask], pipe_c.predict_proba(x[test_mask])[:, 1])
    cohort_pr = average_precision_score(y[test_mask], pipe_c.predict_proba(x[test_mask])[:, 1])

    return {
        "random_auc": random_auc,
        "random_pr_auc": random_pr,
        "cohort_auc": cohort_auc,
        "cohort_pr_auc": cohort_pr,
        "drop": random_auc - cohort_auc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def validate(csv_path: str, out_json: str | None = None) -> int:
    df = pd.read_csv(csv_path)
    all_errors: list[str] = []
    all_warnings: list[str] = []
    report: dict = {"csv_path": csv_path}

    print("=" * 60)
    print("BASIC CHECKS")
    print("=" * 60)
    errs = check_basic(df)
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    if TARGET in df.columns:
        conv_rate = df[TARGET].mean()
        print(f"  Conversion rate: {conv_rate:.1%}")
        report["conversion_rate"] = float(conv_rate)
    else:
        print("  Conversion rate: N/A (target column missing)")
        report["conversion_rate"] = None
    print(f"  Status: {'FAIL' if errs else 'PASS'}")
    all_errors.extend(errs)
    report["shape"] = list(df.shape)

    print("\nSCHEMA CHECKS")
    print("=" * 60)
    errs = check_schema(df)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Status: {'FAIL' if errs else 'PASS'}")
    all_errors.extend(errs)
    report["columns"] = list(df.columns)

    print("\nMISSINGNESS")
    print("=" * 60)
    errs, miss_report = check_missingness(df)
    for col, info in miss_report.items():
        print(f"  {col}: {info['count']} ({info['rate']:.1%})")
    print(f"  Total missing: {df.isnull().sum().sum()}")
    print(f"  Status: {'FAIL' if errs else 'PASS'}")
    all_errors.extend(errs)
    report["missingness"] = miss_report

    print("\nDETERMINISM CHECKS")
    print("=" * 60)
    errs = check_determinism(df)
    print(f"  Status: {'FAIL' if errs else 'PASS'}")
    if errs:
        for e in errs:
            print(f"  * {e}")
    all_errors.extend(errs)

    print("\nACV STATISTICS")
    print("=" * 60)
    errs, acv_stats = check_acv(df)
    if acv_stats:
        print(
            f"  min=${acv_stats['min']:,.0f}  mean=${acv_stats['mean']:,.0f}  "
            f"median=${acv_stats['median']:,.0f}  p95=${acv_stats['p95']:,.0f}  "
            f"max=${acv_stats['max']:,.0f}"
        )
        print(f"  At-max pile-up: {acv_stats['at_max_frac']:.1%}")
    print(f"  Status: {'FAIL' if errs else 'PASS'}")
    all_errors.extend(errs)
    report["acv_stats"] = acv_stats

    print("\nBASELINE MODEL (LR, seed=42, 70/30 stratified)")
    print("=" * 60)
    errs, baseline = check_baseline(df)
    auc = baseline.get("auc", 0.0)
    pr_auc = baseline.get("pr_auc", 0.0)
    base_rate = baseline.get("base_rate", 0.0)
    print(f"  ROC-AUC: {auc:.4f}  PR-AUC: {pr_auc:.4f}  Base rate: {base_rate:.1%}")
    for k in [25, 50, 100]:
        pk = baseline.get(f"precision@{k}")
        lk = baseline.get(f"lift@{k}")
        ck = baseline.get(f"conversions@{k}")
        rk = baseline.get(f"random_conversions@{k}")
        if pk is not None:
            print(f"  P@{k}={pk:.3f}  Lift@{k}={lk:.2f}x  conversions={ck}/{k}  random={rk:.1f}")
    print(f"  Status: {'FAIL' if errs else 'PASS'}")
    all_errors.extend(errs)
    report["baseline"] = baseline

    print("\nVALUE-AWARE RANKING")
    print("=" * 60)
    errs, ev_results = check_value_aware(df)
    for r in ev_results:
        k = r["k"]
        print(
            f"  K={k}: prob=${r['captured_prob']:,.0f} (conv={r['conversions_prob']})  "
            f"ev=${r['captured_ev']:,.0f} (conv={r['conversions_ev']})  "
            f"ACV uplift={r['uplift_pct']:+.1f}%"
        )
    all_errors.extend(errs)
    report["value_aware"] = ev_results

    print("\nCOHORT SPLIT (train A+B, test C)")
    print("=" * 60)
    cohort = check_cohort(df)
    if cohort:
        print(
            f"  Random split:  AUC={cohort['random_auc']:.4f}  PR-AUC={cohort['random_pr_auc']:.4f}"
        )
        print(
            f"  Cohort split:  AUC={cohort['cohort_auc']:.4f}  PR-AUC={cohort['cohort_pr_auc']:.4f}"
        )
        print(f"  AUC drop: {cohort['drop']:+.4f}")
        report["cohort_split"] = cohort
    else:
        print("  Skipped (no acquisition_wave or insufficient cohort sizes)")

    report["errors"] = all_errors
    report["warnings"] = all_warnings

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report written to: {out_json}")

    print(f"\n{'=' * 60}")
    if all_errors:
        print(f"FAILED — {len(all_errors)} error(s):")
        for err in all_errors:
            print(f"  * {err}")
        return 1
    else:
        print("ALL MANDATORY CHECKS PASSED")
        return 0


def main() -> None:
    args = sys.argv[1:]
    out_json = None
    if "--out-json" in args:
        idx = args.index("--out-json")
        if idx + 1 < len(args):
            out_json = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]
        else:
            print("--out-json requires a path", file=sys.stderr)
            sys.exit(1)

    if len(args) != 1:
        print(f"Usage: {sys.argv[0]} CSV_PATH [--out-json PATH]", file=sys.stderr)
        sys.exit(1)

    sys.exit(validate(args[0], out_json=out_json))


if __name__ == "__main__":
    main()
