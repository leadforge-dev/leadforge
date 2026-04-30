#!/usr/bin/env python3
"""Validate a v5 lead scoring intro CSV against the v5 validation spec.

Usage:
    python scripts/validate_v5_dataset.py lead_scoring_intro_v5.csv

Exit code 0 = all mandatory checks pass.
Exit code 1 = at least one mandatory check failed.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "converted"

BANNED_COLUMNS = {
    "current_stage",
    "funnel_stage",
    "conversion_timestamp",
    "is_sql",
    "is_mql",
    "lead_created_at",
}

CAT_FEATURES = [
    "industry",
    "region",
    "company_size",
    "company_revenue",
    "contact_role",
    "seniority",
    "lead_source",
]

BINARY_FEATURES = [
    "opportunity_created",
    "demo_completed",
]

LEAKAGE_TRAP = "__leakage__total_touches_90d"

# Deterministic group thresholds
MIN_GROUP_SIZE = 50
RATE_LOWER = 0.02
RATE_UPPER = 0.98

# AUC bounds (hold-out)
AUC_LOWER = 0.62
AUC_UPPER = 0.90

# Leakage trap robustness thresholds (multi-seed)
TRAP_MEAN_DELTA = 0.03
TRAP_MIN_DELTA = 0.015
TRAP_N_SEEDS = 10

# Missingness
MAX_COL_MISSING_RATE = 0.10

# Duplicates
MAX_DUPLICATE_RATE = 0.01


# ---------------------------------------------------------------------------
# Utility: fit LR on a train/test split, return test metrics
# ---------------------------------------------------------------------------


def _split_and_preprocess(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Split first, then fit preprocessing on train only.

    Returns scaled (x_train, x_test, y_train, y_test).  Label encoding,
    median imputation, and standard scaling are all fit on the training fold
    so that test-fold metrics are truly out-of-sample.
    """
    exclude = set(exclude_cols or [])
    feature_cols = [c for c in df.columns if c != TARGET and c not in exclude]

    x_raw = df[feature_cols].copy()
    y = df[TARGET].astype(int)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_raw, y, test_size=0.30, random_state=seed, stratify=y
    )

    # Encode categoricals: fit LabelEncoder on train, transform both.
    cat_cols = list(x_train_raw.select_dtypes(include=["object", "category"]).columns)
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(x_train_raw[col].astype(str).fillna("__MISSING__"))
        encoders[col] = le
        x_train_raw[col] = le.transform(x_train_raw[col].astype(str).fillna("__MISSING__"))
        # Unseen test categories get mapped to "__MISSING__"
        test_vals = x_test_raw[col].astype(str).fillna("__MISSING__")
        test_vals = test_vals.where(test_vals.isin(le.classes_), "__MISSING__")
        # Ensure __MISSING__ is in classes (it always is since we fillna above)
        x_test_raw[col] = le.transform(test_vals)

    # Select numeric columns and impute with train medians.
    x_train_num = x_train_raw.select_dtypes(include=[np.number]).copy()
    x_test_num = x_test_raw[x_train_num.columns].copy()
    train_medians = x_train_num.median()
    x_train_num = x_train_num.fillna(train_medians)
    x_test_num = x_test_num.fillna(train_medians)

    # Scale.
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train_num)
    x_test_s = scaler.transform(x_test_num)

    return x_train_s, x_test_s, y_train, y_test


def _fit_lr_holdout(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Fit LR on 70/30 hold-out split and return metrics."""
    x_train_s, x_test_s, y_train, y_test = _split_and_preprocess(df, exclude_cols, seed)

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(x_train_s, y_train)
    probs = lr.predict_proba(x_test_s)[:, 1]

    auc = float(roc_auc_score(y_test, probs))
    pr_auc = float(average_precision_score(y_test, probs))

    # Precision@K and Lift@K
    metrics: dict[str, float] = {"auc": auc, "pr_auc": pr_auc}
    n_test = len(y_test)
    for k in [25, 50, 100]:
        if k > n_test:
            continue
        top_k_idx = np.argsort(-probs)[:k]
        top_k_labels = y_test.iloc[top_k_idx]
        prec_k = float(top_k_labels.mean())
        base_rate = float(y_test.mean())
        lift_k = prec_k / base_rate if base_rate > 0 else 0.0
        metrics[f"precision@{k}"] = prec_k
        metrics[f"lift@{k}"] = lift_k

    return metrics


def _fit_lr_auc_only(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    seed: int = 42,
) -> float:
    """Fit LR on hold-out split and return only AUC (for multi-seed checks)."""
    x_train_s, x_test_s, y_train, y_test = _split_and_preprocess(df, exclude_cols, seed)
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(x_train_s, y_train)
    probs = lr.predict_proba(x_test_s)[:, 1]
    return float(roc_auc_score(y_test, probs))


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def check_banned_columns(df: pd.DataFrame) -> list[str]:
    """Check 1: No banned columns."""
    errors = []
    present = BANNED_COLUMNS & set(df.columns)
    if present:
        errors.append(f"Banned columns present: {sorted(present)}")
    id_cols = [c for c in df.columns if c.endswith("_id")]
    if id_cols:
        errors.append(f"ID columns present: {sorted(id_cols)}")
    return errors


def check_deterministic_groups(df: pd.DataFrame) -> list[str]:
    """Check 2: No deterministic feature groups."""
    errors = []
    check_cols = [c for c in CAT_FEATURES + BINARY_FEATURES if c in df.columns]
    for col in check_cols:
        stats = df.groupby(col)[TARGET].agg(["mean", "count"])
        large = stats[stats["count"] >= MIN_GROUP_SIZE]
        for val, row in large.iterrows():
            if row["mean"] < RATE_LOWER:
                errors.append(
                    f"DETERMINISTIC: {col}={val} has {row['mean']:.1%} "
                    f"conversion (n={int(row['count'])})"
                )
            if row["mean"] > RATE_UPPER:
                errors.append(
                    f"DETERMINISTIC: {col}={val} has {row['mean']:.1%} "
                    f"conversion (n={int(row['count'])})"
                )
    return errors


def check_conversion_rate(df: pd.DataFrame) -> list[str]:
    """Check 3: Conversion rate realism."""
    rate = df[TARGET].mean()
    if rate < 0.15 or rate > 0.40:
        return [f"Conversion rate {rate:.1%} outside [15%, 40%]"]
    return []


def check_baseline_auc(df: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    """Check 4: Baseline model AUC on hold-out split (without leakage trap)."""
    metrics = _fit_lr_holdout(df, exclude_cols=[LEAKAGE_TRAP])
    errors = []
    auc = metrics["auc"]
    if auc < AUC_LOWER:
        errors.append(f"Baseline hold-out AUC {auc:.3f} below {AUC_LOWER}")
    if auc > AUC_UPPER:
        errors.append(f"Baseline hold-out AUC {auc:.3f} above {AUC_UPPER}")
    return errors, metrics


def check_leakage_trap_robust(
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, float]]:
    """Check 5: Leakage trap effectiveness across multiple split seeds."""
    if LEAKAGE_TRAP not in df.columns:
        return [f"Leakage trap column '{LEAKAGE_TRAP}' not found"], {}

    deltas = []
    for seed in range(TRAP_N_SEEDS):
        auc_without = _fit_lr_auc_only(df, exclude_cols=[LEAKAGE_TRAP], seed=seed)
        auc_with = _fit_lr_auc_only(df, seed=seed)
        deltas.append(auc_with - auc_without)

    mean_delta = float(np.mean(deltas))
    min_delta = float(np.min(deltas))
    max_delta = float(np.max(deltas))

    errors = []
    if mean_delta < TRAP_MEAN_DELTA:
        errors.append(
            f"Leakage trap mean delta {mean_delta:.4f} below {TRAP_MEAN_DELTA} "
            f"(min={min_delta:.4f}, max={max_delta:.4f})"
        )
    if min_delta < TRAP_MIN_DELTA:
        errors.append(
            f"Leakage trap min delta {min_delta:.4f} below {TRAP_MIN_DELTA} "
            f"across {TRAP_N_SEEDS} seeds"
        )

    stats = {
        "mean_delta": mean_delta,
        "min_delta": min_delta,
        "max_delta": max_delta,
        "deltas": deltas,
    }
    return errors, stats


def check_missingness(df: pd.DataFrame) -> list[str]:
    """Check 6: Missingness structure and bounds."""
    errors = []

    # web_sessions must have nulls
    if "web_sessions" in df.columns:
        if df["web_sessions"].isna().sum() == 0:
            errors.append("web_sessions has no nulls")
        else:
            outbound_mask = df["lead_source"] == "sdr_outbound"
            inbound_mask = df["lead_source"] == "inbound_marketing"
            if not outbound_mask.any():
                errors.append(
                    "web_sessions missingness check requires at least one sdr_outbound row"
                )
            elif not inbound_mask.any():
                errors.append(
                    "web_sessions missingness check requires at least one inbound_marketing row"
                )
            else:
                outbound_rate = df.loc[outbound_mask, "web_sessions"].isna().mean()
                inbound_rate = df.loc[inbound_mask, "web_sessions"].isna().mean()
                if inbound_rate > 0 and outbound_rate / inbound_rate < 3.0:
                    errors.append(
                        f"web_sessions missing ratio outbound/inbound = "
                        f"{outbound_rate / inbound_rate:.1f}x (need >3x)"
                    )
                elif inbound_rate == 0 and outbound_rate == 0:
                    errors.append("web_sessions has no source-conditional missingness")

    # seniority must have nulls
    if "seniority" in df.columns:
        if df["seniority"].isna().sum() == 0:
            errors.append("seniority has no nulls")
        else:
            partner_mask = df["lead_source"] == "partner_referral"
            other_mask = ~partner_mask
            if not partner_mask.any():
                errors.append(
                    "seniority missingness check requires at least one partner_referral row"
                )
            elif not other_mask.any():
                errors.append(
                    "seniority missingness check requires at least one non-partner_referral row"
                )
            else:
                partner_rate = df.loc[partner_mask, "seniority"].isna().mean()
                other_rate = df.loc[other_mask, "seniority"].isna().mean()
                if other_rate > 0 and partner_rate / other_rate < 3.0:
                    errors.append(
                        f"seniority missing ratio partner/other = "
                        f"{partner_rate / other_rate:.1f}x (need >3x)"
                    )

    # days_since_last_touch must have nulls
    if "days_since_last_touch" in df.columns:
        if df["days_since_last_touch"].isna().sum() == 0:
            errors.append("days_since_last_touch has no nulls")

    # Per-column missingness bound
    for col in df.columns:
        if col == TARGET:
            continue
        miss_rate = df[col].isna().mean()
        if miss_rate > MAX_COL_MISSING_RATE:
            errors.append(f"{col} has {miss_rate:.1%} missing (>{MAX_COL_MISSING_RATE:.0%})")

    # Target must never be missing
    if df[TARGET].isna().sum() > 0:
        errors.append(f"Target column '{TARGET}' has missing values!")

    return errors


def check_shape(df: pd.DataFrame) -> list[str]:
    """Check 7: Shape constraints."""
    errors = []
    if len(df) != 1000:
        errors.append(f"Expected 1000 rows, got {len(df)}")
    if len(df.columns) != 19:
        errors.append(f"Expected 19 columns, got {len(df.columns)}")
    return errors


def check_duplicates(df: pd.DataFrame) -> list[str]:
    """Check 8: No excessive duplicate rows."""
    n_dupes = df.duplicated().sum()
    dupe_rate = n_dupes / len(df)
    errors = []
    if dupe_rate > MAX_DUPLICATE_RATE:
        errors.append(f"{n_dupes} duplicate rows ({dupe_rate:.1%}, max {MAX_DUPLICATE_RATE:.0%})")
    return errors


def check_leakage_naming(df: pd.DataFrame) -> list[str]:
    """Check 9: Leakage columns must be explicitly named with __leakage__ prefix."""
    errors = []
    leakage_cols = [c for c in df.columns if c.startswith("__leakage__")]
    if len(leakage_cols) == 0:
        errors.append("No __leakage__ prefixed column found")
    elif len(leakage_cols) > 1:
        errors.append(f"Multiple leakage columns found: {leakage_cols}")
    # total_touches_all should NOT exist (replaced by __leakage__ name)
    if "total_touches_all" in df.columns:
        errors.append("Old leakage trap name 'total_touches_all' still present")
    return errors


def check_acv_range(df: pd.DataFrame) -> list[str]:
    """Check 10: expected_acv within narrative-consistent range."""
    errors = []
    if "expected_acv" in df.columns:
        acv = pd.to_numeric(df["expected_acv"], errors="coerce").dropna()
        if acv.empty:
            errors.append("expected_acv contains no usable numeric values")
            return errors
        if acv.min() < 18_000 - 1:
            errors.append(f"expected_acv min {acv.min():.0f} below narrative floor 18,000")
        if acv.max() > 120_000 + 1:
            errors.append(f"expected_acv max {acv.max():.0f} above narrative cap 120,000")
    return errors


# ---------------------------------------------------------------------------
# Warning checks
# ---------------------------------------------------------------------------


def warn_redundancy(df: pd.DataFrame) -> list[str]:
    """Warning: Column redundancy."""
    warnings = []
    if "inbound_touches" in df.columns and "outbound_touches" in df.columns:
        total = df["inbound_touches"].fillna(0) + df["outbound_touches"].fillna(0)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ("inbound_touches", "outbound_touches", TARGET, LEAKAGE_TRAP):
                continue
            corr = total.corr(df[col].fillna(0))
            if abs(corr) > 0.99:
                warnings.append(f"inbound+outbound correlates {corr:.3f} with {col}")
    return warnings


def warn_low_variance(df: pd.DataFrame) -> list[str]:
    """Warning: Low-variance features."""
    warnings = []
    for col in df.columns:
        if col == TARGET:
            continue
        nunique = df[col].dropna().nunique()
        if nunique < 3 and col not in BINARY_FEATURES:
            warnings.append(f"{col} has only {nunique} unique value(s)")
    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def validate(csv_path: str) -> int:
    """Run all checks and return exit code."""
    df = pd.read_csv(csv_path)
    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Mandatory checks
    print("Check 1:  Banned columns...", end=" ")
    errs = check_banned_columns(df)
    print("FAIL" if errs else "PASS")
    all_errors.extend(errs)

    print("Check 2:  Deterministic groups...", end=" ")
    errs = check_deterministic_groups(df)
    print("FAIL" if errs else "PASS")
    all_errors.extend(errs)

    print("Check 3:  Conversion rate...", end=" ")
    errs = check_conversion_rate(df)
    rate = df[TARGET].mean()
    print(f"{'FAIL' if errs else 'PASS'} ({rate:.1%})")
    all_errors.extend(errs)

    print("Check 4:  Baseline AUC (hold-out)...", end=" ")
    errs, baseline_metrics = check_baseline_auc(df)
    auc = baseline_metrics.get("auc", 0)
    pr_auc = baseline_metrics.get("pr_auc", 0)
    print(f"{'FAIL' if errs else 'PASS'} (AUC={auc:.3f}, PR-AUC={pr_auc:.3f})")
    all_errors.extend(errs)

    if baseline_metrics:
        for k in [25, 50, 100]:
            key_p = f"precision@{k}"
            key_l = f"lift@{k}"
            if key_p in baseline_metrics:
                print(
                    f"          Precision@{k}={baseline_metrics[key_p]:.3f}  "
                    f"Lift@{k}={baseline_metrics[key_l]:.2f}"
                )

    print("Check 5:  Leakage trap (multi-seed)...", end=" ")
    errs, trap_stats = check_leakage_trap_robust(df)
    if trap_stats:
        print(
            f"{'FAIL' if errs else 'PASS'} "
            f"(mean={trap_stats['mean_delta']:.4f}, "
            f"min={trap_stats['min_delta']:.4f}, "
            f"max={trap_stats['max_delta']:.4f})"
        )
    else:
        print("FAIL")
    all_errors.extend(errs)

    print("Check 6:  Missingness...", end=" ")
    errs = check_missingness(df)
    print("FAIL" if errs else "PASS")
    all_errors.extend(errs)

    print("Check 7:  Shape...", end=" ")
    errs = check_shape(df)
    print(f"{'FAIL' if errs else 'PASS'} ({len(df)} rows × {len(df.columns)} cols)")
    all_errors.extend(errs)

    print("Check 8:  Duplicates...", end=" ")
    errs = check_duplicates(df)
    n_dupes = df.duplicated().sum()
    print(f"{'FAIL' if errs else 'PASS'} ({n_dupes} duplicates)")
    all_errors.extend(errs)

    print("Check 9:  Leakage naming...", end=" ")
    errs = check_leakage_naming(df)
    print("FAIL" if errs else "PASS")
    all_errors.extend(errs)

    print("Check 10: ACV range...", end=" ")
    errs = check_acv_range(df)
    if "expected_acv" in df.columns:
        acv = df["expected_acv"].dropna()
        print(f"{'FAIL' if errs else 'PASS'} (range: {acv.min():.0f}–{acv.max():.0f})")
    else:
        print("FAIL (column missing)")
    all_errors.extend(errs)

    # Missingness summary
    print("\nMissingness summary:")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            print(f"  {col}: {n_miss} ({n_miss / len(df):.1%})")
    total_miss = df.isna().sum().sum()
    print(f"  Total: {total_miss} missing values across all columns")

    # Warnings
    print("\nWarning checks:")
    warns = warn_redundancy(df)
    if warns:
        all_warnings.extend(warns)
    warns = warn_low_variance(df)
    if warns:
        all_warnings.extend(warns)

    # Report
    if all_errors:
        print(f"\n{'=' * 60}")
        print(f"FAILED — {len(all_errors)} error(s):")
        for err in all_errors:
            print(f"  ✗ {err}")
    else:
        print(f"\n{'=' * 60}")
        print("ALL MANDATORY CHECKS PASSED")

    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for warn in all_warnings:
            print(f"  ⚠ {warn}")

    return 1 if all_errors else 0


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} CSV_PATH", file=sys.stderr)
        sys.exit(1)
    sys.exit(validate(sys.argv[1]))


if __name__ == "__main__":
    main()
