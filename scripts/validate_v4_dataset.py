#!/usr/bin/env python3
"""Validate a v4 lead scoring intro CSV against the v4 validation spec.

Usage:
    python scripts/validate_v4_dataset.py lead_scoring_intro_v4.csv

Exit code 0 = all mandatory checks pass.
Exit code 1 = at least one mandatory check failed.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

LEAKAGE_TRAP = "total_touches_all"

# Deterministic group thresholds
MIN_GROUP_SIZE = 50
RATE_LOWER = 0.02
RATE_UPPER = 0.98

# AUC bounds
AUC_LOWER = 0.65
AUC_UPPER = 0.90
AUC_TRAP_BOOST = 0.03


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


def _fit_lr(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> float:
    """Fit LR and return AUC."""
    feature_cols = [c for c in df.columns if c != TARGET and c not in (exclude_cols or [])]
    x_df = df[feature_cols].copy()
    y = df[TARGET].astype(int)

    for col in x_df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        x_df[col] = le.fit_transform(x_df[col].astype(str).fillna("__MISSING__"))

    x_df = x_df.select_dtypes(include=[np.number])
    x_df = x_df.fillna(x_df.median())

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df)

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(x_scaled, y)
    probs = lr.predict_proba(x_scaled)[:, 1]
    return float(roc_auc_score(y, probs))


def check_baseline_auc(df: pd.DataFrame) -> tuple[list[str], float]:
    """Check 4: Baseline model AUC without leakage trap."""
    auc = _fit_lr(df, exclude_cols=[LEAKAGE_TRAP])
    errors = []
    if auc < AUC_LOWER:
        errors.append(f"Baseline AUC {auc:.3f} below {AUC_LOWER}")
    if auc > AUC_UPPER:
        errors.append(f"Baseline AUC {auc:.3f} above {AUC_UPPER}")
    return errors, auc


def check_leakage_trap(df: pd.DataFrame, baseline_auc: float) -> list[str]:
    """Check 5: Leakage trap effectiveness."""
    if LEAKAGE_TRAP not in df.columns:
        return [f"Leakage trap column '{LEAKAGE_TRAP}' not found"]
    full_auc = _fit_lr(df)
    boost = full_auc - baseline_auc
    errors = []
    if boost < AUC_TRAP_BOOST:
        errors.append(
            f"Leakage trap boost {boost:.3f} below {AUC_TRAP_BOOST} "
            f"(baseline={baseline_auc:.3f}, full={full_auc:.3f})"
        )
    return errors


def check_missingness(df: pd.DataFrame) -> list[str]:
    """Check 6: Missingness structure."""
    errors = []

    # web_sessions must have nulls
    if "web_sessions" in df.columns:
        if df["web_sessions"].isna().sum() == 0:
            errors.append("web_sessions has no nulls")
        else:
            # Check source-conditional ratio
            outbound_rate = (
                df.loc[df["lead_source"] == "sdr_outbound", "web_sessions"].isna().mean()
            )
            inbound_rate = (
                df.loc[df["lead_source"] == "inbound_marketing", "web_sessions"].isna().mean()
            )
            if inbound_rate > 0 and outbound_rate / inbound_rate < 3.0:
                errors.append(
                    f"web_sessions missing ratio outbound/inbound = "
                    f"{outbound_rate / inbound_rate:.1f}x (need >3x)"
                )
            elif inbound_rate == 0 and outbound_rate > 0:
                pass  # Trivially satisfied
            elif inbound_rate == 0 and outbound_rate == 0:
                errors.append("web_sessions has no source-conditional missingness")

    # seniority must have nulls
    if "seniority" in df.columns:
        if df["seniority"].isna().sum() == 0:
            errors.append("seniority has no nulls")
        else:
            partner_rate = (
                df.loc[df["lead_source"] == "partner_referral", "seniority"].isna().mean()
            )
            other_rate = df.loc[df["lead_source"] != "partner_referral", "seniority"].isna().mean()
            if other_rate > 0 and partner_rate / other_rate < 3.0:
                errors.append(
                    f"seniority missing ratio partner/other = "
                    f"{partner_rate / other_rate:.1f}x (need >3x)"
                )

    # days_since_last_touch must have nulls
    if "days_since_last_touch" in df.columns:
        if df["days_since_last_touch"].isna().sum() == 0:
            errors.append("days_since_last_touch has no nulls")

    # No column should have > 20% missing
    for col in df.columns:
        miss_rate = df[col].isna().mean()
        if miss_rate > 0.20:
            errors.append(f"{col} has {miss_rate:.1%} missing (>20%)")

    return errors


def check_shape(df: pd.DataFrame) -> list[str]:
    """Check 7: Shape constraints."""
    errors = []
    if len(df) != 1000:
        errors.append(f"Expected 1000 rows, got {len(df)}")
    if len(df.columns) != 18:
        errors.append(f"Expected 18 columns, got {len(df.columns)}")
    return errors


# ---------------------------------------------------------------------------
# Warning checks
# ---------------------------------------------------------------------------


def warn_redundancy(df: pd.DataFrame) -> list[str]:
    """Warning 2: Column redundancy."""
    warnings = []
    if "inbound_touches" in df.columns and "outbound_touches" in df.columns:
        total = df["inbound_touches"].fillna(0) + df["outbound_touches"].fillna(0)
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ("inbound_touches", "outbound_touches", TARGET):
                continue
            corr = total.corr(df[col].fillna(0))
            if abs(corr) > 0.99:
                warnings.append(f"inbound+outbound correlates {corr:.3f} with {col}")
    return warnings


def warn_low_variance(df: pd.DataFrame) -> list[str]:
    """Warning 3: Low-variance features."""
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
    print("Check 1: Banned columns...", end=" ")
    errs = check_banned_columns(df)
    print("FAIL" if errs else "PASS")
    all_errors.extend(errs)

    print("Check 2: Deterministic groups...", end=" ")
    errs = check_deterministic_groups(df)
    print("FAIL" if errs else "PASS")
    all_errors.extend(errs)

    print("Check 3: Conversion rate...", end=" ")
    errs = check_conversion_rate(df)
    rate = df[TARGET].mean()
    print(f"{'FAIL' if errs else 'PASS'} ({rate:.1%})")
    all_errors.extend(errs)

    print("Check 4: Baseline AUC...", end=" ")
    errs, baseline_auc = check_baseline_auc(df)
    print(f"{'FAIL' if errs else 'PASS'} (AUC={baseline_auc:.3f})")
    all_errors.extend(errs)

    print("Check 5: Leakage trap...", end=" ")
    errs = check_leakage_trap(df, baseline_auc)
    print("FAIL" if errs else "PASS")
    all_errors.extend(errs)

    print("Check 6: Missingness...", end=" ")
    errs = check_missingness(df)
    print("FAIL" if errs else "PASS")
    all_errors.extend(errs)

    print("Check 7: Shape...", end=" ")
    errs = check_shape(df)
    print(f"{'FAIL' if errs else 'PASS'} ({len(df)} rows × {len(df.columns)} cols)")
    all_errors.extend(errs)

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
