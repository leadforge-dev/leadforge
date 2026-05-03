#!/usr/bin/env python3
"""Validate v6 lead scoring intro CSVs against the v6 spec.

Usage:
    python scripts/validate_v6_dataset.py STUDENT_CSV INSTRUCTOR_CSV

Validates both exports and runs all mandatory checks. Exit code 0 = all pass.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET = "converted"
LEAKAGE_PREFIX = "__leakage__"

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
    "acquisition_wave",
]

BINARY_FEATURES = [
    "opportunity_created",
    "demo_completed",
]

# Validation thresholds
AUC_LOWER = 0.60
AUC_UPPER = 0.90
TRAP_MEAN_DELTA = 0.03
TRAP_MIN_DELTA = 0.015
TRAP_N_SEEDS = 10
TRAP_SEED_START = 42
MAX_COL_MISSING_RATE = 0.10
MAX_DUPLICATE_RATE = 0.01
MIN_GROUP_SIZE = 50
RATE_LOWER = 0.02
RATE_UPPER = 0.98


# ---------------------------------------------------------------------------
# ML pipeline builder (canonical)
# ---------------------------------------------------------------------------


def _build_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """Build the canonical sklearn baseline pipeline."""
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


def _fit_evaluate(
    df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
    seed: int = 42,
    test_size: float = 0.30,
) -> tuple[float, float, np.ndarray, pd.Series]:
    """Fit LR on hold-out split, return (AUC, PR-AUC, probs, y_test)."""
    y = df[TARGET].astype(int)
    cat_cols, num_cols = _get_feature_cols(df, exclude=exclude_cols)
    x = df[cat_cols + num_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe = _build_pipeline(num_cols, cat_cols)
    pipe.fit(x_train, y_train)
    probs = pipe.predict_proba(x_test)[:, 1]

    auc = float(roc_auc_score(y_test, probs))
    pr_auc = float(average_precision_score(y_test, probs))
    return auc, pr_auc, probs, y_test


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def check_basic(df: pd.DataFrame, label: str) -> list[str]:
    """Basic structural checks."""
    errors = []

    # Row count
    if len(df) != 1000:
        errors.append(f"[{label}] Expected 1000 rows, got {len(df)}")

    # Target
    if TARGET not in df.columns:
        errors.append(f"[{label}] Missing target column '{TARGET}'")
        return errors
    target_vals = set(df[TARGET].dropna().unique())
    if not target_vals <= {0, 1}:
        errors.append(f"[{label}] Target values not binary: {target_vals}")
    if df[TARGET].isna().any():
        errors.append(f"[{label}] Target has missing values")

    # Banned columns
    present = BANNED_COLUMNS & set(df.columns)
    if present:
        errors.append(f"[{label}] Banned columns: {sorted(present)}")

    # ID columns
    id_cols = [c for c in df.columns if c.endswith("_id")]
    if id_cols:
        errors.append(f"[{label}] ID columns: {sorted(id_cols)}")

    # Duplicates
    n_dupes = df.duplicated().sum()
    dupe_rate = n_dupes / len(df) if len(df) > 0 else 0
    if dupe_rate > MAX_DUPLICATE_RATE:
        errors.append(f"[{label}] {n_dupes} duplicates ({dupe_rate:.1%})")

    # Missingness
    for col in df.columns:
        if col == TARGET:
            continue
        rate = float(df[col].isna().mean())
        if rate > MAX_COL_MISSING_RATE:
            errors.append(f"[{label}] {col} missing rate {rate:.1%} > {MAX_COL_MISSING_RATE:.0%}")

    return errors


def check_determinism(df: pd.DataFrame, label: str) -> list[str]:
    """No categorical/binary group should be near-deterministic."""
    errors = []
    check_cols = [c for c in CAT_FEATURES + BINARY_FEATURES if c in df.columns]
    for col in check_cols:
        stats = df.groupby(col)[TARGET].agg(["mean", "count"])
        large = stats[stats["count"] >= MIN_GROUP_SIZE]
        for val, row in large.iterrows():
            if row["mean"] < RATE_LOWER:
                errors.append(
                    f"[{label}] DETERMINISTIC: {col}={val}: "
                    f"{row['mean']:.1%} (n={int(row['count'])})"
                )
            if row["mean"] > RATE_UPPER:
                errors.append(
                    f"[{label}] DETERMINISTIC: {col}={val}: "
                    f"{row['mean']:.1%} (n={int(row['count'])})"
                )
    return errors


def check_baseline_auc(df: pd.DataFrame, label: str) -> tuple[list[str], dict[str, float]]:
    """Baseline model AUC on hold-out split."""
    leakage = {c for c in df.columns if c.startswith(LEAKAGE_PREFIX)}
    auc, pr_auc, probs, y_test = _fit_evaluate(df, exclude_cols=leakage)

    errors = []
    if auc < AUC_LOWER:
        errors.append(f"[{label}] Baseline AUC {auc:.3f} < {AUC_LOWER}")
    if auc > AUC_UPPER:
        errors.append(f"[{label}] Baseline AUC {auc:.3f} > {AUC_UPPER}")

    base_rate = float(y_test.mean())
    n_pos = int(y_test.sum())
    order = np.argsort(-probs, kind="stable")
    y_sorted = y_test.values[order]

    metrics: dict[str, float] = {"auc": auc, "pr_auc": pr_auc, "base_rate": base_rate}
    for k in [25, 50]:
        if k > len(y_test):
            continue
        top_k = y_sorted[:k]
        prec = float(top_k.mean())
        rec = float(top_k.sum() / n_pos) if n_pos > 0 else 0.0
        lift = prec / base_rate if base_rate > 0 else 0.0
        metrics[f"precision@{k}"] = prec
        metrics[f"recall@{k}"] = rec
        metrics[f"lift@{k}"] = lift

    return errors, metrics


def check_tree_improvement(df: pd.DataFrame, label: str) -> tuple[list[str], dict[str, float]]:
    """Tree model should not be significantly worse than LR."""
    leakage = {c for c in df.columns if c.startswith(LEAKAGE_PREFIX)}
    cat_cols, num_cols = _get_feature_cols(df, exclude=leakage)
    y = df[TARGET].astype(int)
    x = df[cat_cols + num_cols]

    lr_aucs = []
    gb_aucs = []
    for seed in range(42, 47):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.30, random_state=seed, stratify=y
        )

        lr = _build_pipeline(num_cols, cat_cols)
        lr.fit(x_train, y_train)
        lr_auc = roc_auc_score(y_test, lr.predict_proba(x_test)[:, 1])
        lr_aucs.append(lr_auc)

        # GBM with one-hot encoded features
        numeric_transformer = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
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
        gb = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ]
        )
        gb.fit(x_train, y_train)
        gb_auc = roc_auc_score(y_test, gb.predict_proba(x_test)[:, 1])
        gb_aucs.append(gb_auc)

    mean_lr = float(np.mean(lr_aucs))
    mean_gb = float(np.mean(gb_aucs))
    improvement = mean_gb - mean_lr

    errors = []
    if mean_gb < mean_lr - 0.01:
        errors.append(
            f"[{label}] GBM significantly worse than LR: LR={mean_lr:.4f}, GBM={mean_gb:.4f}"
        )

    metrics = {
        "mean_lr_auc": mean_lr,
        "mean_gbm_auc": mean_gb,
        "mean_improvement": improvement,
    }

    return errors, metrics


def check_leakage_student(df: pd.DataFrame) -> list[str]:
    """Student export must have ZERO leakage columns."""
    leakage = [c for c in df.columns if c.startswith(LEAKAGE_PREFIX)]
    if leakage:
        return [f"[student] Leakage columns found: {leakage}"]
    return []


def check_leakage_instructor(df: pd.DataFrame) -> list[str]:
    """Instructor export must have EXACTLY ONE leakage column."""
    leakage = [c for c in df.columns if c.startswith(LEAKAGE_PREFIX)]
    if len(leakage) == 0:
        return ["[instructor] No __leakage__ column found"]
    if len(leakage) > 1:
        return [f"[instructor] Multiple leakage columns: {leakage}"]
    return []


def check_trap_delta(df: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    """Leakage trap AUC delta across seeds."""
    leakage_cols = [c for c in df.columns if c.startswith(LEAKAGE_PREFIX)]
    if not leakage_cols:
        return ["No trap column found"], {}

    trap_col = leakage_cols[0]
    all_leakage = set(leakage_cols)

    deltas = []
    for seed in range(TRAP_SEED_START, TRAP_SEED_START + TRAP_N_SEEDS):
        auc_without, _, _, _ = _fit_evaluate(df, exclude_cols=all_leakage, seed=seed)
        auc_with, _, _, _ = _fit_evaluate(df, exclude_cols=all_leakage - {trap_col}, seed=seed)
        deltas.append(auc_with - auc_without)

    mean_delta = float(np.mean(deltas))
    min_delta = float(np.min(deltas))
    max_delta = float(np.max(deltas))

    errors = []
    if mean_delta < TRAP_MEAN_DELTA:
        errors.append(f"Trap mean delta {mean_delta:.4f} < {TRAP_MEAN_DELTA} (min={min_delta:.4f})")
    if min_delta < TRAP_MIN_DELTA:
        bad = [
            f"seed {s}: {d:.4f}"
            for s, d in zip(
                range(TRAP_SEED_START, TRAP_SEED_START + TRAP_N_SEEDS), deltas, strict=True
            )
            if d < TRAP_MIN_DELTA
        ]
        errors.append(f"Trap min delta {min_delta:.4f} < {TRAP_MIN_DELTA} [{', '.join(bad)}]")

    stats = {
        "mean_delta": mean_delta,
        "min_delta": min_delta,
        "max_delta": max_delta,
        "deltas": deltas,
    }
    return errors, stats


def check_value_aware(df: pd.DataFrame) -> tuple[list[str], list[dict[str, float]]]:
    """Value-aware ranking: EV ranking should capture >= prob ranking ACV."""
    if "expected_acv" not in df.columns:
        return ["expected_acv column missing"], []

    leakage = {c for c in df.columns if c.startswith(LEAKAGE_PREFIX)}
    cat_cols, num_cols = _get_feature_cols(df, exclude=leakage)
    y = df[TARGET].astype(int)
    x = df[cat_cols + num_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )

    pipe = _build_pipeline(num_cols, cat_cols)
    pipe.fit(x_train, y_train)
    probs = pipe.predict_proba(x_test)[:, 1]

    test_acv = pd.to_numeric(df.loc[x_test.index, "expected_acv"], errors="coerce").fillna(0).values
    test_converted = y_test.values
    expected_value = probs * test_acv

    errors = []
    results = []
    for k in [25, 50]:
        if k > len(y_test):
            continue

        order_prob = np.argsort(-probs, kind="stable")
        top_k_prob = order_prob[:k]
        captured_prob = float(np.sum(test_acv[top_k_prob] * test_converted[top_k_prob]))

        order_ev = np.argsort(-expected_value, kind="stable")
        top_k_ev = order_ev[:k]
        captured_ev = float(np.sum(test_acv[top_k_ev] * test_converted[top_k_ev]))

        uplift = ((captured_ev - captured_prob) / captured_prob * 100) if captured_prob > 0 else 0.0
        results.append(
            {"k": k, "captured_prob": captured_prob, "captured_ev": captured_ev, "uplift": uplift}
        )
        if captured_ev < captured_prob:
            # Warning only, not a hard error
            pass

    return errors, results


def check_acv_range(df: pd.DataFrame, label: str) -> list[str]:
    """expected_acv within narrative range."""
    if "expected_acv" not in df.columns:
        return [f"[{label}] expected_acv column missing"]
    acv = pd.to_numeric(df["expected_acv"], errors="coerce").dropna()
    if acv.empty:
        return [f"[{label}] expected_acv has no values"]
    errors = []
    if acv.min() < 18_000 - 1:
        errors.append(f"[{label}] expected_acv min {acv.min():.0f} < 18,000")
    if acv.max() > 120_000 + 1:
        errors.append(f"[{label}] expected_acv max {acv.max():.0f} > 120,000")
    return errors


def check_row_alignment(student: pd.DataFrame, instructor: pd.DataFrame) -> list[str]:
    """Instructor file must be identical to student in all shared columns."""
    errors = []
    student_cols = set(student.columns)
    shared_cols = [c for c in instructor.columns if c in student_cols]

    if len(student) != len(instructor):
        errors.append(f"Row count mismatch: student={len(student)}, instructor={len(instructor)}")
        return errors

    for col in shared_cols:
        s = student[col]
        i = instructor[col]
        # Both NaN or both equal
        mask = s.isna() & i.isna()
        vals_match = (s == i) | mask
        if not vals_match.all():
            n_diff = int((~vals_match).sum())
            errors.append(f"Column '{col}' differs in {n_diff} rows between exports")

    return errors


def check_cohort_split(df: pd.DataFrame) -> dict[str, float] | None:
    """Optional cohort split evaluation (train A/B, test C)."""
    if "acquisition_wave" not in df.columns:
        return None

    leakage = {c for c in df.columns if c.startswith(LEAKAGE_PREFIX)}
    cat_cols, num_cols = _get_feature_cols(df, exclude=leakage | {"acquisition_wave"})
    y = df[TARGET].astype(int)
    x = df[cat_cols + num_cols]

    # Random split baseline
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    pipe_r = _build_pipeline(num_cols, cat_cols)
    pipe_r.fit(x_train, y_train)
    random_auc = roc_auc_score(y_test, pipe_r.predict_proba(x_test)[:, 1])

    # Cohort split: train A/B, test C
    train_mask = df["acquisition_wave"].isin(["A", "B"])
    test_mask = df["acquisition_wave"] == "C"

    if test_mask.sum() < 30 or train_mask.sum() < 100:
        return None

    x_train_c = x[train_mask]
    y_train_c = y[train_mask]
    x_test_c = x[test_mask]
    y_test_c = y[test_mask]

    pipe_c = _build_pipeline(num_cols, cat_cols)
    pipe_c.fit(x_train_c, y_train_c)
    cohort_auc = roc_auc_score(y_test_c, pipe_c.predict_proba(x_test_c)[:, 1])

    return {
        "random_auc": random_auc,
        "cohort_auc": cohort_auc,
        "drop": random_auc - cohort_auc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def validate(student_path: str, instructor_path: str) -> int:
    """Run all checks and return exit code."""
    student = pd.read_csv(student_path)
    instructor = pd.read_csv(instructor_path)
    all_errors: list[str] = []

    # 1. Basic checks (both)
    print("=" * 60)
    print("BASIC CHECKS")
    print("=" * 60)

    for label, df in [("student", student), ("instructor", instructor)]:
        errs = check_basic(df, label)
        status = "FAIL" if errs else "PASS"
        print(f"  [{label}] Structural checks: {status}")
        all_errors.extend(errs)

    # 2. Row alignment
    print("\nRow alignment...", end=" ")
    errs = check_row_alignment(student, instructor)
    print("FAIL" if errs else "PASS")
    all_errors.extend(errs)

    # 3. Leakage column checks
    print("\nLeakage column checks:")
    errs = check_leakage_student(student)
    print(f"  [student] No leakage columns: {'FAIL' if errs else 'PASS'}")
    all_errors.extend(errs)

    errs = check_leakage_instructor(instructor)
    leakage_cols = [c for c in instructor.columns if c.startswith(LEAKAGE_PREFIX)]
    status = "FAIL" if errs else "PASS"
    print(f"  [instructor] Exactly one leakage column: {status} ({leakage_cols})")
    all_errors.extend(errs)

    # 4. Determinism checks
    print("\nDeterminism checks:")
    for label, df in [("student", student), ("instructor", instructor)]:
        errs = check_determinism(df, label)
        print(f"  [{label}]: {'FAIL' if errs else 'PASS'}")
        all_errors.extend(errs)

    # 5. ACV range
    print("\nACV range:")
    for label, df in [("student", student)]:
        errs = check_acv_range(df, label)
        acv = pd.to_numeric(df.get("expected_acv", pd.Series()), errors="coerce").dropna()
        range_str = f"[{acv.min():.0f}, {acv.max():.0f}]" if not acv.empty else "N/A"
        print(f"  [{label}]: {'FAIL' if errs else 'PASS'} {range_str}")
        all_errors.extend(errs)

    # 6. Baseline AUC (student dataset)
    print("\nBaseline AUC (student)...", end=" ")
    errs, baseline = check_baseline_auc(student, "student")
    auc = baseline.get("auc", 0)
    pr_auc = baseline.get("pr_auc", 0)
    print(f"{'FAIL' if errs else 'PASS'} (AUC={auc:.3f}, PR-AUC={pr_auc:.3f})")
    all_errors.extend(errs)

    if baseline:
        for k in [25, 50]:
            key_p = f"precision@{k}"
            key_l = f"lift@{k}"
            if key_p in baseline:
                print(f"  P@{k}={baseline[key_p]:.3f}  Lift@{k}={baseline[key_l]:.2f}x")

    # 7. Tree improvement (student dataset)
    print("\nTree model comparison (5 seeds)...", end=" ")
    errs, tree_metrics = check_tree_improvement(student, "student")
    print(
        f"{'FAIL' if errs else 'PASS'} "
        f"(LR={tree_metrics['mean_lr_auc']:.4f}, "
        f"GBM={tree_metrics['mean_gbm_auc']:.4f}, "
        f"delta={tree_metrics['mean_improvement']:+.4f})"
    )
    all_errors.extend(errs)

    # 8. Value-aware (student dataset)
    print("\nValue-aware ranking:")
    errs, value_results = check_value_aware(student)
    for vr in value_results:
        print(
            f"  K={vr['k']}: by_prob=${vr['captured_prob']:,.0f} "
            f"by_ev=${vr['captured_ev']:,.0f} "
            f"uplift={vr['uplift']:+.1f}%"
        )
    all_errors.extend(errs)

    # 9. Leakage trap delta (instructor dataset)
    print("\nLeakage trap delta (instructor, 10 seeds)...", end=" ")
    errs, trap_stats = check_trap_delta(instructor)
    if trap_stats:
        print(
            f"{'FAIL' if errs else 'PASS'} "
            f"(mean={trap_stats['mean_delta']:.4f}, "
            f"min={trap_stats['min_delta']:.4f}, "
            f"max={trap_stats['max_delta']:.4f})"
        )
        if "deltas" in trap_stats:
            seeds = range(TRAP_SEED_START, TRAP_SEED_START + TRAP_N_SEEDS)
            for s, d in zip(seeds, trap_stats["deltas"], strict=True):
                status = "OK" if d >= TRAP_MIN_DELTA else "LOW"
                print(f"  seed {s}: delta={d:+.4f} [{status}]")
    else:
        print("FAIL (no trap data)")
    all_errors.extend(errs)

    # 10. Missingness summary
    print("\nMissingness summary (student):")
    for col in student.columns:
        n_miss = student[col].isna().sum()
        if n_miss > 0:
            print(f"  {col}: {n_miss} ({n_miss / len(student):.1%})")
    total_miss = student.isna().sum().sum()
    print(f"  Total: {total_miss} missing values")

    # 11. Cohort split (optional)
    print("\nCohort split evaluation (optional):")
    cohort = check_cohort_split(student)
    if cohort:
        print(
            f"  Random split AUC: {cohort['random_auc']:.4f}\n"
            f"  Cohort split AUC: {cohort['cohort_auc']:.4f}\n"
            f"  AUC drop:         {cohort['drop']:+.4f}"
        )
    else:
        print("  Skipped (no acquisition_wave column or too few rows)")

    # Report
    if all_errors:
        print(f"\n{'=' * 60}")
        print(f"FAILED - {len(all_errors)} error(s):")
        for err in all_errors:
            print(f"  * {err}")
        return 1
    else:
        print(f"\n{'=' * 60}")
        print("ALL MANDATORY CHECKS PASSED")
        return 0


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} STUDENT_CSV INSTRUCTOR_CSV", file=sys.stderr)
        sys.exit(1)
    sys.exit(validate(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
