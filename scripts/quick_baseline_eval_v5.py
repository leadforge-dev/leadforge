#!/usr/bin/env python3
"""Quick baseline evaluation for the v5 lead scoring intro dataset.

Usage:
    python scripts/quick_baseline_eval_v5.py lead_scoring_intro_v5.csv

Runs Logistic Regression and Random Forest on a 70/30 hold-out split,
prints key metrics, and demonstrates leakage trap detection.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET = "converted"
LEAKAGE_TRAP = "__leakage__total_touches_90d"
SEED = 42


def split_and_preprocess(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split first, then fit preprocessing on train only.

    Returns (x_train, x_test, y_train, y_test) with numeric columns,
    label-encoded categoricals, and train-median imputation.
    """
    feature_cols = [c for c in df.columns if c != TARGET and c not in (exclude or [])]
    x_raw = df[feature_cols].copy()
    y = df[TARGET].astype(int)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_raw, y, test_size=0.30, random_state=seed, stratify=y
    )

    cat_cols = list(x_train_raw.select_dtypes(include=["object", "category"]).columns)
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(x_train_raw[col].astype(str).fillna("__MISSING__"))
        x_train_raw[col] = le.transform(x_train_raw[col].astype(str).fillna("__MISSING__"))
        test_vals = x_test_raw[col].astype(str).fillna("__MISSING__")
        test_vals = test_vals.where(test_vals.isin(le.classes_), "__MISSING__")
        x_test_raw[col] = le.transform(test_vals)

    x_train = x_train_raw.select_dtypes(include=[np.number]).copy()
    x_test = x_test_raw[x_train.columns].copy()
    train_medians = x_train.median()
    x_train = x_train.fillna(train_medians)
    x_test = x_test.fillna(train_medians)

    return x_train, x_test, y_train, y_test


def evaluate(name: str, y_true: pd.Series, probs: np.ndarray) -> dict[str, float]:
    """Compute and print metrics."""
    auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    base_rate = y_true.mean()

    metrics = {"AUC": auc, "PR-AUC": pr_auc}
    print(f"\n  {name}")
    print(f"    AUC:    {auc:.3f}")
    print(f"    PR-AUC: {pr_auc:.3f}")

    n_test = len(y_true)
    for k in [25, 50, 100]:
        if k > n_test:
            continue
        top_k_idx = np.argsort(-probs)[:k]
        top_k_labels = y_true.iloc[top_k_idx]
        prec_k = float(top_k_labels.mean())
        lift_k = prec_k / base_rate if base_rate > 0 else 0.0
        metrics[f"P@{k}"] = prec_k
        metrics[f"Lift@{k}"] = lift_k
        print(f"    P@{k:3d}:  {prec_k:.3f}  (Lift: {lift_k:.2f}x)")

    return metrics


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} CSV_PATH", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    print(f"Dataset: {len(df)} rows × {len(df.columns)} cols")
    print(f"Conversion rate: {df[TARGET].mean():.1%}")
    print(f"Missing values: {df.isna().sum().sum()} total")

    # --- Without leakage trap ---
    print(f"\n{'=' * 60}")
    print("BASELINE (without leakage trap)")
    print(f"{'=' * 60}")

    x_train, x_test, y_train, y_test = split_and_preprocess(df, exclude=[LEAKAGE_TRAP])

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    lr = LogisticRegression(max_iter=2000, random_state=SEED)
    lr.fit(x_train_s, y_train)
    evaluate("Logistic Regression", y_test, lr.predict_proba(x_test_s)[:, 1])

    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(x_train, y_train)
    evaluate("Random Forest", y_test, rf.predict_proba(x_test)[:, 1])

    # --- With leakage trap ---
    if LEAKAGE_TRAP in df.columns:
        print(f"\n{'=' * 60}")
        print("WITH LEAKAGE TRAP (for comparison — students should detect this)")
        print(f"{'=' * 60}")

        x_train_f, x_test_f, y_train_f, y_test_f = split_and_preprocess(df)
        scaler_f = StandardScaler()
        x_train_fs = scaler_f.fit_transform(x_train_f)
        x_test_fs = scaler_f.transform(x_test_f)

        lr_f = LogisticRegression(max_iter=2000, random_state=SEED)
        lr_f.fit(x_train_fs, y_train_f)
        m_with = evaluate("LR with trap", y_test_f, lr_f.predict_proba(x_test_fs)[:, 1])

        x_train_n, x_test_n, _, _ = split_and_preprocess(df, exclude=[LEAKAGE_TRAP])
        scaler_n = StandardScaler()
        x_train_ns = scaler_n.fit_transform(x_train_n)
        x_test_ns = scaler_n.transform(x_test_n)
        lr_without = LogisticRegression(max_iter=2000, random_state=SEED)
        lr_without.fit(x_train_ns, y_train_f)
        m_without = evaluate("LR without trap", y_test_f, lr_without.predict_proba(x_test_ns)[:, 1])

        delta = m_with["AUC"] - m_without["AUC"]
        print(f"\n  ** Leakage trap AUC delta: {delta:+.4f} **")
        if delta > 0.02:
            print("  → Detectable improvement — students should investigate why")
        else:
            print("  → Small delta — trap may be hard to detect in single split")

    # --- Feature importance ---
    print(f"\n{'=' * 60}")
    print("FEATURE IMPORTANCE (Random Forest, without trap)")
    print(f"{'=' * 60}")
    importances = sorted(
        zip(x_train.columns, rf.feature_importances_, strict=False),
        key=lambda t: t[1],
        reverse=True,
    )
    for feat, imp in importances:
        bar = "█" * int(imp * 100)
        print(f"  {feat:30s} {imp:.3f} {bar}")

    # --- Expected value demonstration ---
    if "expected_acv" in df.columns:
        print(f"\n{'=' * 60}")
        print("VALUE-AWARE SCORING DEMO")
        print(f"{'=' * 60}")

        # Reuse the baseline split (same seed, same rows)
        x_tr, x_te, y_tr, y_te = split_and_preprocess(df, exclude=[LEAKAGE_TRAP])
        scaler_v = StandardScaler()
        x_tr_s = scaler_v.fit_transform(x_tr)
        x_te_s = scaler_v.transform(x_te)

        lr_v = LogisticRegression(max_iter=2000, random_state=SEED)
        lr_v.fit(x_tr_s, y_tr)
        test_probs = lr_v.predict_proba(x_te_s)[:, 1]

        test_df = df.iloc[x_te.index].copy()
        test_df["pred_prob"] = test_probs
        test_df["expected_value"] = test_df["pred_prob"] * test_df["expected_acv"]

        for k in [25, 50]:
            # Rank by probability
            top_k_prob = test_df.nlargest(k, "pred_prob")
            ev_prob = top_k_prob.loc[top_k_prob[TARGET] == 1, "expected_acv"].sum()

            # Rank by expected value
            top_k_ev = test_df.nlargest(k, "expected_value")
            ev_ev = top_k_ev.loc[top_k_ev[TARGET] == 1, "expected_acv"].sum()

            print(f"\n  Top-{k} leads:")
            print(f"    Ranked by P(convert):     captured ACV = ${ev_prob:,.0f}")
            print(f"    Ranked by expected value:  captured ACV = ${ev_ev:,.0f}")
            diff_pct = ((ev_ev - ev_prob) / ev_prob * 100) if ev_prob > 0 else 0
            print(f"    Difference: {diff_pct:+.1f}%")


if __name__ == "__main__":
    main()
