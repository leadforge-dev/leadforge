#!/usr/bin/env python3
"""Quick baseline evaluation for the v6 lead scoring intro dataset.

Usage:
    python scripts/quick_baseline_eval_v6.py STUDENT_CSV [INSTRUCTOR_CSV]

Runs Logistic Regression, Random Forest, and GBM on a 70/30 hold-out split,
prints key metrics, Top-K/Lift metrics, value-aware ranking, and feature
importance. Optionally detects leakage trap from instructor file.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "converted"
LEAKAGE_PREFIX = "__leakage__"
SEED = 42


def _get_feature_cols(df, exclude=None):
    exclude = (exclude or set()) | {TARGET}
    cat, num = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num.append(col)
        else:
            cat.append(col)
    return cat, num


def _build_pipeline(num_cols, cat_cols, model="lr"):
    num_tr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_tr = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        [("num", num_tr, num_cols), ("cat", cat_tr, cat_cols)], remainder="drop"
    )
    if model == "lr":
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    elif model == "gbm":
        clf = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    else:
        raise ValueError(model)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(name, y_true, probs, test_acv=None):
    auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    base_rate = y_true.mean()

    print(f"\n  {name}")
    print(f"    AUC:    {auc:.3f}")
    print(f"    PR-AUC: {pr_auc:.3f}")

    n_test = len(y_true)
    for k in [25, 50]:
        if k > n_test:
            continue
        order = np.argsort(-probs, kind="stable")
        top_k = y_true.iloc[order[:k]]
        prec = float(top_k.mean())
        lift = prec / base_rate if base_rate > 0 else 0.0
        print(f"    P@{k:3d}:  {prec:.3f}  (Lift: {lift:.2f}x)")

    if test_acv is not None:
        print("\n    Value-aware ranking:")
        expected_value = probs * test_acv
        for k in [25, 50]:
            if k > n_test:
                continue
            order_prob = np.argsort(-probs, kind="stable")[:k]
            order_ev = np.argsort(-expected_value, kind="stable")[:k]
            cap_prob = float(np.sum(test_acv[order_prob] * y_true.values[order_prob]))
            cap_ev = float(np.sum(test_acv[order_ev] * y_true.values[order_ev]))
            uplift = ((cap_ev - cap_prob) / cap_prob * 100) if cap_prob > 0 else 0.0
            print(
                f"    EV@{k:3d}: by_prob=${cap_prob:,.0f}  "
                f"by_ev=${cap_ev:,.0f}  uplift={uplift:+.1f}%"
            )

    return {"AUC": auc, "PR-AUC": pr_auc}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} STUDENT_CSV [INSTRUCTOR_CSV]", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    instructor_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Dataset: {len(df)} rows x {len(df.columns)} cols")
    print(f"Conversion rate: {df[TARGET].mean():.1%}")
    print(f"Missing values: {df.isna().sum().sum()} total")

    leakage = {c for c in df.columns if c.startswith(LEAKAGE_PREFIX)}
    cat_cols, num_cols = _get_feature_cols(df, exclude=leakage)
    y = df[TARGET].astype(int)
    x = df[cat_cols + num_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=SEED, stratify=y
    )

    test_acv = (
        pd.to_numeric(df.loc[x_test.index, "expected_acv"], errors="coerce").fillna(0).values
        if "expected_acv" in df.columns
        else None
    )

    # --- Baselines ---
    print(f"\n{'=' * 60}")
    print("BASELINE MODELS (without leakage)")
    print(f"{'=' * 60}")

    for model_name, model_key in [
        ("Logistic Regression", "lr"),
        ("Random Forest", "rf"),
        ("Gradient Boosting", "gbm"),
    ]:
        pipe = _build_pipeline(num_cols, cat_cols, model=model_key)
        pipe.fit(x_train, y_train)
        probs = pipe.predict_proba(x_test)[:, 1]
        evaluate(model_name, y_test, probs, test_acv=test_acv if model_key == "lr" else None)

    # --- Feature importance (RF) ---
    print(f"\n{'=' * 60}")
    print("FEATURE IMPORTANCE (Random Forest)")
    print(f"{'=' * 60}")
    rf = _build_pipeline(num_cols, cat_cols, model="rf")
    rf.fit(x_train, y_train)
    pre = rf.named_steps["pre"]
    feature_names = num_cols + list(
        pre.named_transformers_["cat"].named_steps["enc"].get_feature_names_out(cat_cols)
    )
    importances = rf.named_steps["clf"].feature_importances_
    top = sorted(zip(feature_names, importances, strict=False), key=lambda t: t[1], reverse=True)
    for feat, imp in top[:15]:
        bar = "*" * int(imp * 100)
        print(f"  {feat:40s} {imp:.3f} {bar}")

    # --- Leakage trap (instructor file) ---
    if instructor_path:
        df_i = pd.read_csv(instructor_path)
        trap_cols = [c for c in df_i.columns if c.startswith(LEAKAGE_PREFIX)]

        if trap_cols:
            print(f"\n{'=' * 60}")
            print("LEAKAGE TRAP DETECTION (instructor file)")
            print(f"{'=' * 60}")

            trap_col = trap_cols[0]
            all_leakage = set(trap_cols)

            cat_i, num_i = _get_feature_cols(df_i, exclude=all_leakage)
            cat_with, num_with = _get_feature_cols(df_i, exclude=all_leakage - {trap_col})
            y_i = df_i[TARGET].astype(int)

            deltas = []
            for seed in range(42, 52):
                x_without = df_i[cat_i + num_i]
                xt, xte, yt, yte = train_test_split(
                    x_without, y_i, test_size=0.30, random_state=seed, stratify=y_i
                )
                pipe_wo = _build_pipeline(num_i, cat_i)
                pipe_wo.fit(xt, yt)
                auc_wo = roc_auc_score(yte, pipe_wo.predict_proba(xte)[:, 1])

                x_with = df_i[cat_with + num_with]
                xt2, xte2, yt2, yte2 = train_test_split(
                    x_with, y_i, test_size=0.30, random_state=seed, stratify=y_i
                )
                pipe_w = _build_pipeline(num_with, cat_with)
                pipe_w.fit(xt2, yt2)
                auc_w = roc_auc_score(yte2, pipe_w.predict_proba(xte2)[:, 1])

                d = auc_w - auc_wo
                deltas.append(d)
                print(f"  Seed {seed}: without={auc_wo:.4f}  with={auc_w:.4f}  delta={d:+.4f}")

            print(f"\n  Mean delta: {np.mean(deltas):+.4f}")
            print(f"  Min delta:  {np.min(deltas):+.4f}")
            if np.mean(deltas) > 0.02:
                print("  -> Detectable leakage signal")


if __name__ == "__main__":
    main()
