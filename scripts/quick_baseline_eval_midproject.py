#!/usr/bin/env python3
"""Quick baseline evaluation for the mid-project lead scoring dataset.

Usage:
    python scripts/quick_baseline_eval_midproject.py CSV_PATH

Runs LR + RF + GBM baselines, value-aware scoring, and feature importance.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from leadforge.pipelines.common import BINARY_FEATURES, CAT_FEATURES, NUM_FEATURES, TARGET
from leadforge.pipelines.ml import LEAKAGE_PREFIX, build_preprocessor, sanitize_categoricals

_EVAL_NUM_FEATURES = NUM_FEATURES + BINARY_FEATURES


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} CSV_PATH", file=sys.stderr)
        sys.exit(1)

    df = sanitize_categoricals(pd.read_csv(sys.argv[1]), CAT_FEATURES)
    leakage = {c for c in df.columns if c.startswith(LEAKAGE_PREFIX)}
    cat_cols = [c for c in CAT_FEATURES if c in df.columns and c not in leakage]
    num_cols = [c for c in _EVAL_NUM_FEATURES if c in df.columns and c not in leakage]

    y = df[TARGET].astype(int)
    x = df[cat_cols + num_cols]

    print(f"Dataset: {len(df)} rows, {len(df.columns)} cols")
    print(f"Conversion rate: {y.mean():.1%}")
    print(f"Features: {len(cat_cols)} cat + {len(num_cols)} num = {len(cat_cols) + len(num_cols)}")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (5-seed average, 70/30 stratified)")
    print("=" * 60)

    models = {
        "LR": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "GBM": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    for name, clf in models.items():
        aucs = []
        for seed in range(42, 47):
            x_tr, x_te, y_tr, y_te = train_test_split(
                x, y, test_size=0.30, random_state=seed, stratify=y
            )
            pipe = Pipeline([("pre", build_preprocessor(num_cols, cat_cols)), ("clf", clone(clf))])
            pipe.fit(x_tr, y_tr)
            aucs.append(roc_auc_score(y_te, pipe.predict_proba(x_te)[:, 1]))
        print(f"  {name:4s}: AUC = {np.mean(aucs):.4f} (std={np.std(aucs):.4f})")

    print("\n" + "=" * 60)
    print("DETAILED METRICS (seed 42)")
    print("=" * 60)

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)
    pipe = Pipeline(
        [
            ("pre", build_preprocessor(num_cols, cat_cols)),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ]
    )
    pipe.fit(x_tr, y_tr)
    probs = pipe.predict_proba(x_te)[:, 1]
    auc = roc_auc_score(y_te, probs)
    pr_auc = average_precision_score(y_te, probs)
    base = y_te.mean()
    print(f"  AUC:    {auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Base rate: {base:.1%}")

    order = np.argsort(-probs)
    y_sorted = y_te.values[order]
    for k in [25, 50, 100]:
        if k <= len(y_te):
            prec = y_sorted[:k].mean()
            rec = y_sorted[:k].sum() / y_te.sum()
            lift = prec / base
            print(f"  P@{k}={prec:.3f}  R@{k}={rec:.3f}  Lift@{k}={lift:.2f}x")

    print("\nValue-aware ranking:")
    test_acv = pd.to_numeric(df.loc[x_te.index, "expected_acv"], errors="coerce").fillna(0).values
    test_conv = y_te.values
    ev = probs * test_acv
    for k in [25, 50]:
        top_prob = np.argsort(-probs)[:k]
        cap_prob = np.sum(test_acv[top_prob] * test_conv[top_prob])
        conv_prob = int(test_conv[top_prob].sum())
        top_ev = np.argsort(-ev)[:k]
        cap_ev = np.sum(test_acv[top_ev] * test_conv[top_ev])
        conv_ev = int(test_conv[top_ev].sum())
        uplift = (cap_ev - cap_prob) / cap_prob * 100 if cap_prob > 0 else 0.0
        print(
            f"  K={k}: prob=${cap_prob:,.0f} (conv={conv_prob})  "
            f"ev=${cap_ev:,.0f} (conv={conv_ev})  uplift={uplift:+.1f}%"
        )

    print("\nFeature importance (GBM):")
    gbm_pipe = Pipeline(
        [
            ("pre", build_preprocessor(num_cols, cat_cols)),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]
    )
    gbm_pipe.fit(x_tr, y_tr)
    importances = gbm_pipe.named_steps["clf"].feature_importances_
    ohe = gbm_pipe.named_steps["pre"].named_transformers_["cat"].named_steps["encoder"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_names
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    for _, row in imp_df.head(15).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")

    print("\nMissingness summary:")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            print(f"  {col}: {n_miss} ({n_miss / len(df):.1%})")


if __name__ == "__main__":
    main()
