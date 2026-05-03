#!/usr/bin/env python3
"""Quick baseline evaluation for v7 lead scoring intro dataset.

Usage:
    python scripts/quick_baseline_eval_v7.py STUDENT_CSV [INSTRUCTOR_CSV]

Runs LR + RF + GBM baselines, value-aware scoring, feature importance,
and optional trap detection on the instructor dataset.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "converted"
LEAKAGE_PREFIX = "__leakage__"

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

NUM_FEATURES = [
    "expected_acv",
    "inbound_touches",
    "outbound_touches",
    "touches_week_1",
    "touches_last_7_days",
    "days_since_first_touch",
    "web_sessions",
    "sales_activities",
    "days_since_last_touch",
]


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def _build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} STUDENT_CSV [INSTRUCTOR_CSV]", file=sys.stderr)
        sys.exit(1)

    student_path = sys.argv[1]
    instructor_path = sys.argv[2] if len(sys.argv) > 2 else None

    df = _sanitize(pd.read_csv(student_path))
    leakage = {c for c in df.columns if c.startswith(LEAKAGE_PREFIX)}
    cat_cols = [c for c in CAT_FEATURES if c in df.columns and c not in leakage]
    num_cols = [c for c in NUM_FEATURES if c in df.columns and c not in leakage]

    y = df[TARGET].astype(int)
    x = df[cat_cols + num_cols]

    print(f"Dataset: {len(df)} rows, {len(df.columns)} cols")
    print(f"Conversion rate: {y.mean():.1%}")
    print(f"Features: {len(cat_cols)} cat + {len(num_cols)} num = {len(cat_cols) + len(num_cols)}")

    # Multi-model comparison
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
            pipe = Pipeline([("pre", _build_preprocessor(num_cols, cat_cols)), ("clf", clf)])
            pipe.fit(x_tr, y_tr)
            aucs.append(roc_auc_score(y_te, pipe.predict_proba(x_te)[:, 1]))
        print(f"  {name:4s}: AUC = {np.mean(aucs):.4f} (std={np.std(aucs):.4f})")

    # Single-seed detailed metrics
    print("\n" + "=" * 60)
    print("DETAILED METRICS (seed 42)")
    print("=" * 60)

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)
    pipe = Pipeline(
        [
            ("pre", _build_preprocessor(num_cols, cat_cols)),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ]
    )
    pipe.fit(x_tr, y_tr)
    probs = pipe.predict_proba(x_te)[:, 1]
    auc = roc_auc_score(y_te, probs)
    pr_auc = average_precision_score(y_te, probs)
    print(f"  AUC:    {auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")

    order = np.argsort(-probs)
    y_sorted = y_te.values[order]
    for k in [25, 50]:
        prec = y_sorted[:k].mean()
        lift = prec / y_te.mean()
        print(f"  P@{k}:  {prec:.3f}  Lift@{k}: {lift:.2f}x")

    # Value-aware
    print("\nValue-aware ranking:")
    test_acv = pd.to_numeric(df.loc[x_te.index, "expected_acv"], errors="coerce").fillna(0).values
    test_conv = y_te.values
    ev = probs * test_acv
    for k in [25, 50]:
        top_prob = np.argsort(-probs)[:k]
        cap_prob = np.sum(test_acv[top_prob] * test_conv[top_prob])
        top_ev = np.argsort(-ev)[:k]
        cap_ev = np.sum(test_acv[top_ev] * test_conv[top_ev])
        uplift = (cap_ev - cap_prob) / cap_prob * 100 if cap_prob > 0 else 0
        print(f"  K={k}: prob=${cap_prob:,.0f}  ev=${cap_ev:,.0f}  uplift={uplift:+.1f}%")

    # Feature importance (GBM)
    print("\nFeature importance (GBM):")
    gbm_pipe = Pipeline(
        [
            ("pre", _build_preprocessor(num_cols, cat_cols)),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]
    )
    gbm_pipe.fit(x_tr, y_tr)
    importances = gbm_pipe.named_steps["clf"].feature_importances_

    # Get feature names from the preprocessor
    pre = gbm_pipe.named_steps["pre"]
    ohe = pre.named_transformers_["cat"].named_steps["encoder"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_names
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    for _, row in imp_df.head(15).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")

    # Trap detection (instructor)
    if instructor_path:
        print("\n" + "=" * 60)
        print("TRAP DETECTION (instructor)")
        print("=" * 60)
        inst = _sanitize(pd.read_csv(instructor_path))
        trap_cols = [c for c in inst.columns if c.startswith(LEAKAGE_PREFIX)]
        if trap_cols:
            trap_col = trap_cols[0]
            trap_conv = inst.loc[inst[TARGET] == 1, trap_col].mean()
            trap_not = inst.loc[inst[TARGET] == 0, trap_col].mean()
            print(f"  Trap column: {trap_col}")
            print(f"  Mean (converted):     {trap_conv:.1f}")
            print(f"  Mean (not converted): {trap_not:.1f}")
            print(f"  Ratio:                {trap_conv / trap_not:.2f}x")


if __name__ == "__main__":
    main()
