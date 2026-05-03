"""Shared ML pipeline utilities for validation and evaluation scripts.

Provides the canonical sklearn pipeline (ColumnTransformer + imputation +
encoding + LogisticRegression) used across dataset validators and baseline
evaluation scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from leadforge.pipelines.common import BINARY_FEATURES, CAT_FEATURES, NUM_FEATURES, TARGET

__all__ = [
    "build_baseline_pipeline",
    "build_preprocessor",
    "fit_evaluate",
    "get_feature_cols",
    "sanitize_categoricals",
]

LEAKAGE_PREFIX = "__leakage__"


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Build the canonical preprocessing ColumnTransformer."""
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
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


def build_baseline_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    seed: int = 42,
) -> Pipeline:
    """Build the canonical sklearn baseline pipeline (preprocessor + LR)."""
    preprocessor = build_preprocessor(num_cols, cat_cols)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)),
        ]
    )


def get_feature_cols(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Partition feature columns into (cat_cols, num_cols).

    Uses the canonical feature lists, falling back to dtype-based detection
    for columns not in the canonical lists (e.g. leakage trap columns).
    """
    exclude = (exclude or set()) | {TARGET}
    cat_cols = [c for c in CAT_FEATURES if c in df.columns and c not in exclude]
    num_cols = [c for c in NUM_FEATURES + BINARY_FEATURES if c in df.columns and c not in exclude]
    # Add any trap columns to numeric if not excluded
    for c in df.columns:
        if c.startswith(LEAKAGE_PREFIX) and c not in exclude:
            num_cols.append(c)
    return cat_cols, num_cols


def sanitize_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Convert pd.NA in categorical columns to None for sklearn compatibility."""
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def fit_evaluate(
    df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
    seed: int = 42,
    test_size: float = 0.30,
) -> tuple[float, float, np.ndarray, pd.Series]:
    """Fit LR on hold-out split, return (AUC, PR-AUC, probs, y_test)."""
    y = df[TARGET].astype(int)
    cat_cols, num_cols = get_feature_cols(df, exclude=exclude_cols)
    df_clean = sanitize_categoricals(df, cat_cols)
    x = df_clean[cat_cols + num_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipe = build_baseline_pipeline(num_cols, cat_cols, seed=42)
    pipe.fit(x_train, y_train)
    probs = pipe.predict_proba(x_test)[:, 1]

    auc = float(roc_auc_score(y_test, probs))
    pr_auc = float(average_precision_score(y_test, probs))
    return auc, pr_auc, probs, y_test
