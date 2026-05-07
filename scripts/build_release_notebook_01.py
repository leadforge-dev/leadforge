"""One-shot builder for ``release/notebooks/01_baseline_lead_scoring.ipynb``.

Run from the repository root::

    python scripts/build_release_notebook_01.py

Cells are assigned deterministic IDs by ``_release_notebook_common`` so
re-running yields a byte-identical file — same audit-artifact-sync
pattern PR 4.1 / 5.1 / 5.2 use for ``release/`` artifacts.  The byte
equality is enforced in CI by ``tests/scripts/test_release_notebook_builders.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``scripts/`` importable regardless of how this file is loaded
# (mirrors ``scripts/package_hf_release.py``'s sys.path dance).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nbformat as nbf  # noqa: E402 — must follow sys.path insert
from _release_notebook_common import (  # noqa: E402 — must follow sys.path insert
    assemble_notebook,
    code,
    md,
    write_notebook,
)

OUT = (
    Path(__file__).resolve().parents[1] / "release" / "notebooks" / "01_baseline_lead_scoring.ipynb"
)


def cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Notebook 01 — Baseline Lead Scoring

            **Dataset:** `leadforge-lead-scoring-v1`, *intermediate* tier (the
            release default).

            **Goal:** train Logistic Regression and Histogram Gradient Boosting
            baselines on the snapshot-safe public bundle, and verify they
            reproduce the cross-seed-median metrics in
            [`release/validation/validation_report.md`](../validation/validation_report.md)
            within the per-metric tolerances fixed by acceptance gate **G13.2**.

            **Public path discipline (G13.3).** This notebook reads only from
            the public bundle at `release/intermediate/`. The instructor
            companion (`release/intermediate_instructor/`, with full-horizon
            event tables, the latent registry, the hidden DAG, and the
            mechanism summary) is **not** loaded — public modelling work must
            never depend on instructor-only artefacts.
            """
        ),
        md("## 1. Setup"),
        code(
            """
            from __future__ import annotations

            import json
            import sys
            from pathlib import Path

            import numpy as np
            import pandas as pd
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                average_precision_score,
                brier_score_loss,
                roc_auc_score,
            )
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            sys.path.insert(0, str(Path.cwd()))
            from _notebook_utils import (
                assert_within_tolerance,
                precision_at_k,
                top_decile_rate,
            )

            SEED = 42
            BUNDLE = Path("../intermediate")          # public student bundle
            TASK = "converted_within_90_days"
            """
        ),
        md(
            """
            ## 2. Reproduction targets

            We pin the cross-seed-median metrics for the *intermediate* tier
            (seeds 42–46) from `release/validation/validation_report.json`.
            The targets live in a sibling file
            (`release/notebooks/_release_targets.json`) so they can't drift
            from the validation report without an audit-sync test failure
            in CI.

            **Per-metric tolerances** are tighter than a flat 5 % band: the
            cross-seed standard deviation in the report is well under 0.02
            on AUC and Brier, and a flat ±0.05 would let a regression slip
            through. Average-precision and the small-`k` `top_decile_rate`
            stay at ±0.05 because their seed-to-seed variance is larger.
            """
        ),
        code(
            """
            with (Path.cwd() / "_release_targets.json").open() as fh:
                targets = json.load(fh)["intermediate"]

            # Re-key the validation report's metric names into the metric
            # names this notebook prints below, so the gate compares apples
            # to apples.
            VALIDATION_REPORT_TARGETS = {
                "lr_auc": targets["lr_auc"],
                "gbm_auc": targets["gbm_auc"],
                "lr_average_precision": targets["lr_average_precision"],
                "lr_brier": targets["brier_score"],
                "lr_top_decile_rate": targets["top_decile_rate"],
            }
            TOLERANCES = {
                "lr_auc": 0.02,                  # G13.2 — tighter than a flat 5%
                "gbm_auc": 0.02,
                "lr_average_precision": 0.05,    # higher seed variance
                "lr_brier": 0.02,
                "lr_top_decile_rate": 0.05,      # small-k variance
            }
            for k, v in VALIDATION_REPORT_TARGETS.items():
                print(f"  target  {k:<24s} {v:.4f}  (tol ±{TOLERANCES[k]:.2f})")
            """
        ),
        md(
            """
            ## 3. Load the bundle

            We load the parquet task splits — the canonical format the
            release ships in. The accompanying `lead_scoring.csv` is a
            convenience export with the same rows but coerced dtypes;
            sticking with parquet preserves nullable `Int64` / `Float64` /
            `boolean` columns the way the validator sees them.
            """
        ),
        code(
            """
            train = pd.read_parquet(BUNDLE / "tasks" / TASK / "train.parquet")
            valid = pd.read_parquet(BUNDLE / "tasks" / TASK / "valid.parquet")
            test = pd.read_parquet(BUNDLE / "tasks" / TASK / "test.parquet")

            with (BUNDLE / "manifest.json").open() as fh:
                manifest = json.load(fh)

            assert manifest["exposure_mode"] == "student_public", (
                "this notebook expects the public bundle"
            )
            assert manifest["relational_snapshot_safe"] is True

            print(f"Train: {len(train):,} rows")
            print(f"Valid: {len(valid):,} rows  (held out — not used here)")
            print(f"Test:  {len(test):,} rows")
            print()
            print(f"Bundle exposure_mode: {manifest['exposure_mode']}")
            print(f"Bundle snapshot_day:  {manifest['snapshot_day']}")
            print(f"Bundle horizon_days:  {manifest['horizon_days']}")
            print()
            print("Conversion rates:")
            for name, df in [("train", train), ("valid", valid), ("test", test)]:
                print(f"  {name}: {df[TASK].mean():.1%}")
            """
        ),
        md(
            """
            ## 4. Feature selection

            We use the **same feature set as `release/validation/validation_report.json`**
            so the gate in section 7 is a real reproduction check rather
            than a related-but-different number. That means we drop only
            the IDs and the label — every other column in `train` (including
            `total_touches_all`, the documented leakage trap) goes into the
            pipeline.

            **About `total_touches_all`.** The feature dictionary flags it
            with `leakage_risk = True`: it counts touches over the full
            90-day horizon, which is post-snapshot data. The validation
            report keeps it in the panel anyway because (a) its standalone
            AUC is barely above 0.55 (see the *post_snapshot_aggregates*
            baseline column in the report) and (b) the report exists to
            measure the v1 dataset's *as-shipped* difficulty, leakage trap
            included. **Notebook 03** *(coming in PR 6.2)* walks through
            what dropping the trap does to performance and how to detect
            similar traps from feature audits alone.
            """
        ),
        code(
            """
            feat_dict = pd.read_csv(BUNDLE / "feature_dictionary.csv")
            trap_cols = feat_dict.loc[
                feat_dict["leakage_risk"].astype(bool), "name"
            ].tolist()
            ID_COLS = ["account_id", "contact_id", "lead_id", "lead_created_at"]
            # Mirrors ``release_quality._partition_columns`` — IDs + label only.
            EXCLUDE = set(ID_COLS + [TASK])

            feature_cols = [c for c in train.columns if c not in EXCLUDE]
            cat_cols = [
                c
                for c in feature_cols
                if not (
                    pd.api.types.is_bool_dtype(train[c])
                    or pd.api.types.is_numeric_dtype(train[c])
                )
            ]
            num_cols = [c for c in feature_cols if c not in cat_cols]

            print(f"Leakage-trap columns kept (see narrative above): {trap_cols}")
            print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
            print(f"Numeric features    ({len(num_cols)}): {num_cols}")
            """
        ),
        md(
            """
            ## 5. Preprocessing pipeline

            Mirrors `leadforge.validation.release_quality._build_pipeline`
            so the notebook's metric panel and the validation report's
            metric panel agree by construction:

            - numeric: median-impute, then `StandardScaler`
            - categorical: most-frequent-impute, then dense `OneHotEncoder`
              with `handle_unknown="ignore"`
            """
        ),
        code(
            """
            def _sanitize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                for c in cat_cols:
                    out[c] = out[c].astype(object).where(out[c].notna(), None)
                return out

            x_train = _sanitize_categoricals(train[feature_cols])
            x_test = _sanitize_categoricals(test[feature_cols])
            y_train = train[TASK].astype("boolean").fillna(False).astype(int).to_numpy()
            y_test = test[TASK].astype("boolean").fillna(False).astype(int).to_numpy()

            numeric_t = Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )
            categorical_t = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_t, num_cols),
                    ("cat", categorical_t, cat_cols),
                ],
                remainder="drop",
            )
            """
        ),
        md("## 6. Train baselines and score the test split"),
        code(
            """
            lr_pipe = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000, solver="lbfgs", random_state=SEED
                        ),
                    ),
                ]
            )
            gbm_pipe = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("classifier", HistGradientBoostingClassifier(random_state=SEED)),
                ]
            )

            lr_pipe.fit(x_train, y_train)
            gbm_pipe.fit(x_train, y_train)

            lr_probs = lr_pipe.predict_proba(x_test)[:, 1]
            gbm_probs = gbm_pipe.predict_proba(x_test)[:, 1]

            metrics = {
                "lr_auc": float(roc_auc_score(y_test, lr_probs)),
                "gbm_auc": float(roc_auc_score(y_test, gbm_probs)),
                "lr_average_precision": float(average_precision_score(y_test, lr_probs)),
                "lr_brier": float(brier_score_loss(y_test, lr_probs)),
                "lr_top_decile_rate": top_decile_rate(lr_probs, y_test),
                # Print-only; not pinned (the validation report tracks
                # ``top_decile_rate`` instead, which we gate above).
                "lr_precision_at_50": precision_at_k(lr_probs, y_test, 50),
                "lr_precision_at_100": precision_at_k(lr_probs, y_test, 100),
                "lr_precision_at_200": precision_at_k(lr_probs, y_test, 200),
            }
            for k, v in metrics.items():
                print(f"  {k:<24s} {v:.4f}")
            """
        ),
        md(
            """
            ## 7. Tolerance check (G13.2)

            The notebook's printed metrics must match the cross-seed medians
            in `validation_report.json` to within the per-metric tolerances
            declared in section 2. If a future change breaks this, the
            assertion below fails — and CI catches it, because the same
            cell runs under `nbclient` in the `notebooks` job.
            """
        ),
        code(
            """
            assert_within_tolerance(
                observed=metrics,
                target=VALIDATION_REPORT_TARGETS,
                tolerances=TOLERANCES,
                label="notebook 01 vs validation_report.json (intermediate tier)",
            )
            print("OK — all gated metrics are within their per-metric tolerance.")
            """
        ),
        md(
            """
            ## 8. Decile lift chart

            Standard sanity-check for ranking quality: sort the test set by
            score, bucket into deciles, plot the per-decile conversion rate
            vs the base rate.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt

            order = np.argsort(-lr_probs, kind="stable")
            y_sorted = y_test[order]
            n = len(y_test)
            edges = np.linspace(0, n, 11, dtype=int)
            decile_rate = np.array(
                [y_sorted[edges[i] : edges[i + 1]].mean() for i in range(10)]
            )
            base_rate = y_test.mean()

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(range(1, 11), decile_rate, color="#3b82f6")
            ax.axhline(
                base_rate,
                color="#ef4444",
                linestyle="--",
                label=f"base rate ({base_rate:.1%})",
            )
            ax.set_xticks(range(1, 11))
            ax.set_xlabel("Score decile (1 = highest)")
            ax.set_ylabel("Conversion rate")
            ax.set_title("LR decile lift — intermediate tier (seed 42)")
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 9. Calibration plot

            Reliability diagram: bin predicted probabilities into 10 equal-
            width buckets, plot mean predicted vs mean observed. The
            validation report's reference reliability plot for the
            intermediate tier lives at
            `release/validation/figures/calibration_intermediate.png`.
            """
        ),
        code(
            """
            edges = np.linspace(0.0, 1.0, 11)
            mean_pred = []
            mean_actual = []
            for i in range(10):
                lo, hi = edges[i], edges[i + 1]
                mask = (lr_probs >= lo) & ((lr_probs <= hi) if i == 9 else (lr_probs < hi))
                if mask.sum() == 0:
                    continue
                mean_pred.append(lr_probs[mask].mean())
                mean_actual.append(y_test[mask].mean())

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", label="perfect calibration")
            ax.plot(mean_pred, mean_actual, marker="o", color="#3b82f6", label="LR")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed conversion rate")
            ax.set_title("Calibration — LR, intermediate tier (seed 42)")
            ax.legend(loc="upper left")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 10. Next

            - **Notebook 02** — engineer features by joining the snapshot-
              safe relational tables under `release/intermediate/tables/`,
              then measure the lift over the flat-CSV LR baseline above.
            - **Notebook 03** *(coming in PR 6.2)* — leakage and time-window
              walkthrough; works through what `total_touches_all` does to
              your AUC if you forget to drop it.
            - **Notebook 04** *(coming in PR 6.2)* — value-aware ranking
              (`expected_acv` × P(convert)), threshold selection, and the
              cohort-shift stress test.
            """
        ),
    ]


def main() -> None:
    write_notebook(OUT, assemble_notebook(cells()))


if __name__ == "__main__":
    main()
