"""One-shot builder for ``release/notebooks/01_baseline_lead_scoring.ipynb``.

Run from the repository root::

    python scripts/_build_release_notebook_01.py

Produces a cleared notebook (no execution_count, no outputs) with stable
metadata.  Re-running yields a byte-identical file — same audit-artifact-
sync pattern PR 4.1 / 5.1 / 5.2 use for ``release/`` artifacts.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

OUT = (
    Path(__file__).resolve().parents[1] / "release" / "notebooks" / "01_baseline_lead_scoring.ipynb"
)


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(dedent(source).strip("\n"))
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """
            # Notebook 01 — Baseline Lead Scoring

            **Dataset:** `leadforge-lead-scoring-v1`, *intermediate* tier (the
            release default).

            **Goal:** train Logistic Regression and Histogram Gradient Boosting
            baselines on the snapshot-safe public bundle, and verify they
            reproduce the cross-seed-median metrics in
            [`release/validation/validation_report.md`](../validation/validation_report.md)
            within the **±0.05** tolerance fixed by acceptance gate **G13.2**.

            **Public path discipline (G13.3).** This notebook reads only from
            the public bundle at `release/intermediate/`. The instructor
            companion (`release/intermediate_instructor/`, with full-horizon
            event tables, the latent registry, the hidden DAG, and the
            mechanism summary) is **not** loaded — public modelling work must
            never depend on instructor-only artefacts.
            """
        ),
        md(
            """
            ## 1. Setup
            """
        ),
        code(
            """
            from __future__ import annotations

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
            from _notebook_utils import assert_within_tolerance, precision_at_k

            SEED = 42
            BUNDLE = Path("../intermediate")          # public student bundle
            TASK = "converted_within_90_days"
            """
        ),
        md(
            """
            ## 2. Reproduction targets

            We pin the cross-seed median metrics from
            `release/validation/validation_report.md` (intermediate tier).
            Each is a Logistic Regression score on the test split unless
            otherwise noted; ranking-based metrics use stable argsort on the
            LR probability so ties resolve identically to the validator.
            """
        ),
        code(
            """
            # Cross-seed medians from release/validation/validation_report.md
            # (intermediate tier; seeds 42-46).  See ``$.tiers.intermediate.medians``
            # in the JSON sibling for the source of truth.
            VALIDATION_REPORT_TARGETS = {
                "lr_auc": 0.8859,
                "gbm_auc": 0.8755,
                "lr_average_precision": 0.5752,
                "lr_brier": 0.1096,
                "lr_precision_at_100": 0.59,
            }
            TOLERANCE = 0.05  # G13.2
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
            import json

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

            The feature dictionary flags one column as a deliberate **leakage
            trap**: `total_touches_all` counts touches over the full 90-day
            horizon (post-snapshot data). We drop it from the baseline so
            the comparison against the validation report is honest.

            Notebook 03 (the leakage walkthrough, shipping in PR 6.2) exists
            specifically to show what happens if you keep it.
            """
        ),
        code(
            """
            feat_dict = pd.read_csv(BUNDLE / "feature_dictionary.csv")
            trap_cols = feat_dict.loc[
                feat_dict["leakage_risk"].astype(bool), "name"
            ].tolist()
            ID_COLS = ["account_id", "contact_id", "lead_id", "lead_created_at"]
            EXCLUDE = set(ID_COLS + trap_cols + [TASK])

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

            print(f"Leakage-trap columns dropped: {trap_cols}")
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
            y_train = train[TASK].astype("boolean").fillna(False).astype(int).values
            y_test = test[TASK].astype("boolean").fillna(False).astype(int).values

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
        md(
            """
            ## 6. Train baselines and score the test split
            """
        ),
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
            in `validation_report.md` to within ±0.05. If a future change
            breaks this, the assertion below fails — and CI catches it,
            because the same cell runs under `nbclient` in the
            `notebooks` job.
            """
        ),
        code(
            """
            assert_within_tolerance(
                observed=metrics,
                target=VALIDATION_REPORT_TARGETS,
                tolerances=TOLERANCE,
                label="notebook 01 vs validation_report.md (intermediate tier)",
            )
            print("OK — all reported metrics are within ±0.05 of the validation report medians.")
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
            - **Notebook 03** *(PR 6.2)* — leakage and time-window
              walkthrough; works through what `total_touches_all` does to
              your AUC if you forget to drop it.
            - **Notebook 04** *(PR 6.2)* — value-aware ranking
              (`expected_acv` × P(convert)), threshold selection, and the
              cohort-shift stress test.
            """
        ),
    ]
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }
    return nb


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False)
    OUT.write_text(text + "\n", encoding="utf-8")
    # Run ruff format on the emitted notebook so the builder's output
    # matches the project's pre-commit hook byte-for-byte.  Without this
    # step a contributor running the builder would see pre-commit
    # reformat their notebook on commit, defeating the audit-artifact-
    # sync invariant.
    subprocess.run(["ruff", "format", str(OUT)], check=True)  # noqa: S603, S607
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
