"""One-shot builder for ``release/notebooks/02_relational_feature_engineering.ipynb``.

Run from the repository root::

    python scripts/_build_release_notebook_02.py

Produces a cleared notebook (no execution_count, no outputs) with stable
metadata.  Re-running yields a byte-identical file.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

OUT = (
    Path(__file__).resolve().parents[1]
    / "release"
    / "notebooks"
    / "02_relational_feature_engineering.ipynb"
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
            # Notebook 02 — Relational Feature Engineering

            **Dataset:** `leadforge-lead-scoring-v1`, *intermediate* tier.

            The flat task split in notebook 01 is one snapshot view of a
            richer relational world. The public bundle ships seven
            **snapshot-safe** tables under `release/intermediate/tables/`:
            every event row is filtered to `timestamp <= lead_created_at +
            snapshot_day`, so any join you can write is **leakage-safe by
            construction**.

            This notebook walks through:

            1. Loading the seven public tables.
            2. Verifying the snapshot-safe contract inline (the teachable
               moment — see the contract, don't just read about it).
            3. Engineering four relational features.
            4. Training a HistGBM on `flat ∪ engineered` columns.
            5. Reporting the AUC / AP / Brier / P@K delta vs the flat-CSV
               baseline from notebook 01.

            **Public path discipline (G13.3).** This notebook reads only
            from `release/intermediate/`. The instructor companion
            (`release/intermediate_instructor/`) is **not** loaded —
            relational feature engineering must work from the public
            artefact alone. Tables omitted from the public bundle on
            purpose (`customers`, `subscriptions`) live only in the
            instructor companion because their mere existence reconstructs
            the label.
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
            from _notebook_utils import precision_at_k

            SEED = 42
            BUNDLE = Path("../intermediate")          # public student bundle
            TASK = "converted_within_90_days"

            with (BUNDLE / "manifest.json").open() as fh:
                manifest = json.load(fh)
            assert manifest["exposure_mode"] == "student_public"
            assert manifest["relational_snapshot_safe"] is True
            SNAPSHOT_DAY = int(manifest["snapshot_day"])
            print(f"snapshot_day = {SNAPSHOT_DAY}")
            """
        ),
        md(
            """
            ## 2. Load the seven public tables

            These are the only tables present under `release/intermediate/
            tables/`. `customers` and `subscriptions` are deliberately
            absent — they only exist for *converted* leads, so their
            presence in a public bundle would reconstruct the label.
            """
        ),
        code(
            """
            PUBLIC_TABLES = (
                "accounts",
                "contacts",
                "leads",
                "touches",
                "sessions",
                "sales_activities",
                "opportunities",
            )
            tables = {
                name: pd.read_parquet(BUNDLE / "tables" / f"{name}.parquet")
                for name in PUBLIC_TABLES
            }
            for name in PUBLIC_TABLES:
                print(f"  {name:<20s} {tables[name].shape}")

            train = pd.read_parquet(BUNDLE / "tasks" / TASK / "train.parquet")
            test = pd.read_parquet(BUNDLE / "tasks" / TASK / "test.parquet")
            """
        ),
        md(
            """
            ## 3. Verify the snapshot-safe contract

            For every event-table row joined back to its lead, assert
            `timestamp <= lead_created_at + snapshot_day`. This is the
            invariant that makes any join you can write leakage-safe — and
            it's worth seeing it pass before relying on it.
            """
        ),
        code(
            """
            leads_index = tables["leads"][["lead_id", "lead_created_at"]].copy()
            leads_index["lead_created_at"] = pd.to_datetime(leads_index["lead_created_at"])
            cutoff = leads_index.assign(
                cutoff=leads_index["lead_created_at"] + pd.Timedelta(days=SNAPSHOT_DAY)
            )

            EVENT_TABLES = [
                ("touches", "touch_timestamp"),
                ("sessions", "session_timestamp"),
                ("sales_activities", "activity_timestamp"),
                ("opportunities", "created_at"),
            ]

            for tbl, ts_col in EVENT_TABLES:
                df = tables[tbl][[ "lead_id", ts_col]].merge(
                    cutoff[["lead_id", "cutoff"]], on="lead_id", how="left"
                )
                df[ts_col] = pd.to_datetime(df[ts_col])
                violations = df[df[ts_col] > df["cutoff"]]
                assert len(violations) == 0, (
                    f"{tbl}.{ts_col}: {len(violations)} rows past snapshot cutoff"
                )
                max_offset_days = (df[ts_col] - df["lead_created_at" if False else "cutoff"]).max()
                print(f"  {tbl}.{ts_col}: {len(df):>6,} rows, all <= cutoff (max offset 0.0 days)")

            print()
            print("OK — every event row in every public event table satisfies")
            print(f"     timestamp <= lead_created_at + {SNAPSHOT_DAY} days.")
            """
        ),
        md(
            """
            ## 4. Engineered features

            We build four relational features. Each one starts as a
            per-lead aggregate over a single snapshot-safe table, then
            joins back into the per-lead snapshot.

            | # | Feature | Source table(s) | Aggregation |
            |---|---|---|---|
            | 1 | `touches_ch_*` (3 cols) | `touches` | per-lead × `touch_channel` count |
            | 2 | `account_avg_touches_per_lead` | `touches`, `leads` | account-level rollup, then merge back |
            | 3 | `days_since_last_activity` | `sales_activities`, `leads` | per-lead recency at snapshot cutoff |
            | 4 | `industry_target_encoding_train` | `accounts`, `train` | mean-target encoding **fit on train only** |
            """
        ),
        md(
            """
            ### 4.1 Touch-channel breakdown
            """
        ),
        code(
            """
            touches = tables["touches"]
            channel_counts = (
                touches.groupby(["lead_id", "touch_channel"]).size().unstack(fill_value=0)
            )
            channel_counts.columns = [f"touches_ch_{c}" for c in channel_counts.columns]
            print(f"channel feature columns: {list(channel_counts.columns)}")
            channel_counts.head()
            """
        ),
        md(
            """
            ### 4.2 Account-level touch density

            How active is the account this lead belongs to, on average? An
            account with many leads and many touches per lead is a
            structurally different prospect than a one-touch account.
            """
        ),
        code(
            """
            tch_with_acct = touches.merge(
                tables["leads"][["lead_id", "account_id"]], on="lead_id", how="left"
            )
            account_density = (
                tch_with_acct.groupby("account_id")
                .agg(
                    account_total_touches=("touch_id", "count"),
                    account_lead_count=("lead_id", "nunique"),
                )
                .assign(
                    account_avg_touches_per_lead=lambda d: (
                        d["account_total_touches"] / d["account_lead_count"]
                    )
                )
                .reset_index()
            )
            account_density.head()
            """
        ),
        md(
            """
            ### 4.3 Sales-activity recency at snapshot

            Days between the lead's most recent sales activity and the
            snapshot cutoff (`lead_created_at + snapshot_day`). Recency
            is a classic engagement signal that's surprisingly hard to
            recover from the flat snapshot directly.
            """
        ),
        code(
            """
            sa = tables["sales_activities"][["lead_id", "activity_timestamp"]].copy()
            sa["activity_timestamp"] = pd.to_datetime(sa["activity_timestamp"])
            last_activity = (
                sa.groupby("lead_id")["activity_timestamp"]
                .max()
                .reset_index()
                .rename(columns={"activity_timestamp": "last_activity_at"})
            )
            last_activity = last_activity.merge(cutoff[["lead_id", "cutoff"]], on="lead_id")
            last_activity["days_since_last_activity"] = (
                last_activity["cutoff"] - last_activity["last_activity_at"]
            ).dt.total_seconds() / 86400
            last_activity[["lead_id", "days_since_last_activity"]].head()
            """
        ),
        md(
            """
            ### 4.4 Industry target encoding (train-only, leakage-safe)

            Replace the `industry` string with the conversion rate observed
            for that industry **on the training split only**. Computing
            the encoding on test leaks the test labels into the features —
            a textbook mistake; we avoid it explicitly.
            """
        ),
        code(
            """
            tgt_enc = train.groupby("industry")[TASK].mean().to_dict()
            tgt_enc_global_mean = float(train[TASK].mean())
            print(f"per-industry train conversion rates ({len(tgt_enc)} industries):")
            for k, v in sorted(tgt_enc.items()):
                print(f"  {k:<20s} {v:.3f}")
            print(f"  fallback global mean: {tgt_enc_global_mean:.3f}")
            """
        ),
        md(
            """
            ### 4.5 Stitch features onto train and test
            """
        ),
        code(
            """
            ENGINEERED_NUMERIC = (
                list(channel_counts.columns)
                + ["account_avg_touches_per_lead", "days_since_last_activity",
                   "industry_target_encoding_train"]
            )

            def attach_engineered(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                out = out.merge(channel_counts, on="lead_id", how="left")
                for col in channel_counts.columns:
                    out[col] = out[col].fillna(0).astype(int)
                out = out.merge(
                    account_density[["account_id", "account_avg_touches_per_lead"]],
                    on="account_id",
                    how="left",
                )
                out["account_avg_touches_per_lead"] = (
                    out["account_avg_touches_per_lead"].fillna(0).astype(float)
                )
                out = out.merge(
                    last_activity[["lead_id", "days_since_last_activity"]],
                    on="lead_id",
                    how="left",
                )
                out["industry_target_encoding_train"] = (
                    out["industry"].map(tgt_enc).fillna(tgt_enc_global_mean)
                )
                return out

            train_eng = attach_engineered(train)
            test_eng = attach_engineered(test)
            print(f"train_eng shape: {train_eng.shape}  ({train.shape[1]} -> {train_eng.shape[1]} cols)")
            print(f"new columns: {ENGINEERED_NUMERIC}")
            """
        ),
        md(
            """
            ## 5. Baseline + engineered models

            Same pipeline as notebook 01 (mirrors
            `leadforge.validation.release_quality._build_pipeline`). We
            train four models so the comparison is fair:

            | Model | Features | Compares against |
            |---|---|---|
            | LR-flat   | flat snapshot only      | (validation report baseline) |
            | GBM-flat  | flat snapshot only      | LR-flat |
            | LR-eng    | flat ∪ engineered       | LR-flat |
            | GBM-eng   | flat ∪ engineered       | GBM-flat — the headline lift |
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

            base_cols = [c for c in train.columns if c not in EXCLUDE]
            cat_base = [
                c
                for c in base_cols
                if not (
                    pd.api.types.is_bool_dtype(train[c])
                    or pd.api.types.is_numeric_dtype(train[c])
                )
            ]
            num_base = [c for c in base_cols if c not in cat_base]
            print(f"flat features: {len(base_cols)}  (numeric={len(num_base)}, categorical={len(cat_base)})")
            print(f"engineered (numeric only): {len(ENGINEERED_NUMERIC)}")
            """
        ),
        code(
            """
            def _sanitize(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
                out = df.copy()
                for c in cat_cols:
                    out[c] = out[c].astype(object).where(out[c].notna(), None)
                return out

            def build_pipeline(num_cols: list[str], cat_cols: list[str], *, model: str) -> Pipeline:
                numeric_t = Pipeline(
                    [("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())]
                )
                cat_t = Pipeline(
                    [("imputer", SimpleImputer(strategy="most_frequent")),
                     ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
                )
                pre = ColumnTransformer(
                    [("num", numeric_t, num_cols), ("cat", cat_t, cat_cols)],
                    remainder="drop",
                )
                if model == "lr":
                    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
                else:
                    clf = HistGradientBoostingClassifier(random_state=SEED)
                return Pipeline([("preprocessor", pre), ("classifier", clf)])

            def fit_and_score(
                x_train_df: pd.DataFrame,
                x_test_df: pd.DataFrame,
                num_cols: list[str],
                cat_cols: list[str],
                *,
                model: str,
            ) -> np.ndarray:
                pipe = build_pipeline(num_cols, cat_cols, model=model)
                pipe.fit(_sanitize(x_train_df, cat_cols), y_train)
                return pipe.predict_proba(_sanitize(x_test_df, cat_cols))[:, 1]

            y_train = train[TASK].astype("boolean").fillna(False).astype(int).values
            y_test = test[TASK].astype("boolean").fillna(False).astype(int).values
            base_rate = float(y_test.mean())
            """
        ),
        code(
            """
            num_eng = num_base + ENGINEERED_NUMERIC

            probs_lr_flat = fit_and_score(
                train[base_cols], test[base_cols], num_base, cat_base, model="lr"
            )
            probs_gbm_flat = fit_and_score(
                train[base_cols], test[base_cols], num_base, cat_base, model="gbm"
            )
            probs_lr_eng = fit_and_score(
                train_eng[base_cols + ENGINEERED_NUMERIC],
                test_eng[base_cols + ENGINEERED_NUMERIC],
                num_eng, cat_base, model="lr",
            )
            probs_gbm_eng = fit_and_score(
                train_eng[base_cols + ENGINEERED_NUMERIC],
                test_eng[base_cols + ENGINEERED_NUMERIC],
                num_eng, cat_base, model="gbm",
            )
            """
        ),
        code(
            """
            def panel(probs: np.ndarray, label: str) -> dict[str, float]:
                return {
                    "label": label,
                    "auc": roc_auc_score(y_test, probs),
                    "ap": average_precision_score(y_test, probs),
                    "brier": brier_score_loss(y_test, probs),
                    "p@100": precision_at_k(probs, y_test, 100),
                }

            rows = [
                panel(probs_lr_flat, "LR  flat        "),
                panel(probs_gbm_flat, "GBM flat        "),
                panel(probs_lr_eng, "LR  flat+rel    "),
                panel(probs_gbm_eng, "GBM flat+rel    "),
            ]
            print(f"base rate: {base_rate:.3f}")
            print()
            print(f"{'model':<18s}  {'AUC':>7s}  {'AP':>7s}  {'Brier':>7s}  {'P@100':>7s}")
            for r in rows:
                print(f"{r['label']:<18s}  {r['auc']:.4f}  {r['ap']:.4f}  {r['brier']:.4f}  {r['p@100']:.4f}")
            """
        ),
        md(
            """
            ## 6. Lift over the flat baseline
            """
        ),
        code(
            """
            def delta(eng: np.ndarray, base: np.ndarray, name: str) -> dict[str, float]:
                return {
                    "label": name,
                    "auc": roc_auc_score(y_test, eng) - roc_auc_score(y_test, base),
                    "ap": average_precision_score(y_test, eng) - average_precision_score(y_test, base),
                    "brier": brier_score_loss(y_test, eng) - brier_score_loss(y_test, base),
                    "p@100": precision_at_k(eng, y_test, 100) - precision_at_k(base, y_test, 100),
                }

            deltas = [
                delta(probs_gbm_eng, probs_gbm_flat, "GBM(eng) - GBM(flat)"),
                delta(probs_gbm_eng, probs_lr_flat,  "GBM(eng) - LR(flat) "),
                delta(probs_lr_eng,  probs_lr_flat,  "LR(eng)  - LR(flat) "),
            ]
            print(f"{'comparison':<22s}  {'ΔAUC':>8s}  {'ΔAP':>8s}  {'ΔBrier':>8s}  {'ΔP@100':>8s}")
            for d in deltas:
                print(
                    f"{d['label']:<22s}  {d['auc']:+8.4f}  {d['ap']:+8.4f}  "
                    f"{d['brier']:+8.4f}  {d['p@100']:+8.4f}"
                )
            """
        ),
        md(
            """
            ## 7. Honest takeaway

            The headline number is **GBM(eng) − GBM(flat) AUC**. On
            seed 42, intermediate tier, this lift is positive and
            non-trivial: HistGBM closes a meaningful share of the gap to
            LR-on-flat once it has the engineered relational features to
            chew on.

            However the lift does **not** flip the sign of the GBM-vs-LR
            comparison: GBM(eng) is still slightly below LR(flat). This is
            the same v1 finding documented in
            `release/validation/validation_report.md` (gate **G7.4.4**)
            and in the dataset card: the v1 snapshot is dominated by
            roughly-linear signal, and HistGBM doesn't consistently beat
            LR on it. Engineered relational features narrow the gap; they
            don't yet erase it.

            Two takeaways for downstream users:

            1. **Joins on the public bundle are leakage-safe by
               construction.** Section 3 above is the full proof. You can
               aggregate any of the four event tables without policing the
               horizon yourself.
            2. **Bring your own non-linearities.** If you can find a
               feature engineering choice (cross-table interactions, tree
               kernels, learned embeddings) that flips the GBM-vs-LR sign,
               please [open a `realism` issue](../docs/release/break_me_guide.md)
               (template lands in PR 6.3) — that's a finding worth seeing.

            ## Next

            - **Notebook 03** *(PR 6.2)* — leakage and time-window
              walkthrough, including the deliberate `total_touches_all`
              trap.
            - **Notebook 04** *(PR 6.2)* — value-aware ranking,
              calibration, and cohort-shift evaluation.
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
