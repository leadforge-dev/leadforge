"""One-shot builder for ``release/notebooks/02_relational_feature_engineering.ipynb``.

Run from the repository root::

    python scripts/build_release_notebook_02.py

Cells are assigned deterministic IDs by ``_release_notebook_common`` so
re-running yields a byte-identical file — same audit-artifact-sync
pattern PR 4.1 / 5.1 / 5.2 use for ``release/`` artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nbformat as nbf  # noqa: E402 — must follow sys.path insert
from _release_notebook_common import (  # noqa: E402 — must follow sys.path insert
    assemble_notebook,
    code,
    md,
    write_notebook,
)

OUT = (
    Path(__file__).resolve().parents[1]
    / "release"
    / "notebooks"
    / "02_relational_feature_engineering.ipynb"
)


def cells() -> list[nbf.NotebookNode]:
    return [
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
               moment — see the contract pass, don't just read about it).
            3. Engineering four relational features, with train-only
               discipline for any aggregation that crosses splits.
            4. Training a HistGBM on `flat ∪ engineered` columns.
            5. Reporting the AUC / AP / Brier / P@K delta vs the flat-CSV
               baseline, with a tolerance gate that fails CI if the
               headline lift regresses.

            **Public path discipline (G13.3).** This notebook reads only
            from `release/intermediate/`. The instructor companion
            (`release/intermediate_instructor/`) is **not** loaded —
            relational feature engineering must work from the public
            artefact alone. Tables omitted from the public bundle on
            purpose (`customers`, `subscriptions`) live only in the
            instructor companion because their mere existence reconstructs
            the label.

            **Leakage-trap discipline.** Unlike notebook 01 (which
            reproduces the validation report's panel verbatim and
            therefore keeps `total_touches_all`), notebook 02 **drops**
            the trap from the flat baseline. We're teaching feature
            engineering here; mixing a known-leaky column into the
            "before" panel would muddy the relational lift attribution.
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
            from _notebook_utils import assert_within_tolerance, precision_at_k

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
            `timestamp <= lead_created_at + snapshot_day`. The reported
            **headroom** is the *minimum* gap any event row leaves between
            its timestamp and the snapshot cutoff — a non-negative number
            when the contract holds. Showing the actual minimum (rather
            than just "all <= cutoff") makes the receipt honest: if a
            future regeneration ever shaves the contract close, you'll see
            the headroom shrink before the assertion fires.
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
                df = tables[tbl][["lead_id", ts_col]].merge(
                    cutoff[["lead_id", "cutoff"]], on="lead_id", how="left"
                )
                df[ts_col] = pd.to_datetime(df[ts_col])
                violations = df[df[ts_col] > df["cutoff"]]
                assert len(violations) == 0, (
                    f"{tbl}.{ts_col}: {len(violations)} rows past snapshot cutoff"
                )
                min_headroom = (df["cutoff"] - df[ts_col]).min()
                min_headroom_days = float(min_headroom.total_seconds()) / 86400.0
                print(
                    f"  {tbl}.{ts_col}: {len(df):>6,} rows; "
                    f"min headroom under cutoff: {min_headroom_days:6.2f} days"
                )

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
            joins back into the per-lead snapshot. Aggregations that pool
            across leads (account-level density, target encoding) are
            **fit on the train split only** and applied to test via
            join — same train-only discipline that prevents target leakage
            in mean encoding.

            | # | Feature | Source table(s) | Aggregation |
            |---|---|---|---|
            | 1 | `touches_ch_*` (3 cols) | `touches` | per-lead × `touch_channel` count |
            | 2 | `account_avg_touches_per_lead` | `touches`, `leads`, train lead set | account-level rollup over **train leads only**, then merge back |
            | 3 | `days_since_last_activity` | `sales_activities`, `leads` | per-lead recency at snapshot cutoff |
            | 4 | `industry_target_encoding_train` | `accounts`, train | mean-target encoding **fit on train only** |
            """
        ),
        md("### 4.1 Touch-channel breakdown"),
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

            **Train-only.** We compute the account-level rollup using only
            train leads' touches. Test leads in train-only-empty accounts
            fall back to 0 via `fillna`. This avoids letting test rows
            influence the train feature distribution — same discipline
            applied to mean-target encoding in 4.4.
            """
        ),
        code(
            """
            train_lead_ids = set(train["lead_id"].tolist())
            tch_with_acct = touches.merge(
                tables["leads"][["lead_id", "account_id"]], on="lead_id", how="left"
            )
            tch_train = tch_with_acct[tch_with_acct["lead_id"].isin(train_lead_ids)]
            account_density = (
                tch_train.groupby("account_id")
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
            print(
                f"account_density rows: {len(account_density):,}  "
                f"(accounts represented in train touches)"
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
        md("### 4.5 Stitch features onto train and test"),
        code(
            """
            ENGINEERED_NUMERIC = (
                list(channel_counts.columns)
                + [
                    "account_avg_touches_per_lead",
                    "days_since_last_activity",
                    "industry_target_encoding_train",
                ]
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
            print(
                f"train_eng shape: {train_eng.shape}  "
                f"({train.shape[1]} -> {train_eng.shape[1]} cols)"
            )
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
            | LR-flat   | flat snapshot, no trap            | (validation report baseline) |
            | GBM-flat  | flat snapshot, no trap            | LR-flat |
            | LR-eng    | flat ∪ engineered, no trap, no raw `industry` | LR-flat |
            | GBM-eng   | flat ∪ engineered, no trap, no raw `industry` | GBM-flat — the headline lift |

            The `+rel` pipelines drop the raw `industry` categorical
            because the train-only target encoding already represents it
            as a numeric column — feeding both would feed the same column
            twice.
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
            # Drop raw `industry` from the +rel categorical list so the LR
            # pipeline doesn't see it twice (one-hot + target-encoded).
            cat_eng = [c for c in cat_base if c != "industry"]
            print(
                f"flat features: {len(base_cols)}  "
                f"(numeric={len(num_base)}, categorical={len(cat_base)})"
            )
            print(f"engineered (numeric only): {len(ENGINEERED_NUMERIC)}")
            print(f"+rel categorical list drops: {set(cat_base) - set(cat_eng)}")
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
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                )
                cat_t = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
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

            y_train = train[TASK].astype("boolean").fillna(False).astype(int).to_numpy()
            y_test = test[TASK].astype("boolean").fillna(False).astype(int).to_numpy()
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
                num_eng, cat_eng, model="lr",
            )
            probs_gbm_eng = fit_and_score(
                train_eng[base_cols + ENGINEERED_NUMERIC],
                test_eng[base_cols + ENGINEERED_NUMERIC],
                num_eng, cat_eng, model="gbm",
            )
            """
        ),
        code(
            """
            def panel(probs: np.ndarray, label: str) -> dict[str, float]:
                return {
                    "label": label,
                    "auc": float(roc_auc_score(y_test, probs)),
                    "ap": float(average_precision_score(y_test, probs)),
                    "brier": float(brier_score_loss(y_test, probs)),
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
                print(
                    f"{r['label']:<18s}  {r['auc']:.4f}  {r['ap']:.4f}  "
                    f"{r['brier']:.4f}  {r['p@100']:.4f}"
                )
            """
        ),
        md(
            """
            ## 6. Lift over the flat baseline

            *Sign convention.* `ΔAUC`, `ΔAP`, `ΔP@100`: **higher is
            better** (positive = engineered features helped). `ΔBrier`:
            **lower is better** (Brier is a loss, so negative ΔBrier =
            improved calibration).
            """
        ),
        code(
            """
            def delta(eng: np.ndarray, base: np.ndarray, name: str) -> dict[str, float]:
                return {
                    "label": name,
                    "auc": float(roc_auc_score(y_test, eng) - roc_auc_score(y_test, base)),
                    "ap": float(
                        average_precision_score(y_test, eng) - average_precision_score(y_test, base)
                    ),
                    "brier": float(
                        brier_score_loss(y_test, eng) - brier_score_loss(y_test, base)
                    ),
                    "p@100": float(
                        precision_at_k(eng, y_test, 100) - precision_at_k(base, y_test, 100)
                    ),
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
            ## 7. Tolerance gate

            Pin the four model AUCs and the headline lift to per-metric
            tolerances so a regression in any pipeline component (feature
            engineering, leakage discipline, sklearn upgrade) breaks CI.
            Targets and tolerances are seed-42-specific by design — this
            notebook is reproducible-by-seed, not a cross-seed median.
            """
        ),
        code(
            """
            # Single-seed (seed=42) AUCs observed on the as-shipped
            # intermediate bundle.  Tolerances allow ±0.02 around each
            # baseline (well outside numerical jitter, well inside the
            # band that would let GBM(eng) silently drop below GBM(flat)).
            NB02_TARGETS = {
                "lr_flat_auc":  0.8737,
                "gbm_flat_auc": 0.8432,
                "lr_eng_auc":   0.8763,
                "gbm_eng_auc":  0.8579,
                "headline_lift_auc": 0.0147,  # GBM(eng) - GBM(flat)
            }
            NB02_TOLERANCES = {
                "lr_flat_auc":  0.02,
                "gbm_flat_auc": 0.02,
                "lr_eng_auc":   0.02,
                "gbm_eng_auc":  0.02,
                "headline_lift_auc": 0.015,   # tighter — sign-aware below
            }

            observed = {
                "lr_flat_auc":  rows[0]["auc"],
                "gbm_flat_auc": rows[1]["auc"],
                "lr_eng_auc":   rows[2]["auc"],
                "gbm_eng_auc":  rows[3]["auc"],
                "headline_lift_auc": deltas[0]["auc"],
            }
            assert_within_tolerance(
                observed=observed,
                target=NB02_TARGETS,
                tolerances=NB02_TOLERANCES,
                label="notebook 02 metric panel (seed 42, intermediate)",
            )
            assert observed["headline_lift_auc"] > 0.0, (
                "GBM(eng) − GBM(flat) AUC went non-positive — relational "
                "lift disappeared; investigate before merging."
            )
            print("OK — all panel metrics within tolerance and headline lift is positive.")
            """
        ),
        md(
            """
            ## 8. Honest takeaway

            On seed 42 the GBM(eng) − GBM(flat) AUC lift is small
            (+0.0147). Cross-seed variance for `gbm_auc` on this bundle
            is ~0.027 (see `release/validation/validation_report.json`,
            `tiers.intermediate.spreads.gbm_auc`), so a single-seed lift
            of this size is **suggestive, not conclusive**. Confirming a
            real signal needs a seed sweep — see the cohort-shift / seed
            harness coming in PR 6.2's notebook 04.

            The lift also does **not** flip the sign of the GBM-vs-LR
            comparison: GBM(eng) is still slightly below LR(flat). This
            is the same v1 finding documented in
            `release/validation/validation_report.md` (gate **G7.4.4**)
            and the dataset card: the v1 snapshot is dominated by
            roughly-linear signal, and HistGBM doesn't consistently beat
            LR on it. Engineered relational features narrow the gap; on
            this seed they don't yet erase it.

            Two takeaways for downstream users:

            1. **Joins on the public bundle are leakage-safe by
               construction.** Section 3 above is the full proof. You can
               aggregate any of the four event tables without policing the
               horizon yourself.
            2. **Bring your own non-linearities.** If a feature
               engineering choice (cross-table interactions, tree
               kernels, learned embeddings, bigger seed sweeps) flips the
               GBM-vs-LR sign reliably, that's a finding worth filing —
               the *break_me_guide* template lands in PR 6.3.

            ## Next

            - **Notebook 03** *(coming in PR 6.2)* — leakage and
              time-window walkthrough, including the deliberate
              `total_touches_all` trap notebook 01 keeps and this notebook
              drops.
            - **Notebook 04** *(coming in PR 6.2)* — value-aware ranking,
              calibration, and cohort-shift evaluation with a seed sweep.
            """
        ),
    ]


def main() -> None:
    write_notebook(OUT, assemble_notebook(cells()))


if __name__ == "__main__":
    main()
