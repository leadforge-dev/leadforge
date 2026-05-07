"""One-shot builder for ``release/notebooks/04_lift_calibration_value_ranking.ipynb``.

Run from the repository root::

    python scripts/build_release_notebook_04.py

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
    builder_arg_parser,
    code,
    md,
    write_notebook,
)

DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "release"
    / "notebooks"
    / "04_lift_calibration_value_ranking.ipynb"
)


def cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Notebook 04 — Lift, Calibration, Value-Aware Ranking

            **Dataset:** `leadforge-lead-scoring-v1`, *intermediate* tier.

            AUC ranks well; that doesn't mean it ranks *for the right
            thing*. Sales teams care about three additional concerns
            AUC alone never tells you about:

            1. **Calibration.** Are predicted probabilities trustworthy
               as point estimates, or just as a ranking?
            2. **Value-aware ranking.** A 30 %-likely lead worth $200K
               is more valuable than a 60 %-likely one worth $20K.
               Ranking by P(convert) wastes ACV; ranking by
               P(convert) × `expected_acv` doesn't.
            3. **Robustness.** Does the model still work next quarter
               (cohort shift)? How tight is the metric on the test set
               you have (bootstrap)?

            We answer all three on the public bundle, plus a threshold-
            selection walkthrough that maps a fixed sales-capacity
            constraint to an operating point. The notebook closes with
            a tolerance gate that pins the cohort-shift result to the
            published validation report — if a regeneration ever
            silently changes the cohort-degradation behaviour, CI
            catches it.

            **Public path discipline (G13.3).** Loads only
            `release/intermediate/` (the public student bundle).
            Instructor-only artefacts (the latent registry, full-horizon
            event tables, hidden DAG) are never read.

            **Trap discipline.** The headline LR / GBM panel drops
            `total_touches_all` (per notebook 02's leakage discipline)
            so the metrics it reports are honest production numbers.
            The cohort-shift section deliberately *keeps* the trap to
            reproduce the validation report's cohort-shift block — the
            report's panel is the as-shipped one, and we want a
            comparable number, not a cleaner one.
            """
        ),
        md("## 1. Setup"),
        code(
            """
            from __future__ import annotations

            import json
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
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
            from _notebook_utils import assert_within_tolerance

            SEED = 42
            BUNDLE = Path("../intermediate")          # public student bundle
            TASK = "converted_within_90_days"
            TRAP = "total_touches_all"

            with (BUNDLE / "manifest.json").open() as fh:
                manifest = json.load(fh)
            assert manifest["exposure_mode"] == "student_public"
            assert manifest["relational_snapshot_safe"] is True

            train = pd.read_parquet(BUNDLE / "tasks" / TASK / "train.parquet")
            test = pd.read_parquet(BUNDLE / "tasks" / TASK / "test.parquet")
            print(f"train rows: {len(train):,}")
            print(f"test rows:  {len(test):,}")
            """
        ),
        md(
            """
            ## 2. Train the headline LR + GBM panel

            Same preprocessing as notebooks 01 / 02 (mirrors
            `leadforge.validation.release_quality._build_pipeline`).
            We drop the documented leakage trap `total_touches_all`
            here so the calibration / lift / value plots in sections
            3–6 reflect an honest production model. The cohort-shift
            section in section 7 uses the validator's full-panel
            posture (trap kept) so its number is comparable to the
            published validation report.
            """
        ),
        code(
            """
            ID_COLS = ["account_id", "contact_id", "lead_id", "lead_created_at"]
            EXCLUDE_HEADLINE = set(ID_COLS + [TASK, TRAP])
            headline_cols = [c for c in train.columns if c not in EXCLUDE_HEADLINE]
            cat_cols = [
                c
                for c in headline_cols
                if not (
                    pd.api.types.is_bool_dtype(train[c])
                    or pd.api.types.is_numeric_dtype(train[c])
                )
            ]
            num_cols = [c for c in headline_cols if c not in cat_cols]
            print(f"headline panel: {len(headline_cols)} cols (trap dropped)")

            def _sanitize(df: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
                out = df.copy()
                for c in cats:
                    out[c] = out[c].astype(object).where(out[c].notna(), None)
                return out

            def build_pipeline(num: list[str], cat: list[str], *, model: str) -> Pipeline:
                pre = ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            num,
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "encoder",
                                        OneHotEncoder(
                                            handle_unknown="ignore", sparse_output=False
                                        ),
                                    ),
                                ]
                            ),
                            cat,
                        ),
                    ],
                    remainder="drop",
                )
                clf = (
                    LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
                    if model == "lr"
                    else HistGradientBoostingClassifier(random_state=SEED)
                )
                return Pipeline([("preprocessor", pre), ("classifier", clf)])

            y_train = train[TASK].astype("boolean").fillna(False).astype(int).to_numpy()
            y_test = test[TASK].astype("boolean").fillna(False).astype(int).to_numpy()
            base_rate = float(y_test.mean())

            x_train = _sanitize(train[headline_cols], cat_cols)
            x_test = _sanitize(test[headline_cols], cat_cols)

            lr_pipe = build_pipeline(num_cols, cat_cols, model="lr").fit(x_train, y_train)
            gbm_pipe = build_pipeline(num_cols, cat_cols, model="gbm").fit(x_train, y_train)
            lr_probs = lr_pipe.predict_proba(x_test)[:, 1]
            gbm_probs = gbm_pipe.predict_proba(x_test)[:, 1]

            print(f"  base rate: {base_rate:.3f}")
            print(f"  LR   AUC: {roc_auc_score(y_test, lr_probs):.4f}   "
                  f"AP: {average_precision_score(y_test, lr_probs):.4f}   "
                  f"Brier: {brier_score_loss(y_test, lr_probs):.4f}")
            print(f"  GBM  AUC: {roc_auc_score(y_test, gbm_probs):.4f}   "
                  f"AP: {average_precision_score(y_test, gbm_probs):.4f}   "
                  f"Brier: {brier_score_loss(y_test, gbm_probs):.4f}")
            """
        ),
        md(
            """
            ## 3. Calibration / reliability diagram

            Bin LR's predicted probabilities into ten equal-width
            buckets, plot mean predicted vs mean observed. A perfectly
            calibrated model lies on the diagonal; LR after
            `StandardScaler + LogisticRegression` is usually close.
            We also surface `max_bin_error` — the worst gap across
            non-empty bins — which the validation report tracks
            (`tiers.intermediate.medians.calibration_max_bin_error`).
            """
        ),
        code(
            """
            edges = np.linspace(0.0, 1.0, 11)
            mean_pred: list[float] = []
            mean_actual: list[float] = []
            bin_n: list[int] = []
            for i in range(10):
                lo, hi = edges[i], edges[i + 1]
                mask = (lr_probs >= lo) & (
                    (lr_probs <= hi) if i == 9 else (lr_probs < hi)
                )
                if mask.sum() == 0:
                    continue
                mean_pred.append(float(lr_probs[mask].mean()))
                mean_actual.append(float(y_test[mask].mean()))
                bin_n.append(int(mask.sum()))

            max_bin_err = max(
                abs(p - a) for p, a in zip(mean_pred, mean_actual, strict=False)
            )
            print(f"max bin error (LR): {max_bin_err:.4f}")
            for p, a, n in zip(mean_pred, mean_actual, bin_n, strict=False):
                print(f"  pred={p:.3f}  actual={a:.3f}  n={n:>4d}")

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
            ## 4. Lift and cumulative gains

            Two complementary curves:

            * **Cumulative gains** — fraction of positives captured as
              you sweep the score threshold. Top 10 % of the ranked
              list captures ~26 % of converted leads on this seed (vs
              the 10 % a random ranker would catch).
            * **Lift at *k* %** — `top_k_conversion_rate / base_rate`.
              Lift = 2 means "the top 1 % of leads convert at twice
              the base rate."

            Both metrics are in `release/validation/validation_report.json`
            (`per_seed[0].cumulative_gains` and `per_seed[0].lift_at_pct`)
            so the reproduction is auditable.
            """
        ),
        code(
            """
            order = np.argsort(-lr_probs, kind="stable")
            y_sorted = y_test[order]
            n = len(y_test)
            n_pos = int(y_test.sum())

            # Cumulative gains: fraction of positives captured by top-pct.
            pcts = np.arange(0, 101, 10)
            gains = []
            for pct in pcts:
                k = max(1, int(round(n * pct / 100.0)))
                if pct == 0:
                    gains.append(0.0)
                else:
                    gains.append(float(y_sorted[:k].sum() / n_pos))

            # Lift at 1 / 5 / 10 %.
            lifts = {}
            for pct in [1.0, 5.0, 10.0]:
                k = max(1, int(round(n * pct / 100.0)))
                lifts[pct] = float(y_sorted[:k].mean() / base_rate)

            for pct, lift in lifts.items():
                print(f"  lift @ top {pct:>4.0f}%: {lift:.3f}x")

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(pcts, gains, marker="o", color="#3b82f6", label="LR")
            axes[0].plot([0, 100], [0, 1], color="#9ca3af", linestyle="--", label="random")
            axes[0].set_xlabel("Top-pct of ranked leads")
            axes[0].set_ylabel("Cumulative conversion capture")
            axes[0].set_title("Cumulative gains")
            axes[0].legend(loc="lower right")

            axes[1].bar(
                [str(int(p)) for p in lifts],
                list(lifts.values()),
                color="#3b82f6",
            )
            axes[1].axhline(1.0, color="#ef4444", linestyle="--", label="random (lift=1)")
            axes[1].set_xlabel("Top-pct of ranked leads")
            axes[1].set_ylabel("Lift over base rate")
            axes[1].set_title("Lift at top-pct")
            axes[1].legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 5. Value-aware ranking — `expected_acv` × P(convert)

            Sales reps don't have infinite capacity, so the right
            objective is rarely "maximise conversion count" — it's
            "maximise revenue captured per outreach slot." The bundle
            ships an `expected_acv` column (opportunity ACV when
            available, else revenue-band midpoint heuristic) which
            makes value-aware ranking trivial:

            $$ \\text{score}_\\text{value} = P(\\text{convert}) \\times
            \\text{expected\\_acv} $$

            We compare two top-K policies — rank by P(convert) only
            vs rank by score_value — and report
            `expected_acv_capture_at_k = sum(acv * y) over top-K /
            sum(acv * y) over the whole test`. The validation report's
            `per_seed[0].expected_acv_capture_at_k` is the reference.
            """
        ),
        code(
            """
            acv = pd.to_numeric(test["expected_acv"], errors="coerce").fillna(0.0).to_numpy()
            value_score = lr_probs * acv

            # Pre-compute the ranking orders once — argsort is O(N log N)
            # and the order doesn't change as K varies, so the ~30 plot
            # points below should not pay for ~30 sorts.
            total_converted_acv = float(np.sum(acv * y_test))
            assert total_converted_acv > 0, "no converted-ACV in the test set"
            order_p = np.argsort(-lr_probs, kind="stable")
            order_v = np.argsort(-value_score, kind="stable")
            captured_p = np.cumsum(acv[order_p] * y_test[order_p]) / total_converted_acv
            captured_v = np.cumsum(acv[order_v] * y_test[order_v]) / total_converted_acv

            def acv_capture(use_value: bool, k: int) -> float:
                # 1-indexed cumulative-capture lookup (k=1 = first slot).
                series = captured_v if use_value else captured_p
                if k <= 0 or k > len(series):
                    return float("nan")
                return float(series[k - 1])

            print(f"{'top-K':<6s}  {'cap by P(conv)':>14s}  {'cap by P×ACV':>13s}  {'gain':>7s}")
            value_gains = {}
            for k in (50, 100, 200):
                cap_p = acv_capture(False, k)
                cap_v = acv_capture(True, k)
                value_gains[k] = cap_v - cap_p
                print(f"  top {k:<3d}  {cap_p:>14.4f}  {cap_v:>13.4f}  {cap_v - cap_p:+7.4f}")

            # Plot side-by-side ACV capture for K in 10..300.  Cheap now
            # — every point is a single cumulative-array lookup.
            ks = np.arange(10, 301, 10)
            cap_p_curve = [acv_capture(False, int(k)) for k in ks]
            cap_v_curve = [acv_capture(True, int(k)) for k in ks]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(ks, cap_p_curve, marker="o", color="#9ca3af", label="rank by P(convert)")
            ax.plot(ks, cap_v_curve, marker="o", color="#3b82f6", label="rank by P(convert)×ACV")
            ax.set_xlabel("top-K leads contacted")
            ax.set_ylabel("Fraction of converted-ACV captured")
            ax.set_title("ACV capture vs top-K (rank by P only vs P × ACV)")
            ax.legend(loc="lower right")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 6. Threshold selection for fixed top-K capacity

            Sales rarely has the patience for "score everything, run
            stats." The realistic ask is: *"My team can work 50 leads
            this week. Set a probability threshold that selects ~50
            from the test population."*

            We sweep the probability threshold across the LR score
            distribution and report **count, precision, and recall**
            above threshold for each step, then pick the threshold
            whose count is closest to the requested capacity.
            """
        ),
        code(
            """
            CAPACITY = 50
            n_pos_test = max(int(y_test.sum()), 1)

            sorted_probs = np.sort(lr_probs)[::-1]
            # The K-th highest probability is the smallest threshold that
            # admits AT LEAST K leads via ``probs >= threshold``.  If
            # several leads share that probability, the inclusive
            # comparison can admit more than K — that's a property of
            # threshold-based selection, not a bug.  The
            # ``actually_above`` readout below makes the realised count
            # visible so the operator can see when ties are inflating
            # the slate (and decide whether to break them with a
            # secondary score).
            threshold = float(sorted_probs[CAPACITY - 1])
            mask = lr_probs >= threshold
            n_above = int(mask.sum())
            prec = float(y_test[mask].mean()) if n_above > 0 else float("nan")
            recall = float(y_test[mask].sum() / n_pos_test)
            print(
                f"capacity={CAPACITY}  threshold={threshold:.3f}  "
                f"actually_above={n_above}  precision={prec:.3f}  recall={recall:.3f}"
            )

            # Threshold sweep — show what happens around the operating
            # point so the threshold choice is informed, not magic.
            thresholds = np.linspace(
                float(np.quantile(lr_probs, 0.5)), float(np.max(lr_probs)), 30
            )
            counts = []
            precs = []
            recalls = []
            for t in thresholds:
                m = lr_probs >= t
                n_t = int(m.sum())
                counts.append(n_t)
                precs.append(float(y_test[m].mean()) if n_t > 0 else 0.0)
                recalls.append(float(y_test[m].sum() / n_pos_test))

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].plot(thresholds, counts, marker="o", color="#3b82f6")
            axes[0].axhline(CAPACITY, color="#ef4444", linestyle="--", label=f"capacity={CAPACITY}")
            axes[0].axvline(threshold, color="#10b981", linestyle="--", label=f"chosen ({threshold:.3f})")
            axes[0].set_xlabel("threshold")
            axes[0].set_ylabel("# leads above threshold")
            axes[0].set_title("count above")
            axes[0].legend(fontsize=8)

            axes[1].plot(thresholds, precs, marker="o", color="#3b82f6")
            axes[1].axhline(base_rate, color="#9ca3af", linestyle="--", label=f"base rate ({base_rate:.3f})")
            axes[1].axvline(threshold, color="#10b981", linestyle="--", label=f"chosen ({threshold:.3f})")
            axes[1].set_xlabel("threshold")
            axes[1].set_ylabel("precision above threshold")
            axes[1].set_title("precision above")
            axes[1].legend(fontsize=8)

            axes[2].plot(thresholds, recalls, marker="o", color="#3b82f6")
            axes[2].axvline(threshold, color="#10b981", linestyle="--", label=f"chosen ({threshold:.3f})")
            axes[2].set_xlabel("threshold")
            axes[2].set_ylabel("recall above threshold")
            axes[2].set_title("recall above")
            axes[2].legend(fontsize=8)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 7. Cohort-shift evaluation

            The bundle's train/test split is a uniform random split of
            leads. A more realistic stress test is "train on the first
            85 % of leads chronologically, score the last 15 %" —
            because in production you always have to predict the
            *future*, never a held-out random sample of the past.

            We mirror the validator's cohort-shift logic
            (`leadforge.validation.release_quality.measure_cohort_shift_from_bundle`)
            exactly: pool train + test, sort by `lead_created_at` with
            `lead_id` as a stable tiebreak, train HistGBM on the first
            85 % (`COHORT_TRAIN_FRAC = 0.85`) and score the last 15 %.
            Both random and cohort splits use the full feature panel
            **including** the trap, matching the report's posture so
            the numbers compare directly. The HistGBM uses
            `random_state=0` here (the validator's
            `DEFAULT_MODEL_RANDOM_STATE = 0`) rather than the
            notebook's default `SEED=42` — the report's cohort-shift
            block reproduces to four decimals only when both knobs
            match.

            The expected behaviour for the v1 intermediate tier is
            *no* degradation — the report shows the cohort split AUC
            running ~0.015 *higher* than the random split. That's a
            surprise worth surfacing: the v1 simulator's intermediate
            world doesn't drift over its 90-day horizon, so cohort
            order isn't a stressor here. The intro and advanced
            tiers show small positive degradations (intro +0.016,
            advanced +0.010) — see
            `release/validation/validation_report.json` ⇒
            `cohort_shift`.
            """
        ),
        code(
            """
            # Constants mirror leadforge.validation.release_quality so
            # the numbers reproduce the report's cohort-shift block.
            COHORT_TRAIN_FRAC = 0.85
            COHORT_MODEL_SEED = 0

            # Cohort-shift uses the validator's full panel (trap kept).
            EXCLUDE_FULL = set(ID_COLS + [TASK])
            full_cols = [c for c in train.columns if c not in EXCLUDE_FULL]
            cat_full = [
                c
                for c in full_cols
                if not (
                    pd.api.types.is_bool_dtype(train[c])
                    or pd.api.types.is_numeric_dtype(train[c])
                )
            ]
            num_full = [c for c in full_cols if c not in cat_full]

            def _gbm_pipeline_for_cohort() -> Pipeline:
                # Local builder so the validator's ``model_random_state=0``
                # is used here, while the headline panel above keeps
                # ``random_state=SEED`` for the section-2 LR/GBM models.
                pre = ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            num_full,
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    (
                                        "encoder",
                                        OneHotEncoder(
                                            handle_unknown="ignore", sparse_output=False
                                        ),
                                    ),
                                ]
                            ),
                            cat_full,
                        ),
                    ],
                    remainder="drop",
                )
                clf = HistGradientBoostingClassifier(random_state=COHORT_MODEL_SEED)
                return Pipeline([("preprocessor", pre), ("classifier", clf)])

            # Random split AUC = HistGBM on the bundle's existing split.
            rand_pipe = _gbm_pipeline_for_cohort().fit(
                _sanitize(train[full_cols], cat_full), y_train
            )
            random_split_auc = float(
                roc_auc_score(
                    y_test,
                    rand_pipe.predict_proba(_sanitize(test[full_cols], cat_full))[:, 1],
                )
            )

            # Chronological resplit: pool, sort by lead_created_at +
            # lead_id (stable tiebreak), take first 85 % as train, last
            # 15 % as test.  Mirrors ``measure_cohort_shift_from_bundle``.
            pooled = pd.concat([train, test], ignore_index=True)
            ts = pd.to_datetime(pooled["lead_created_at"], errors="coerce")
            assert not ts.isna().any(), "expected every lead to have a parseable lead_created_at"
            sort_frame = pd.DataFrame(
                {"_ts": ts.values, "_lid": pooled["lead_id"].astype(str).values}
            )
            order = sort_frame.sort_values(["_ts", "_lid"], kind="stable").index.to_numpy()
            cutoff = int(round(len(pooled) * COHORT_TRAIN_FRAC))
            early = pooled.iloc[order[:cutoff]]
            late = pooled.iloc[order[cutoff:]]
            y_early = early[TASK].astype("boolean").fillna(False).astype(int).to_numpy()
            y_late = late[TASK].astype("boolean").fillna(False).astype(int).to_numpy()

            cohort_pipe = _gbm_pipeline_for_cohort().fit(
                _sanitize(early[full_cols], cat_full), y_early
            )
            cohort_split_auc = float(
                roc_auc_score(
                    y_late,
                    cohort_pipe.predict_proba(_sanitize(late[full_cols], cat_full))[:, 1],
                )
            )
            auc_degradation = random_split_auc - cohort_split_auc
            print(f"random_split_auc:  {random_split_auc:.4f}")
            print(f"cohort_split_auc:  {cohort_split_auc:.4f}")
            print(f"auc_degradation:   {auc_degradation:+.4f}  (positive = cohort is harder)")
            """
        ),
        md(
            """
            ## 8. Bootstrap robustness — within-bundle metric variance

            Cross-seed metric variance (the validation report's
            `tiers.intermediate.spreads.gbm_auc = 0.027`) is the
            cleanest answer to "how confident is this AUC?", but it
            requires regenerating the bundle from N seeds — something
            a public-bundle consumer (Kaggle / HF) can't easily do.

            The within-bundle proxy is **non-parametric bootstrap of
            the test set**. We resample the 750 test rows with
            replacement, re-rank using the model probabilities we
            already have, and recompute AUC / AP. 200 resamples is
            enough to read a confidence band off the distribution.

            The bootstrap variance is **smaller** than the cross-seed
            variance — it captures sampling noise on a single
            generated world, not generation-process noise across
            seeds — but it's the right number for the question
            "given *this* test set, how stable is the AUC?"
            """
        ),
        code(
            """
            N_BOOT = 200
            rng = np.random.default_rng(SEED)

            boot_lr_auc = np.empty(N_BOOT)
            boot_gbm_auc = np.empty(N_BOOT)
            boot_lr_ap = np.empty(N_BOOT)
            n_test = len(y_test)
            for i in range(N_BOOT):
                idx = rng.integers(0, n_test, n_test)
                if y_test[idx].sum() == 0 or y_test[idx].sum() == n_test:
                    # Degenerate resample (all-positive or all-negative)
                    # — ``roc_auc_score`` is undefined here.  We mark
                    # the iteration NaN and let ``_summary`` filter it
                    # out; with n_test=750 and base rate ~22 %, the
                    # probability of a degenerate draw is ~10⁻¹⁰⁰, so
                    # this branch is dead in practice.  Kept as a
                    # defensive safety net for tiny test sets.
                    boot_lr_auc[i] = np.nan
                    boot_gbm_auc[i] = np.nan
                    boot_lr_ap[i] = np.nan
                    continue
                boot_lr_auc[i] = roc_auc_score(y_test[idx], lr_probs[idx])
                boot_gbm_auc[i] = roc_auc_score(y_test[idx], gbm_probs[idx])
                boot_lr_ap[i] = average_precision_score(y_test[idx], lr_probs[idx])

            def _summary(arr: np.ndarray, name: str) -> None:
                arr = arr[~np.isnan(arr)]
                lo, med, hi = np.quantile(arr, [0.025, 0.5, 0.975])
                print(
                    f"  {name:<14s}  median={med:.4f}  "
                    f"95% CI=[{lo:.4f}, {hi:.4f}]  IQR={(np.quantile(arr,0.75)-np.quantile(arr,0.25)):.4f}"
                )

            print(f"bootstrap on test set, n_iters={N_BOOT}, seed={SEED}:")
            _summary(boot_lr_auc, "LR AUC")
            _summary(boot_gbm_auc, "GBM AUC")
            _summary(boot_lr_ap, "LR AP")

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(boot_lr_auc, bins=30, color="#3b82f6", alpha=0.7, label="LR AUC")
            ax.hist(boot_gbm_auc, bins=30, color="#9ca3af", alpha=0.7, label="GBM AUC")
            ax.axvline(roc_auc_score(y_test, lr_probs), color="#1d4ed8", linestyle="--", label="LR (point)")
            ax.axvline(roc_auc_score(y_test, gbm_probs), color="#374151", linestyle="--", label="GBM (point)")
            ax.set_xlabel("AUC")
            ax.set_ylabel("# bootstrap draws")
            ax.set_title(f"Bootstrap AUC distribution (n={N_BOOT})")
            ax.legend(loc="upper left", fontsize=8)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 9. Tolerance gate (G13.2)

            Three groups of pinned values:

            * **Cohort-shift block** — pinned to
              `release/notebooks/_release_targets.json`'s
              `cohort_shift.intermediate`, which is itself audit-synced
              against `validation_report.json`'s `cohort_shift.intermediate`
              by `tests/release/notebooks/test_release_targets_match_report.py`.
              That audit-sync is what makes the "this notebook
              reproduces the report" claim meaningful.
            * **Calibration / lift / value-capture** — pinned inline
              against the seed-42 single-run values from the
              validation report's `per_seed[0]` block. Tolerances
              widen for small-K metrics (P@K, value capture) because
              their seed-to-seed variance is larger.
            * **Bootstrap medians** — pinned inline against the
              seed-42 point estimates (the bootstrap median converges
              to the data-specific value, not to the cross-seed
              median).

            The headline lift sign-check (`gbm_auc > lr_auc - eps` was
            *not* asserted — the v1 dataset documents the surprising
            finding that LR ≥ GBM on intermediate; see
            `release/validation/validation_report.md` gate G7.4.4).
            """
        ),
        code(
            """
            with (Path.cwd() / "_release_targets.json").open() as fh:
                release_targets = json.load(fh)
            cohort_targets = release_targets["cohort_shift"]["intermediate"]

            cohort_observed = {
                "random_split_auc":  random_split_auc,
                "cohort_split_auc":  cohort_split_auc,
                "auc_degradation":   auc_degradation,
            }
            assert_within_tolerance(
                observed=cohort_observed,
                target=cohort_targets,
                tolerances={
                    # ±0.02 on AUCs — well outside numerical jitter,
                    # well inside the band that would let the
                    # cohort-shift sign flip silently.
                    "random_split_auc":  0.02,
                    "cohort_split_auc":  0.02,
                    # Wider on the difference because both AUCs are
                    # within tolerance, so the difference can drift up
                    # to ±0.04 in the worst case.
                    "auc_degradation":   0.04,
                },
                label="notebook 04 cohort-shift vs validation_report (intermediate)",
            )

            # Inline pins for the seed-42 single-run values *of the
            # without-trap headline panel*.  These are not the report's
            # published numbers (the report keeps the trap) — the
            # report-level pin lives in section 9's cohort-shift block,
            # which is the only metric this notebook reproduces against
            # the report.  Notebook 02 trains the same trap-dropped LR
            # and reports the same AUCs, so these values are also
            # cross-checked there.
            NB04_TARGETS = {
                "lr_auc":             0.8737,
                "gbm_auc":            0.8432,
                "lr_max_bin_err":     0.1344,
                "lift_at_5pct":       2.4819,
                "lift_at_10pct":      2.7536,
                "acv_cap_50":         0.1615,
                "acv_cap_100":        0.3702,
                # Bootstrap medians converge to the seed-42 point
                # estimates within sampling noise.
                "boot_lr_auc_median":  0.8757,
                "boot_gbm_auc_median": 0.8440,
            }
            NB04_TOLERANCES = {
                "lr_auc":             0.02,
                "gbm_auc":            0.02,
                "lr_max_bin_err":     0.05,
                "lift_at_5pct":       0.30,
                "lift_at_10pct":      0.30,
                "acv_cap_50":         0.05,
                "acv_cap_100":        0.05,
                "boot_lr_auc_median":  0.03,
                "boot_gbm_auc_median": 0.03,
            }
            observed = {
                "lr_auc":        float(roc_auc_score(y_test, lr_probs)),
                "gbm_auc":       float(roc_auc_score(y_test, gbm_probs)),
                "lr_max_bin_err": float(max_bin_err),
                "lift_at_5pct":  lifts[5.0],
                "lift_at_10pct": lifts[10.0],
                "acv_cap_50":    acv_capture(False, 50),
                "acv_cap_100":   acv_capture(False, 100),
                "boot_lr_auc_median":  float(np.nanmedian(boot_lr_auc)),
                "boot_gbm_auc_median": float(np.nanmedian(boot_gbm_auc)),
            }
            assert_within_tolerance(
                observed=observed,
                target=NB04_TARGETS,
                tolerances=NB04_TOLERANCES,
                label="notebook 04 metric panel (seed 42, intermediate)",
            )

            # Sign-aware: value-aware ranking should be measurably
            # better, not just non-worse.  On this seed every K shows
            # +0.20 ACV-capture gain; the threshold sits well below
            # that but well above noise so a regeneration that erodes
            # the value-aware lift fails here.
            MIN_VALUE_GAIN = 0.05
            for k, gain in value_gains.items():
                assert gain > MIN_VALUE_GAIN, (
                    f"value-aware ranking lift at top-{k} collapsed: "
                    f"{gain:+.4f} <= {MIN_VALUE_GAIN:.4f} — "
                    "the P×ACV story is no longer load-bearing"
                )
            print("OK — cohort-shift, calibration, lift, value-capture, and bootstrap all pinned.")
            """
        ),
        md(
            """
            ## 10. Summary

            * The LR baseline is well-calibrated (max bin error ≈ 0.13
              on the trap-dropped headline panel, vs ~0.19 on the
              with-trap panel the validation report tracks) and lifts
              the top decile to ~2.75× the base rate.
            * Value-aware ranking (P × ACV) captures more revenue per
              top-K slot than P-only ranking — the gap depends on K
              but is positive across all sizes we tested.
            * Cohort shift is **negative** on the intermediate tier
              (the late cohort is *easier*, not harder); the report
              documents this, and the notebook reproduces it. The
              intro and advanced tiers show small positive
              degradations.
            * Bootstrap on the existing test split gives a within-
              bundle confidence band that's tighter than the cross-seed
              spread the validation report computes — useful for "how
              confident is this single AUC" questions, not for "how
              much does the bundle move across seeds."

            ## Where to go next

            1. Try cohort-shifted training in production: refit weekly
               on the trailing 60-day window, score the next 7 days.
            2. If you have real ACV data, swap the `expected_acv`
               heuristic for it and recompute section 5 — the revenue
               capture story should sharpen.
            3. The break-me playbook in
               [`docs/release/break_me_guide.md`](https://github.com/leadforge-dev/leadforge/blob/main/docs/release/break_me_guide.md)
               catalogues additional stress tests (target-encoding
               leakage, train-test contamination, cohort-by-segment)
               and how to detect each from a single bundle.
            """
        ),
    ]


def main() -> None:
    args = builder_arg_parser(
        default_out=DEFAULT_OUT,
        description="Build release/notebooks/04_lift_calibration_value_ranking.ipynb",
    ).parse_args()
    write_notebook(args.out, assemble_notebook(cells()))


if __name__ == "__main__":
    main()
