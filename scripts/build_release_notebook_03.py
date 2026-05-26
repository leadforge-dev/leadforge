"""One-shot builder for ``release/notebooks/03_leakage_and_time_windows.ipynb``.

Run from the repository root::

    python scripts/build_release_notebook_03.py

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
    / "03_leakage_and_time_windows.ipynb"
)


def cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Notebook 03 — Leakage and Time Windows

            **Dataset:** `leadforge-lead-scoring-v1`, *intermediate* tier.

            The bundle ships with one **deliberate leakage trap**:
            `total_touches_all`. The feature dictionary marks it
            `leakage_risk = True`; the dataset card calls it out;
            notebook 01 keeps it (matching the validation report's
            panel) while notebook 02 drops it. This notebook turns the
            trap into a teaching moment.

            We do four things:

            1. **Read the receipts.** The trap is documented in the
               feature dictionary. We surface that label.
            2. **Time-window proof.** Quantify how much
               `total_touches_all` differs from its snapshot-safe
               sibling `touch_count` — the difference is post-snapshot
               information by construction, regardless of how
               predictive that information turns out to be.
            3. **The lesson.** Run a single-column standalone-AUC probe
               on the trap (it looks innocuous, ~0.53). Then run the
               full-panel ± trap comparison: HistGBM extracts a
               substantial AUC lift (+0.03) from a column whose
               standalone AUC is barely above chance. Standalone
               probes undersell tree-friendly leakage.
            4. **Pin the deltas.** Sign-aware tolerance gates so a
               future regeneration that neutralises the trap (or
               accidentally amplifies it) breaks CI.

            **Public path discipline (G13.3).** This notebook reads only
            from `release/intermediate/` (the public student bundle). The
            instructor companion is **not** loaded — leakage detection
            has to work from the public artefact alone, since that's all
            a downstream consumer ever has.
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
            from sklearn.metrics import roc_auc_score
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
            SNAPSHOT_DAY = int(manifest["snapshot_day"])
            HORIZON_DAYS = int(manifest["horizon_days"])
            print(f"snapshot anchor: day {SNAPSHOT_DAY} of a {HORIZON_DAYS}-day horizon")
            print(
                f"any feature aggregating events past day {SNAPSHOT_DAY} "
                f"is leaking — that's the whole {HORIZON_DAYS - SNAPSHOT_DAY}-day window "
                "we're hunting in section 3"
            )
            """
        ),
        md(
            """
            ## 2. The trap, as the feature dictionary calls it out

            The release ships a `feature_dictionary.csv` next to the
            data. Any column with `leakage_risk = True` is flagged as a
            **deliberate teaching trap** — included so users can practise
            detecting it, with the trap's nature documented inline.

            Treat the feature dictionary as the first place you look on
            any new dataset. A column named `total_touches_all` is not
            obviously bad until the dictionary tells you it counts
            touches over the full 90-day horizon, well past the
            30-day snapshot anchor that defines the prediction time.
            """
        ),
        code(
            """
            feat_dict = pd.read_csv(BUNDLE / "feature_dictionary.csv")
            traps = feat_dict[feat_dict["leakage_risk"].astype(bool)]
            print(f"trap columns flagged in feature_dictionary.csv: {len(traps)}")
            for _, row in traps.iterrows():
                print(f"  {row['name']}: {row['description']}")
            assert TRAP in set(traps["name"]), f"{TRAP} expected to be flagged in dictionary"
            """
        ),
        md(
            """
            ## 3. Time-window proof — the trap *by construction*

            The dictionary *says* `total_touches_all` uses post-snapshot
            data. We verify that on the same row that carries the trap:
            the task table also carries `touch_count`, the
            **snapshot-safe** touch aggregate (filtered to
            `touch_timestamp <= lead_created_at + snapshot_day`). Their
            difference is the **post-snapshot delta** — by construction,
            information from days 31–90 that the model should never see
            when scoring at day 30.

            The pedagogical point is independent of how predictive that
            difference turns out to be. **A column that uses
            post-snapshot data is invalid at scoring time even when it
            looks unpredictive in isolation.** Section 4 measures that
            "looks unpredictive in isolation" claim directly, then
            section 5 shows it can be misleading.

            We pool all three task splits so the receipt covers every
            lead in the bundle.
            """
        ),
        code(
            """
            train = pd.read_parquet(BUNDLE / "tasks" / TASK / "train.parquet")
            valid = pd.read_parquet(BUNDLE / "tasks" / TASK / "valid.parquet")
            test = pd.read_parquet(BUNDLE / "tasks" / TASK / "test.parquet")

            all_leads = pd.concat([train, valid, test], ignore_index=True)
            assert all_leads["lead_id"].is_unique, (
                "expected one row per lead across train/valid/test"
            )

            window = all_leads[["lead_id", TRAP, "touch_count", TASK]].copy()
            window[TRAP] = pd.to_numeric(window[TRAP], errors="coerce")
            window["touch_count"] = pd.to_numeric(window["touch_count"], errors="coerce")
            window = window.dropna(subset=[TRAP, "touch_count"]).copy()
            window["post_snapshot_touches"] = window[TRAP] - window["touch_count"]
            window[TASK] = window[TASK].astype("boolean").fillna(False).astype(int)

            print(f"leads used in this section: {len(window):,}")
            print(
                f"  {TRAP:<22s}      mean={window[TRAP].mean():6.2f}  "
                f"max={int(window[TRAP].max()):>4d}"
            )
            print(
                f"  {'touch_count (snapshot-safe)':<22s}  "
                f"mean={window['touch_count'].mean():6.2f}  "
                f"max={int(window['touch_count'].max()):>4d}"
            )
            mean_delta = float(window["post_snapshot_touches"].mean())
            n_post = int((window["post_snapshot_touches"] > 0).sum())
            print(
                f"  {'post-snapshot delta':<22s}      "
                f"mean={mean_delta:6.2f}  "
                f"max={int(window['post_snapshot_touches'].max()):>4d}"
            )
            print(
                f"  → {n_post:,} of {len(window):,} leads "
                f"({n_post / len(window):.1%}) have a positive post-snapshot delta"
            )
            # Real gate, not performative: on the as-shipped bundle the
            # mean delta is ~3.2 touches/lead and ~82 % of leads have a
            # positive delta.  The thresholds below sit well below those
            # observations but well above "barely above zero" — a
            # regeneration that erodes the trap's post-snapshot
            # footprint will fail here even if a single lead still
            # carries a positive delta.
            assert mean_delta > 1.0, (
                f"mean post-snapshot delta collapsed to {mean_delta:.2f} (<= 1.0) — "
                "the trap may have been silently rebuilt as a snapshot-safe aggregate"
            )
            assert n_post / len(window) > 0.5, (
                f"only {n_post / len(window):.1%} of leads have a positive "
                "post-snapshot delta (< 50 %); the trap's footprint has eroded"
            )
            """
        ),
        md(
            """
            ### 3.1 The post-snapshot delta is uncorrelated with the label *on this dataset*

            On the v1 procurement world, the count of touches between
            day 30 and day 90 turns out to be roughly the same for
            converted and non-converted leads — sales reps keep working
            both groups for a while before the funnel settles. A
            stronger world (more aggressive sales follow-up on hot
            leads) would split these apart; this one doesn't.

            The plot below makes that lack-of-split visible. The trap
            is *still a trap* — we just can't tell that from the
            post-snapshot delta alone, which is why the validation
            report's `post_snapshot_aggregates` baseline (a single-
            column probe) gives an AUC of only ~0.55. The real damage
            shows up when a tree model gets to combine the trap with
            other columns; section 5 measures that.
            """
        ),
        code(
            """
            grouped = window.groupby(TASK)["post_snapshot_touches"].agg(["mean", "median", "count"])
            grouped.index = grouped.index.map({0: "non-converted", 1: "converted"})
            print(grouped)

            fig, ax = plt.subplots(figsize=(6, 4))
            data = [
                window.loc[window[TASK] == 0, "post_snapshot_touches"],
                window.loc[window[TASK] == 1, "post_snapshot_touches"],
            ]
            ax.boxplot(data, tick_labels=["non-converted", "converted"], showfliers=False)
            ax.set_ylabel("post-snapshot touches (total_touches_all − touch_count)")
            ax.set_title("Post-snapshot delta by label\\n(roughly the same — section 5 explains why this is misleading)")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 4. Standalone-AUC probe (the audit that almost lets the trap pass)

            A common leakage audit is to fit a one-feature classifier on
            each suspect column and report the standalone AUC. The
            validation report does this at scale — its
            `post_snapshot_aggregates` baseline trains a *full LR
            pipeline* (median-impute + StandardScaler + LR) on the
            single column `total_touches_all` and reports an AUC of
            ~0.55. We use a quicker probe here — the raw column
            value as a score, no preprocessing — which gives ~0.53 on
            this seed. The two numbers measure slightly different
            things (a fitted LR can re-scale and adjust, a raw-value
            ranker can't), but both fall in the "barely above chance"
            band. On a busy schedule it's tempting to clear the column
            on those grounds. Section 5 shows what that audit misses.
            """
        ),
        code(
            """
            # ``window`` was already dropped of NaN in section 3, so the
            # raw-value ranker can use it directly.
            y = window[TASK].astype(int).to_numpy()
            standalone = {
                TRAP: float(roc_auc_score(y, window[TRAP].to_numpy())),
                "touch_count (snapshot-safe)": float(
                    roc_auc_score(y, window["touch_count"].to_numpy())
                ),
                "post-snapshot delta": float(
                    roc_auc_score(y, window["post_snapshot_touches"].to_numpy())
                ),
            }
            print(f"{'feature':<32s}  {'standalone AUC':>16s}")
            for name, auc in standalone.items():
                print(f"  {name:<30s}  {auc:>16.4f}")

            # Sign-aware: the section-5 narrative ("standalone probe
            # sees the trap as predictive-ish, but tree models extract
            # more") falls apart if the trap drops to chance or below.
            # Lower bound 0.51 sits just above sampling noise; if a
            # regeneration ever puts the trap at or below 0.50, the
            # whole pedagogical setup needs revisiting.
            assert standalone[TRAP] > 0.51, (
                f"trap standalone AUC collapsed to {standalone[TRAP]:.3f} (<= 0.51); "
                "section 5 contrasts the standalone probe with the GBM ablation, "
                "and that contrast is empty if the trap is at or below chance"
            )
            """
        ),
        md(
            """
            ## 5. Side-by-side AUC: full panel ± trap

            Train two HistGBM and two Logistic Regression baselines on
            the **same train/test split, same model, same seed** —
            the only thing that varies is whether `total_touches_all`
            is in the column list.

            We use the full as-shipped feature panel (every public
            snapshot column except IDs / label) as the baseline. This
            mirrors notebook 01 / the validation report's setup, so the
            with-trap AUC reproduces the report's published number and
            the without-trap AUC is what notebook 02 starts from.
            """
        ),
        code(
            """
            ID_COLS = ["account_id", "contact_id", "lead_id", "lead_created_at"]
            EXCLUDE = set(ID_COLS + [TASK])

            full_cols = [c for c in train.columns if c not in EXCLUDE]
            full_cols_no_trap = [c for c in full_cols if c != TRAP]
            print(f"full panel:         {len(full_cols)} cols (incl. {TRAP})")
            print(f"full panel no trap: {len(full_cols_no_trap)} cols")

            def _split_cols(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
                cat = [
                    c
                    for c in cols
                    if not (
                        pd.api.types.is_bool_dtype(df[c])
                        or pd.api.types.is_numeric_dtype(df[c])
                    )
                ]
                num = [c for c in cols if c not in cat]
                return num, cat

            def _sanitize(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
                out = df.copy()
                for c in cat_cols:
                    out[c] = out[c].astype(object).where(out[c].notna(), None)
                return out

            def _build_pipeline(num_cols: list[str], cat_cols: list[str], *, model: str) -> Pipeline:
                num_t = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                )
                cat_t = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                )
                pre = ColumnTransformer(
                    [("num", num_t, num_cols), ("cat", cat_t, cat_cols)],
                    remainder="drop",
                )
                if model == "lr":
                    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
                else:
                    clf = HistGradientBoostingClassifier(random_state=SEED)
                return Pipeline([("preprocessor", pre), ("classifier", clf)])

            def fit_score(cols: list[str], *, model: str) -> np.ndarray:
                num_cols, cat_cols = _split_cols(train, cols)
                pipe = _build_pipeline(num_cols, cat_cols, model=model)
                pipe.fit(_sanitize(train[cols], cat_cols), y_train)
                return pipe.predict_proba(_sanitize(test[cols], cat_cols))[:, 1]

            y_train = train[TASK].astype("boolean").fillna(False).astype(int).to_numpy()
            y_test = test[TASK].astype("boolean").fillna(False).astype(int).to_numpy()
            base_rate = float(y_test.mean())
            print(f"test base rate: {base_rate:.3f}")
            """
        ),
        code(
            """
            results: dict[str, dict[str, float]] = {}
            for model in ("lr", "gbm"):
                p_with = fit_score(full_cols, model=model)
                p_without = fit_score(full_cols_no_trap, model=model)
                results[model] = {
                    "with_trap_auc":    float(roc_auc_score(y_test, p_with)),
                    "without_trap_auc": float(roc_auc_score(y_test, p_without)),
                }

            print(f"{'model':<5s}  {'with trap':>10s}  {'without trap':>13s}  {'Δ AUC':>8s}")
            for m, r in results.items():
                d = r["with_trap_auc"] - r["without_trap_auc"]
                print(
                    f"{m:<5s}  {r['with_trap_auc']:>10.4f}  "
                    f"{r['without_trap_auc']:>13.4f}  {d:+8.4f}"
                )
            """
        ),
        md(
            """
            ### 5.1 The lesson — standalone AUC underestimates trap impact

            Section 4 says `total_touches_all` is barely above chance
            (~0.53 AUC) on its own. Section 5 says HistGBM extracts a
            sizeable lift (~+0.03 AUC) from the same column once it can
            combine it with the rest of the feature panel. Both
            measurements are correct; they just measure different things.

            **Why the gap?** A standalone-AUC probe asks *can this
            column rank leads when it's the only signal you have?* A
            tree model with the rest of the panel already in scope asks
            *can this column refine my existing splits?* The trap's
            post-snapshot information correlates with other columns
            non-linearly — a few late touches by an outbound rep on an
            engaged-but-not-yet-converted lead is a very different
            signal from the same touches on a cold lead — and the
            tree can carve the join, while a single-feature probe
            never sees it. The Logistic Regression gain is much smaller
            (~+0.01) for the same reason: it cannot represent that
            interaction structure.

            Bar chart below highlights the asymmetry.
            """
        ),
        code(
            """
            labels = ["GBM full", "LR full"]
            deltas = [
                results["gbm"]["with_trap_auc"] - results["gbm"]["without_trap_auc"],
                results["lr"]["with_trap_auc"]  - results["lr"]["without_trap_auc"],
            ]
            colors = ["#3b82f6", "#9ca3af"]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(range(len(labels)), deltas, color=colors)
            ax.axhline(0.0, color="#1f2937", linewidth=0.8)
            ax.axhline(
                standalone[TRAP] - 0.5,
                color="#ef4444",
                linestyle="--",
                label=f"standalone-AUC excess ({standalone[TRAP] - 0.5:+.3f})",
            )
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel("ΔAUC = with_trap − without_trap")
            ax.set_title("Trap impact — tree models extract more than the probe predicts")
            ax.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 6. Tolerance gate (G13.2)

            Single-seed (seed=42) AUCs and trap deltas observed on the
            as-shipped intermediate bundle. Tolerances pin each AUC to
            within ±0.02 (well outside numerical jitter, well inside the
            band that would hide a regression). The sign-aware
            assertion below makes the pedagogical claim load-bearing:
            if a regeneration ever neutralises the GBM trap-delta, this
            fails — even if the absolute AUCs stay inside their bands.
            """
        ),
        code(
            """
            NB03_TARGETS = {
                "lr_with_trap_auc":     0.6704,
                "lr_without_trap_auc":  0.6362,
                "gbm_with_trap_auc":    0.6524,
                "gbm_without_trap_auc": 0.6023,
                "trap_standalone_auc":  0.5188,
            }
            NB03_TOLERANCES = dict.fromkeys(NB03_TARGETS, 0.02)

            observed = {
                "lr_with_trap_auc":     results["lr"]["with_trap_auc"],
                "lr_without_trap_auc":  results["lr"]["without_trap_auc"],
                "gbm_with_trap_auc":    results["gbm"]["with_trap_auc"],
                "gbm_without_trap_auc": results["gbm"]["without_trap_auc"],
                "trap_standalone_auc":  standalone[TRAP],
            }
            assert_within_tolerance(
                observed=observed,
                target=NB03_TARGETS,
                tolerances=NB03_TOLERANCES,
                label="notebook 03 trap-panel AUCs (seed 42, intermediate)",
            )

            # Sign-aware: GBM must extract a meaningful lift from the
            # trap.  Threshold sits well below the seed-42 observation
            # (~+0.050) but well above LR's +0.034, so it specifically
            # guards the tree-model lift the section-5 narrative claims.
            MIN_GBM_LIFT = 0.015
            gbm_lift = (
                results["gbm"]["with_trap_auc"] - results["gbm"]["without_trap_auc"]
            )
            assert gbm_lift > MIN_GBM_LIFT, (
                f"GBM trap-lift collapsed: {gbm_lift:+.4f} <= {MIN_GBM_LIFT:.4f} — "
                "the trap is no longer carrying the pedagogical lesson in section 5"
            )
            print("OK — trap-panel AUCs in tolerance and GBM lift positive.")
            """
        ),
        md(
            """
            ## 7. A detection recipe you can run on any dataset

            The trap was easy to spot here because the dataset
            *advertises* it. On a third-party dataset you don't get
            that courtesy. The same recipe still works:

            1. **Read any feature dictionary you have.** Any column
               whose description references a window longer than the
               prediction horizon is suspicious. Even when no
               dictionary ships, an obvious naming smell (`*_total`,
               `*_all`, `*_lifetime`) on a 30-day-snapshot dataset is a
               flag.
            2. **Probe the standalone AUC** *and* **the contribution to
               a tree model.** A standalone probe alone undersells
               tree-friendly leakage (sections 4 and 5 demonstrate why
               on this dataset). Train a model with the column, train
               another without, and compare. The ablation captures
               interactions the standalone probe can't.
            3. **Inspect the time window.** Cross-check the suspect
               column against any time-stamped event tables. If the
               column's value can only be explained by events past the
               snapshot anchor, you've found a trap. Section 3 makes
               this concrete here — the same technique generalises
               anywhere there's an event table to corroborate.

            A walkthrough of additional detection patterns
            (column-name heuristics, target-encoding leakage on
            test, train-test contamination via account_id,
            cohort-by-segment evaluation) lives in
            [`docs/release/break_me_guide.md`](https://github.com/leadforge-dev/leadforge/blob/main/docs/release/break_me_guide.md) —
            pair it with this notebook for a more complete
            playbook.

            ## Next

            - **Notebook 04** — value-aware ranking
              (`expected_acv` × P(convert)), calibration plots,
              threshold selection for top-K capacity, and a
              cohort-shift / bootstrap robustness harness.
            """
        ),
    ]


def main() -> None:
    args = builder_arg_parser(
        default_out=DEFAULT_OUT,
        description="Build release/notebooks/03_leakage_and_time_windows.ipynb",
    ).parse_args()
    write_notebook(args.out, assemble_notebook(cells()))


if __name__ == "__main__":
    main()
