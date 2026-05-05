#!/usr/bin/env python3
"""Probe public relational tables for ``converted_within_90_days`` leakage.

Reproduces the ChatGPT v2 finding on the alpha ``student_public`` bundles
under ``release/{intro,intermediate,advanced}/``.

The script reports four orthogonal pieces of evidence:

1. **Deterministic reconstruction paths** (no model fit, just joins):

       A. Direct read of ``leads.converted_within_90_days``
          — the label is published in cleartext.
       B. Opportunity outcome
          — lead has any opportunity with ``close_outcome == "closed_won"``.
       C. Customer existence
          — lead -> opportunities -> customers (any joined customer).
       D. Subscription existence
          — lead -> opportunities -> customers -> subscriptions.
       E. Deterministic OR (B ∨ C ∨ D)
          — the headline join-only reconstruction.

2. **Phase-2-success ablation** — same paths after simulating Phase 2's
   redaction (drop ``converted_within_90_days``/``conversion_timestamp`` from
   ``leads``; drop ``close_outcome``/``closed_at`` from ``opportunities``;
   omit ``customers`` and ``subscriptions``). This is what PR 2.1 needs to
   size against.

3. **Bonus model probes** — 5-fold CV LR + HistGBM on join-derived features.
   Reported in two variants:

       - ``with_close_outcome_aggregates`` — includes ``any_closed_won``
         (which is just Path B aggregated; trivially perfect).
       - ``without_close_outcome_aggregates`` — only counts and ACV
         aggregates. Tells us whether non-trivial relational features
         carry the leak independently of ``close_outcome``.

4. **Snapshot-window probe (G4.4)** — for each event table, count rows
   with ``timestamp > lead_created_at + horizon_days``. If horizon_days
   isn't readable from the manifest, falls back to 90.

The deterministic reconstruction is exposed as
:func:`deterministic_relational_reconstruction` so PR 3.1 can lift it
verbatim into ``leadforge/validation/leakage_probes.py``.

Exit codes:

    0 — probe ran to completion; no threshold violation
    1 — missing/unreadable required tables
    2 — a deterministic-path or bonus-model accuracy/AUC exceeded
        ``--max-accuracy`` (intended for Phase 2 CI gating)

Usage::

    python scripts/probe_relational_leakage.py release/intermediate
    python scripts/probe_relational_leakage.py release/intro --json
    python scripts/probe_relational_leakage.py release/intermediate \\
        --max-accuracy 0.65   # Phase-2 CI mode
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Re-export from the canonical package location.  PR 2.1 lifted this
# function into ``leadforge/validation/relational_leakage.py``; the
# script keeps it accessible at ``probe_module.deterministic_relational_reconstruction``
# (and at the CLI level) so callers and existing tests remain stable.
from leadforge.validation.relational_leakage import (
    deterministic_relational_reconstruction,
)

REQUIRED_TABLES = ("leads", "opportunities")
DEFAULT_HORIZON_DAYS = 90


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Accuracy / precision / recall / F1 / counts for boolean predictions.

    F1 is ``0.0`` when precision or recall is exactly ``0.0`` (well-defined zero
    skill, not undefined); ``NaN`` only when precision or recall is itself
    ``NaN`` (undefined — no positive predictions or no positives to find).
    """
    tp = int(((y_pred) & (y_true)).sum())
    fp = int(((y_pred) & (~y_true)).sum())
    fn = int(((~y_pred) & (y_true)).sum())
    tn = int(((~y_pred) & (~y_true)).sum())
    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    # F1 logic:
    #   - either precision or recall is NaN -> F1 is NaN (undefined input)
    #   - precision + recall == 0 -> F1 = 0 (defined: zero skill)
    #   - otherwise -> harmonic mean
    if math.isnan(precision) or math.isnan(recall):
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _build_relational_features(
    leads: pd.DataFrame,
    opportunities: pd.DataFrame,
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    *,
    include_close_outcome_aggregates: bool,
) -> pd.DataFrame:
    """Aggregate features per lead from public relational tables.

    When ``include_close_outcome_aggregates`` is False, the trivially-perfect
    ``any_closed_won`` / ``any_closed`` columns are omitted. The resulting
    feature set is the load-bearing "non-trivial relational" probe.
    """
    agg_kwargs: dict[str, tuple[str, str | Any]] = {
        "n_opps": ("opportunity_id", "count"),
        "max_acv": ("estimated_acv", "max"),
        "mean_acv": ("estimated_acv", "mean"),
    }
    if include_close_outcome_aggregates and "close_outcome" in opportunities.columns:
        agg_kwargs["any_closed_won"] = (
            "close_outcome",
            lambda s: int((s == "closed_won").any()),
        )
        agg_kwargs["any_closed"] = (
            "close_outcome",
            lambda s: int(s.notna().any()),
        )

    if len(opportunities) > 0:
        opp_agg = opportunities.groupby("lead_id").agg(**agg_kwargs).reset_index()
        opp_to_lead = dict(
            zip(opportunities["opportunity_id"], opportunities["lead_id"], strict=False)
        )
    else:
        opp_agg = pd.DataFrame(columns=["lead_id", *agg_kwargs.keys()])
        opp_to_lead = {}

    customers_with_lead = customers.copy()
    customers_with_lead["lead_id"] = customers_with_lead["opportunity_id"].map(opp_to_lead)
    cust_agg = customers_with_lead.groupby("lead_id").size().rename("n_customers").reset_index()

    cust_to_opp = (
        dict(zip(customers["customer_id"], customers["opportunity_id"], strict=False))
        if len(customers) > 0
        else {}
    )
    subs = subscriptions.copy()
    subs["opportunity_id"] = subs["customer_id"].map(cust_to_opp)
    subs["lead_id"] = subs["opportunity_id"].map(opp_to_lead)
    sub_agg = subs.groupby("lead_id").size().rename("n_subscriptions").reset_index()

    feats = (
        leads[["lead_id"]]
        .merge(opp_agg, on="lead_id", how="left")
        .merge(cust_agg, on="lead_id", how="left")
        .merge(sub_agg, on="lead_id", how="left")
    )
    fill_defaults = {
        "n_opps": 0,
        "max_acv": 0.0,
        "mean_acv": 0.0,
        "n_customers": 0,
        "n_subscriptions": 0,
    }
    if "any_closed_won" in feats.columns:
        fill_defaults["any_closed_won"] = 0
        fill_defaults["any_closed"] = 0
    feats = feats.fillna(fill_defaults)
    return feats.set_index("lead_id")


def _bonus_model_metrics(
    leads: pd.DataFrame,
    opportunities: pd.DataFrame,
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    *,
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]] | None:
    """5-fold CV AUC + AP for LR and HistGBM on join-derived features.

    Returns a nested dict keyed by feature-set variant
    (``with_close_outcome_aggregates`` vs ``without_close_outcome_aggregates``)
    then by model name. The "without" variant is the load-bearing probe —
    the "with" variant trivially achieves AUC 1.0 because ``any_closed_won``
    is Path B aggregated.

    Returns ``None`` if scikit-learn is unavailable.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    if "converted_within_90_days" not in leads.columns:
        return None

    out: dict[str, dict[str, dict[str, float]]] = {}
    for variant_name, include_co_agg in (
        ("with_close_outcome_aggregates", True),
        ("without_close_outcome_aggregates", False),
    ):
        features = _build_relational_features(
            leads,
            opportunities,
            customers,
            subscriptions,
            include_close_outcome_aggregates=include_co_agg,
        )
        y = (
            leads.set_index("lead_id")["converted_within_90_days"]
            .astype(int)
            .reindex(features.index)
        )

        models: dict[str, Pipeline] = {
            "logistic_regression": Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
            ),
            "hist_gbm": Pipeline([("clf", HistGradientBoostingClassifier(random_state=seed))]),
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        variant_results: dict[str, dict[str, float]] = {}
        for name, pipe in models.items():
            aucs: list[float] = []
            aps: list[float] = []
            for train_idx, test_idx in skf.split(features.values, y.values):
                x_tr, x_te = features.values[train_idx], features.values[test_idx]
                y_tr, y_te = y.values[train_idx], y.values[test_idx]
                pipe.fit(x_tr, y_tr)
                proba = pipe.predict_proba(x_te)[:, 1]
                aucs.append(roc_auc_score(y_te, proba))
                aps.append(average_precision_score(y_te, proba))
            variant_results[name] = {
                "auc_mean": float(sum(aucs) / len(aucs)),
                "auc_min": float(min(aucs)),
                "auc_max": float(max(aucs)),
                "ap_mean": float(sum(aps) / len(aps)),
                "n_features": int(features.shape[1]),
            }
        out[variant_name] = variant_results
    return out


def _phase2_success_view(
    leads: pd.DataFrame, opportunities: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate Phase-2 redaction so the same probe runs on a virtual fix.

    Drops public-leakage columns from ``leads``/``opportunities`` and
    omits ``customers``/``subscriptions`` entirely. Used to measure the
    join-only reconstruction *that PR 2.1 must defeat*, distinct from the
    trivially-perfect Path A.
    """
    drop_from_leads = [
        c for c in ("converted_within_90_days", "conversion_timestamp") if c in leads.columns
    ]
    leads_safe = leads.drop(columns=drop_from_leads)
    opps_safe = opportunities.drop(
        columns=[c for c in ("close_outcome", "closed_at") if c in opportunities.columns]
    )
    empty_customers = pd.DataFrame(
        {
            "customer_id": pd.Series(dtype=str),
            "opportunity_id": pd.Series(dtype=str),
            "account_id": pd.Series(dtype=str),
        }
    )
    empty_subscriptions = pd.DataFrame(
        {
            "subscription_id": pd.Series(dtype=str),
            "customer_id": pd.Series(dtype=str),
            "plan_name": pd.Series(dtype=str),
        }
    )
    return leads_safe, opps_safe, empty_customers, empty_subscriptions


def _g4_4_snapshot_window_probe(
    leads: pd.DataFrame,
    tables_dir: Path,
    horizon_days: int,
) -> dict[str, dict[str, int]]:
    """Per-event-table: count rows with timestamp > lead_created_at + horizon_days.

    Implements G4.4 directly. Skips tables that aren't on disk.
    """
    leads_anchor = leads[["lead_id", "lead_created_at"]].copy()
    leads_anchor["lead_created_at"] = pd.to_datetime(leads_anchor["lead_created_at"])
    horizon = pd.Timedelta(days=horizon_days)

    event_tables = (
        ("touches", "touch_timestamp"),
        ("sessions", "session_timestamp"),
        ("sales_activities", "activity_timestamp"),
        ("opportunities", "created_at"),
    )
    out: dict[str, dict[str, int]] = {}
    for tbl, ts_col in event_tables:
        path = tables_dir / f"{tbl}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["lead_id", ts_col])
        df[ts_col] = pd.to_datetime(df[ts_col])
        merged = df.merge(leads_anchor, on="lead_id", how="left")
        bad = (merged[ts_col] > merged["lead_created_at"] + horizon).sum()
        out[tbl] = {"violations": int(bad), "total_rows": int(len(df))}
    return out


def _read_horizon_days(bundle_dir: Path) -> int:
    manifest = bundle_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            return int(data.get("horizon_days") or DEFAULT_HORIZON_DAYS)
        except (json.JSONDecodeError, ValueError, TypeError):
            return DEFAULT_HORIZON_DAYS
    return DEFAULT_HORIZON_DAYS


def _read_optional_table(tables_dir: Path, name: str, columns: list[str]) -> pd.DataFrame:
    path = tables_dir / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({c: pd.Series(dtype=str) for c in columns})


def probe_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Run every probe against a single bundle directory; return a result dict.

    Tolerates missing ``customers.parquet`` / ``subscriptions.parquet`` —
    that's exactly the Phase 2 success state, and the same script must run
    cleanly on regenerated bundles.
    """
    tables_dir = bundle_dir / "tables"
    if not tables_dir.is_dir():
        raise FileNotFoundError(f"missing tables/ under {bundle_dir}")
    missing_required = [t for t in REQUIRED_TABLES if not (tables_dir / f"{t}.parquet").exists()]
    if missing_required:
        raise FileNotFoundError(f"missing required tables in {tables_dir}: {missing_required}")

    leads = pd.read_parquet(tables_dir / "leads.parquet")
    opportunities = pd.read_parquet(tables_dir / "opportunities.parquet")
    customers = _read_optional_table(
        tables_dir, "customers", ["customer_id", "opportunity_id", "account_id"]
    )
    subscriptions = _read_optional_table(
        tables_dir, "subscriptions", ["subscription_id", "customer_id", "plan_name"]
    )

    paths = deterministic_relational_reconstruction(leads, opportunities, customers, subscriptions)

    base: dict[str, Any] = {
        "bundle": str(bundle_dir),
        "n_leads": int(len(leads)),
        "n_opportunities": int(len(opportunities)),
        "n_customers": int(len(customers)),
        "n_subscriptions": int(len(subscriptions)),
        "leakage_columns_present": {
            "leads.converted_within_90_days": "converted_within_90_days" in leads.columns,
            "leads.conversion_timestamp": "conversion_timestamp" in leads.columns,
            "opportunities.close_outcome": "close_outcome" in opportunities.columns,
            "opportunities.closed_at": "closed_at" in opportunities.columns,
            "tables/customers.parquet": (tables_dir / "customers.parquet").exists(),
            "tables/subscriptions.parquet": (tables_dir / "subscriptions.parquet").exists(),
        },
    }

    horizon_days = _read_horizon_days(bundle_dir)
    base["horizon_days"] = horizon_days
    base["g4_4_snapshot_window"] = _g4_4_snapshot_window_probe(leads, tables_dir, horizon_days)

    # Path prediction rates — useful even without ground truth (post-Phase-2,
    # B/C/D should all be 0.0; non-zero would mean the redaction is incomplete).
    base["path_prediction_rates"] = {col: float(paths[col].mean()) for col in paths.columns}

    if "converted_within_90_days" not in leads.columns:
        base["note"] = (
            "leads.parquet has no converted_within_90_days column — "
            "scored metrics (accuracy/precision/recall/F1) skipped; "
            "path_prediction_rates above is the post-Phase-2 verification view."
        )
        return base

    y_true = (
        leads.set_index("lead_id")["converted_within_90_days"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    y_true = y_true.reindex(paths.index)
    base["conversion_rate"] = float(y_true.mean())

    base["deterministic_paths"] = {
        col: _binary_metrics(y_true, paths[col]) for col in paths.columns
    }

    # Phase-2 success ablation — re-run the deterministic probe under the
    # virtual redaction. Path E should drop to 0 hits if Phase 2 works.
    leads_p2, opps_p2, cust_p2, subs_p2 = _phase2_success_view(leads, opportunities)
    paths_p2 = deterministic_relational_reconstruction(leads_p2, opps_p2, cust_p2, subs_p2)
    base["phase2_ablation_paths"] = {
        col: _binary_metrics(y_true, paths_p2[col]) for col in paths_p2.columns
    }

    base["bonus_model"] = _bonus_model_metrics(leads, opportunities, customers, subscriptions)
    return base


def _format_metrics_row(name: str, m: dict[str, float], width: int = 32) -> str:
    return (
        f"    {name:<{width}} "
        f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
        f"{m['recall']:>6.3f} {m['f1']:>6.3f}"
    )


def _print_human(result: dict[str, Any]) -> None:
    print(f"\n=== {result['bundle']} ===")
    print(
        f"n_leads={result['n_leads']} "
        f"n_opps={result.get('n_opportunities')} "
        f"n_customers={result.get('n_customers')} "
        f"n_subscriptions={result.get('n_subscriptions')} "
        f"conversion_rate={result.get('conversion_rate', float('nan')):.3f} "
        f"horizon_days={result.get('horizon_days')}"
    )
    if "path_prediction_rates" in result:
        print("  Path prediction rates (fraction of leads each path flags positive):")
        for path, rate in result["path_prediction_rates"].items():
            print(f"    {path:<32} {rate:>6.3f}")

    if "note" in result:
        print(f"  {result['note']}")
        return

    print("  Deterministic reconstruction (vs converted_within_90_days):")
    print(f"    {'path':<32} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
    for path, m in result["deterministic_paths"].items():
        print(_format_metrics_row(path, m))

    print("  Phase-2-success ablation (label/close_outcome/customers/subs redacted):")
    print(f"    {'path':<32} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
    for path, m in result["phase2_ablation_paths"].items():
        print(_format_metrics_row(path, m))

    bonus = result["bonus_model"]
    if bonus is None:
        print("  Bonus model: scikit-learn not installed — skipped.")
    else:
        print("  Bonus model (5-fold CV on join-derived features):")
        for variant, models in bonus.items():
            print(f"    [{variant}]")
            for name, m in models.items():
                print(
                    f"      {name:<22} AUC={m['auc_mean']:.3f} "
                    f"(min={m['auc_min']:.3f} max={m['auc_max']:.3f})  "
                    f"AP={m['ap_mean']:.3f}  n_feat={m['n_features']}"
                )

    horizon = result["horizon_days"]
    print(f"  G4.4 snapshot-window (timestamp > lead_created_at + {horizon}d):")
    for tbl, m in result["g4_4_snapshot_window"].items():
        flag = "PASS" if m["violations"] == 0 else "FAIL"
        print(f"    {tbl:<22} {flag} ({m['violations']}/{m['total_rows']} rows past horizon)")

    print("  Leakage-column / file presence (G4.1-G4.3):")
    for k, v in result["leakage_columns_present"].items():
        flag = "PRESENT" if v else "absent"
        print(f"    {k:<40} {flag}")


def _max_observed_accuracy(result: dict[str, Any]) -> float:
    """Maximum reconstruction accuracy across deterministic and bonus probes.

    Excludes Phase-2 ablation (that's the virtual fix, not the current state).
    Used for ``--max-accuracy`` gating.
    """
    candidates: list[float] = []
    for m in result.get("deterministic_paths", {}).values():
        candidates.append(m["accuracy"])
    bonus = result.get("bonus_model") or {}
    for variant in bonus.values():
        for m in variant.values():
            candidates.append(m["auc_mean"])
    return max(candidates) if candidates else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe public relational tables for converted_within_90_days leakage"
    )
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help=(
            "Path to a bundle directory (must contain tables/{leads,opportunities}.parquet; "
            "customers/subscriptions optional)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text",
    )
    parser.add_argument(
        "--max-accuracy",
        type=float,
        default=None,
        help=(
            "Threshold for reconstruction accuracy / bonus-model AUC. "
            "Exit 2 if any probe exceeds this. Intended for Phase 2 CI gating."
        ),
    )
    args = parser.parse_args(argv)

    try:
        result = probe_bundle(args.bundle_dir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_human(result)

    if args.max_accuracy is not None:
        observed = _max_observed_accuracy(result)
        if observed > args.max_accuracy:
            print(
                f"\nGATE FAILURE: observed reconstruction = {observed:.3f} "
                f"> --max-accuracy {args.max_accuracy:.3f}",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
