#!/usr/bin/env python3
"""Probe public relational tables for ``converted_within_90_days`` leakage.

Reproduces the ChatGPT v2 finding: in the alpha ``student_public`` bundles
under ``release/{intro,intermediate,advanced}/``, joining
``leads`` to ``opportunities`` / ``customers`` / ``subscriptions`` reconstructs
the target end-to-end.

Reports five deterministic reconstruction paths plus a bonus "public
relational only" model trained on join-derived features:

    A. Direct read of ``leads.converted_within_90_days``         (sanity floor)
    B. Opportunity outcome           (lead has any closed_won opportunity)
    C. Customer existence            (lead -> opportunities -> customers)
    D. Subscription existence        (lead -> opportunities -> customers -> subscriptions)
    E. Deterministic OR              (B OR C OR D)
    F. LR / HistGBM on join-derived features (5-fold CV AUC + AP)

Path E is the headline reconstruction; the implementation is exposed as
:func:`deterministic_relational_reconstruction` so PR 3.1 can lift it
verbatim into ``leadforge/validation/leakage_probes.py``.

Exit code: 0 on success (any probe outcome), 1 on missing tables / shape errors.

Usage::

    python scripts/probe_relational_leakage.py release/intermediate
    python scripts/probe_relational_leakage.py release/intro --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_TABLES = ("leads", "opportunities", "customers", "subscriptions")


def deterministic_relational_reconstruction(
    leads: pd.DataFrame,
    opportunities: pd.DataFrame,
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct ``converted_within_90_days`` from public relational joins.

    Returns a DataFrame indexed by ``lead_id`` with five boolean columns,
    one per reconstruction path (A-E). Path E is the union of B, C, D and
    is the headline relational-leakage prediction.

    No hidden state, no model fit — pure joins. Designed to be lifted into
    ``leadforge/validation/leakage_probes.py`` (PR 3.1) as the relational
    leakage probe.
    """
    leads_idx = leads.set_index("lead_id", drop=False)

    # Path A — the label itself, if present in public leads.
    if "converted_within_90_days" in leads.columns:
        path_a = leads_idx["converted_within_90_days"].astype(bool)
    else:
        path_a = pd.Series(False, index=leads_idx.index, name="converted_within_90_days")

    # Path B — any opportunity with close_outcome == "closed_won".
    if "close_outcome" in opportunities.columns:
        won_leads = set(
            opportunities.loc[opportunities["close_outcome"] == "closed_won", "lead_id"]
        )
    else:
        won_leads = set()
    path_b = leads_idx["lead_id"].isin(won_leads)

    # Path C — lead has any joined customer (via opportunity_id -> opportunity.lead_id).
    opp_to_lead = dict(zip(opportunities["opportunity_id"], opportunities["lead_id"], strict=True))
    customer_leads = {
        opp_to_lead[opp_id] for opp_id in customers["opportunity_id"] if opp_id in opp_to_lead
    }
    path_c = leads_idx["lead_id"].isin(customer_leads)

    # Path D — lead has any joined subscription (sub -> customer -> opportunity -> lead).
    cust_to_opp = dict(zip(customers["customer_id"], customers["opportunity_id"], strict=True))
    sub_leads: set[str] = set()
    for cust_id in subscriptions["customer_id"]:
        opp_id = cust_to_opp.get(cust_id)
        if opp_id is None:
            continue
        lead_id = opp_to_lead.get(opp_id)
        if lead_id is not None:
            sub_leads.add(lead_id)
    path_d = leads_idx["lead_id"].isin(sub_leads)

    # Path E — deterministic OR of B, C, D.
    path_e = path_b | path_c | path_d

    out = pd.DataFrame(
        {
            "path_a_direct_label": path_a.values,
            "path_b_opportunity_won": path_b.values,
            "path_c_customer_exists": path_c.values,
            "path_d_subscription_exists": path_d.values,
            "path_e_or_b_c_d": path_e.values,
        },
        index=leads_idx.index,
    )
    return out


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Accuracy / precision / recall / F1 for boolean predictions."""
    tp = int(((y_pred) & (y_true)).sum())
    fp = int(((y_pred) & (~y_true)).sum())
    fn = int(((~y_pred) & (y_true)).sum())
    tn = int(((~y_pred) & (~y_true)).sum())
    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision and recall and precision + recall)
        else float("nan")
    )
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
) -> pd.DataFrame:
    """Aggregate features per lead from public relational tables."""
    has_close_outcome = "close_outcome" in opportunities.columns
    opp_agg = (
        opportunities.groupby("lead_id")
        .agg(
            n_opps=("opportunity_id", "count"),
            max_acv=("estimated_acv", "max"),
            mean_acv=("estimated_acv", "mean"),
            any_closed_won=(
                "close_outcome",
                lambda s: int((s == "closed_won").any()) if has_close_outcome else 0,
            ),
            any_closed=(
                "close_outcome",
                lambda s: int(s.notna().any()) if has_close_outcome else 0,
            ),
        )
        .reset_index()
    )
    opp_to_lead = dict(zip(opportunities["opportunity_id"], opportunities["lead_id"], strict=True))
    customers = customers.copy()
    customers["lead_id"] = customers["opportunity_id"].map(opp_to_lead)
    cust_agg = customers.groupby("lead_id").size().rename("n_customers").reset_index()

    cust_to_opp = dict(zip(customers["customer_id"], customers["opportunity_id"], strict=True))
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
    feats = feats.fillna(
        {
            "n_opps": 0,
            "max_acv": 0.0,
            "mean_acv": 0.0,
            "any_closed_won": 0,
            "any_closed": 0,
            "n_customers": 0,
            "n_subscriptions": 0,
        }
    )
    return feats.set_index("lead_id")


def _bonus_model_metrics(
    leads: pd.DataFrame,
    opportunities: pd.DataFrame,
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    seed: int = 42,
) -> dict[str, dict[str, float]] | None:
    """5-fold CV AUC + AP for LR and HistGBM on join-derived features.

    Returns ``None`` if scikit-learn is not installed (lets the deterministic
    paths still report).
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None

    features = _build_relational_features(leads, opportunities, customers, subscriptions)
    if "converted_within_90_days" not in leads.columns:
        return None
    y = leads.set_index("lead_id")["converted_within_90_days"].astype(int).reindex(features.index)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results: dict[str, dict[str, float]] = {}

    for name, mk_model in (
        ("logistic_regression", lambda: LogisticRegression(max_iter=1000)),
        ("hist_gbm", lambda: HistGradientBoostingClassifier(random_state=seed)),
    ):
        aucs: list[float] = []
        aps: list[float] = []
        for train_idx, test_idx in skf.split(features.values, y.values):
            x_tr, x_te = features.values[train_idx], features.values[test_idx]
            y_tr, y_te = y.values[train_idx], y.values[test_idx]
            if name == "logistic_regression":
                scaler = StandardScaler()
                x_tr = scaler.fit_transform(x_tr)
                x_te = scaler.transform(x_te)
            model = mk_model()
            model.fit(x_tr, y_tr)
            proba = model.predict_proba(x_te)[:, 1]
            aucs.append(roc_auc_score(y_te, proba))
            aps.append(average_precision_score(y_te, proba))
        results[name] = {
            "auc_mean": float(sum(aucs) / len(aucs)),
            "auc_min": float(min(aucs)),
            "auc_max": float(max(aucs)),
            "ap_mean": float(sum(aps) / len(aps)),
        }
    return results


def probe_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Run every probe against a single bundle directory; return a result dict."""
    tables_dir = bundle_dir / "tables"
    if not tables_dir.is_dir():
        raise FileNotFoundError(f"missing tables/ under {bundle_dir}")
    missing = [t for t in REQUIRED_TABLES if not (tables_dir / f"{t}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"missing required tables in {tables_dir}: {missing}")

    leads = pd.read_parquet(tables_dir / "leads.parquet")
    opportunities = pd.read_parquet(tables_dir / "opportunities.parquet")
    customers = pd.read_parquet(tables_dir / "customers.parquet")
    subscriptions = pd.read_parquet(tables_dir / "subscriptions.parquet")

    paths = deterministic_relational_reconstruction(leads, opportunities, customers, subscriptions)

    if "converted_within_90_days" not in leads.columns:
        return {
            "bundle": str(bundle_dir),
            "n_leads": int(len(leads)),
            "error": "leads.parquet has no converted_within_90_days column — cannot score paths",
        }

    y_true = (
        leads.set_index("lead_id")["converted_within_90_days"].astype(bool).reindex(paths.index)
    )
    conversion_rate = float(y_true.mean())

    path_metrics: dict[str, dict[str, float]] = {}
    for col in paths.columns:
        path_metrics[col] = _binary_metrics(y_true, paths[col])

    bonus = _bonus_model_metrics(leads, opportunities, customers, subscriptions)

    return {
        "bundle": str(bundle_dir),
        "n_leads": int(len(leads)),
        "n_opportunities": int(len(opportunities)),
        "n_customers": int(len(customers)),
        "n_subscriptions": int(len(subscriptions)),
        "conversion_rate": conversion_rate,
        "deterministic_paths": path_metrics,
        "bonus_model": bonus,
        "leakage_columns_present": {
            "leads.converted_within_90_days": "converted_within_90_days" in leads.columns,
            "leads.conversion_timestamp": "conversion_timestamp" in leads.columns,
            "opportunities.close_outcome": "close_outcome" in opportunities.columns,
            "opportunities.closed_at": "closed_at" in opportunities.columns,
            "customers.parquet present": True,
            "subscriptions.parquet present": True,
        },
    }


def _print_human(result: dict[str, Any]) -> None:
    print(f"\n=== {result['bundle']} ===")
    print(
        f"n_leads={result['n_leads']} "
        f"n_opps={result.get('n_opportunities')} "
        f"n_customers={result.get('n_customers')} "
        f"n_subscriptions={result.get('n_subscriptions')} "
        f"conversion_rate={result.get('conversion_rate', float('nan')):.3f}"
    )
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print("  Deterministic reconstruction paths (vs converted_within_90_days):")
    print(f"    {'path':<32} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}")
    for path, m in result["deterministic_paths"].items():
        print(
            f"    {path:<32} {m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f}"
        )
    bonus = result["bonus_model"]
    if bonus is None:
        print("  Bonus model: scikit-learn not installed — skipped.")
    else:
        print("  Bonus model (5-fold CV on join-derived features):")
        for name, m in bonus.items():
            print(
                f"    {name:<22} AUC={m['auc_mean']:.3f} "
                f"(min={m['auc_min']:.3f} max={m['auc_max']:.3f})  "
                f"AP={m['ap_mean']:.3f}"
            )
    print("  Leakage-column presence (G4.1-G4.3 indicators):")
    for k, v in result["leakage_columns_present"].items():
        flag = "PRESENT" if v else "absent"
        print(f"    {k:<40} {flag}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help=(
            "Path to a bundle directory (must contain tables/{leads,opportunities,"
            "customers,subscriptions}.parquet)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text",
    )
    args = parser.parse_args(argv)

    try:
        result = probe_bundle(args.bundle_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
