"""Entity ID generation.

All IDs are stable, opaque, namespace-unique, and deterministic for a given
(recipe, config, seed) triple.  Callers derive a dedicated RNG substream via
``RNGRoot.child()`` and pass a monotonically increasing counter to
:func:`make_id`.

Canonical prefixes
------------------
acct_   — Account
cnt_    — Contact
lead_   — Lead
touch_  — Touch
sess_   — Session
act_    — SalesActivity
opp_    — Opportunity
cust_   — Customer
sub_    — Subscription
rep_    — Sales rep (internal)
"""

from __future__ import annotations

# Canonical prefix registry — single source of truth used by tests and
# simulation code alike.
ID_PREFIXES: dict[str, str] = {
    "account": "acct",
    "contact": "cnt",
    "lead": "lead",
    "touch": "touch",
    "session": "sess",
    "sales_activity": "act",
    "opportunity": "opp",
    "customer": "cust",
    "subscription": "sub",
    "rep": "rep",
}

_PAD_WIDTH = 6  # e.g. acct_000001


def make_id(prefix: str, n: int) -> str:
    """Return a zero-padded entity ID string.

    Args:
        prefix: The namespace prefix (e.g. ``"acct"``).
        n: A 1-based counter for this entity type within one generation run.

    Returns:
        A string of the form ``"<prefix>_<n:06d>"``; e.g. ``"acct_000001"``.

    Raises:
        ValueError: if *n* is not a positive integer.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")
    return f"{prefix}_{n:0{_PAD_WIDTH}d}"
