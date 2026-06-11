"""Lifecycle customer population generation.

:func:`build_customer_population` is the single public entry point.  It
produces a cohort of customers (accounts + lifecycle customer entities +
latent state) for the pLTV simulation.

Design decisions reflected here
---------------------------------
**D3 (independent generation)**: customers are generated self-contained — not
derived from a lead-scoring bundle's converted leads.  A seam for future
chained generation exists via the nullable ``opportunity_id`` field on
:class:`~leadforge.schemes.lifecycle.entities.CustomerLifecycleRow`.

**D4 (staggered start dates + fixed observation date)**: customers are acquired
across an ``acquisition_window_weeks`` window ending at the absolute
``observation_date``.  Each customer receives a uniformly-sampled start date
within that window, so tenure-at-observation naturally varies from near-zero
(cold-start) to the full window length.

RNG substreams
--------------
All randomness derives from two named :class:`~leadforge.core.rng.RNGRoot`
substreams so population and latent draws are independently stable:

- ``lifecycle_population_accounts`` — account entity generation.
- ``lifecycle_population_customers`` — customer entity + latent generation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from leadforge.core.ids import ID_PREFIXES, make_id
from leadforge.core.rng import RNGRoot
from leadforge.schema.entities import AccountRow
from leadforge.schemes.lifecycle.entities import CustomerLifecycleRow

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class CustomerLatentState:
    """Hidden ground-truth latent variables for the lifecycle population.

    Each mapping is ``entity_id → {trait_name: float_in_[0,1]}``.
    """

    account_latents: dict[str, dict[str, float]] = field(default_factory=dict)
    customer_latents: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class CustomerPopulationResult:
    """Output of one :func:`build_customer_population` call."""

    accounts: list[AccountRow]
    customers: list[CustomerLifecycleRow]
    latent_state: CustomerLatentState
    # ISO-8601 date at which snapshots and labels are anchored.
    observation_date: str = ""


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Shared with the lead-scoring account generator so account firmographics are
# drawn from the same distribution (accounts are the same entity in both worlds).
_EMPLOYEE_BANDS = ("200-499", "500-999", "1000-1999", "2000+")
_EMPLOYEE_BAND_WEIGHTS = (0.35, 0.35, 0.20, 0.10)

_REVENUE_BANDS = ("$1M-$10M", "$10M-$50M", "$50M-$200M", "$200M+")
_REVENUE_BAND_WEIGHTS = (0.25, 0.40, 0.25, 0.10)

_PROCESS_MATURITY_BANDS = ("low", "medium", "high")
_PROCESS_MATURITY_BAND_WEIGHTS = (0.30, 0.45, 0.25)
_PROCESS_MATURITY_MEANS = {"low": 0.25, "medium": 0.50, "high": 0.75}

# Industries drawn from the ICP defined in the procurement narrative.
_ICP_INDUSTRIES = (
    "manufacturing",
    "logistics",
    "professional_services",
    "healthcare_non_clinical",
)

_GEOGRAPHIES = ("US", "UK")

# Subscription plans with MRR ranges (USD/month) indexed by employee band.
# Larger accounts tend to land on higher-ACV plans.
_PLAN_BY_EMPLOYEE_BAND: dict[str, tuple[str, ...]] = {
    "200-499": ("starter", "starter", "growth"),
    "500-999": ("starter", "growth", "growth"),
    "1000-1999": ("growth", "growth", "enterprise"),
    "2000+": ("growth", "enterprise", "enterprise"),
}

_MRR_RANGE_BY_PLAN: dict[str, tuple[int, int]] = {
    "starter": (1_000, 3_500),
    "growth": (3_500, 9_000),
    "enterprise": (9_000, 25_000),
}

_CONTRACT_TERMS_MONTHS = (12, 24)
_CONTRACT_TERM_WEIGHTS = (0.65, 0.35)

# Number of CSM reps assigned to customers.
_N_CSMS = 8

# Calendar base: observation date is derived relative to this anchor.
# Matches the lead-scoring world base date so any future cohort-linking
# remains temporally coherent.
_WORLD_BASE_DATE = date(2024, 1, 1)

# Default acquisition window in weeks before the observation date.
_DEFAULT_ACQUISITION_WINDOW_WEEKS = 52

# Motif-family-specific additive bias on the 0.50 latent mean.
# Five retention motif families (see docs/ltv/design.md §6.1).
_MOTIF_LATENT_BIAS: dict[str, dict[str, float]] = {
    "product_led_retention": {
        "latent_product_fit": 0.12,
        "latent_adoption_velocity": 0.06,
    },
    "relationship_led_retention": {
        "latent_champion_strength": 0.14,
        "latent_organizational_stability": 0.06,
    },
    "expansion_led_growth": {
        "latent_adoption_velocity": 0.16,
        "latent_product_fit": 0.06,
    },
    "payment_fragile": {
        "latent_budget_stability": -0.18,
        "latent_organizational_stability": -0.06,
    },
    "churner_dominated": {
        "latent_product_fit": -0.14,
        "latent_champion_strength": -0.10,
    },
}

LIFECYCLE_MOTIF_FAMILIES: tuple[str, ...] = tuple(_MOTIF_LATENT_BIAS.keys())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_customer_population(
    n_customers: int,
    seed: int,
    motif_family: str = "product_led_retention",
    *,
    n_accounts: int | None = None,
    observation_date: str | None = None,
    acquisition_window_weeks: int = _DEFAULT_ACQUISITION_WINDOW_WEEKS,
) -> CustomerPopulationResult:
    """Generate accounts and lifecycle customers with their latent states.

    All randomness is derived from named :class:`~leadforge.core.rng.RNGRoot`
    substreams, making the result fully deterministic for a given
    ``(n_customers, seed, motif_family, n_accounts, observation_date,
    acquisition_window_weeks)``.

    Args:
        n_customers: Number of customer entities to generate.
        seed: Master RNG seed.
        motif_family: One of the five lifecycle retention motif families
            (see :data:`LIFECYCLE_MOTIF_FAMILIES`).  Controls the mean of
            latent traits, making the simulated world structurally coherent.
        n_accounts: Number of account entities to generate.  Defaults to
            ``max(n_customers // 3, 1)`` — on average ~3 customers per
            account, reflecting enterprise B2B upsell / multi-product.
        observation_date: ISO-8601 date at which the snapshot and labels are
            anchored (the "as-of" date for the pLTV model).  Defaults to
            ``_WORLD_BASE_DATE + 1 year``.
        acquisition_window_weeks: Width of the customer acquisition window
            (weeks before ``observation_date``).  Customer start dates are
            sampled uniformly within this window, producing the tenure
            variation needed for a realistic cold-start subpopulation.

    Returns:
        A :class:`CustomerPopulationResult` containing the account list,
        customer list, and latent state.

    Raises:
        ValueError: if ``motif_family`` is not one of the registered families.
    """
    if motif_family not in _MOTIF_LATENT_BIAS:
        raise ValueError(
            f"Unknown lifecycle motif family {motif_family!r}. "
            f"Valid families: {sorted(_MOTIF_LATENT_BIAS)}"
        )

    if n_accounts is None:
        n_accounts = max(n_customers // 3, 1)

    obs_date: date
    if observation_date is None:
        obs_date = _WORLD_BASE_DATE + timedelta(weeks=acquisition_window_weeks + 4)
    else:
        obs_date = date.fromisoformat(observation_date)

    acq_start: date = obs_date - timedelta(weeks=acquisition_window_weeks)

    root = RNGRoot(seed)
    bias = _MOTIF_LATENT_BIAS.get(motif_family, {})

    accounts, acct_latents = _generate_accounts(
        n=n_accounts,
        bias=bias,
        rng=root.child("lifecycle_population_accounts"),
    )

    customers, cust_latents = _generate_customers(
        n=n_customers,
        accounts=accounts,
        bias=bias,
        acq_start=acq_start,
        obs_date=obs_date,
        rng=root.child("lifecycle_population_customers"),
    )

    return CustomerPopulationResult(
        accounts=accounts,
        customers=customers,
        latent_state=CustomerLatentState(
            account_latents=acct_latents,
            customer_latents=cust_latents,
        ),
        observation_date=obs_date.isoformat(),
    )


# ---------------------------------------------------------------------------
# Account generation
# ---------------------------------------------------------------------------


def _generate_accounts(
    n: int,
    bias: dict[str, float],
    rng: random.Random,
) -> tuple[list[AccountRow], dict[str, dict[str, float]]]:
    """Generate *n* account entities with latent traits."""
    rows: list[AccountRow] = []
    latents: dict[str, dict[str, float]] = {}

    for i in range(1, n + 1):
        acct_id = make_id(ID_PREFIXES["account"], i)
        industry = rng.choice(_ICP_INDUSTRIES)
        region = rng.choice(_GEOGRAPHIES)
        employee_band = rng.choices(_EMPLOYEE_BANDS, weights=_EMPLOYEE_BAND_WEIGHTS, k=1)[0]
        revenue_band = rng.choices(_REVENUE_BANDS, weights=_REVENUE_BAND_WEIGHTS, k=1)[0]
        maturity_band = rng.choices(
            _PROCESS_MATURITY_BANDS, weights=_PROCESS_MATURITY_BAND_WEIGHTS, k=1
        )[0]
        days_before = rng.randint(30, 730)
        created_at = (_WORLD_BASE_DATE - timedelta(days=days_before)).isoformat()

        rows.append(
            AccountRow(
                account_id=acct_id,
                company_name=f"Company {acct_id}",
                industry=industry,
                region=region,
                employee_band=employee_band,
                estimated_revenue_band=revenue_band,
                process_maturity_band=maturity_band,
                created_at=created_at,
            )
        )
        latents[acct_id] = {
            "latent_account_fit": _sample_latent(rng, 0.50 + bias.get("latent_account_fit", 0.0)),
            "latent_budget_readiness": _sample_latent(
                rng, 0.50 + bias.get("latent_budget_readiness", 0.0)
            ),
            "latent_process_maturity": _sample_latent(
                rng, _PROCESS_MATURITY_MEANS[maturity_band], std=0.15
            ),
        }

    return rows, latents


# ---------------------------------------------------------------------------
# Customer generation
# ---------------------------------------------------------------------------


def _generate_customers(
    n: int,
    accounts: list[AccountRow],
    bias: dict[str, float],
    acq_start: date,
    obs_date: date,
    rng: random.Random,
) -> tuple[list[CustomerLifecycleRow], dict[str, dict[str, float]]]:
    """Generate *n* lifecycle customer entities with latent traits.

    Customer start dates are sampled uniformly in ``[acq_start, obs_date)``,
    realising the staggered-start design (D4).  The nullable ``opportunity_id``
    is left ``None`` (independent generation, D3); it exists as the seam for
    future chained generation from a lead-scoring bundle.
    """
    acq_span_days = (obs_date - acq_start).days
    csm_ids = [make_id("rep", i) for i in range(1, _N_CSMS + 1)]

    rows: list[CustomerLifecycleRow] = []
    latents: dict[str, dict[str, float]] = {}

    for i in range(1, n + 1):
        cust_id = make_id(ID_PREFIXES["customer"], i)
        account = rng.choice(accounts)

        # Staggered start date: uniform within the acquisition window.
        days_offset = rng.randint(0, max(acq_span_days - 1, 0))
        start = acq_start + timedelta(days=days_offset)

        plan = rng.choice(_PLAN_BY_EMPLOYEE_BAND.get(account.employee_band, ("growth",)))
        mrr_lo, mrr_hi = _MRR_RANGE_BY_PLAN[plan]
        initial_mrr = rng.randint(mrr_lo, mrr_hi)
        contract_months = rng.choices(_CONTRACT_TERMS_MONTHS, weights=_CONTRACT_TERM_WEIGHTS, k=1)[
            0
        ]
        csm_rep = rng.choice(csm_ids)

        rows.append(
            CustomerLifecycleRow(
                customer_id=cust_id,
                account_id=account.account_id,
                customer_start_at=start.isoformat(),
                initial_plan=plan,
                initial_mrr=initial_mrr,
                contract_term_months=contract_months,
                csm_rep_id=csm_rep,
                opportunity_id=None,  # seam for future chaining (D3)
            )
        )
        latents[cust_id] = {
            "latent_product_fit": _sample_latent(rng, 0.50 + bias.get("latent_product_fit", 0.0)),
            "latent_adoption_velocity": _sample_latent(
                rng, 0.50 + bias.get("latent_adoption_velocity", 0.0)
            ),
            "latent_budget_stability": _sample_latent(
                rng, 0.50 + bias.get("latent_budget_stability", 0.0)
            ),
            "latent_champion_strength": _sample_latent(
                rng, 0.50 + bias.get("latent_champion_strength", 0.0)
            ),
            "latent_organizational_stability": _sample_latent(
                rng, 0.50 + bias.get("latent_organizational_stability", 0.0)
            ),
        }

    return rows, latents


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sample_latent(rng: random.Random, mean: float = 0.50, std: float = 0.20) -> float:
    """Draw a latent trait value in [0, 1] from a clipped Gaussian."""
    mean = max(0.10, min(0.90, mean))
    return max(0.0, min(1.0, rng.gauss(mean, std)))
