"""Population generation — accounts, contacts, leads, and latent states.

:func:`build_population` is the single entry point consumed by the
simulation layer.  All randomness derives from named :class:`~leadforge.core.rng.RNGRoot`
substreams so the full population is deterministic given ``config.seed``.

Latent state
------------
Each entity carries hidden ground-truth traits that drive simulation
mechanics but are **never** directly exposed in ``student_public`` mode:

- **account** — ``latent_account_fit``, ``latent_budget_readiness``,
  ``latent_process_maturity``
- **contact** — ``latent_problem_awareness``, ``latent_contact_authority``,
  ``latent_responsiveness``, ``latent_engagement_propensity``
- **lead** — ``latent_sales_friction``

All values are floats in [0, 1] sampled from a clipped Gaussian.  The
active motif family shifts the mean of selected traits to create a
structurally coherent world (e.g. ``fit_dominant`` raises the mean of
``latent_account_fit``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from leadforge.core.exceptions import InvalidConfigError
from leadforge.core.ids import ID_PREFIXES, make_id
from leadforge.core.models import GenerationConfig
from leadforge.core.rng import RNGRoot
from leadforge.schema.entities import AccountRow, ContactRow, LeadRow

if TYPE_CHECKING:
    from leadforge.narrative.spec import NarrativeSpec
    from leadforge.structure.graph import WorldGraph


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class LatentState:
    """Hidden ground-truth latent variables for all entities in one world.

    Each mapping is ``entity_id → {trait_name: float_in_[0,1]}``.
    """

    account_latents: dict[str, dict[str, float]] = field(default_factory=dict)
    contact_latents: dict[str, dict[str, float]] = field(default_factory=dict)
    lead_latents: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class PopulationResult:
    """Output of one :func:`build_population` call."""

    accounts: list[AccountRow]
    contacts: list[ContactRow]
    leads: list[LeadRow]
    latent_state: LatentState


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_EMPLOYEE_BANDS = ("200-499", "500-999", "1000-1999", "2000+")
_EMPLOYEE_BAND_WEIGHTS = (0.35, 0.35, 0.20, 0.10)

_REVENUE_BANDS = ("$1M-$10M", "$10M-$50M", "$50M-$200M", "$200M+")
_REVENUE_BAND_WEIGHTS = (0.25, 0.40, 0.25, 0.10)

_PROCESS_MATURITY_BANDS = ("low", "medium", "high")
_PROCESS_MATURITY_BAND_WEIGHTS = (0.30, 0.45, 0.25)
_PROCESS_MATURITY_MEANS = {"low": 0.25, "medium": 0.50, "high": 0.75}

_SENIORITY_LEVELS = ("individual_contributor", "manager", "director", "vp", "c_suite")
_SENIORITY_WEIGHTS = (0.25, 0.30, 0.25, 0.15, 0.05)

_EMAIL_DOMAIN_TYPES = ("corporate", "personal", "unknown")
_EMAIL_DOMAIN_WEIGHTS = (0.80, 0.12, 0.08)

# Base reference date: all leads are created within a 30-day window starting here.
_WORLD_BASE_DATE = date(2024, 1, 1)

# Number of internal sales-rep entities used for lead assignment.
_N_REPS = 10

# Motif-family-specific additive bias on the default 0.50 latent mean.
# Only traits explicitly listed are shifted; all others stay at 0.50.
_MOTIF_LATENT_BIAS: dict[str, dict[str, float]] = {
    "fit_dominant": {
        "latent_account_fit": 0.10,
        "latent_budget_readiness": 0.05,
    },
    "intent_dominant": {
        "latent_engagement_propensity": 0.12,
        "latent_problem_awareness": 0.10,
    },
    "sales_execution_sensitive": {
        "latent_sales_friction": 0.12,
        "latent_responsiveness": -0.08,
    },
    "demo_trial_mediated": {
        "latent_engagement_propensity": 0.08,
        "latent_problem_awareness": 0.06,
    },
    "buying_committee_friction": {
        "latent_contact_authority": -0.10,
        "latent_sales_friction": 0.15,
    },
}

# Map GTM channel names → GtmMotionSpec attribute names.
_CHANNEL_TO_SHARE_ATTR: dict[str, str] = {
    "inbound_marketing": "inbound_share",
    "sdr_outbound": "outbound_share",
    "partner_referral": "partner_share",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_population(
    config: GenerationConfig,
    narrative: NarrativeSpec,
    world_graph: WorldGraph,
) -> PopulationResult:
    """Generate accounts, contacts, leads, and their latent states.

    All randomness is derived from named substreams of ``RNGRoot(config.seed)``
    so the result is fully deterministic for a given
    ``(config, narrative, world_graph.motif_family)``.

    Args:
        config: Fully resolved generation configuration (counts, seed, etc.).
        narrative: Parsed narrative spec providing ICP industries, geographies,
            personas, and GTM channel mix.  Must have non-empty
            ``market.icp_industries``, ``market.geographies``, ``personas``,
            and ``gtm_motion.channels``.
        world_graph: The sampled hidden world graph; its ``motif_family`` is used
            to apply latent-trait mean biases that make the world structurally
            coherent.

    Returns:
        A :class:`PopulationResult` containing the three entity lists and the
        full :class:`LatentState`.

    Raises:
        InvalidConfigError: If any required narrative collection is empty.
    """
    _validate_narrative(narrative)
    root = RNGRoot(config.seed)
    bias = _MOTIF_LATENT_BIAS.get(world_graph.motif_family, {})

    accounts, acct_latents = _generate_accounts(
        n=config.n_accounts,
        narrative=narrative,
        bias=bias,
        rng=root.child("population_accounts"),
    )

    contacts, cont_latents = _generate_contacts(
        n=config.n_contacts,
        accounts=accounts,
        narrative=narrative,
        bias=bias,
        rng=root.child("population_contacts"),
    )

    leads, lead_latents = _generate_leads(
        n=config.n_leads,
        contacts=contacts,
        narrative=narrative,
        bias=bias,
        rng=root.child("population_leads"),
    )

    return PopulationResult(
        accounts=accounts,
        contacts=contacts,
        leads=leads,
        latent_state=LatentState(
            account_latents=acct_latents,
            contact_latents=cont_latents,
            lead_latents=lead_latents,
        ),
    )


# ---------------------------------------------------------------------------
# Account generation
# ---------------------------------------------------------------------------


def _generate_accounts(
    n: int,
    narrative: NarrativeSpec,
    bias: dict[str, float],
    rng: random.Random,
) -> tuple[list[AccountRow], dict[str, dict[str, float]]]:
    industries = list(narrative.market.icp_industries)
    geographies = list(narrative.market.geographies)

    rows: list[AccountRow] = []
    latents: dict[str, dict[str, float]] = {}

    for i in range(1, n + 1):
        acct_id = make_id(ID_PREFIXES["account"], i)

        industry = rng.choice(industries)
        region = rng.choice(geographies)
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
            # Correlated with observable band; not directly biased by motif.
            "latent_process_maturity": _sample_latent(
                rng, _PROCESS_MATURITY_MEANS[maturity_band], std=0.15
            ),
        }

    return rows, latents


# ---------------------------------------------------------------------------
# Contact generation
# ---------------------------------------------------------------------------


def _generate_contacts(
    n: int,
    accounts: list[AccountRow],
    narrative: NarrativeSpec,
    bias: dict[str, float],
    rng: random.Random,
) -> tuple[list[ContactRow], dict[str, dict[str, float]]]:
    personas = list(narrative.personas)

    rows: list[ContactRow] = []
    latents: dict[str, dict[str, float]] = {}

    for i in range(1, n + 1):
        cnt_id = make_id(ID_PREFIXES["contact"], i)
        account = rng.choice(accounts)

        persona = rng.choice(personas)
        job_title = rng.choice(list(persona.title_variants))
        role_function = persona.role
        buyer_role = persona.decision_authority
        seniority = rng.choices(_SENIORITY_LEVELS, weights=_SENIORITY_WEIGHTS, k=1)[0]
        email_domain = rng.choices(_EMAIL_DOMAIN_TYPES, weights=_EMAIL_DOMAIN_WEIGHTS, k=1)[0]

        # Contacts are created at or shortly after their account.
        acct_date = date.fromisoformat(account.created_at)
        days_after = rng.randint(0, 30)
        created_at = (acct_date + timedelta(days=days_after)).isoformat()

        rows.append(
            ContactRow(
                contact_id=cnt_id,
                account_id=account.account_id,
                job_title=job_title,
                role_function=role_function,
                seniority=seniority,
                buyer_role=buyer_role,
                email_domain_type=email_domain,
                created_at=created_at,
            )
        )
        latents[cnt_id] = {
            "latent_problem_awareness": _sample_latent(
                rng, 0.50 + bias.get("latent_problem_awareness", 0.0)
            ),
            "latent_contact_authority": _sample_latent(
                rng, 0.50 + bias.get("latent_contact_authority", 0.0)
            ),
            "latent_responsiveness": _sample_latent(
                rng, 0.50 + bias.get("latent_responsiveness", 0.0)
            ),
            "latent_engagement_propensity": _sample_latent(
                rng, 0.50 + bias.get("latent_engagement_propensity", 0.0)
            ),
        }

    return rows, latents


# ---------------------------------------------------------------------------
# Lead generation
# ---------------------------------------------------------------------------


def _generate_leads(
    n: int,
    contacts: list[ContactRow],
    narrative: NarrativeSpec,
    bias: dict[str, float],
    rng: random.Random,
) -> tuple[list[LeadRow], dict[str, dict[str, float]]]:
    channels, channel_weights = _channel_weights(narrative)
    rep_ids = [make_id(ID_PREFIXES["rep"], i) for i in range(1, _N_REPS + 1)]

    rows: list[LeadRow] = []
    latents: dict[str, dict[str, float]] = {}

    for i in range(1, n + 1):
        lead_id = make_id(ID_PREFIXES["lead"], i)
        contact = rng.choice(contacts)

        lead_source = rng.choices(channels, weights=channel_weights, k=1)[0]
        days_offset = rng.randint(0, 29)
        lead_created_at = (_WORLD_BASE_DATE + timedelta(days=days_offset)).isoformat()
        owner_rep_id = rng.choice(rep_ids)

        rows.append(
            LeadRow(
                lead_id=lead_id,
                contact_id=contact.contact_id,
                account_id=contact.account_id,
                lead_created_at=lead_created_at,
                lead_source=lead_source,
                first_touch_channel=lead_source,
                current_stage="mql",
                owner_rep_id=owner_rep_id,
                is_mql=True,
                is_sql=False,
                converted_within_90_days=False,
                conversion_timestamp=None,
            )
        )
        latents[lead_id] = {
            "latent_sales_friction": _sample_latent(
                rng, 0.50 + bias.get("latent_sales_friction", 0.0)
            ),
        }

    return rows, latents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_narrative(narrative: NarrativeSpec) -> None:
    """Raise :exc:`InvalidConfigError` if any collection required by population
    generation is empty."""
    checks: list[tuple[object, str]] = [
        (narrative.market.icp_industries, "narrative.market.icp_industries"),
        (narrative.market.geographies, "narrative.market.geographies"),
        (narrative.personas, "narrative.personas"),
        (narrative.gtm_motion.channels, "narrative.gtm_motion.channels"),
    ]
    for collection, name in checks:
        if not collection:
            raise InvalidConfigError(f"{name} must not be empty")


def _sample_latent(rng: random.Random, mean: float = 0.50, std: float = 0.20) -> float:
    """Draw a latent trait value in [0, 1] from a clipped Gaussian."""
    mean = max(0.10, min(0.90, mean))
    return max(0.0, min(1.0, rng.gauss(mean, std)))


def _channel_weights(narrative: NarrativeSpec) -> tuple[list[str], list[float]]:
    """Return (channels, weights) lists ordered as in the GTM spec.

    If the per-channel share attributes sum to zero (all shares are 0),
    falls back to a uniform distribution so ``random.choices`` never
    receives an all-zero weight list.
    """
    gtm = narrative.gtm_motion
    channels: list[str] = []
    weights: list[float] = []
    for ch in gtm.channels:
        attr = _CHANNEL_TO_SHARE_ATTR.get(ch)
        channels.append(ch)
        weights.append(float(getattr(gtm, attr)) if attr else 0.0)
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    else:
        # All shares are zero — fall back to uniform.
        uniform = 1.0 / len(channels)
        weights = [uniform] * len(channels)
    return channels, weights
