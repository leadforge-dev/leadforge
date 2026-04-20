"""Typed narrative specification models.

A ``NarrativeSpec`` is the fully parsed, validated in-memory representation of a
recipe's ``narrative.yaml``.  Every downstream layer (schema, simulation,
rendering) anchors to these objects rather than raw YAML dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from leadforge.core.exceptions import InvalidRecipeError

# ---------------------------------------------------------------------------
# Leaf dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompanySpec:
    """The simulated vendor company."""

    name: str
    founded_year: int
    hq_city: str
    hq_country: str
    stage: str
    employee_range: tuple[int, int]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompanySpec:
        _require_keys(
            data,
            {"name", "founded_year", "hq_city", "hq_country", "stage", "employee_range"},
            "company",
        )
        er = data["employee_range"]
        if not (
            isinstance(er, (list, tuple))
            and len(er) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in er)
        ):
            raise InvalidRecipeError(
                f"company.employee_range must be a [min, max] int pair, got {er!r}"
            )
        return cls(
            name=str(data["name"]),
            founded_year=_pos_int(data["founded_year"], "company.founded_year"),
            hq_city=str(data["hq_city"]),
            hq_country=str(data["hq_country"]),
            stage=str(data["stage"]),
            employee_range=(int(er[0]), int(er[1])),
        )


@dataclass(frozen=True)
class ProductSpec:
    """The simulated product being sold."""

    name: str
    category: str
    deployment: str
    pricing_model: str
    acv_range_usd: tuple[int, int]
    contract_terms_months: tuple[int, ...]
    free_trial_available: bool
    demo_available: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProductSpec:
        _require_keys(
            data,
            {
                "name",
                "category",
                "deployment",
                "pricing_model",
                "acv_range_usd",
                "contract_terms_months",
                "free_trial_available",
                "demo_available",
            },
            "product",
        )
        acv = data["acv_range_usd"]
        if not (
            isinstance(acv, (list, tuple))
            and len(acv) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in acv)
        ):
            raise InvalidRecipeError(
                f"product.acv_range_usd must be a [min, max] int pair, got {acv!r}"
            )
        terms = data["contract_terms_months"]
        if not isinstance(terms, (list, tuple)) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in terms
        ):
            raise InvalidRecipeError(
                f"product.contract_terms_months must be a list of ints, got {terms!r}"
            )
        return cls(
            name=str(data["name"]),
            category=str(data["category"]),
            deployment=str(data["deployment"]),
            pricing_model=str(data["pricing_model"]),
            acv_range_usd=(int(acv[0]), int(acv[1])),
            contract_terms_months=tuple(int(t) for t in terms),
            free_trial_available=bool(data["free_trial_available"]),
            demo_available=bool(data["demo_available"]),
        )


@dataclass(frozen=True)
class MarketSpec:
    """The target market definition."""

    icp_employee_range: tuple[int, int]
    icp_industries: tuple[str, ...]
    geographies: tuple[str, ...]
    avg_deal_size_usd: int
    avg_sales_cycle_days: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarketSpec:
        _require_keys(
            data,
            {
                "icp_employee_range",
                "icp_industries",
                "geographies",
                "avg_deal_size_usd",
                "avg_sales_cycle_days",
            },
            "market",
        )
        er = data["icp_employee_range"]
        if not (
            isinstance(er, (list, tuple))
            and len(er) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in er)
        ):
            raise InvalidRecipeError(
                f"market.icp_employee_range must be a [min, max] int pair, got {er!r}"
            )
        return cls(
            icp_employee_range=(int(er[0]), int(er[1])),
            icp_industries=tuple(str(i) for i in data["icp_industries"]),
            geographies=tuple(str(g) for g in data["geographies"]),
            avg_deal_size_usd=_pos_int(data["avg_deal_size_usd"], "market.avg_deal_size_usd"),
            avg_sales_cycle_days=_pos_int(
                data["avg_sales_cycle_days"], "market.avg_sales_cycle_days"
            ),
        )


@dataclass(frozen=True)
class GtmMotionSpec:
    """Go-to-market channels and approximate share mix."""

    channels: tuple[str, ...]
    inbound_share: float
    outbound_share: float
    partner_share: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GtmMotionSpec:
        _require_keys(
            data,
            {"channels", "inbound_share", "outbound_share", "partner_share"},
            "gtm_motion",
        )
        return cls(
            channels=tuple(str(c) for c in data["channels"]),
            inbound_share=float(data["inbound_share"]),
            outbound_share=float(data["outbound_share"]),
            partner_share=float(data["partner_share"]),
        )


@dataclass(frozen=True)
class PersonaSpec:
    """A buyer persona present in the simulated market."""

    role: str
    title_variants: tuple[str, ...]
    decision_authority: str
    typical_involvement: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonaSpec:
        _require_keys(
            data,
            {"role", "title_variants", "decision_authority", "typical_involvement"},
            "personas[]",
        )
        return cls(
            role=str(data["role"]),
            title_variants=tuple(str(t) for t in data["title_variants"]),
            decision_authority=str(data["decision_authority"]),
            typical_involvement=str(data["typical_involvement"]),
        )


@dataclass(frozen=True)
class FunnelStageSpec:
    """A single named stage in the sales funnel."""

    name: str
    label: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FunnelStageSpec:
        _require_keys(data, {"name", "label"}, "funnel_stages[]")
        return cls(name=str(data["name"]), label=str(data["label"]))


# ---------------------------------------------------------------------------
# Root spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NarrativeSpec:
    """Complete parsed narrative for one generation run."""

    company: CompanySpec
    product: ProductSpec
    market: MarketSpec
    gtm_motion: GtmMotionSpec
    personas: tuple[PersonaSpec, ...]
    funnel_stages: tuple[FunnelStageSpec, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NarrativeSpec:
        """Parse and validate a raw narrative YAML payload.

        Raises:
            InvalidRecipeError: on missing keys, wrong types, or invalid values.
        """
        _require_keys(
            data,
            {"company", "product", "market", "gtm_motion", "personas", "funnel_stages"},
            "narrative",
        )
        personas_raw = data["personas"]
        if not isinstance(personas_raw, list):
            raise InvalidRecipeError(
                f"narrative.personas must be a list, got {type(personas_raw).__name__!r}"
            )
        funnel_raw = data["funnel_stages"]
        if not isinstance(funnel_raw, list):
            raise InvalidRecipeError(
                f"narrative.funnel_stages must be a list, got {type(funnel_raw).__name__!r}"
            )

        return cls(
            company=CompanySpec.from_dict(data["company"]),
            product=ProductSpec.from_dict(data["product"]),
            market=MarketSpec.from_dict(data["market"]),
            gtm_motion=GtmMotionSpec.from_dict(data["gtm_motion"]),
            personas=tuple(PersonaSpec.from_dict(p) for p in personas_raw),
            funnel_stages=tuple(FunnelStageSpec.from_dict(s) for s in funnel_raw),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_keys(data: dict[str, Any], required: set[str], context: str) -> None:
    missing = required - data.keys()
    if missing:
        raise InvalidRecipeError(
            f"Narrative section '{context}' is missing required keys: {sorted(missing)}"
        )


def _pos_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRecipeError(f"'{name}' must be a positive int, got {type(value).__name__!r}")
    if value <= 0:
        raise InvalidRecipeError(f"'{name}' must be positive, got {value}")
    return int(value)
