"""Tests for the leadforge mechanism layer (M6)."""

from __future__ import annotations

import json
import random

import pytest

from leadforge.mechanisms.base import MechanismAssignment, MechanismContext, MechanismSummary
from leadforge.mechanisms.categorical import CHANNEL_QUALITY_SCORES, CategoricalInfluence
from leadforge.mechanisms.counts import PoissonIntensity, RecencyDecayIntensity
from leadforge.mechanisms.hazards import ConversionHazard
from leadforge.mechanisms.influence import (
    AdditiveInfluence,
    InteractionTerm,
    LogisticInfluence,
    SaturatingInfluence,
    ThresholdInfluence,
)
from leadforge.mechanisms.measurement import NoisyCategorization, NoisyProxy, ProxyCompression
from leadforge.mechanisms.policies import assign_mechanisms, mechanism_params_for_motif
from leadforge.mechanisms.scores import LatentScore
from leadforge.mechanisms.static import BoundedNumericDraw, CategoricalDraw, MixtureDraw
from leadforge.mechanisms.transitions import HazardTransition, StageSequence
from leadforge.structure.motifs import MOTIF_FAMILY_NAMES

_LATENTS = {
    "latent_account_fit": 0.7,
    "latent_budget_readiness": 0.6,
    "latent_engagement_propensity": 0.8,
    "latent_problem_awareness": 0.5,
    "latent_contact_authority": 0.6,
    "latent_responsiveness": 0.55,
    "latent_sales_friction": 0.3,
    "latent_process_maturity": 0.5,
}
_CTX = MechanismContext(latents=_LATENTS, stage="mql", t=5)


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)  # noqa: S311


# ===========================================================================
# MechanismContext
# ===========================================================================


def test_context_defaults() -> None:
    ctx = MechanismContext()
    assert ctx.latents == {}
    assert ctx.stage is None
    assert ctx.t == 0
    assert ctx.extra == {}


# ===========================================================================
# Static mechanisms
# ===========================================================================


def test_categorical_draw_returns_valid_category() -> None:
    mech = CategoricalDraw(["a", "b", "c"], [1.0, 2.0, 1.0])
    results = {mech.sample(_CTX, _rng(i)) for i in range(50)}
    assert results <= {"a", "b", "c"}


def test_categorical_draw_weights_normalised() -> None:
    mech = CategoricalDraw(["x", "y"], [3.0, 1.0])
    assert abs(sum(mech._weights) - 1.0) < 1e-9


def test_categorical_draw_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        CategoricalDraw([], [])


def test_categorical_draw_mismatched_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        CategoricalDraw(["a"], [1.0, 2.0])


def test_categorical_draw_serialise() -> None:
    mech = CategoricalDraw(["a", "b"], [1.0, 1.0])
    d = mech.to_dict()
    assert d["name"] == "categorical_draw"
    assert set(d["categories"]) == {"a", "b"}


def test_bounded_numeric_draw_in_range() -> None:
    mech = BoundedNumericDraw(0.0, 1.0, 0.5, 0.2)
    for i in range(200):
        v = mech.sample(_CTX, _rng(i))
        assert 0.0 <= v <= 1.0


def test_bounded_numeric_draw_lo_ge_hi_raises() -> None:
    with pytest.raises(ValueError, match="lo"):
        BoundedNumericDraw(lo=1.0, hi=0.5)


def test_mixture_draw_in_range() -> None:
    mech = MixtureDraw([(0.2, 0.1), (0.8, 0.1)], [1.0, 1.0])
    for i in range(200):
        v = mech.sample(_CTX, _rng(i))
        assert 0.0 <= v <= 1.0


def test_mixture_draw_serialise_roundtrip() -> None:
    mech = MixtureDraw([(0.3, 0.15), (0.7, 0.15)], [2.0, 1.0])
    d = mech.to_dict()
    assert d["name"] == "mixture_draw"
    assert len(d["components"]) == 2
    assert abs(sum(d["mix_weights"]) - 1.0) < 1e-9


# ===========================================================================
# Influence mechanisms
# ===========================================================================


def test_additive_influence_clips_to_unit() -> None:
    mech = AdditiveInfluence({"latent_account_fit": 2.0}, bias=0.5)
    v = mech.sample(_CTX, _rng())
    assert 0.0 <= v <= 1.0


def test_logistic_influence_in_unit() -> None:
    mech = LogisticInfluence({"latent_account_fit": 3.0}, bias=-1.0)
    v = mech.sample(_CTX, _rng())
    assert 0.0 < v < 1.0


def test_logistic_influence_zero_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature"):
        LogisticInfluence({}, temperature=0.0)


def test_saturating_influence_in_unit() -> None:
    mech = SaturatingInfluence({"latent_engagement_propensity": 2.0})
    v = mech.sample(_CTX, _rng())
    assert 0.0 <= v <= 1.0


def test_threshold_influence_binary() -> None:
    mech = ThresholdInfluence({"latent_account_fit": 1.0}, threshold=0.5)
    v = mech.sample(_CTX, _rng())
    assert v in (0.0, 1.0)


def test_interaction_term_clips_to_unit() -> None:
    mech = InteractionTerm("latent_account_fit", "latent_contact_authority", weight=2.0)
    v = mech.sample(_CTX, _rng())
    assert 0.0 <= v <= 1.0


def test_influence_serialise() -> None:
    for mech in [
        AdditiveInfluence({"k": 1.0}),
        LogisticInfluence({"k": 1.0}),
        SaturatingInfluence({"k": 1.0}),
        ThresholdInfluence({"k": 1.0}),
        InteractionTerm("a", "b"),
    ]:
        d = mech.to_dict()
        assert "name" in d
        json.dumps(d)  # must be JSON-serialisable


# ===========================================================================
# Latent score
# ===========================================================================


def test_latent_score_in_unit() -> None:
    mech = LatentScore({"latent_account_fit": 2.0, "latent_sales_friction": -1.0})
    v = mech.sample(_CTX, _rng())
    assert 0.0 < v < 1.0


def test_latent_score_monotone_in_positive_key() -> None:
    mech = LatentScore({"latent_account_fit": 2.0})
    low_ctx = MechanismContext(latents={"latent_account_fit": 0.1})
    high_ctx = MechanismContext(latents={"latent_account_fit": 0.9})
    assert mech.score(low_ctx.latents) < mech.score(high_ctx.latents)


def test_latent_score_empty_weights_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        LatentScore({})


def test_latent_score_missing_key_treated_as_zero() -> None:
    mech = LatentScore({"missing_key": 1.0}, bias=0.0)
    # With only the missing key contributing 0, score = sigmoid(0) = 0.5
    assert abs(mech.score({}) - 0.5) < 1e-9


# ===========================================================================
# Conversion hazard
# ===========================================================================


def test_conversion_hazard_returns_bool() -> None:
    score = LatentScore({"latent_account_fit": 2.0}, bias=-1.5)
    hazard = ConversionHazard(score, base_rate=0.01, scale=0.05)
    result = hazard.sample(_CTX, _rng())
    assert isinstance(result, bool)


def test_conversion_hazard_probability_in_range() -> None:
    score = LatentScore({"latent_account_fit": 2.0}, bias=-1.5)
    hazard = ConversionHazard(score, base_rate=0.01, scale=0.05)
    p = hazard.daily_probability(_LATENTS)
    assert 0.0 <= p <= hazard._max_daily_rate


def test_conversion_hazard_higher_fit_higher_prob() -> None:
    score = LatentScore({"latent_account_fit": 3.0}, bias=-1.5)
    hazard = ConversionHazard(score, base_rate=0.005, scale=0.08)
    low = hazard.daily_probability({"latent_account_fit": 0.1})
    high = hazard.daily_probability({"latent_account_fit": 0.9})
    assert high > low


def test_conversion_hazard_invalid_params() -> None:
    score = LatentScore({"k": 1.0})
    with pytest.raises(ValueError, match="base_rate"):
        ConversionHazard(score, base_rate=1.5)
    with pytest.raises(ValueError, match="scale"):
        ConversionHazard(score, scale=-0.1)


def test_conversion_hazard_serialise() -> None:
    score = LatentScore({"latent_account_fit": 1.0})
    hazard = ConversionHazard(score)
    d = hazard.to_dict()
    assert d["name"] == "conversion_hazard"
    json.dumps(d)


# ===========================================================================
# Stage sequence + hazard transition
# ===========================================================================


def test_stage_sequence_next_stage() -> None:
    seq = StageSequence()
    assert seq.next_stage("mql") == "sal"
    assert seq.next_stage("sal") == "sql"
    assert seq.next_stage("closed_won") is None
    assert seq.next_stage("closed_lost") is None
    assert seq.next_stage("unknown") is None


def test_stage_sequence_is_terminal() -> None:
    seq = StageSequence()
    assert seq.is_terminal("closed_won")
    assert seq.is_terminal("closed_lost")
    assert not seq.is_terminal("mql")


def test_stage_sequence_sample_returns_next() -> None:
    seq = StageSequence()
    ctx = MechanismContext(stage="sql")
    assert seq.sample(ctx, _rng()) == "demo_scheduled"


def test_hazard_transition_returns_bool() -> None:
    score = LatentScore({"latent_engagement_propensity": 2.0})
    trans = HazardTransition(score, base_rate=0.05, scale=0.15)
    ctx = MechanismContext(latents=_LATENTS, extra={"dwell_days": 5})
    assert isinstance(trans.sample(ctx, _rng()), bool)


def test_hazard_transition_min_dwell_blocks() -> None:
    score = LatentScore({"latent_account_fit": 5.0})
    trans = HazardTransition(score, base_rate=0.99, scale=0.0, min_dwell_days=10)
    assert trans.daily_probability(_LATENTS, dwell=3) == 0.0


def test_hazard_transition_invalid_params() -> None:
    score = LatentScore({"k": 1.0})
    with pytest.raises(ValueError, match="base_rate"):
        HazardTransition(score, base_rate=-0.1)
    with pytest.raises(ValueError, match="min_dwell"):
        HazardTransition(score, min_dwell_days=-1)


def test_hazard_transition_serialise() -> None:
    score = LatentScore({"latent_account_fit": 1.0})
    trans = HazardTransition(score)
    d = trans.to_dict()
    assert d["name"] == "hazard_transition"
    json.dumps(d)


# ===========================================================================
# Count mechanisms
# ===========================================================================


def test_poisson_intensity_non_negative() -> None:
    mech = PoissonIntensity(base_rate=0.5, weights={"latent_engagement_propensity": 0.3})
    for i in range(100):
        assert mech.sample(_CTX, _rng(i)) >= 0


def test_poisson_intensity_expected_count_positive() -> None:
    mech = PoissonIntensity(base_rate=0.4)
    assert mech.expected_count(_LATENTS) > 0


def test_poisson_intensity_invalid_rate() -> None:
    with pytest.raises(ValueError, match="positive"):
        PoissonIntensity(base_rate=0.0)


def test_recency_decay_decreases_with_time() -> None:
    mech = RecencyDecayIntensity(base_rate=1.0, decay_factor=0.9)
    assert mech.expected_count(0) > mech.expected_count(10) > mech.expected_count(50)


def test_recency_decay_floor_respected() -> None:
    mech = RecencyDecayIntensity(base_rate=1.0, decay_factor=0.5, floor_rate=0.05)
    assert mech.expected_count(1000) >= 0.05


def test_recency_decay_invalid_factor() -> None:
    with pytest.raises(ValueError, match="decay_factor"):
        RecencyDecayIntensity(base_rate=1.0, decay_factor=0.0)


# ===========================================================================
# Categorical influence
# ===========================================================================


def test_categorical_influence_known_key() -> None:
    mech = CategoricalInfluence("channel", CHANNEL_QUALITY_SCORES)
    ctx = MechanismContext(extra={"channel": "partner_referral"})
    assert mech.sample(ctx, _rng()) == pytest.approx(0.70)


def test_categorical_influence_missing_key_returns_default() -> None:
    mech = CategoricalInfluence("channel", CHANNEL_QUALITY_SCORES, default=0.5)
    ctx = MechanismContext(extra={})
    assert mech.sample(ctx, _rng()) == pytest.approx(0.5)


def test_categorical_influence_unknown_value_returns_default() -> None:
    mech = CategoricalInfluence("channel", CHANNEL_QUALITY_SCORES, default=0.5)
    ctx = MechanismContext(extra={"channel": "unknown_channel"})
    assert mech.sample(ctx, _rng()) == pytest.approx(0.5)


# ===========================================================================
# Measurement mechanisms
# ===========================================================================


def test_noisy_proxy_in_unit_or_none() -> None:
    mech = NoisyProxy("latent_account_fit", noise_std=0.1, missing_rate=0.1)
    results = [mech.sample(_CTX, _rng(i)) for i in range(200)]
    non_none = [v for v in results if v is not None]
    assert all(0.0 <= v <= 1.0 for v in non_none)
    assert any(v is None for v in results)  # missingness fires at 10%


def test_noisy_proxy_zero_noise_close_to_true() -> None:
    mech = NoisyProxy("latent_account_fit", noise_std=0.0, missing_rate=0.0)
    v = mech.sample(_CTX, _rng())
    assert v == pytest.approx(_LATENTS["latent_account_fit"])


def test_noisy_proxy_invalid_params() -> None:
    with pytest.raises(ValueError, match="noise_std"):
        NoisyProxy("k", noise_std=-0.1)
    with pytest.raises(ValueError, match="missing_rate"):
        NoisyProxy("k", missing_rate=1.5)


def test_noisy_categorization_valid_category() -> None:
    cats = ["low", "medium", "high"]
    mech = NoisyCategorization("tier", cats, confusion_prob=0.1, missing_rate=0.0)
    ctx = MechanismContext(extra={"tier": "medium"})
    results = {mech.sample(ctx, _rng(i)) for i in range(100)}
    assert results <= set(cats)


def test_noisy_categorization_missing_fires() -> None:
    mech = NoisyCategorization("tier", ["a", "b"], confusion_prob=0.0, missing_rate=1.0)
    assert mech.sample(_CTX, _rng()) is None


def test_proxy_compression_correct_band() -> None:
    mech = ProxyCompression(
        "latent_account_fit",
        thresholds=[0.33, 0.67],
        labels=["low", "medium", "high"],
    )
    low_ctx = MechanismContext(latents={"latent_account_fit": 0.1})
    mid_ctx = MechanismContext(latents={"latent_account_fit": 0.5})
    high_ctx = MechanismContext(latents={"latent_account_fit": 0.9})
    assert mech.sample(low_ctx, _rng()) == "low"
    assert mech.sample(mid_ctx, _rng()) == "medium"
    assert mech.sample(high_ctx, _rng()) == "high"


def test_proxy_compression_bad_labels_count() -> None:
    with pytest.raises(ValueError, match="labels"):
        ProxyCompression("k", thresholds=[0.5], labels=["a"])


def test_proxy_compression_unsorted_thresholds() -> None:
    with pytest.raises(ValueError, match="increasing"):
        ProxyCompression("k", thresholds=[0.7, 0.3], labels=["a", "b", "c"])


# ===========================================================================
# Policies / MechanismAssignment
# ===========================================================================


@pytest.mark.parametrize("motif", MOTIF_FAMILY_NAMES)
def test_assign_mechanisms_returns_assignment(motif: str) -> None:
    assignment = assign_mechanisms(motif, _rng())
    assert isinstance(assignment, MechanismAssignment)
    assert assignment.motif_family == motif


@pytest.mark.parametrize("motif", MOTIF_FAMILY_NAMES)
def test_assignment_mechanisms_are_callable(motif: str) -> None:
    assignment = assign_mechanisms(motif, _rng())
    ctx = MechanismContext(latents=_LATENTS, stage="mql", t=3, extra={"dwell_days": 3})
    assert isinstance(assignment.conversion_hazard.sample(ctx, _rng()), bool)
    assert isinstance(assignment.stage_transition.sample(ctx, _rng()), bool)
    assert isinstance(assignment.touch_intensity.sample(ctx, _rng()), int)
    proxy = assignment.measurement.sample(ctx, _rng())
    assert proxy is None or 0.0 <= proxy <= 1.0


@pytest.mark.parametrize("motif", MOTIF_FAMILY_NAMES)
def test_assignment_summary_serialisable(motif: str) -> None:
    assignment = assign_mechanisms(motif, _rng())
    summary = assignment.summary()
    assert isinstance(summary, MechanismSummary)
    d = summary.to_dict()
    json.dumps(d)  # must not raise
    assert d["motif_family"] == motif


@pytest.mark.parametrize("motif", MOTIF_FAMILY_NAMES)
def test_assignment_summary_roundtrip(motif: str) -> None:
    assignment = assign_mechanisms(motif, _rng())
    summary = assignment.summary()
    restored = MechanismSummary.from_dict(summary.to_dict())
    assert restored.motif_family == motif
    assert restored.conversion_hazard == summary.conversion_hazard


def test_assign_mechanisms_deterministic() -> None:
    a1 = assign_mechanisms("fit_dominant", random.Random(7))  # noqa: S311
    a2 = assign_mechanisms("fit_dominant", random.Random(7))  # noqa: S311
    assert a1.summary().to_dict() == a2.summary().to_dict()


def test_assign_unknown_motif_falls_back_gracefully() -> None:
    assignment = assign_mechanisms("nonexistent_motif", _rng())
    assert assignment.motif_family == "nonexistent_motif"
    ctx = MechanismContext(latents=_LATENTS, extra={"dwell_days": 5})
    assert isinstance(assignment.conversion_hazard.sample(ctx, _rng()), bool)


def test_mechanism_params_for_motif_contains_expected_keys() -> None:
    params = mechanism_params_for_motif("fit_dominant")
    assert "conversion_score_weights" in params
    assert "hazard_params" in params
    assert "transition_score_weights" in params
    assert "touch_base_rate" in params


# ===========================================================================
# Fit-dominant vs buying-committee-friction hazard ordering
# ===========================================================================


def test_fit_dominant_higher_conversion_rate_than_friction() -> None:
    """Across a range of high-fit latent states, fit_dominant worlds should
    have higher daily conversion probability than buying_committee_friction."""
    high_fit_latents = dict(_LATENTS)
    high_fit_latents["latent_account_fit"] = 0.9
    high_fit_latents["latent_sales_friction"] = 0.2

    fit_p = []
    fric_p = []
    for seed in range(20):
        fit_asgn = assign_mechanisms("fit_dominant", random.Random(seed))  # noqa: S311
        fric_asgn = assign_mechanisms("buying_committee_friction", random.Random(seed))  # noqa: S311
        fit_p.append(fit_asgn.conversion_hazard.daily_probability(high_fit_latents))
        fric_p.append(fric_asgn.conversion_hazard.daily_probability(high_fit_latents))

    assert sum(fit_p) / len(fit_p) > sum(fric_p) / len(fric_p)
