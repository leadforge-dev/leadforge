"""Tests for core enums."""

from leadforge.core.enums import DifficultyProfile, ExposureMode


def test_exposure_mode_values() -> None:
    assert ExposureMode.student_public.value == "student_public"
    assert ExposureMode.research_instructor.value == "research_instructor"


def test_exposure_mode_from_string() -> None:
    assert ExposureMode("student_public") is ExposureMode.student_public
    assert ExposureMode("research_instructor") is ExposureMode.research_instructor


def test_difficulty_profile_values() -> None:
    assert DifficultyProfile.intro.value == "intro"
    assert DifficultyProfile.intermediate.value == "intermediate"
    assert DifficultyProfile.advanced.value == "advanced"


def test_difficulty_profile_from_string() -> None:
    assert DifficultyProfile("intermediate") is DifficultyProfile.intermediate
