from enum import StrEnum


class ExposureMode(StrEnum):
    """Controls how much hidden world truth is published in a bundle."""

    student_public = "student_public"
    research_instructor = "research_instructor"


class DifficultyProfile(StrEnum):
    """Named difficulty preset for a generation run."""

    intro = "intro"
    intermediate = "intermediate"
    advanced = "advanced"
