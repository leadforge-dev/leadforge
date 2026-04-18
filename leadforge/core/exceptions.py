class LeadforgeError(Exception):
    """Base exception for all leadforge errors."""


class InvalidRecipeError(LeadforgeError):
    """Raised when a recipe identifier is unknown or its files are malformed."""


class InvalidConfigError(LeadforgeError):
    """Raised when a GenerationConfig fails validation."""


class GraphConstructionError(LeadforgeError):
    """Raised when the hidden world graph cannot be constructed or validated."""


class SimulationError(LeadforgeError):
    """Raised when world simulation fails."""


class RenderError(LeadforgeError):
    """Raised when bundle rendering fails."""


class ValidationError(LeadforgeError):
    """Raised when bundle artifact validation fails."""
