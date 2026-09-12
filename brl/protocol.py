"""Public simulator protocol constants shared by local validation and policies."""
from __future__ import annotations

OFFICIAL_BEARING_DECIMALS = 2
OFFICIAL_BEARING_ERROR_DEG = 1.0
# The exact rounding allowance is 0.005 degrees; round the envelope upward to
# 1.01 degrees so geometric containment does not depend on a boundary ulp.
OFFICIAL_BEARING_ENVELOPE_DEG = 1.01


class ActionIOError(RuntimeError):
    """A transport/protocol failure that policy fallbacks must not consume."""


class GeometryNumericalError(RuntimeError):
    """A numerical geometry failure for which the certified fallback is valid."""


class CertificateViolation(RuntimeError):
    """Observed feedback contradicts a claimed geometric clear certificate."""


class DeadlineExceeded(ActionIOError):
    """The official real-time budget can no longer safely accept an action."""


def quantize_bearing_deg(value: float, decimals: int = OFFICIAL_BEARING_DECIMALS) -> float:
    """Match the official two-decimal bearing response while preserving [0, 360)."""
    return float(round(float(value), int(decimals)) % 360.0)
