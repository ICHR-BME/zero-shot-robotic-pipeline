"""Pure decision rules for finished-print zero-shot inspection.

This module intentionally has no OpenCV, PyTorch, FAISS, camera, robot, or GUI
imports.  The main project can therefore unit-test the quality decision without
loading any AI model or opening hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["APPROVED", "REJECTED", "UNKNOWN"]
Reason = Literal[
    "WITHIN_QUALITY_THRESHOLD",
    "QUALITY_DISTANCE_EXCEEDED",
    "IDENTITY_DISTANCE_EXCEEDED",
    "WRONG_PART",
]


@dataclass(frozen=True)
class MatchDecision:
    """Classification produced from the nearest-reference match."""

    decision: Decision
    reason: Reason


def classify_match(
    *,
    distance: float,
    detected_part: str,
    quality_threshold: float,
    identity_threshold: float,
    expected_part: str | None = None,
) -> MatchDecision:
    """Convert one FAISS distance into a production decision.

    Lower L2 distance means a closer DINOv2 match.  The quality threshold must
    be no larger than the identity threshold:

    * distance > identity threshold -> UNKNOWN
    * recognized as a different known part -> REJECTED / WRONG_PART
    * distance > quality threshold -> REJECTED / QUALITY_DISTANCE_EXCEEDED
    * otherwise -> APPROVED

    ``UNKNOWN`` is a valid inspection result, not a software error.
    """

    if quality_threshold < 0 or identity_threshold < 0:
        raise ValueError("Inspection thresholds must be non-negative.")
    if quality_threshold > identity_threshold:
        raise ValueError(
            "quality_threshold must be less than or equal to identity_threshold."
        )
    if distance < 0:
        raise ValueError("FAISS distance must be non-negative.")
    if not detected_part:
        raise ValueError("detected_part cannot be blank.")

    if distance > identity_threshold:
        return MatchDecision("UNKNOWN", "IDENTITY_DISTANCE_EXCEEDED")

    if expected_part and detected_part != expected_part:
        return MatchDecision("REJECTED", "WRONG_PART")

    if distance > quality_threshold:
        return MatchDecision("REJECTED", "QUALITY_DISTANCE_EXCEEDED")

    return MatchDecision("APPROVED", "WITHIN_QUALITY_THRESHOLD")
