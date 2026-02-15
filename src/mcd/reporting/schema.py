"""Data schemas for reporting."""

from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    predicted_label: str
    distance: float
    threshold: float
    score: float
    drift: bool