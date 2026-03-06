from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2


class ThresholdStrategy(Protocol):
    """Strategy interface for computing drift thresholds from Mahalanobis distances."""

    def compute(self, distances: NDArray[np.float64] | Sequence[float], *, feature_dim: int) -> float:
        """Compute a distance threshold given observed distances and feature space dimension."""
        ...


@dataclass(frozen=True)
class QuantileThresholdStrategy:
    """Empirical quantile-based threshold strategy."""

    quantile: float = 0.95

    def compute(self, distances: NDArray[np.float64] | Sequence[float], *, feature_dim: int) -> float:
        if not 0.0 < self.quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {self.quantile}")

        arr = np.asarray(distances, dtype=float)
        if arr.ndim != 1:
            raise ValueError("distances must be a 1D array")
        if arr.size == 0:
            raise ValueError("distances must not be empty")

        return float(np.quantile(arr, self.quantile))


@dataclass(frozen=True)
class ChiSquareThresholdStrategy:
    """Analytical chi-square-based threshold for Mahalanobis distances.

    Under Gaussian assumptions, the squared Mahalanobis distance follows a chi-square
    distribution with df equal to the feature dimension. This strategy computes a
    distance threshold corresponding to a given upper-tail probability.
    """

    alpha: float = 0.99

    def compute(self, distances: NDArray[np.float64] | Sequence[float], *, feature_dim: int) -> float:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        threshold_sq = chi2.ppf(self.alpha, df=feature_dim)
        if not np.isfinite(threshold_sq):
            raise ValueError("failed to compute finite chi-square quantile")

        return float(np.sqrt(threshold_sq))


def compute_thresholds(distances: Sequence[float], quantile: float = 0.95) -> float:
    """Backward-compatible helper: compute threshold as empirical quantile of distances."""
    strategy = QuantileThresholdStrategy(quantile=quantile)
    # feature_dim is unused for quantile-based strategy, but required by the protocol
    return strategy.compute(distances, feature_dim=1)
