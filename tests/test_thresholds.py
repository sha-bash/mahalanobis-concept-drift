import numpy as np

from src.mcd.modeling.thresholds import (
    ChiSquareThresholdStrategy,
    QuantileThresholdStrategy,
)


def test_quantile_threshold_monotonic_in_quantile() -> None:
    distances = np.linspace(0.0, 10.0, num=101)

    low = QuantileThresholdStrategy(quantile=0.5).compute(distances, feature_dim=1)
    high = QuantileThresholdStrategy(quantile=0.9).compute(distances, feature_dim=1)

    assert low <= high


def test_quantile_threshold_approximates_fraction_of_points() -> None:
    rng = np.random.default_rng(seed=42)
    distances = rng.normal(loc=0.0, scale=1.0, size=10_000)
    distances = np.abs(distances)

    strategy = QuantileThresholdStrategy(quantile=0.9)
    threshold = strategy.compute(distances, feature_dim=1)

    fraction_below = np.mean(distances <= threshold)

    assert 0.87 <= fraction_below <= 0.93


def test_chi_square_threshold_increases_with_alpha() -> None:
    distances = [0.0, 1.0, 2.0]  # not used by strategy, but required by interface
    dim = 5

    low = ChiSquareThresholdStrategy(alpha=0.9).compute(distances, feature_dim=dim)
    high = ChiSquareThresholdStrategy(alpha=0.99).compute(distances, feature_dim=dim)

    assert low < high
