import numpy as np

from src.mcd.modeling.mahalanobis import mahalanobis_distance


def test_mahalanobis_distance_1d_identity_covariance() -> None:
    x = np.array([2.0])
    mean = np.array([0.0])
    cov = np.array([[1.0]])

    dist = mahalanobis_distance(x, mean, cov)

    assert np.isclose(dist, 2.0)


def test_mahalanobis_distance_matches_euclidean_for_identity_covariance() -> None:
    x = np.array([1.0, 2.0, -1.0])
    mean = np.zeros_like(x)
    cov = np.eye(3)

    dist = mahalanobis_distance(x, mean, cov)
    euclidean = np.linalg.norm(x - mean)

    assert np.isclose(dist, euclidean)
