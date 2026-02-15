import numpy as np

def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Compute Mahalanobis distance."""
    diff = x - mean
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(diff.T @ inv_cov @ diff) 
