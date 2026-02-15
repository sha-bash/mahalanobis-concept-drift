"""Covariance estimation and inversion utilities."""

import numpy as np


def estimate_covariance(X: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Estimate covariance matrix with diagonal regularization.
    
    Args:
        X: Data matrix (n_samples, n_features)
        reg: Regularization strength
        
    Returns:
        Regularized covariance matrix
    """
    cov = np.cov(X.T)
    if cov.ndim == 1:  # Handle 1D case
        cov = np.array([[cov[0]]])
    cov += reg * np.eye(cov.shape[0])
    return cov


def invert_covariance(cov: np.ndarray, reg: float = 1e-6, max_iter: int = 6) -> np.ndarray:
    """
    Invert covariance matrix with escalating regularization if needed.
    
    Args:
        cov: Covariance matrix
        reg: Initial regularization strength
        max_iter: Max iterations of escalation
        
    Returns:
        Inverted covariance matrix
    """
    current_reg = reg
    for _ in range(max_iter):
        try:
            cov_reg = cov + current_reg * np.eye(cov.shape[0])
            return np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            current_reg *= 10
    # Final attempt with very high regularization
    cov_reg = cov + (current_reg * 10) * np.eye(cov.shape[0])
    return np.linalg.inv(cov_reg)