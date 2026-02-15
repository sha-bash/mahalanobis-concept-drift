"""Dimensionality reduction utilities."""

import numpy as np
from sklearn.decomposition import PCA


def project_2d(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Project data to 2D using PCA.
    
    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of components
        
    Returns:
        Projected data (n_samples, n_components)
    """
    if X.shape[0] < 2:
        return X[:, :n_components] if X.shape[1] >= n_components else X
    pca = PCA(n_components=min(n_components, X.shape[1]))
    return pca.fit_transform(X)