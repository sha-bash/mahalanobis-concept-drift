"""Scatter plot utilities."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_scatter_2d(X: np.ndarray, labels: np.ndarray = None, title: str = "2D Projection") -> Figure:
    """
    Create scatter plot of 2D data.
    
    Args:
        X: 2D points (n_samples, 2)
        labels: Optional cluster labels
        title: Plot title
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is not None:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30)
        fig.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=30)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    plt.tight_layout()
    return fig