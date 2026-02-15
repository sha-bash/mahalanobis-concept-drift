import numpy as np
from typing import List

def compute_thresholds(distances: List[float], quantile: float = 0.95) -> float:
    """Compute threshold as quantile of distances."""
    return np.quantile(distances, quantile) 
