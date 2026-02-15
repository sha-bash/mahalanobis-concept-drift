"""Drift detection utilities."""


def detect_drift(distance: float, threshold: float) -> bool:
    """
    Detect drift based on Mahalanobis distance vs threshold.
    
    Args:
        distance: Mahalanobis distance
        threshold: Drift threshold
        
    Returns:
        True if distance > threshold (drift detected), else False
    """
    return distance > threshold