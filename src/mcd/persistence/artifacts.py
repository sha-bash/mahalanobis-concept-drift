import joblib
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def save_artifact(data: Any, path: str) -> None:
    """Save artifact using joblib."""
    try:
        joblib.dump(data, path)
        logger.info(f"Saved artifact to {path}")
    except Exception as e:
        logger.error(f"Failed to save artifact to {path}: {e}")
        raise

def load_artifact(path: str) -> Any:
    """Load artifact using joblib."""
    try:
        data = joblib.load(path)
        logger.info(f"Loaded artifact from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load artifact from {path}: {e}")
        raise

def save_label_mapping(mapping: Dict[str, int], path: str) -> None:
    """Save label to index mapping as JSON."""
    try:
        with open(path, 'w') as f:
            json.dump(mapping, f, indent=2)
        logger.info(f"Saved label mapping to {path}")
    except Exception as e:
        logger.error(f"Failed to save label mapping to {path}: {e}")
        raise

def load_label_mapping(path: str) -> Dict[str, int]:
    """Load label to index mapping from JSON."""
    try:
        with open(path, 'r') as f:
            mapping = json.load(f)
        logger.info(f"Loaded label mapping from {path}")
        return mapping
    except Exception as e:
        logger.error(f"Failed to load label mapping from {path}: {e}")
        raise 
