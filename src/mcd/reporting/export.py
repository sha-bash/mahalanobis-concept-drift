"""Export utilities for predictions and metrics."""

import json
import csv
from typing import List, Dict, Any


def save_predictions_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Save predictions to CSV.
    
    Args:
        path: Output CSV path
        rows: List of prediction dicts
    """
    if not rows:
        return
    keys = rows[0].keys()
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_json(path: str, payload: Dict[str, Any]) -> None:
    """
    Save metrics to JSON.
    
    Args:
        path: Output JSON path
        payload: Metrics dict
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)