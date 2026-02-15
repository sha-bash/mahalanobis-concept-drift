"""Test artifact save/load stability."""

import tempfile
import csv
import numpy as np
from pathlib import Path
from src.mcd.io import load_labeled_tickets_csv
from src.mcd.modeling.classifier import MahalanobisDriftDetector


def test_save_load_stability():
    """Ensure save/load preserves model state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "data.csv"
        rows = [
            {"subject": "Issue A", "body": "Details A", "queue": "q1"},
            {"subject": "Issue B", "body": "Details B", "queue": "q2"},
            {"subject": "Issue C", "body": "Details C", "queue": "q1"},
            {"subject": "Issue D", "body": "Details D", "queue": "q2"},
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "body", "queue"])
            writer.writeheader()
            writer.writerows(rows)
        
        texts, labels, _, _ = load_labeled_tickets_csv(str(csv_path), "queue")
        
        # Train
        d1 = MahalanobisDriftDetector(min_cluster_size=1, threshold_quantile=0.95)
        d1.fit(texts, labels)
        
        # Multiple predictions before save
        preds_before = [d1.predict(t) for t in texts[:2]]
        
        # Save
        model_path = Path(tmpdir) / "model.joblib"
        d1.save(str(model_path))
        
        # Load
        d2 = MahalanobisDriftDetector.load(str(model_path))
        
        # Verify same predictions
        preds_after = [d2.predict(t) for t in texts[:2]]
        
        for (l1, dist1, th1, dr1), (l2, dist2, th2, dr2) in zip(preds_before, preds_after):
            assert l1 == l2, f"Labels differ: {l1} vs {l2}"
            assert abs(dist1 - dist2) < 1e-5, f"Distances differ: {dist1} vs {dist2}"
            assert abs(th1 - th2) < 1e-5, f"Thresholds differ: {th1} vs {th2}"
            assert dr1 == dr2, f"Drift flags differ: {dr1} vs {dr2}"


if __name__ == "__main__":
    test_save_load_stability()
    print("✓ Roundtrip test passed")