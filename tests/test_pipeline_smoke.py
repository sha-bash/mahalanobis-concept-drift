"""Real smoke test for pipeline."""

import tempfile
import csv
import numpy as np
from pathlib import Path
from src.mcd.io import load_labeled_tickets_csv
from src.mcd.modeling.classifier import MahalanobisDriftDetector


def test_pipeline_with_synthetic_data():
    """Test fit/predict/save/load with synthetic CSV."""
    # Create temp CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_data.csv"
        
        # Write minimal CSV
        rows = [
            {"subject": "Payment issue", "body": "Cannot process my card", "queue": "billing"},
            {"subject": "Technical problem", "body": "App crashes on login", "queue": "technical"},
            {"subject": "Refund request", "body": "Need money back", "queue": "billing"},
            {"subject": "Bug report", "body": "Search function broken", "queue": "technical"},
            {"subject": "Account error", "body": "Cannot reset password", "queue": "account"},
            {"subject": "Upgrade issue", "body": "Plan change failed", "queue": "billing"},
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "body", "queue"])
            writer.writeheader()
            writer.writerows(rows)
        
        # Load
        texts, labels, label_to_idx, idx_to_label = load_labeled_tickets_csv(str(csv_path), "queue")
        assert len(texts) == 6
        assert len(labels) == 6
        assert len(label_to_idx) == 3
        
        # Train
        detector = MahalanobisDriftDetector(min_cluster_size=1)
        detector.fit(texts, labels)
        assert len(detector.cluster_means) >= 2
        
        # Predict
        pred_label, distance, threshold, is_drift = detector.predict(texts[0])
        assert isinstance(pred_label, str)
        assert isinstance(distance, (float, np.floating))
        assert isinstance(threshold, (float, np.floating))
        assert isinstance(is_drift, (bool, np.bool_))
        
        # Save/Load
        model_path = Path(tmpdir) / "test_model.joblib"
        detector.save(str(model_path))
        assert model_path.exists()
        
        loaded = MahalanobisDriftDetector.load(str(model_path))
        pred_label2, distance2, threshold2, is_drift2 = loaded.predict(texts[0])
        assert pred_label == pred_label2
        assert abs(distance - distance2) < 1e-5
        

if __name__ == "__main__":
    test_pipeline_with_synthetic_data()
    print("✓ Smoke test passed")