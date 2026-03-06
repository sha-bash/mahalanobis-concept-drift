import logging
from typing import Dict, List, Tuple

import numpy as np

from src.mcd.embedding.sbert import SBERT
from src.mcd.modeling.covariance import estimate_covariance, invert_covariance
from src.mcd.modeling.drift import detect_drift
from src.mcd.modeling.thresholds import QuantileThresholdStrategy, ThresholdStrategy

logger = logging.getLogger(__name__)


class MahalanobisDriftDetector:
    def __init__(
        self,
        embedder=None,
        threshold_quantile: float = 0.99,
        min_cluster_size: int = 10,
        threshold_strategy: ThresholdStrategy | None = None,
    ) -> None:
        self.embedder = embedder or SBERT()
        self.label_to_index: Dict[str, int] = {}
        self.index_to_label: Dict[int, str] = {}
        self.cluster_means: List[np.ndarray] = []
        self.cluster_covs: List[np.ndarray] = []
        self.thresholds: List[float] = []
        self.regularization = 1e-6
        self.threshold_quantile = threshold_quantile
        self.min_cluster_size = min_cluster_size
        # Use provided strategy or default to empirical quantile based on threshold_quantile
        self.threshold_strategy: ThresholdStrategy = threshold_strategy or QuantileThresholdStrategy(
            quantile=threshold_quantile
        )

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """Fit the model on labeled texts."""
        unique_labels = sorted(set(labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        embeddings = self.embedder.embed(texts)
        n_clusters = len(unique_labels)

        self.cluster_means = []
        self.cluster_covs = []
        self.thresholds = []

        for i in range(n_clusters):
            cluster_texts = [texts[j] for j in range(len(texts)) if labels[j] == unique_labels[i]]
            if len(cluster_texts) < self.min_cluster_size:
                logger.warning(f"Skipping cluster {unique_labels[i]} with size {len(cluster_texts)} < {self.min_cluster_size}")
                continue
            cluster_embeddings = self.embedder.embed(cluster_texts)

            mean = np.mean(cluster_embeddings, axis=0)
            self.cluster_means.append(mean)

            cov = estimate_covariance(cluster_embeddings, self.regularization)
            self.cluster_covs.append(cov)

            distances = []
            for emb in cluster_embeddings:
                diff = emb - mean
                inv_cov = invert_covariance(cov, self.regularization)
                dist = np.sqrt(diff.T @ inv_cov @ diff)
                distances.append(dist)

            feature_dim = cluster_embeddings.shape[1]
            threshold = self.threshold_strategy.compute(distances, feature_dim=feature_dim)
            self.thresholds.append(threshold)

        logger.info(f"Fitted model with {len(self.cluster_means)} clusters")

    def predict(self, text: str) -> Tuple[str, float, float, bool]:
        """Predict cluster and detect drift for a single text."""
        embedding = self.embedder.embed([text])[0]
        
        min_dist = float('inf')
        predicted_cluster = -1
        
        for i, (mean, cov) in enumerate(zip(self.cluster_means, self.cluster_covs)):
            diff = embedding - mean
            inv_cov = invert_covariance(cov, self.regularization)
            dist = np.sqrt(diff.T @ inv_cov @ diff)
            if dist < min_dist:
                min_dist = dist
                predicted_cluster = i
        
        threshold = self.thresholds[predicted_cluster]
        is_drift = detect_drift(min_dist, threshold)
        predicted_label = self.index_to_label[predicted_cluster]
        
        return predicted_label, min_dist, threshold, is_drift

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float, float, bool]]:
        """Predict batch of texts."""
        embeddings = self.embedder.embed(texts)
        results = []
        for embedding in embeddings:
            min_dist = float('inf')
            predicted_cluster = -1
            
            for i, (mean, cov) in enumerate(zip(self.cluster_means, self.cluster_covs)):
                diff = embedding - mean
                inv_cov = invert_covariance(cov, self.regularization)
                dist = np.sqrt(diff.T @ inv_cov @ diff)
                if dist < min_dist:
                    min_dist = dist
                    predicted_cluster = i
            
            threshold = self.thresholds[predicted_cluster]
            is_drift = detect_drift(min_dist, threshold)
            predicted_label = self.index_to_label[predicted_cluster]
            
            results.append((predicted_label, min_dist, threshold, is_drift))
        return results

    def save(self, path: str) -> None:
        """Save model."""
        from src.mcd.persistence.artifacts import save_artifact, save_label_mapping
        data = {
            'label_to_index': self.label_to_index,
            'cluster_means': self.cluster_means,
            'cluster_covs': self.cluster_covs,
            'thresholds': self.thresholds,
            'regularization': self.regularization,
            'threshold_quantile': self.threshold_quantile,
            'min_cluster_size': self.min_cluster_size
        }
        save_artifact(data, path)
        mapping_path = path.replace('.joblib', '_mapping.json')
        save_label_mapping(self.label_to_index, mapping_path)

    @classmethod
    def load(cls, path: str) -> 'MahalanobisDriftDetector':
        """Load model."""
        from src.mcd.persistence.artifacts import load_artifact, load_label_mapping
        data = load_artifact(path)
        mapping_path = path.replace('.joblib', '_mapping.json')
        label_to_index = load_label_mapping(mapping_path)

        instance = cls(threshold_quantile=data.get('threshold_quantile', 0.99))
        instance.label_to_index = label_to_index
        instance.index_to_label = {v: k for k, v in label_to_index.items()}
        instance.cluster_means = data['cluster_means']
        instance.cluster_covs = data['cluster_covs']
        instance.thresholds = data['thresholds']
        instance.regularization = data['regularization']
        instance.threshold_quantile = data.get('threshold_quantile', 0.99)
        instance.min_cluster_size = data.get('min_cluster_size', 10)
        return instance