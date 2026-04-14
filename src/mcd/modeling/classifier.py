from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from src.mcd.embedding.sbert import SBERT
from src.mcd.modeling.covariance import estimate_covariance, invert_covariance
from src.mcd.preprocessing import normalize_ticket_input, preprocess_text
from src.mcd.modeling.drift import detect_drift
from src.mcd.modeling.thresholds import QuantileThresholdStrategy, ThresholdStrategy

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

_DEFAULT_PROJECT_BATCH = 512


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required when using a projector. Install with: pip install torch"
        ) from e
    return torch


class MahalanobisDriftDetector:
    def __init__(
        self,
        embedder: Any = None,
        threshold_quantile: float = 0.99,
        min_cluster_size: int = 10,
        threshold_strategy: ThresholdStrategy | None = None,
        projector: nn.Module | None = None,
        projector_batch_size: int = _DEFAULT_PROJECT_BATCH,
        regularization: float = 1e-6,
    ) -> None:
        self.embedder = embedder or SBERT()
        self.label_to_index: Dict[str, int] = {}
        self.index_to_label: Dict[int, str] = {}
        self.cluster_means: List[NDArray[np.float64]] = []
        self.cluster_covs: List[NDArray[np.float64]] = []
        self.thresholds: List[float] = []
        self.regularization = float(regularization)
        self.threshold_quantile = threshold_quantile
        self.min_cluster_size = min_cluster_size
        self.threshold_strategy: ThresholdStrategy = threshold_strategy or QuantileThresholdStrategy(
            quantile=threshold_quantile
        )
        self.projector = projector
        self.projector_batch_size = projector_batch_size

    def _project_numpy_embeddings(self, embeddings: NDArray[Any]) -> NDArray[np.float64]:
        """Map embeddings through the optional PyTorch projector (eval, no grad, batched)."""
        if self.projector is None:
            return np.asarray(embeddings, dtype=np.float64)
        torch = _require_torch()
        self.projector.eval()
        arr = np.asarray(embeddings, dtype=np.float32)
        out_chunks: list[NDArray[np.float64]] = []
        with torch.no_grad():
            for start in range(0, arr.shape[0], self.projector_batch_size):
                batch = arr[start : start + self.projector_batch_size]
                tensor = torch.as_tensor(batch, dtype=torch.float32)
                proj = self.projector(tensor).cpu().numpy()
                out_chunks.append(proj)
        return np.vstack(out_chunks).astype(np.float64, copy=False)

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """Fit the model on labeled texts."""
        unique_labels = sorted(set(labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

        embeddings = self.embedder.embed(texts)
        embeddings = self._project_numpy_embeddings(embeddings)

        self.cluster_means = []
        self.cluster_covs = []
        self.thresholds = []

        label_arr = np.array(labels)
        for cluster_label in unique_labels:
            mask = label_arr == cluster_label
            cluster_embeddings = embeddings[mask]
            if cluster_embeddings.shape[0] < self.min_cluster_size:
                logger.warning(
                    "Skipping cluster %s with size %s < %s",
                    cluster_label,
                    cluster_embeddings.shape[0],
                    self.min_cluster_size,
                )
                continue

            mean = np.mean(cluster_embeddings, axis=0)
            self.cluster_means.append(mean)

            cov = estimate_covariance(cluster_embeddings, self.regularization)
            self.cluster_covs.append(cov)

            distances = []
            for emb in cluster_embeddings:
                diff = emb - mean
                inv_cov = invert_covariance(cov, self.regularization)
                dist = float(np.sqrt(diff.T @ inv_cov @ diff))
                distances.append(dist)

            feature_dim = int(cluster_embeddings.shape[1])
            threshold = self.threshold_strategy.compute(distances, feature_dim=feature_dim)
            self.thresholds.append(threshold)

        logger.info("Fitted model with %s clusters", len(self.cluster_means))

    def predict(self, text: str) -> Tuple[str, float, float, bool]:
        """Predict cluster and detect drift for a single text."""
        text = preprocess_text(normalize_ticket_input(text))
        embedding = self.embedder.embed([text])
        embedding = self._project_numpy_embeddings(embedding)[0]

        min_dist = float("inf")
        predicted_cluster = -1

        for i, (mean, cov) in enumerate(zip(self.cluster_means, self.cluster_covs)):
            diff = embedding - mean
            inv_cov = invert_covariance(cov, self.regularization)
            dist = float(np.sqrt(diff.T @ inv_cov @ diff))
            if dist < min_dist:
                min_dist = dist
                predicted_cluster = i

        threshold = self.thresholds[predicted_cluster]
        is_drift = detect_drift(min_dist, threshold)
        predicted_label = self.index_to_label[predicted_cluster]

        return predicted_label, min_dist, threshold, is_drift

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float, float, bool]]:
        """Predict batch of texts."""
        cleaned = [preprocess_text(normalize_ticket_input(t)) for t in texts]
        embeddings = self.embedder.embed(cleaned)
        embeddings = self._project_numpy_embeddings(embeddings)
        results = []
        for embedding in embeddings:
            min_dist = float("inf")
            predicted_cluster = -1

            for i, (mean, cov) in enumerate(zip(self.cluster_means, self.cluster_covs)):
                diff = embedding - mean
                inv_cov = invert_covariance(cov, self.regularization)
                dist = float(np.sqrt(diff.T @ inv_cov @ diff))
                if dist < min_dist:
                    min_dist = dist
                    predicted_cluster = i

            threshold = self.thresholds[predicted_cluster]
            is_drift = detect_drift(min_dist, threshold)
            predicted_label = self.index_to_label[predicted_cluster]

            results.append((predicted_label, min_dist, threshold, is_drift))
        return results

    @staticmethod
    def _projector_paths(model_path: str) -> Tuple[Path, Path]:
        base = Path(model_path)
        stem = base.stem
        parent = base.parent
        return parent / f"{stem}_projector.pt", parent / f"{stem}_projector_config.json"

    def save(self, path: str) -> None:
        """Save model and optional projector weights plus architecture JSON."""
        from src.mcd.persistence.artifacts import save_artifact, save_label_mapping

        has_projector = self.projector is not None
        data: Dict[str, Any] = {
            "label_to_index": self.label_to_index,
            "cluster_means": self.cluster_means,
            "cluster_covs": self.cluster_covs,
            "thresholds": self.thresholds,
            "regularization": self.regularization,
            "threshold_quantile": self.threshold_quantile,
            "min_cluster_size": self.min_cluster_size,
            "has_projector": has_projector,
            "projector_batch_size": self.projector_batch_size,
        }
        save_artifact(data, path)
        mapping_path = path.replace(".joblib", "_mapping.json")
        save_label_mapping(self.label_to_index, mapping_path)

        if has_projector:
            torch = _require_torch()
            from src.mcd.projection.projector import DeepMahalanobisProjector

            if not isinstance(self.projector, DeepMahalanobisProjector):
                raise TypeError("projector must be a DeepMahalanobisProjector instance for save()")
            pt_path, json_path = self._projector_paths(path)
            cfg = self.projector.architecture_dict()
            cfg["schema"] = 1
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            torch.save(self.projector.state_dict(), pt_path)

    @classmethod
    def load(cls, path: str) -> MahalanobisDriftDetector:
        """Load model; restores projector if *_projector.pt and *_projector_config.json exist."""
        from src.mcd.persistence.artifacts import load_artifact, load_label_mapping

        data = load_artifact(path)
        mapping_path = path.replace(".joblib", "_mapping.json")
        label_to_index = load_label_mapping(mapping_path)

        instance = cls(
            threshold_quantile=data.get("threshold_quantile", 0.99),
            min_cluster_size=data.get("min_cluster_size", 10),
            regularization=float(data.get("regularization", 1e-6)),
        )
        instance.label_to_index = label_to_index
        instance.index_to_label = {v: k for k, v in label_to_index.items()}
        instance.cluster_means = data["cluster_means"]
        instance.cluster_covs = data["cluster_covs"]
        instance.thresholds = data["thresholds"]
        instance.regularization = data["regularization"]
        instance.threshold_quantile = data.get("threshold_quantile", 0.99)
        instance.min_cluster_size = data.get("min_cluster_size", 10)
        instance.projector_batch_size = int(data.get("projector_batch_size", _DEFAULT_PROJECT_BATCH))

        pt_path, json_path = cls._projector_paths(path)
        if pt_path.exists() or json_path.exists():
            if not pt_path.exists():
                raise FileNotFoundError(
                    f"Projector weights missing: expected {pt_path} alongside config {json_path}"
                )
            if not json_path.exists():
                raise FileNotFoundError(
                    f"Projector config missing: expected {json_path} alongside weights {pt_path}"
                )
            torch = _require_torch()
            from src.mcd.projection.projector import DeepMahalanobisProjector

            with open(json_path, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg.pop("schema", None)
            projector = DeepMahalanobisProjector.from_architecture_dict(cfg)
            load_kw: Dict[str, Any] = {"map_location": torch.device("cpu")}
            try:
                state = torch.load(pt_path, weights_only=True, **load_kw)
            except TypeError:
                state = torch.load(pt_path, **load_kw)
            projector.load_state_dict(state)
            projector.eval()
            instance.projector = projector

        return instance
