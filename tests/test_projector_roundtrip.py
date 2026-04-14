"""Save/load roundtrip for MahalanobisDriftDetector with DeepMahalanobisProjector."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from src.mcd.modeling.classifier import MahalanobisDriftDetector
from src.mcd.projection.projector import DeepMahalanobisProjector


def _text_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class _FakeEmbedder:
    """Text-deterministic embeddings (same string always maps to the same vector)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        rows = []
        for t in texts:
            rng = np.random.RandomState(_text_seed(t) % (2**32 - 1))
            rows.append(rng.randn(self.dim))
        return np.vstack(rows).astype(np.float64)


def test_save_load_with_projector_roundtrip() -> None:
    torch.manual_seed(0)
    texts = ["a ticket", "b ticket", "c ticket", "d ticket"]
    labels = ["q1", "q1", "q2", "q2"]
    embed_dim = 16
    projector = DeepMahalanobisProjector(
        input_dim=embed_dim,
        hidden_dims=[8],
        output_dim=4,
        dropout=0.0,
    )

    d1 = MahalanobisDriftDetector(
        embedder=_FakeEmbedder(embed_dim),
        projector=projector,
        min_cluster_size=2,
        threshold_quantile=0.95,
    )
    d1.fit(texts, labels)

    sample = texts[:2]
    preds_before = [d1.predict(t) for t in sample]

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "model.joblib"
        d1.save(str(model_path))
        d2 = MahalanobisDriftDetector.load(str(model_path))
        # Artifacts do not persist embedder; use the same fake embedder as d1.
        d2.embedder = _FakeEmbedder(embed_dim)
        preds_after = [d2.predict(t) for t in sample]

    for (l1, dist1, th1, dr1), (l2, dist2, th2, dr2) in zip(preds_before, preds_after):
        assert l1 == l2
        assert abs(dist1 - dist2) < 1e-5
        assert abs(th1 - th2) < 1e-5
        assert dr1 == dr2
