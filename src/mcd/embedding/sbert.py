from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from .base import Embedder

class SBERT(Embedder):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts into vectors."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings 
