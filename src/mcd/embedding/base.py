from abc import ABC, abstractmethod
import numpy as np
from typing import List

class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts into vectors."""
        pass 
