from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    text_columns: List[str] = None
    label_column: str = "queue"
    threshold_quantile: float = 0.99
    min_cluster_size: int = 10

    def __post_init__(self):
        if self.text_columns is None:
            self.text_columns = ["subject", "body"] 
