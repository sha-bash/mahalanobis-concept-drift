"""Text preprocessing utilities."""

import re
from typing import List


def preprocess_text(text: str) -> str:
    """
    Preprocess a single text: strip, collapse whitespace, remove control chars.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    # Remove control characters and null bytes
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def preprocess_texts(texts: List[str]) -> List[str]:
    """Preprocess a list of texts."""
    return [preprocess_text(t) for t in texts]