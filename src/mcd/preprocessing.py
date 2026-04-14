"""Text preprocessing utilities."""

import re
from typing import List


def normalize_ticket_input(text: str) -> str:
    """Strip common email-style headers so text matches CSV training format.

    Training uses ``f\"{subject}\\n\\n{body}\"`` without ``Subject:`` / ``Body:`` labels.
    If those labels are present (e.g. pasted from Streamlit), merge into the same shape
    before :func:`preprocess_text`.
    """
    text = text.strip()
    if not text:
        return text
    subject_m = re.search(
        r"(?is)^\s*Subject:\s*(.+?)(?=^\s*Body:|\Z)",
        text,
        flags=re.MULTILINE,
    )
    body_m = re.search(r"(?is)Body:\s*(.+)$", text)
    if subject_m and body_m:
        subj = subject_m.group(1).strip()
        bod = body_m.group(1).strip()
        if subj or bod:
            return f"{subj}\n\n{bod}".strip()
    return text


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