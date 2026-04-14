"""Ticket input normalization."""

from src.mcd.preprocessing import normalize_ticket_input, preprocess_text


def test_normalize_subject_body_strips_labels() -> None:
    raw = """Subject: Password reset email never arrives
Body: I use the forgot-password link but I do not get any email."""
    norm = normalize_ticket_input(raw)
    assert "Subject:" not in norm
    assert "Body:" not in norm
    assert "Password reset" in norm
    assert "forgot-password" in norm


def test_preprocess_after_normalize_matches_training_shape() -> None:
    raw = "Subject: A\n\nBody: B line"
    out = preprocess_text(normalize_ticket_input(raw))
    assert out == "A B line"
