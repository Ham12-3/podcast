"""Small text helpers shared by extractors, the script writer and the pipeline."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Tidy whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = text.replace("­", "")  # soft hyphens from PDFs/web pages
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def slugify(value: str, max_len: int = 60) -> str:
    """Make a safe file name: 'How Noise-Cancelling Works!' -> 'how-noise-cancelling-works'."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value[:max_len].rstrip("-") or "podcast"


def word_count(text: str) -> int:
    return len(text.split())
