from __future__ import annotations

import re
from typing import Iterable, List

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'])")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\u00A0", " ", text)  # non-breaking space
    return text.strip()


def split_sentences(text: str) -> List[str]:
    parts = SENTENCE_SPLIT_REGEX.split(text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def chunk_sentences(sentences: List[str], max_tokens: int = 500) -> List[str]:
    # naive tokenization by words
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for s in sentences:
        words = s.split()
        if current_len + len(words) > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(s)
        current_len += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks
