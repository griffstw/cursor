from __future__ import annotations

import io
from typing import Iterable, List


def load_text_from_pdf_bytes(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        # pypdf not available; gracefully skip PDF extraction
        return ""
    reader = PdfReader(io.BytesIO(data))
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(t.strip() for t in texts if t.strip())


def load_text_from_txt_bytes(data: bytes, encoding: str = "utf-8") -> str:
    try:
        return data.decode(encoding)
    except Exception:
        return data.decode("latin-1", errors="ignore")


def sniff_and_load(filename: str, data: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return load_text_from_pdf_bytes(data)
    if lower.endswith(".txt") or lower.endswith(".text"):
        return load_text_from_txt_bytes(data)
    # naive fallback: try utf-8 decode
    return load_text_from_txt_bytes(data)
