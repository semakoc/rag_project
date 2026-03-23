"""
PDF ingestion, text cleaning, normalization, and chunking for the RAG pipeline.
"""

from __future__ import annotations

import re
from typing import Optional

import pdfplumber

try:
    import pymupdf
except ImportError:  # pragma: no cover
    pymupdf = None


def extract_text_from_pdf(path: str) -> str:
    """
    Read a PDF file and return its text as one string.

    The function tries PyMuPDF first because it is usually faster. If that
    returns an empty string (some PDFs scan poorly with one library), it
    falls back to pdfplumber so we still get usable text when possible.
    """
    raw = ""
    if pymupdf is not None:
        try:
            doc = pymupdf.open(path)
            parts: list[str] = []
            for page in doc:
                parts.append(page.get_text())
            doc.close()
            raw = "\n".join(parts)
        except OSError as exc:
            raise OSError(f"Could not read PDF with PyMuPDF: {path}") from exc
    if raw.strip():
        return raw
    try:
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except OSError as exc:
        raise OSError(f"Could not read PDF with pdfplumber: {path}") from exc


def _strip_non_ascii(text: str) -> str:
    """Keep only ASCII characters so downstream tools see a consistent alphabet."""
    return text.encode("ascii", errors="ignore").decode("ascii")


def clean_text(text: Optional[str]) -> str:
    """
    Clean extracted text for downstream use.

    Removes lines that are only a page number, collapses extra spaces and
    newlines, and drops non-ASCII characters so the corpus is consistent.
    If the input is missing or empty, returns an empty string safely.
    """
    if text is None or not str(text).strip():
        return ""
    lines = str(text).splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"\d+", stripped):
            continue
        cleaned_lines.append(stripped)
    joined = "\n".join(cleaned_lines)
    joined = re.sub(r"[ \t]+", " ", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return _strip_non_ascii(joined.strip())


def normalize_text(text: str, lowercase: bool = True) -> str:
    """
    Normalize punctuation and spacing so similar phrases match better.

    Optionally lowercases the text. Converts curly quotes to straight quotes,
    long dashes to simple hyphens, and trims spaces on each line.
    """
    if not text:
        return ""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    lines = [ln.strip() for ln in text.splitlines()]
    out = "\n".join(lines)
    if lowercase:
        return out.lower()
    return out


def _merge_short_last_chunk(chunks: list[str], chunk_size: int, max_extra: int = 10) -> list[str]:
    """
    If the final window is very short, join it to the previous chunk when the
    combined size stays within chunk_size + max_extra words.
    """
    if len(chunks) < 2:
        return chunks
    last_words = chunks[-1].split()
    if len(last_words) >= 20 or not last_words:
        return chunks
    merged = chunks[-2] + " " + chunks[-1]
    if len(merged.split()) <= chunk_size + max_extra:
        return chunks[:-2] + [merged]
    return chunks


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word windows for embedding.

    Words are split on whitespace. Each chunk uses up to `chunk_size` words,
    and the next chunk starts `chunk_size - overlap` words later so nearby
    ideas stay connected across chunks.     Short documents become a single chunk; empty input returns an empty list.
    For long text, each window uses up to `chunk_size` words; a very short
    trailing window may be merged into the prior chunk when that keeps the
    combined size within chunk_size plus a small margin, so small fragments
    are avoided when possible.
    """
    if not text or not text.strip():
        return []
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    stride = max(1, chunk_size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += stride
    return _merge_short_last_chunk(chunks, chunk_size)


def process_pdf(path: str, lowercase: bool = True) -> list[str]:
    """
    Run the full preprocessing path on one PDF: extract, clean, normalize, chunk.

    Returns a list of text chunks ready to be embedded and indexed.
    """
    raw = extract_text_from_pdf(path)
    cleaned = clean_text(raw)
    normalized = normalize_text(cleaned, lowercase=lowercase)
    return chunk_text(normalized)
