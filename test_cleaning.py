#!/usr/bin/env python3
"""
Standalone diagnostic: run a PDF through each preprocessing stage and validate chunks.
"""

from __future__ import annotations

import logging
import re
import sys

logging.basicConfig(level=logging.INFO)

from src.preprocess import (
    chunk_text,
    clean_text,
    extract_text_from_pdf,
    normalize_text,
    process_pdf,
)

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"

_SMART_QUOTE_BYTES = "\x91\x92\x93\x94"


def _check_no_digit_only_chunks(chunks: list[str]) -> bool:
    """Fail if any chunk is only digits (standalone page-number style)."""
    for ch in chunks:
        s = ch.strip()
        if s and re.fullmatch(r"\d+", s):
            return False
    return True


def _check_no_triple_spaces(chunks: list[str]) -> bool:
    return all("   " not in c for c in chunks)


def _check_no_smart_quote_bytes(chunks: list[str]) -> bool:
    return all(c not in _SMART_QUOTE_BYTES for chunk in chunks for c in chunk)


def _check_ascii_only(chunks: list[str]) -> bool:
    for ch in chunks:
        try:
            ch.encode("ascii")
        except UnicodeEncodeError:
            return False
    return True


def _check_no_empty_chunks(chunks: list[str]) -> bool:
    return all(bool(c.strip()) for c in chunks)


def _check_at_least_one_chunk(chunks: list[str]) -> bool:
    return len(chunks) >= 1


def main() -> None:
    print(f"File: {PDF_PATH}\n")
    print("=" * 60)
    print("Step 1: extract_text_from_pdf() — raw extracted text (repr)")
    print("=" * 60)
    raw = extract_text_from_pdf(PDF_PATH)
    print(repr(raw))

    print("\n" + "=" * 60)
    print("Step 2: clean_text() — cleaned result (repr)")
    print("=" * 60)
    cleaned = clean_text(raw)
    print(repr(cleaned))

    print("\n" + "=" * 60)
    print("Step 3: normalize_text() — normalized result (repr)")
    print("=" * 60)
    normalized = normalize_text(cleaned, lowercase=True)
    print(repr(normalized))

    print("\n" + "=" * 60)
    print("Step 4: chunk_text() — numbered chunks")
    print("=" * 60)
    step4_chunks = chunk_text(normalized)
    if not step4_chunks:
        print("(no chunks)")
    for i, ch in enumerate(step4_chunks, start=1):
        print(f"--- chunk {i} ---")
        print(ch)

    print("\n" + "=" * 60)
    print("Step 5: process_pdf() — final chunks and count")
    print("=" * 60)
    final_chunks = process_pdf(PDF_PATH, lowercase=True)
    print(f"Total chunks: {len(final_chunks)}")
    for i, ch in enumerate(final_chunks, start=1):
        print(f"--- final chunk {i} ---")
        print(ch)

    print("\n" + "=" * 60)
    print("Automated checks (final chunks from process_pdf)")
    print("=" * 60)

    checks: list[tuple[str, bool]] = [
        ("Page numbers removed (no standalone digit-only chunks)", _check_no_digit_only_chunks(final_chunks)),
        ("No excessive whitespace (no triple spaces)", _check_no_triple_spaces(final_chunks)),
        (
            "Smart quotes normalized (no \\x93 \\x94 \\x91 \\x92 characters)",
            _check_no_smart_quote_bytes(final_chunks),
        ),
        ("All text is ASCII clean", _check_ascii_only(final_chunks)),
        ("No empty chunks", _check_no_empty_chunks(final_chunks)),
        ("At least one chunk was produced", _check_at_least_one_chunk(final_chunks)),
    ]

    failed = 0
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{status}] {name}")

    print()
    if failed == 0:
        print("✅ All checks passed — safe to load real documents")
    else:
        print(f"❌ {failed} checks failed — review preprocess.py before loading")


if __name__ == "__main__":
    main()
