import pytest

from src.preprocess import clean_text, normalize_text


def test_clean_text_removes_page_numbers():
    raw = "Some text here.\n42\nMore text after the page number."
    out = clean_text(raw)
    assert "42" not in out.split()
    assert "Some text here." in out
    assert "More text after" in out


def test_clean_text_handles_empty_string():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_normalize_text_lowercases():
    assert normalize_text("HELLO World", lowercase=True) == "hello world"


def test_normalize_text_fixes_smart_quotes():
    s = "\u201chello\u201d"
    assert normalize_text(s, lowercase=True) == '"hello"'


def test_normalize_text_no_lowercase_option():
    assert normalize_text("HELLO", lowercase=False) == "HELLO"
