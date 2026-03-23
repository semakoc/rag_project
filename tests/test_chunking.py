from src.preprocess import chunk_text


def test_chunk_returns_list():
    text = "one two three four five " * 10
    out = chunk_text(text, chunk_size=20, overlap=5)
    assert isinstance(out, list)


def test_chunk_sizes_within_bounds():
    words = ["word"] * 500
    text = " ".join(words)
    chunk_size = 100
    overlap = 20
    out = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    for ch in out:
        assert len(ch.split()) <= chunk_size + 10


def test_overlap_creates_continuity():
    words = [f"w{i}" for i in range(200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert len(chunks) >= 2
    first = set(chunks[0].split())
    second = set(chunks[1].split())
    assert first & second


def test_chunk_short_text_returns_one_chunk():
    text = "Just a few words here."
    assert chunk_text(text, chunk_size=400, overlap=50) == [text.strip()]


def test_chunk_empty_text_returns_empty_list():
    assert chunk_text("", chunk_size=100, overlap=10) == []
    assert chunk_text("   ", chunk_size=100, overlap=10) == []
