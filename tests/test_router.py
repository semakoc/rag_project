import pytest

from src.embed import build_faiss_index, embed_chunks, load_embedding_model
from src.router import route_query, validate_group


def _make_pipeline(group: str, chunks: list[str]) -> dict:
    model = load_embedding_model()
    embeddings = embed_chunks(chunks, model=model)
    index = build_faiss_index(embeddings)
    return {
        "group": group,
        "chunks": chunks,
        "index": index,
        "model": model,
        "doc_count": 1,
        "chunk_count": len(chunks),
    }


def test_pro_life_routes_to_pro_choice():
    pl = _make_pipeline(
        "pro_life",
        ["pro-life snippet about protecting life"],
    )
    pc = _make_pipeline(
        "pro_choice",
        [
            "bodily autonomy is central to pro-choice views",
            "private medical decisions matter",
        ],
    )
    out = route_query("pro_life", "autonomy and medical privacy", pl, pc)
    assert out["source_group"] == "pro_choice"
    assert len(out["context_chunks"]) >= 1


def test_pro_choice_routes_to_pro_life():
    pl = _make_pipeline(
        "pro_life",
        [
            "unborn child deserves protection from conception onward",
            "many pro-life advocates emphasize sanctity of life",
        ],
    )
    pc = _make_pipeline("pro_choice", ["pro-choice view on reproductive rights"])
    out = route_query("pro_choice", "protection and sanctity of life", pl, pc)
    assert out["source_group"] == "pro_life"
    assert len(out["context_chunks"]) >= 1


def test_invalid_group_raises_error():
    with pytest.raises(ValueError):
        route_query("other", "q", {}, {})


def test_validate_group_returns_true_for_valid():
    assert validate_group("pro_life") is True
    assert validate_group("pro_choice") is True


def test_validate_group_returns_false_for_invalid():
    assert validate_group("invalid") is False
