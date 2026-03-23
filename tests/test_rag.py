import pytest

from src.embed import build_faiss_index, embed_chunks, load_embedding_model
from src.rag import dual_rag, keyword_search, semantic_search

sample_chunks = [
    "life is precious and must be protected from conception",
    "bodily autonomy is a fundamental human right",
    "both perspectives care deeply about human dignity",
    "medical decisions should remain private and personal",
    "communities support families through difficult choices",
]


@pytest.fixture(scope="module")
def rag_setup():
    model = load_embedding_model()
    embeddings = embed_chunks(sample_chunks, model=model)
    index = build_faiss_index(embeddings)
    return model, index


def test_semantic_search_returns_results(rag_setup):
    model, index = rag_setup
    results = semantic_search(
        "human dignity and respect",
        index,
        sample_chunks,
        model,
        top_k=3,
    )
    assert len(results) >= 1
    assert all(isinstance(t, tuple) and len(t) == 2 for t in results)


def test_semantic_search_returns_correct_count(rag_setup):
    model, index = rag_setup
    results = semantic_search("autonomy", index, sample_chunks, model, top_k=2)
    assert len(results) <= 2


def test_keyword_search_finds_exact_match():
    results = keyword_search("autonomy fundamental", sample_chunks, top_k=3)
    texts = [r[0] for r in results]
    assert any("autonomy" in t.lower() for t in texts)


def test_keyword_search_returns_results():
    results = keyword_search("dignity", sample_chunks, top_k=5)
    assert len(results) >= 1


def test_dual_rag_deduplicates(rag_setup):
    model, index = rag_setup
    chunks = sample_chunks + [sample_chunks[0]]
    embeddings = embed_chunks(chunks, model=model)
    idx = build_faiss_index(embeddings)
    out = dual_rag("dignity", idx, chunks, model, top_k=5)
    assert len(set(out)) == len(out)


def test_dual_rag_returns_top_k(rag_setup):
    model, index = rag_setup
    out = dual_rag("families", index, sample_chunks, model, top_k=3)
    assert len(out) <= 3


def test_dual_rag_empty_chunks_returns_empty(rag_setup):
    model, index = rag_setup
    assert dual_rag("anything", index, [], model, top_k=3) == []
