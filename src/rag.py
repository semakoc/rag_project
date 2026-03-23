"""
Semantic search, keyword search, and dual fusion for retrieval.
"""

from __future__ import annotations

import re
from typing import Any

from src.embed import embed_chunks

# Common English words to skip so keyword scores focus on meaningful terms.
_STOPWORDS = frozenset(
    """
    a an and are as at be been being but by for from had has have he her
    hers him his how i if in into is it its me more my no not of on or our
    out s she so such t than that the their them then there these they this
    to too was we were what when where which who whom why will with you your
    """.split()
)


def _tokenize_lower(text: str) -> list[str]:
    """Split text into lowercase word tokens for matching."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _keyword_terms(query: str) -> list[str]:
    """Keep query words that are not stopwords so scoring focuses on content."""
    return [w for w in _tokenize_lower(query) if w not in _STOPWORDS and len(w) > 1]


def _min_max_normalize(values: list[float], *, lower_is_better: bool) -> list[float]:
    """Scale a list of scores into [0, 1] for fair mixing."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    if lower_is_better:
        return [(hi - v) / (hi - lo) for v in values]
    return [(v - lo) / (hi - lo) for v in values]


def semantic_search(
    query: str,
    index: Any,
    chunks: list[str],
    model: Any,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Find chunks whose embeddings are closest to the query embedding.

    Uses the FAISS L2 index: smaller distance means a better semantic match.
    Results are sorted from smallest distance to largest.
    """
    if not chunks or getattr(index, "ntotal", 0) == 0:
        return []
    query_vec = embed_chunks([query], model=model)
    k = min(top_k, index.ntotal, len(chunks))
    distances, indices = index.search(query_vec, k)
    out: list[tuple[str, float]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        out.append((chunks[int(idx)], float(dist)))
    out.sort(key=lambda x: x[1])
    return out


def keyword_search(
    query: str,
    chunks: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Rank chunks by overlap between query words and chunk words.

    Scores favor chunks that contain more of the query terms, adjusted by
    chunk length so long passages are not unfairly favored. Matching ignores
    case and skips very common stopwords.
    """
    if not chunks:
        return []
    terms = _keyword_terms(query)
    if not terms:
        terms = _tokenize_lower(query)
    scored: list[tuple[str, float]] = []
    for chunk in chunks:
        cwords = _tokenize_lower(chunk)
        if not cwords:
            continue
        hits = sum(cwords.count(t) for t in terms) if terms else 0.0
        score = hits / float(len(cwords))
        scored.append((chunk, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def dual_rag(
    query: str,
    index: Any,
    chunks: list[str],
    model: Any,
    top_k: int = 5,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[str]:
    """
    Blend semantic and keyword retrieval, then return the best unique chunks.

    Each method contributes normalized scores that are combined with weights,
    duplicates are merged, and the highest combined scores win.
    """
    if not chunks:
        return []
    fetch = min(top_k * 2, len(chunks))
    sem_raw = semantic_search(query, index, chunks, model, top_k=fetch)
    key_raw = keyword_search(query, chunks, top_k=fetch)
    sem_dists = [d for _, d in sem_raw]
    key_scores = [s for _, s in key_raw]
    sem_norms = _min_max_normalize(sem_dists, lower_is_better=True)
    key_norms = _min_max_normalize(key_scores, lower_is_better=False)
    sem_map: dict[str, float] = {}
    for (ch, _), n in zip(sem_raw, sem_norms):
        sem_map[ch] = max(sem_map.get(ch, 0.0), n)
    key_map: dict[str, float] = {}
    for (ch, _), n in zip(key_raw, key_norms):
        key_map[ch] = max(key_map.get(ch, 0.0), n)
    all_chunks = set(sem_map) | set(key_map)
    combined: list[tuple[str, float]] = []
    for ch in all_chunks:
        score = semantic_weight * sem_map.get(ch, 0.0) + keyword_weight * key_map.get(
            ch, 0.0
        )
        combined.append((ch, score))
    combined.sort(key=lambda x: x[1], reverse=True)
    seen: set[str] = set()
    result: list[str] = []
    for ch, _ in combined:
        if ch in seen:
            continue
        seen.add(ch)
        result.append(ch)
        if len(result) >= top_k:
            break
    return result
