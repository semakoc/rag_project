"""
Sentence-transformer embeddings and FAISS vector index helpers.
"""

from __future__ import annotations

import faiss
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_model = None


def load_embedding_model():
    """
    Load the sentence-transformer model once and reuse it.

    The model is cached in memory so repeated calls are cheap during a session.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The sentence-transformers package is required for embeddings. "
            "Install it with: pip install sentence-transformers"
        ) from exc
    _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


def embed_chunks(chunks: list[str], model=None) -> np.ndarray:
    """
    Turn text chunks into dense vectors for similarity search.

    Returns a numpy array with one row per chunk. Uses batch encoding for speed.
    If no model is passed, the default shared model is loaded automatically.
    """
    if not chunks:
        enc_model = model or load_embedding_model()
        dim = enc_model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    enc_model = model or load_embedding_model()
    vectors = enc_model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """
    Build a flat L2 index over embedding rows for nearest-neighbor search.

    FAISS stores the vectors and can quickly find the closest ones to a query.
    """
    if embeddings.size == 0:
        dim = 384
        return faiss.IndexFlatL2(dim)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index, path: str) -> None:
    """Write a FAISS index to disk so it can be loaded later without rebuilding."""
    try:
        faiss.write_index(index, path)
    except OSError as exc:
        raise OSError(f"Could not save FAISS index to {path}") from exc


def load_index(path: str):
    """Load a FAISS index previously saved with save_index."""
    try:
        return faiss.read_index(path)
    except OSError as exc:
        raise OSError(f"Could not load FAISS index from {path}") from exc
