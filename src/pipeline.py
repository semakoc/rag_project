"""
Build and query a per-group RAG pipeline from PDF and text files.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.embed import build_faiss_index, embed_chunks, load_embedding_model
from src.preprocess import chunk_text, clean_text, normalize_text, process_pdf
from src.rag import dual_rag

logger = logging.getLogger(__name__)


def _read_text_file(path: Path) -> str:
    """Load a text file as a string, trying common encodings."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        try:
            return path.read_text(encoding="latin-1")
        except OSError as exc:
            raise OSError(f"Could not read text file: {path}") from exc


def _chunks_from_plain_file(path: Path, lowercase: bool) -> list[str]:
    """Clean, normalize, and chunk a plain text file."""
    raw = _read_text_file(path)
    cleaned = clean_text(raw)
    normalized = normalize_text(cleaned, lowercase=lowercase)
    return chunk_text(normalized)


def _collect_chunks(folder: Path) -> tuple[list[str], int]:
    """
    Walk the folder for PDFs and text files and return all chunks plus doc count.

    PDFs use the PDF pipeline; .txt files are read directly.
    """
    chunks: list[str] = []
    doc_count = 0
    if not folder.is_dir():
        return chunks, doc_count
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() == ".pdf":
            try:
                chunks.extend(process_pdf(str(path)))
                doc_count += 1
            except OSError as exc:
                logger.warning("Skipping PDF %s: %s", path, exc)
        elif path.suffix.lower() == ".txt":
            try:
                chunks.extend(_chunks_from_plain_file(path, lowercase=True))
                doc_count += 1
            except OSError as exc:
                logger.warning("Skipping text file %s: %s", path, exc)
    return chunks, doc_count


def _empty_pipeline(group_name: str) -> dict:
    """Return a usable pipeline object when there is nothing to index yet."""
    model = load_embedding_model()
    index = build_faiss_index(embed_chunks([], model=model))
    return {
        "group": group_name,
        "chunks": [],
        "index": index,
        "model": model,
        "doc_count": 0,
        "chunk_count": 0,
    }


def build_pipeline(folder_path: str, group_name: str) -> dict:
    """
    Ingest every PDF and text file in a folder and build embeddings + FAISS.

    The returned dictionary holds chunks, the index, the embedding model, and
    simple counts for logging. Empty folders produce an empty chunk list and a
    placeholder index so the app can still start.
    """
    folder = Path(folder_path)
    chunks, doc_count = _collect_chunks(folder)
    if not chunks:
        logger.warning(
            "No chunks built for %s in %s — using empty index.",
            group_name,
            folder_path,
        )
        pipe = _empty_pipeline(group_name)
        pipe["doc_count"] = doc_count
        return pipe
    model = load_embedding_model()
    embeddings = embed_chunks(chunks, model=model)
    index = build_faiss_index(embeddings)
    return {
        "group": group_name,
        "chunks": chunks,
        "index": index,
        "model": model,
        "doc_count": doc_count,
        "chunk_count": len(chunks),
    }


def query_pipeline(pipeline: dict, query: str, top_k: int = 5) -> list[str]:
    """
    Run dual retrieval (semantic + keyword fusion) on a built pipeline.

    Returns the top text snippets to pass into the language model.
    """
    chunks: list[str] = pipeline.get("chunks", [])
    index = pipeline["index"]
    model = pipeline["model"]
    return dual_rag(query, index, chunks, model, top_k=top_k)
