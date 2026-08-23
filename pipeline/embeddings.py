"""Semantic search over articles: embeddings + cosine similarity.

Embeddings are generated once over `title + ". " + summary` using
sentence-transformers/all-MiniLM-L6-v2, then persisted as a numpy array
(`article_embeddings.npy`) plus a CSV row-order index
(`article_embeddings_index.csv`, article_id per row) so they can be reloaded
without re-running the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

_model_cache = {}


def _get_model():
    from sentence_transformers import SentenceTransformer

    if "model" not in _model_cache:
        _model_cache["model"] = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model_cache["model"]


def article_text(title: str, summary: str) -> str:
    title = (title or "").strip()
    summary = (summary or "").strip()
    if title and summary:
        return f"{title}. {summary}"
    return title or summary


def generate_article_embeddings(articles: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """articles must have `article_id`, `title`, `summary` columns.
    Returns (embeddings [n, dim] float32 L2-normalized, article_ids in row order).
    """
    model = _get_model()
    texts = [article_text(t, s) for t, s in zip(articles["title"], articles["summary"])]
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return embeddings, list(articles["article_id"])


def save_embeddings(embeddings: np.ndarray, article_ids: list[str], npy_path=None, index_path=None) -> None:
    npy_path = Path(npy_path or config.EMBEDDINGS_NPY_PATH)
    index_path = Path(index_path or config.EMBEDDINGS_INDEX_PATH)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, embeddings)
    pd.DataFrame({"article_id": article_ids}).to_csv(index_path, index=False)


def load_embeddings(npy_path=None, index_path=None) -> tuple[np.ndarray, list[str]]:
    npy_path = Path(npy_path or config.EMBEDDINGS_NPY_PATH)
    index_path = Path(index_path or config.EMBEDDINGS_INDEX_PATH)
    embeddings = np.load(npy_path)
    article_ids = pd.read_csv(index_path)["article_id"].tolist()
    return embeddings, article_ids


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """embeddings assumed L2-normalized -> cosine similarity is just the dot product."""
    return embeddings @ embeddings.T


def find_similar_articles(
    query_text: str,
    embeddings: np.ndarray,
    article_ids: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Embed `query_text` and return the top_k most similar article_ids with
    cosine similarity scores, most similar first."""
    model = _get_model()
    query_vec = model.encode([query_text], show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm
    scores = (embeddings @ query_vec.T).ravel()
    top_idx = np.argsort(-scores)[:top_k]
    return [(article_ids[i], float(scores[i])) for i in top_idx]


def compute_top_similar_articles(
    embeddings: np.ndarray, article_ids: list[str], k: int = config.TOP_SIMILAR_ARTICLES_K
) -> dict[str, list[str]]:
    """For every article, the top-k *other* articles by cosine similarity."""
    sims = cosine_similarity_matrix(embeddings)
    n = len(article_ids)
    result: dict[str, list[str]] = {}
    for i in range(n):
        row = sims[i].copy()
        row[i] = -np.inf  # exclude self
        top_idx = np.argsort(-row)[:k]
        result[article_ids[i]] = [article_ids[j] for j in top_idx]
    return result
