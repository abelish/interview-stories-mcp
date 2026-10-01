"""Ranked story search: keyword (BM25F), semantic (local embeddings), and hybrid (both, blended).

Hybrid is the default. Keyword search matches words, with tags counting most, then the title, then
the STAR-L text. Semantic search matches meaning, so "a coworker" finds a story about "a fellow senior
engineer". Hybrid blends both scores. If the embedding model can't load (for example offline on
first run), hybrid falls back to keyword search and retries the model later.

The mode and model were chosen with the retrieval eval (evals/retrieval.py).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from stories import storage
from stories.storage import InvalidStoryError, Story
from stories.text import tokenize

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

Mode = Literal["hybrid", "keyword", "semantic"]
DEFAULT_LIMIT = 5

# --- Keyword search (BM25F) ----------------------------------------------------------

# BM25 parameters. K1 controls how quickly repeated words stop adding score (the standard default).
# FIELD_B controls how much a longer-than-average field is penalized: fully for prose (the standard
# 0.75), less for short titles, and not at all for tags, which are a set of labels. A story with
# more tags shouldn't rank lower for a tag it has.
K1 = 1.2
FIELD_B = {"tags": 0.0, "title": 0.5, "body": 0.75}

FIELD_WEIGHTS = {"tags": 3.0, "title": 2.0, "body": 1.0}


def _fields(story: Story) -> dict[str, list[str]]:
    body = " ".join([story.situation, story.task, story.action, story.result, story.learning])
    return {
        "tags": tokenize(" ".join(story.tags)),
        "title": tokenize(story.title),
        "body": tokenize(body),
    }


def keyword_scores(query: str, stories: Sequence[Story]) -> list[float]:
    """Each story's BM25F score as a share of the most the query could score, from 0 to 1.

    Each query word adds at most its idf, so dividing by the sum of the query words' idfs measures how
    much of the query a story matches, weighted by how rare each word is. A match on a word most stories
    share stays small, instead of counting as much as a match on the word that picks out one story.
    """
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms or not stories:
        return [0.0] * len(stories)

    docs = [{field: Counter(tokens) for field, tokens in _fields(s).items()} for s in stories]
    lengths = [{field: sum(counts.values()) for field, counts in doc.items()} for doc in docs]
    avg_length = {field: max(sum(ln[field] for ln in lengths) / len(docs), 1e-9) for field in FIELD_WEIGHTS}
    n = len(docs)
    idf = {}
    for term in terms:
        doc_freq = sum(1 for d in docs if any(d[f][term] for f in FIELD_WEIGHTS))
        idf[term] = math.log(1 + (n - doc_freq + 0.5) / (doc_freq + 0.5))
    most = sum(idf.values())

    scores = []
    for doc, length in zip(docs, lengths, strict=True):
        score = 0.0
        for term in terms:
            weighted_tf = sum(
                weight * doc[field][term] / (1 - FIELD_B[field] + FIELD_B[field] * length[field] / avg_length[field])
                for field, weight in FIELD_WEIGHTS.items()
            )
            score += idf[term] * weighted_tf / (K1 + weighted_tf)
        scores.append(score / most)
    return scores


def _by_score(stories: Sequence[Story], scores: Sequence[float], keep_zero: bool = True) -> list[Story]:
    """Stories by descending score. Ties keep the stories' original order."""
    order = sorted(range(len(stories)), key=lambda i: (-scores[i], i))
    return [stories[i] for i in order if keep_zero or scores[i] > 0]


def keyword_rank(query: str, stories: Sequence[Story]) -> list[Story]:
    """Stories matching any query word, best first. Ties keep the stories' original order."""
    return _by_score(stories, keyword_scores(query, stories), keep_zero=False)


# --- Semantic search (local embeddings) ---------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# A persistent cache: fastembed's default is the system temp dir, which may be cleaned out.
MODEL_CACHE_DIR = Path(os.environ.get("STORIES_MODEL_CACHE", Path.home() / ".cache" / "interview-stories-mcp"))
RETRY_AFTER_SECONDS = 60.0


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> list[NDArray[np.float32]]: ...
    def embed_query(self, text: str) -> NDArray[np.float32]: ...


class FastEmbedder:
    """bge-small via fastembed (ONNX, CPU). Downloads the model once on first use."""

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path = MODEL_CACHE_DIR) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))

    def embed_passages(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [np.asarray(v, dtype=np.float32) for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return np.asarray(next(iter(self._model.query_embed([text]))), dtype=np.float32)


def story_text(story: Story) -> str:
    """The text a story is embedded as."""
    parts = [story.situation, story.task, story.action, story.result, story.learning]
    return f"{story.title}. Tags: {', '.join(story.tags)}. " + " ".join(parts)


def _unit(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class SemanticIndex:
    """Loads the embedder lazily and caches story embeddings by content, so edits are re-embedded."""

    def __init__(self, load_embedder: Callable[[], Embedder] | None = None) -> None:
        self._load_embedder = load_embedder or FastEmbedder
        self._embedder: Embedder | None = None
        self._failed_at: float | None = None
        self._vectors: dict[str, NDArray[np.float32]] = {}
        self._lock = threading.Lock()

    def embedder(self) -> Embedder | None:
        """The embedder, or None if it can't load right now. Retries a failed load after a while."""
        with self._lock:
            if self._embedder is not None:
                return self._embedder
            if self._failed_at is not None and time.monotonic() - self._failed_at < RETRY_AFTER_SECONDS:
                return None
            try:
                self._embedder = self._load_embedder()
                self._failed_at = None
            except Exception as exc:
                log.warning("Embedding model unavailable, using keyword search only: %s", exc)
                log.debug("Embedding model load failure", exc_info=True)
                self._failed_at = time.monotonic()
            return self._embedder

    def rank(self, query: str, stories: Sequence[Story]) -> list[Story] | None:
        """All stories by similarity to the query, or None if the model isn't available."""
        similarities = self.similarities(query, stories)
        return None if similarities is None else _by_score(stories, similarities.tolist())

    def similarities(self, query: str, stories: Sequence[Story]) -> NDArray[np.float32] | None:
        """Each story's cosine similarity to the query, or None if the model isn't available."""
        embedder = self.embedder()
        if embedder is None:
            return None
        if not stories:
            return np.zeros(0, dtype=np.float32)
        keys = [hashlib.sha256(story_text(s).encode()).hexdigest() for s in stories]
        with self._lock:
            missing = [(k, s) for k, s in zip(keys, stories, strict=True) if k not in self._vectors]
            if missing:
                vectors = embedder.embed_passages([story_text(s) for _, s in missing])
                self._vectors.update({k: _unit(v) for (k, _), v in zip(missing, vectors, strict=True)})
            # Keep the cache to the current stories so edits and deletes don't accumulate.
            self._vectors = {k: self._vectors[k] for k in keys}
            matrix = np.stack([self._vectors[k] for k in keys])
        return matrix @ _unit(embedder.embed_query(query))


_index = SemanticIndex()


def warm_up() -> None:
    """Load the embedding model ahead of the first search (downloading it if needed)."""
    _index.embedder()


# --- Hybrid ------------------------------------------------------------------------------

# How much the keyword score counts against the semantic one. Equal weight, rather than a value tuned
# to the retrieval eval's 52 queries, which would fit their noise.
KEYWORD_WEIGHT = 0.5


def blend(keyword: Sequence[float], semantic: NDArray[np.float32]) -> NDArray[np.float64]:
    """Combine keyword scores (0 to 1) with semantic similarities rescaled to 0 to 1 within the query.

    Scores, not ranks, are blended, so how strongly a story matches counts and not just where it
    places. Rank fusion let a weak keyword match on a word most stories share outrank the closest
    story by meaning, when that story happened to share no words with the question.
    """
    similarity = semantic.astype(np.float64)
    span = float(similarity.max() - similarity.min()) if similarity.size else 0.0
    rescaled = (similarity - similarity.min()) / span if span else np.zeros_like(similarity)
    return KEYWORD_WEIGHT * np.asarray(keyword, dtype=np.float64) + (1 - KEYWORD_WEIGHT) * rescaled


def search_stories(
    query: str,
    limit: int = DEFAULT_LIMIT,
    mode: Mode = "hybrid",
    index: SemanticIndex | None = None,
) -> list[Story]:
    """The stories that best fit the query, best first, at most limit of them."""
    if not query.strip():
        raise InvalidStoryError("query can't be blank.")
    if limit < 1:
        raise InvalidStoryError("limit must be at least 1.")
    stories = storage.list_stories()
    index = index or _index
    if mode == "keyword":
        return keyword_rank(query, stories)[:limit]
    similarities = index.similarities(query, stories)
    if similarities is None:
        ranked = keyword_rank(query, stories)
    elif mode == "semantic":
        ranked = _by_score(stories, similarities.tolist())
    else:
        ranked = _by_score(stories, blend(keyword_scores(query, stories), similarities).tolist())
    return ranked[:limit]
