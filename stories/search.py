"""Ranked story search: keyword (BM25F), semantic (local embeddings), and hybrid (both, fused).

Hybrid is the default. Keyword search matches words, with tags counting most, then the title, then
the STAR-L text. Semantic search matches meaning, so "a coworker" finds a story about "a fellow senior
engineer". Hybrid fuses both rankings. If the embedding model can't load (for example offline on
first run), hybrid falls back to keyword search and retries the model later.

The mode and model were chosen with the retrieval eval (evals/retrieval.py).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import snowballstemmer

from stories import storage
from stories.storage import InvalidStoryError, Story

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

# Common English words, plus words that appear in almost every interview question and say nothing
# about which story fits ("Tell me about a time you...", "Describe a situation where...").
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves
    tell describe give example time times walk share situation ever
    """.split()  # noqa: SIM905 - a word block is easier to read and edit than a long list
)

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_stemmer = snowballstemmer.stemmer("english")


# Stemming dominates search time and snowballstemmer doesn't cache, while the vocabulary is small
# and repeats on every search. The lock is there because the stemmer object isn't thread safe.
_stem_lock = threading.Lock()


@functools.lru_cache(maxsize=100_000)
def _stem(word: str) -> str:
    with _stem_lock:
        return _stemmer.stemWord(word)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stopwords, and stem."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def _fields(story: Story) -> dict[str, list[str]]:
    body = " ".join([story.situation, story.task, story.action, story.result, story.learning])
    return {
        "tags": tokenize(" ".join(story.tags)),
        "title": tokenize(story.title),
        "body": tokenize(body),
    }


def keyword_rank(query: str, stories: Sequence[Story]) -> list[Story]:
    """Stories matching any query word, best first. Ties keep the stories' original order."""
    terms = list(dict.fromkeys(tokenize(query)))
    if not terms or not stories:
        return []

    docs = [{field: Counter(tokens) for field, tokens in _fields(s).items()} for s in stories]
    lengths = [{field: sum(counts.values()) for field, counts in doc.items()} for doc in docs]
    avg_length = {field: max(sum(ln[field] for ln in lengths) / len(docs), 1e-9) for field in FIELD_WEIGHTS}
    doc_freq = {term: sum(1 for d in docs if any(d[f][term] for f in FIELD_WEIGHTS)) for term in terms}
    n = len(docs)

    scored: list[tuple[float, Story]] = []
    for story, doc, length in zip(stories, docs, lengths, strict=True):
        score = 0.0
        for term in terms:
            weighted_tf = sum(
                weight * doc[field][term] / (1 - FIELD_B[field] + FIELD_B[field] * length[field] / avg_length[field])
                for field, weight in FIELD_WEIGHTS.items()
            )
            if weighted_tf:
                idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                score += idf * weighted_tf / (K1 + weighted_tf)
        if score > 0:
            scored.append((score, story))
    return [story for _, story in sorted(scored, key=lambda pair: -pair[0])]


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
        embedder = self.embedder()
        if embedder is None:
            return None
        if not stories:
            return []
        keys = [hashlib.sha256(story_text(s).encode()).hexdigest() for s in stories]
        with self._lock:
            missing = [(k, s) for k, s in zip(keys, stories, strict=True) if k not in self._vectors]
            if missing:
                vectors = embedder.embed_passages([story_text(s) for _, s in missing])
                self._vectors.update({k: _unit(v) for (k, _), v in zip(missing, vectors, strict=True)})
            # Keep the cache to the current stories so edits and deletes don't accumulate.
            self._vectors = {k: self._vectors[k] for k in keys}
            matrix = np.stack([self._vectors[k] for k in keys])
        similarities = matrix @ _unit(embedder.embed_query(query))
        order = sorted(range(len(stories)), key=lambda i: (-float(similarities[i]), i))
        return [stories[i] for i in order]


_index = SemanticIndex()


def warm_up() -> None:
    """Load the embedding model ahead of the first search (downloading it if needed)."""
    _index.embedder()


# --- Hybrid ------------------------------------------------------------------------------

# Reciprocal rank fusion constant. 60 is the standard value from the original RRF paper.
RRF_K = 60


def fuse(*rankings: Sequence[Story]) -> list[Story]:
    """Reciprocal rank fusion: a story ranked high in any list ranks high overall.

    Ties keep the order in which stories first appear across the rankings.
    """
    scores: dict[str, float] = {}
    stories: dict[str, Story] = {}
    for ranking in rankings:
        for rank, story in enumerate(ranking, start=1):
            scores[story.id] = scores.get(story.id, 0.0) + 1 / (RRF_K + rank)
            stories.setdefault(story.id, story)
    return [stories[story_id] for story_id in sorted(scores, key=lambda story_id: -scores[story_id])]


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
    keyword = keyword_rank(query, stories)
    if mode == "keyword":
        ranked = keyword
    else:
        semantic = index.rank(query, stories)
        if semantic is None:
            ranked = keyword
        elif mode == "semantic":
            ranked = semantic
        else:
            ranked = fuse(keyword, semantic)
    return ranked[:limit]
