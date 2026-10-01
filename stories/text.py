"""Turning text into the words that matter: shared by keyword search and duplicate detection."""

from __future__ import annotations

import functools
import re
import threading

import snowballstemmer

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

# A word may contain apostrophes (straight or curly), so "didn't" stays one word instead of "didn" and "t".
_WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
# Contractions are function words ("didn't", "you've", "we'll"), so they say nothing about which story fits.
_CONTRACTION = re.compile(r"(?:n't|'(?:ve|re|ll|d|m|t))$")
_stemmer = snowballstemmer.stemmer("english")


# Stemming dominates search time and snowballstemmer doesn't cache, while the vocabulary is small
# and repeats on every search. The lock is there because the stemmer object isn't thread safe.
_stem_lock = threading.Lock()


@functools.lru_cache(maxsize=100_000)
def _stem(word: str) -> str:
    with _stem_lock:
        return _stemmer.stemWord(word)


def _normalize(word: str) -> str | None:
    """The word without its apostrophes, or None for a contraction. A possessive is the word it's attached to."""
    word = word.replace("\u2019", "'")
    if word.endswith("'s"):
        word = word[:-2]
    elif _CONTRACTION.search(word):
        return None
    return word.replace("'", "")


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop contractions and stopwords, and stem."""
    words = (_normalize(w) for w in _WORD.findall(text.lower()))
    return [_stem(w) for w in words if w and w not in STOPWORDS]


def word_overlap(a: str, b: str) -> float:
    """The share of the shorter text's distinct words that the other text also uses, from 0 to 1."""
    words_a, words_b = set(tokenize(a)), set(tokenize(b))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))
