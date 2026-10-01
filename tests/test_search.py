from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stories import search, storage
from stories.search import SemanticIndex
from stories.storage import Story
from tests.helpers import add_story

# --- A deterministic fake embedder --------------------------------------------------------

VOCAB = ("conflict", "peer", "manager", "deadline", "mentor", "customer")


class FakeEmbedder:
    """Embeds text as counts of a few vocabulary words, so similarity is predictable."""

    passages_embedded: list[str]

    def __init__(self) -> None:
        self.passages_embedded = []

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        words = text.lower()
        return np.array([words.count(w) for w in VOCAB], dtype=np.float32) + 1e-3

    def embed_passages(self, texts: list[str]) -> list[np.ndarray]:
        self.passages_embedded.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def index(embedder: FakeEmbedder) -> SemanticIndex:
    return SemanticIndex(load_embedder=lambda: embedder)


def _ids(stories: list[Story]) -> list[str]:
    return [s.id for s in stories]


# --- Keyword ranking ------------------------------------------------------------------------


def test_keyword_rank_matches_any_query_word() -> None:
    story = add_story(tags=["conflict"], action="I sat down with my peer.")

    assert _ids(search.keyword_rank("conflict with a peer", storage.list_stories())) == [story.id]


def test_keyword_rank_does_not_require_adjacent_words() -> None:
    story = add_story(situation="We missed the deadline by a week.")

    assert _ids(search.keyword_rank("missed deadline", storage.list_stories())) == [story.id]


def test_keyword_rank_matches_word_forms_via_stemming() -> None:
    story = add_story(action="I mentored a new graduate.")

    assert _ids(search.keyword_rank("mentoring", storage.list_stories())) == [story.id]


def test_keyword_rank_weights_tags_over_title_over_body() -> None:
    # Titles of similar length, so BM25's length normalization doesn't dominate the field weights.
    in_body = add_story(title="Launch retrospective", tags=["other"], situation="There was friction over negotiation.")
    in_title = add_story(title="Contract negotiation", tags=["other"])
    in_tags = add_story(title="Launch retrospective", tags=["negotiation"])

    ranked = search.keyword_rank("negotiation", storage.list_stories())

    assert _ids(ranked) == [in_tags.id, in_title.id, in_body.id]


def test_keyword_rank_does_not_penalize_stories_for_having_more_tags() -> None:
    few = add_story(title="Launch retrospective", tags=["negotiation"])
    many = add_story(title="Launch retrospective", tags=["negotiation", "conflict", "deadline", "ownership"])

    ranked = search.keyword_rank("negotiation", storage.list_stories())

    assert _ids(ranked) == [few.id, many.id]  # an exact tie, kept in original order


def test_keyword_rank_prefers_rarer_words() -> None:
    common = add_story(title="Deadline one", tags=["deadline"])
    add_story(title="Deadline two", tags=["deadline"])
    rare = add_story(title="Mentor", tags=["mentoring"])

    ranked = search.keyword_rank("deadline mentoring", storage.list_stories())

    assert ranked[0].id == rare.id
    assert common.id in _ids(ranked)


def test_keyword_rank_leaves_out_non_matches_and_keeps_order_on_ties() -> None:
    first = add_story(title="First", tags=["conflict"])
    add_story(title="Unrelated", tags=["other"], situation="s", task="t", action="a", result="r", learning="l")
    second = add_story(title="Second", tags=["conflict"])

    assert _ids(search.keyword_rank("conflict", storage.list_stories())) == [first.id, second.id]


@pytest.mark.parametrize("query", ["", "   ", "tell me about a time you"])
def test_keyword_rank_with_no_meaningful_words_matches_nothing(query: str) -> None:
    add_story()

    assert search.keyword_rank(query, storage.list_stories()) == []


def test_keyword_rank_searches_every_star_l_part() -> None:
    story = add_story(learning="Flag schedule risk early.")

    assert _ids(search.keyword_rank("schedule risk", storage.list_stories())) == [story.id]


# --- Semantic ranking -----------------------------------------------------------------------


def test_semantic_rank_orders_all_stories_by_similarity(index: SemanticIndex) -> None:
    customer = add_story(title="Customer", tags=["customer"], situation="A customer customer issue.")
    mentor = add_story(title="Mentor", tags=["mentor"], situation="I was a mentor to a mentor.")

    ranked = index.rank("mentor", storage.list_stories())

    assert ranked is not None
    assert _ids(ranked) == [mentor.id, customer.id]


def test_semantic_index_embeds_each_story_once(index: SemanticIndex, embedder: FakeEmbedder) -> None:
    add_story(title="One")
    add_story(title="Two")

    index.rank("conflict", storage.list_stories())
    index.rank("deadline", storage.list_stories())

    assert len(embedder.passages_embedded) == 2


def test_semantic_index_re_embeds_edited_stories(index: SemanticIndex, embedder: FakeEmbedder) -> None:
    story = add_story(title="Before")
    index.rank("conflict", storage.list_stories())

    storage.update_story(story.id, title="After")
    index.rank("conflict", storage.list_stories())

    assert len(embedder.passages_embedded) == 2
    assert embedder.passages_embedded[1].startswith("After.")


def test_semantic_index_returns_none_when_model_fails_and_retries_later(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []
    embedder = FakeEmbedder()

    def load() -> FakeEmbedder:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("no network")
        return embedder

    clock = [1000.0]
    monkeypatch.setattr(search.time, "monotonic", lambda: clock[0])
    index = SemanticIndex(load_embedder=load)
    add_story()

    assert index.rank("conflict", storage.list_stories()) is None
    assert index.rank("conflict", storage.list_stories()) is None  # too soon to retry
    assert len(attempts) == 1

    clock[0] += search.RETRY_AFTER_SECONDS
    assert index.rank("conflict", storage.list_stories()) is not None
    assert len(attempts) == 2


# --- Blending keyword and semantic scores ----------------------------------------------------------


def test_keyword_scores_are_the_share_of_the_query_matched() -> None:
    both = add_story(title="Mentor", tags=["mentoring", "deadline"])
    one = add_story(title="Mentor", tags=["mentoring"])
    none = add_story(title="Unrelated", tags=["other"])

    stories = storage.list_stories()
    scores = dict(zip(_ids(stories), search.keyword_scores("mentoring deadline", stories), strict=True))

    assert scores[none.id] == 0.0
    assert 0.0 < scores[one.id] < scores[both.id] < 1.0


def test_blend_weights_keyword_and_rescaled_semantic_equally() -> None:
    # Semantic similarities are rescaled to 0..1 within the query: 0.9 -> 1 and 0.5 -> 0.
    blended = search.blend([0.0, 0.8], np.array([0.9, 0.5], dtype=np.float32))

    assert blended == pytest.approx([0.5, 0.4])


def test_blend_does_not_let_weak_keyword_matches_bury_a_strong_semantic_match() -> None:
    # The right story shares no words with the question, while the others match a common word weakly.
    blended = search.blend([0.0, 0.1, 0.1, 0.1], np.array([0.9, 0.3, 0.35, 0.3], dtype=np.float32))

    assert int(np.argmax(blended)) == 0


def test_blend_with_identical_similarities_uses_keyword_scores() -> None:
    blended = search.blend([0.2, 0.6], np.array([0.4, 0.4], dtype=np.float32))

    assert int(np.argmax(blended)) == 1


# --- search_stories -------------------------------------------------------------------------------


@pytest.fixture
def peer_conflict() -> Story:
    return add_story(
        title="Disagreeing on an API design",
        tags=["conflict", "collaboration"],
        action="I set up a design review with the other senior engineer, my peer on the platform team.",
    )


@pytest.fixture
def slipped_date() -> Story:
    return add_story(title="Launch slipped", tags=["failure"], situation="We missed the deadline by a week.")


def test_natural_language_query_finds_story(peer_conflict: Story, slipped_date: Story, index: SemanticIndex) -> None:
    found = search.search_stories("conflict with a peer", index=index)

    assert found[0].id == peer_conflict.id


def test_query_words_need_not_be_adjacent(peer_conflict: Story, slipped_date: Story, index: SemanticIndex) -> None:
    found = search.search_stories("missed deadline", index=index)

    assert found[0].id == slipped_date.id


def test_search_respects_limit(index: SemanticIndex) -> None:
    for i in range(4):
        add_story(title=f"Story {i}")

    assert len(search.search_stories("conflict", limit=2, index=index)) == 2
    assert len(search.search_stories("conflict", limit=10, index=index)) == 4


def test_hybrid_returns_semantic_matches_that_share_no_words(index: SemanticIndex) -> None:
    add_story(title="Coaching", tags=["mentor"], situation="I was a mentor.")

    assert search.search_stories("zzz-unrelated-word", index=index) != []
    assert search.search_stories("zzz-unrelated-word", mode="keyword", index=index) == []


def test_hybrid_falls_back_to_keyword_when_model_unavailable(peer_conflict: Story, slipped_date: Story) -> None:
    def fail() -> FakeEmbedder:
        raise OSError("offline")

    index = SemanticIndex(load_embedder=fail)

    assert _ids(search.search_stories("missed deadline", index=index)) == [slipped_date.id]


def test_keyword_mode_never_loads_the_model(peer_conflict: Story) -> None:
    def fail() -> FakeEmbedder:
        raise AssertionError("keyword mode must not load the model")

    assert _ids(search.search_stories("conflict", mode="keyword", index=SemanticIndex(load_embedder=fail))) == [
        peer_conflict.id
    ]


def test_semantic_mode_uses_only_embeddings(index: SemanticIndex) -> None:
    mentor = add_story(title="Mentor", tags=["mentor"], situation="mentor mentor")

    assert _ids(search.search_stories("zzz mentor", mode="semantic", index=index))[0] == mentor.id


@pytest.mark.parametrize(("query", "limit", "message"), [("  ", 5, "query can't be blank"), ("x", 0, "limit")])
def test_search_rejects_bad_input(query: str, limit: int, message: str, index: SemanticIndex) -> None:
    with pytest.raises(storage.InvalidStoryError, match=message):
        search.search_stories(query, limit=limit, index=index)


def test_search_with_no_stories_returns_nothing(index: SemanticIndex) -> None:
    assert search.search_stories("conflict", index=index) == []


def test_search_tolerates_records_missing_learning(stories_path: Path, index: SemanticIndex) -> None:
    record = {
        "id": "legacy",
        "title": "Old story",
        "tags": ["conflict"],
        "situation": "s",
        "task": "t",
        "action": "a",
        "result": "r",
        "created_at": "x",
        "updated_at": "x",
    }
    stories_path.write_text(json.dumps([record]), encoding="utf-8")

    assert _ids(search.search_stories("conflict", index=index)) == ["legacy"]


# --- The real model ------------------------------------------------------------------------------


def test_real_model_ranks_a_paraphrase_first() -> None:
    """Uses the actual bge-small model (downloaded once and cached) to check the semantic path end to end."""
    peer = add_story(
        title="Settling an API design dispute",
        tags=["collaboration"],
        situation="Another senior engineer and I had opposite views on the design.",
    )
    add_story(title="Faster builds", tags=["initiative"], situation="Our CI pipeline took forty minutes.")

    ranked = search.SemanticIndex().rank("a disagreement with a coworker", storage.list_stories())

    assert ranked is not None, "embedding model failed to load"
    assert ranked[0].id == peer.id
