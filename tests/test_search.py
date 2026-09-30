from __future__ import annotations

import pytest

from stories import storage
from tests.helpers import add_story

WHOLE_PHRASE_ONLY = pytest.mark.xfail(
    strict=True, reason="Bug: search only matches the whole query as one exact substring"
)


@pytest.fixture
def peer_conflict() -> storage.Story:
    return add_story(
        title="Disagreeing on an API design",
        tags=["conflict", "collaboration"],
        action="I set up a design review with the other senior engineer, my peer on the platform team.",
    )


@pytest.fixture
def slipped_date() -> storage.Story:
    return add_story(title="Launch slipped", tags=["failure"], situation="We missed the deadline by a week.")


@WHOLE_PHRASE_ONLY
def test_natural_language_query_finds_story(peer_conflict: storage.Story) -> None:
    assert [s.id for s in storage.search_stories("conflict with a peer")] == [peer_conflict.id]


@WHOLE_PHRASE_ONLY
def test_query_words_need_not_be_adjacent(slipped_date: storage.Story) -> None:
    assert [s.id for s in storage.search_stories("missed deadline")] == [slipped_date.id]


def test_unrelated_query_finds_nothing(peer_conflict: storage.Story, slipped_date: storage.Story) -> None:
    assert storage.search_stories("negotiation") == []
