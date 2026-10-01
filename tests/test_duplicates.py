from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals.retrieval import load_corpus
from stories import storage

CONTENT = ("title", "tags", "situation", "task", "action", "result", "learning")

# What Claude saved in the agent eval's capture-duplicate case: the user retelling a story already in
# the bank (missed-launch in the eval corpus) in their own words.
RETOLD_MISSED_LAUNCH: dict[str, Any] = {
    "title": "Late Risk Escalation on Mobile Checkout Rebuild",
    "tags": ["failure", "leadership", "communication", "risk-management", "deadlines"],
    "situation": "I led the rebuild of our mobile checkout flow with a team of five engineers. We had a public "
    "launch date tied to the holiday campaign.",
    "task": "As the lead, I was responsible for delivering the rebuild on time for the holiday launch and for "
    "reporting project status accurately.",
    "action": "I kept reporting the project as green even while integration testing was slipping. I only raised "
    "the risk three weeks before launch, and then we cut scope to get it shipped.",
    "result": "We launched eight days late, with reduced scope.",
    "learning": "I learned to report risk as soon as I see it.",
}


@pytest.fixture
def corpus(stories_path: Path) -> list[dict[str, Any]]:
    records = load_corpus()
    stories_path.write_text(json.dumps(records), encoding="utf-8")
    return records


def test_a_retold_story_is_refused_and_nothing_is_written(corpus: list[dict[str, Any]], stories_path: Path) -> None:
    before = stories_path.read_text(encoding="utf-8")

    with pytest.raises(storage.DuplicateStoryError) as caught:
        storage.add_story(**RETOLD_MISSED_LAUNCH)

    assert caught.value.existing.id == "missed-launch"
    assert "Missing the launch date for the mobile checkout" in str(caught.value)
    assert stories_path.read_text(encoding="utf-8") == before


def test_a_retold_story_is_saved_when_duplicates_are_allowed(corpus: list[dict[str, Any]]) -> None:
    saved = storage.add_story(**RETOLD_MISSED_LAUNCH, allow_duplicate=True)

    assert storage.get_story(saved.id) == saved
    assert len(storage.list_stories()) == len(corpus) + 1


def test_no_two_distinct_stories_in_the_eval_corpus_look_like_duplicates(stories_path: Path) -> None:
    # Many of these stories share a theme (three are about disagreeing), which must not count.
    records = load_corpus()
    for index, record in enumerate(records):
        stories_path.write_text(json.dumps(records[:index] + records[index + 1 :]), encoding="utf-8")

        storage.add_story(**{k: record[k] for k in CONTENT})


def test_a_story_is_never_a_duplicate_in_an_empty_bank() -> None:
    assert storage.add_story(**RETOLD_MISSED_LAUNCH).title == RETOLD_MISSED_LAUNCH["title"]
