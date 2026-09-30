from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stories import storage
from tests.helpers import add_story

STAR_L_TEXT_FIELDS = ("title", "situation", "task", "action", "result", "learning")
# Required when adding. Learning may be left for later, so a story is saved rather than lost.
REQUIRED_ON_ADD = ("title", "situation", "task", "action", "result")
BLANKS = ["", "   ", "\n\t "]


# --- add_story -----------------------------------------------------------------


@pytest.mark.parametrize("field", REQUIRED_ON_ADD)
@pytest.mark.parametrize("blank", BLANKS)
def test_add_rejects_blank_text(field: str, blank: str, stories_path: Path) -> None:
    with pytest.raises(storage.InvalidStoryError, match=f"^{field} can't be blank"):
        add_story(**{field: blank})

    assert not stories_path.exists()


@pytest.mark.parametrize("blank", BLANKS)
def test_add_saves_a_draft_when_learning_is_blank(blank: str, stories_path: Path) -> None:
    story = add_story(learning=blank)

    assert story.learning == ""
    assert storage.get_story(story.id) == story
    assert json.loads(stories_path.read_text(encoding="utf-8"))[0]["learning"] == ""


def test_add_saves_a_draft_when_learning_is_omitted() -> None:
    story = storage.add_story(title="Missed launch", tags=["failure"], situation="s", task="t", action="a", result="r")

    assert story.learning == ""


def test_add_trims_outer_whitespace_but_keeps_line_breaks() -> None:
    story = add_story(title="  Missed launch \n", action="  First I paused.\n\nThen I replanned.  ")

    assert story.title == "Missed launch"
    assert story.action == "First I paused.\n\nThen I replanned."
    assert storage.get_story(story.id) == story


@pytest.mark.parametrize("tags", [[], [""], ["  ", "_", "-"]])
def test_add_requires_at_least_one_real_tag(tags: list[str], stories_path: Path) -> None:
    with pytest.raises(storage.InvalidStoryError, match="at least one tag"):
        add_story(tags=tags)

    assert not stories_path.exists()


def test_add_stores_story_as_plain_data_not_shared_with_the_result(stories_path: Path) -> None:
    story = add_story()
    story.tags.append("mutated-after-save")

    assert "mutated-after-save" not in json.loads(stories_path.read_text(encoding="utf-8"))[0]["tags"]


# --- Tag normalization ------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (["Conflict"], ["conflict"]),
        (["Conflict Resolution"], ["conflict-resolution"]),
        (["conflict_resolution"], ["conflict-resolution"]),
        (["  conflict -- resolution  "], ["conflict-resolution"]),
        (["-leadership-"], ["leadership"]),
        (["Leadership", "leadership", "LEADERSHIP"], ["leadership"]),
        (["failure", "", "ownership", "Failure"], ["failure", "ownership"]),
        (["Café Launch"], ["café-launch"]),
    ],
)
def test_tags_are_normalized(given: list[str], expected: list[str]) -> None:
    assert storage.normalize_tags(given) == expected
    assert add_story(tags=given).tags == expected


# --- update_story --------------------------------------------------------------


@pytest.mark.parametrize("field", STAR_L_TEXT_FIELDS)
def test_update_rejects_blanking_a_field(field: str, stories_path: Path) -> None:
    story = add_story()
    before = stories_path.read_bytes()

    with pytest.raises(storage.InvalidStoryError, match=f"^{field} can't be blank"):
        blank: dict[str, Any] = {field: "  "}
        storage.update_story(story.id, **blank)

    assert stories_path.read_bytes() == before


def test_update_rejects_empty_tags(stories_path: Path) -> None:
    story = add_story()
    before = stories_path.read_bytes()

    with pytest.raises(storage.InvalidStoryError, match="at least one tag"):
        storage.update_story(story.id, tags=[" "])

    assert stories_path.read_bytes() == before


def test_update_with_no_fields_is_rejected(stories_path: Path) -> None:
    story = add_story()
    before = stories_path.read_bytes()

    with pytest.raises(storage.InvalidStoryError, match="Nothing to update"):
        storage.update_story(story.id)

    assert stories_path.read_bytes() == before


def test_update_cleans_given_fields() -> None:
    story = add_story()

    updated = storage.update_story(story.id, title="  Renamed  ", tags=["Team Conflict", "team_conflict"])

    assert updated is not None
    assert updated.title == "Renamed"
    assert updated.tags == ["team-conflict"]


def test_update_rejects_unknown_fields() -> None:
    story = add_story()

    with pytest.raises(TypeError):
        storage.update_story(story.id, notes="not a story field")  # type: ignore[call-arg]


def test_validation_errors_share_a_base_with_file_errors() -> None:
    assert issubclass(storage.InvalidStoryError, storage.StoryError)
    assert issubclass(storage.StoriesFileError, storage.StoryError)
