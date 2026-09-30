from __future__ import annotations

import json
from pathlib import Path

from stories import storage


def _add(title: str = "Missed launch", tags: list[str] | None = None) -> storage.Story:
    return storage.add_story(
        title=title,
        tags=tags if tags is not None else ["failure", "ownership"],
        situation="Team of five, launch slipped a week.",
        task="I owned the release plan.",
        action="I reset expectations with stakeholders and cut scope.",
        result="Shipped two weeks later with no further slips.",
    )


def test_data_path_honors_env_override(stories_path: Path) -> None:
    assert storage.data_path() == stories_path


def test_list_is_empty_when_file_missing(stories_path: Path) -> None:
    assert not stories_path.exists()
    assert storage.list_stories() == []


def test_add_persists_story(stories_path: Path) -> None:
    story = _add()

    assert story.id
    assert story.created_at == story.updated_at
    on_disk = json.loads(stories_path.read_text(encoding="utf-8"))
    assert [s["id"] for s in on_disk] == [story.id]


def test_add_creates_missing_parent_dir(tmp_path: Path, monkeypatch) -> None:
    nested = tmp_path / "a" / "b" / "stories.json"
    monkeypatch.setenv("STORIES_PATH", str(nested))

    _add()

    assert nested.exists()


def test_get_returns_story_or_none() -> None:
    story = _add()

    assert storage.get_story(story.id) == story
    assert storage.get_story("missing") is None


def test_list_returns_all_in_insertion_order() -> None:
    first = _add("First")
    second = _add("Second")

    assert [s.id for s in storage.list_stories()] == [first.id, second.id]


def test_update_changes_only_given_fields() -> None:
    story = _add()

    updated = storage.update_story(story.id, title="New title", tags=None)

    assert updated is not None
    assert updated.title == "New title"
    assert updated.tags == story.tags
    assert updated.situation == story.situation
    assert updated.created_at == story.created_at
    assert updated.updated_at >= story.updated_at
    assert storage.get_story(story.id) == updated


def test_update_missing_returns_none() -> None:
    assert storage.update_story("missing", title="x") is None


def test_delete_removes_story() -> None:
    keep = _add("Keep")
    drop = _add("Drop")

    assert storage.delete_story(drop.id) is True
    assert [s.id for s in storage.list_stories()] == [keep.id]


def test_delete_missing_returns_false() -> None:
    _add()

    assert storage.delete_story("missing") is False


def test_search_matches_exact_substring_case_insensitively() -> None:
    story = _add(tags=["Conflict"])

    assert [s.id for s in storage.search_stories("conflict")] == [story.id]
    assert [s.id for s in storage.search_stories("STAKEHOLDERS")] == [story.id]
    assert storage.search_stories("unrelated") == []


def test_unicode_round_trips(stories_path: Path) -> None:
    story = _add(title="Café launch — résumé")

    assert storage.get_story(story.id) == story
    assert "Café" in stories_path.read_text(encoding="utf-8")
