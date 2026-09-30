from __future__ import annotations

import json
from pathlib import Path

from stories import storage
from tests.helpers import add_story as _add


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
    first = _add(title="First")
    second = _add(title="Second")

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
    keep = _add(title="Keep")
    drop = _add(title="Drop")

    assert storage.delete_story(drop.id) is True
    assert [s.id for s in storage.list_stories()] == [keep.id]


def test_delete_missing_returns_false() -> None:
    _add()

    assert storage.delete_story("missing") is False


def test_unicode_round_trips(stories_path: Path) -> None:
    story = _add(title="Café launch — résumé")

    assert storage.get_story(story.id) == story
    assert "Café" in stories_path.read_text(encoding="utf-8")


def test_add_persists_learning(stories_path: Path) -> None:
    story = _add()

    on_disk = json.loads(stories_path.read_text(encoding="utf-8"))
    assert on_disk[0]["learning"] == story.learning
    assert storage.get_story(story.id) == story


def test_update_changes_learning() -> None:
    story = _add()

    updated = storage.update_story(story.id, learning="Over-communicate early.")

    assert updated is not None
    assert updated.learning == "Over-communicate early."
    assert updated.result == story.result


def test_record_saved_before_star_l_loads_with_empty_learning(stories_path: Path) -> None:
    legacy = {
        "id": "legacy",
        "title": "Old story",
        "tags": ["conflict"],
        "situation": "s",
        "task": "t",
        "action": "a",
        "result": "r",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    stories_path.write_text(json.dumps([legacy]), encoding="utf-8")

    story = storage.get_story("legacy")
    assert story is not None
    assert story.learning == ""
    assert [s.id for s in storage.list_stories()] == ["legacy"]

    updated = storage.update_story("legacy", learning="Now filled in.")
    assert updated is not None
    assert updated.learning == "Now filled in."
    assert json.loads(stories_path.read_text(encoding="utf-8"))[0]["learning"] == "Now filled in."
