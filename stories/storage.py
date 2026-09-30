from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stories.json"


def data_path() -> Path:
    """Where stories are stored. Override with the STORIES_PATH env var."""
    override = os.environ.get("STORIES_PATH")
    return Path(override) if override else DEFAULT_DATA_PATH


@dataclass
class Story:
    id: str
    title: str
    tags: list[str]
    situation: str
    task: str
    action: str
    result: str
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_raw() -> list[dict]:
    path = data_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_raw(stories: list[dict]) -> None:
    path = data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stories, f, indent=2, ensure_ascii=False)
        f.write("\n")


def list_stories() -> list[Story]:
    return [Story(**s) for s in _load_raw()]


def get_story(story_id: str) -> Story | None:
    for s in _load_raw():
        if s["id"] == story_id:
            return Story(**s)
    return None


def search_stories(query: str) -> list[Story]:
    query_lower = query.lower()
    matches = []
    for s in _load_raw():
        haystack = " ".join(
            [s["title"], " ".join(s["tags"]), s["situation"], s["task"], s["action"], s["result"]]
        ).lower()
        if query_lower in haystack:
            matches.append(Story(**s))
    return matches


def add_story(title: str, tags: list[str], situation: str, task: str, action: str, result: str) -> Story:
    stories = _load_raw()
    now = _now()
    story = Story(
        id=str(uuid.uuid4()),
        title=title,
        tags=tags,
        situation=situation,
        task=task,
        action=action,
        result=result,
        created_at=now,
        updated_at=now,
    )
    stories.append(story.__dict__)
    _save_raw(stories)
    return story


def update_story(story_id: str, **fields: object) -> Story | None:
    stories = _load_raw()
    for s in stories:
        if s["id"] == story_id:
            s.update({k: v for k, v in fields.items() if v is not None})
            s["updated_at"] = _now()
            _save_raw(stories)
            return Story(**s)
    return None


def delete_story(story_id: str) -> bool:
    stories = _load_raw()
    remaining = [s for s in stories if s["id"] != story_id]
    if len(remaining) == len(stories):
        return False
    _save_raw(remaining)
    return True
