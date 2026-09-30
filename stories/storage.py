from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stories.json"


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
    return datetime.now(timezone.utc).isoformat()


def _load_raw() -> list[dict]:
    if not DATA_PATH.exists():
        return []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_raw(stories: list[dict]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", encoding="utf-8") as f:
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
