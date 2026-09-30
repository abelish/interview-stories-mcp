from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stories.json"

LOCK_TIMEOUT_SECONDS = 10.0
# os.replace can briefly fail on Windows while OneDrive, antivirus, or an editor has the file open.
REPLACE_ATTEMPTS = 6
REPLACE_BACKOFF_SECONDS = 0.05

_TEXT_FIELDS = ("title", "situation", "task", "action", "result", "learning", "created_at", "updated_at")


class StoriesFileError(Exception):
    """The stories file can't be locked, read, or parsed. The message is safe to show the user."""


def data_path() -> Path:
    """Where stories are stored. Override with the STORIES_PATH env var."""
    override = os.environ.get("STORIES_PATH")
    return Path(override) if override else DEFAULT_DATA_PATH


@dataclass
class Story:
    """An interview story in STAR-L format: situation, task, action, result, learning."""

    id: str
    title: str
    tags: list[str]
    situation: str
    task: str
    action: str
    result: str
    learning: str
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def _locked() -> Generator[Path, None, None]:
    """Hold the cross-process lock for the stories file.

    Every read and write goes through this. Writers need it so concurrent load-modify-save cycles
    don't lose updates. Readers need it because on Windows a reader holding the file open makes the
    writer's os.replace fail.
    """
    path = data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(f"{path}.lock", timeout=LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
    except Timeout as exc:
        raise StoriesFileError(
            f"Timed out after {LOCK_TIMEOUT_SECONDS:g}s waiting for another process to release {path}"
        ) from exc
    try:
        yield path
    finally:
        lock.release()


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise StoriesFileError(
            f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}: {exc.msg}). "
            "Fix or restore the file. It won't be overwritten until then."
        ) from exc
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise StoriesFileError(f"{path} must contain a JSON array of story objects.")
    for index, record in enumerate(data):
        if not isinstance(record.get("id"), str) or not record["id"]:
            raise StoriesFileError(f"The story at position {index} in {path} has no id.")
    return data


def _save(path: Path, stories: list[dict[str, Any]]) -> None:
    """Write atomically: readers and crashes see either the old file or the new one, never a partial one."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(stories, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(Path(tmp_name), path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def _replace_with_retry(src: Path, dst: Path) -> None:
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_BACKOFF_SECONDS * 2**attempt)


def _to_story(raw: dict[str, Any]) -> Story:
    """Build a Story from a stored record, tolerating hand edits.

    Unknown keys are ignored here but stay in the file. Missing text fields (including learning on
    records saved before STAR-L) load as empty so they can be filled in.
    """
    text = {field: "" if raw.get(field) is None else str(raw[field]) for field in _TEXT_FIELDS}
    tags = raw.get("tags")
    return Story(
        id=raw["id"],
        tags=[str(t) for t in tags] if isinstance(tags, list) else [],
        **text,
    )


def list_stories() -> list[Story]:
    with _locked() as path:
        return [_to_story(s) for s in _load(path)]


def get_story(story_id: str) -> Story | None:
    with _locked() as path:
        for s in _load(path):
            if s["id"] == story_id:
                return _to_story(s)
    return None


def search_stories(query: str) -> list[Story]:
    query_lower = query.lower()
    matches = []
    for story in list_stories():
        haystack = " ".join(
            [
                story.title,
                " ".join(story.tags),
                story.situation,
                story.task,
                story.action,
                story.result,
                story.learning,
            ]
        ).lower()
        if query_lower in haystack:
            matches.append(story)
    return matches


def add_story(
    title: str,
    tags: list[str],
    situation: str,
    task: str,
    action: str,
    result: str,
    learning: str,
) -> Story:
    now = _now()
    story = Story(
        id=str(uuid.uuid4()),
        title=title,
        tags=tags,
        situation=situation,
        task=task,
        action=action,
        result=result,
        learning=learning,
        created_at=now,
        updated_at=now,
    )
    with _locked() as path:
        stories = _load(path)
        stories.append(story.__dict__)
        _save(path, stories)
    return story


def update_story(story_id: str, **fields: object) -> Story | None:
    with _locked() as path:
        stories = _load(path)
        for s in stories:
            if s["id"] == story_id:
                s.update({k: v for k, v in fields.items() if v is not None})
                s["updated_at"] = _now()
                _save(path, stories)
                return _to_story(s)
    return None


def delete_story(story_id: str) -> bool:
    with _locked() as path:
        stories = _load(path)
        remaining = [s for s in stories if s["id"] != story_id]
        if len(remaining) == len(stories):
            return False
        _save(path, remaining)
    return True
