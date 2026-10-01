from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from stories.text import word_overlap

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "stories.json"

LOCK_TIMEOUT_SECONDS = 10.0
# os.replace can briefly fail on Windows while OneDrive, antivirus, or an editor has the file open.
REPLACE_ATTEMPTS = 6
REPLACE_BACKOFF_SECONDS = 0.05

_CONTENT_FIELDS = ("title", "situation", "task", "action", "result", "learning")
_TEXT_FIELDS = (*_CONTENT_FIELDS, "created_at", "updated_at")
_TAG_SEPARATORS = re.compile(r"[\s_-]+")

# A new story sharing at least this share of its words with a saved one is taken as a retelling of it.
# Retellings measured 0.6-0.8 and distinct stories at most 0.3, even ones on the same theme.
DUPLICATE_WORD_OVERLAP = 0.45


class StoryError(Exception):
    """Base for storage errors. The message is safe to show the user."""


class StoriesFileError(StoryError):
    """The stories file can't be locked, read, or parsed."""


class InvalidStoryError(StoryError):
    """A story field is blank or otherwise invalid."""


class DuplicateStoryError(StoryError):
    """A new story looks like one already saved, so it wasn't saved."""

    def __init__(self, existing: Story) -> None:
        super().__init__(f"This looks like a story already saved: {existing.title!r} (id {existing.id}).")
        self.existing = existing


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


def _clean_text(field: str, value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise InvalidStoryError(f"{field} can't be blank.")
    return cleaned


def normalize_tags(tags: list[str]) -> list[str]:
    """Lowercase, hyphenate spaces and underscores, and drop blanks and duplicates, keeping order.

    "Conflict Resolution", "conflict_resolution", and "conflict-resolution" all become "conflict-resolution".
    """
    cleaned = (_TAG_SEPARATORS.sub("-", tag.strip().lower()).strip("-") for tag in tags)
    unique = list(dict.fromkeys(tag for tag in cleaned if tag))
    if not unique:
        raise InvalidStoryError("tags needs at least one tag naming a scenario the story covers.")
    return unique


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


def add_story(
    title: str,
    tags: list[str],
    situation: str,
    task: str,
    action: str,
    result: str,
    learning: str = "",
    *,
    allow_duplicate: bool = False,
) -> Story:
    """Save a new story. Every part is required except learning, which may be left for later.

    A story saved without a learning shows up with needs_learning in list_stories, so it can be
    filled in once the user says what they took away, instead of being invented or lost.

    Raises DuplicateStoryError, saving nothing, if the story looks like one already saved, unless
    allow_duplicate is set. The check runs under the file lock, so two clients can't both save it.
    """
    now = _now()
    story = Story(
        id=str(uuid.uuid4()),
        title=_clean_text("title", title),
        tags=normalize_tags(tags),
        situation=_clean_text("situation", situation),
        task=_clean_text("task", task),
        action=_clean_text("action", action),
        result=_clean_text("result", result),
        learning=learning.strip(),
        created_at=now,
        updated_at=now,
    )
    with _locked() as path:
        stories = _load(path)
        if not allow_duplicate and (existing := _likely_duplicate(story, [_to_story(s) for s in stories])):
            raise DuplicateStoryError(existing)
        stories.append(asdict(story))
        _save(path, stories)
    return story


def _words(story: Story) -> str:
    parts = [story.situation, story.task, story.action, story.result, story.learning]
    return " ".join([story.title, *story.tags, *parts])


def _likely_duplicate(story: Story, saved: list[Story]) -> Story | None:
    """The saved story that story most likely retells, or None if it doesn't look like any of them."""
    overlaps = [(word_overlap(_words(story), _words(s)), s) for s in saved]
    best = max(overlaps, key=lambda pair: pair[0], default=None)
    return best[1] if best and best[0] >= DUPLICATE_WORD_OVERLAP else None


def update_story(
    story_id: str,
    *,
    title: str | None = None,
    tags: list[str] | None = None,
    situation: str | None = None,
    task: str | None = None,
    action: str | None = None,
    result: str | None = None,
    learning: str | None = None,
) -> Story | None:
    """Change the given fields, leaving the rest alone. Returns None if there's no such story."""
    given = {
        "title": title,
        "situation": situation,
        "task": task,
        "action": action,
        "result": result,
        "learning": learning,
    }
    changes: dict[str, Any] = {field: _clean_text(field, v) for field, v in given.items() if v is not None}
    if tags is not None:
        changes["tags"] = normalize_tags(tags)
    if not changes:
        raise InvalidStoryError("Nothing to update. Give at least one field to change.")

    with _locked() as path:
        stories = _load(path)
        for s in stories:
            if s["id"] == story_id:
                s.update(changes)
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
