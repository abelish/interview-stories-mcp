"""Storage must survive bad records, failed saves, and concurrent writers without losing stories."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from stories import storage
from tests.helpers import add_story

ROOT = Path(__file__).resolve().parent.parent


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _record(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "rec",
        "title": "Hand-edited story",
        "tags": ["conflict"],
        "situation": "s",
        "task": "t",
        "action": "a",
        "result": "r",
        "learning": "l",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    } | overrides


# --- Records that don't exactly match the schema -----------------------------


@pytest.mark.xfail(strict=True, reason="Bug: Story(**record) raises TypeError on unknown keys")
def test_record_with_unknown_key_loads(stories_path: Path) -> None:
    _write_records(stories_path, [_record(notes="added by hand")])

    assert [s.id for s in storage.list_stories()] == ["rec"]


@pytest.mark.xfail(strict=True, reason="Bug: Story(**record) raises TypeError on missing keys")
def test_record_missing_a_star_l_part_loads_with_it_empty(stories_path: Path) -> None:
    record = _record()
    del record["task"]
    _write_records(stories_path, [record])

    story = storage.get_story("rec")
    assert story is not None
    assert story.task == ""


# --- Malformed files ----------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="Bug: malformed JSON surfaces as a bare JSONDecodeError")
def test_malformed_file_raises_clear_error(stories_path: Path) -> None:
    stories_path.write_text("[{not json", encoding="utf-8")

    with pytest.raises(Exception, match="not valid JSON") as excinfo:
        storage.list_stories()
    assert str(stories_path) in str(excinfo.value)


def test_malformed_file_is_never_overwritten(stories_path: Path) -> None:
    stories_path.write_text("[{not json", encoding="utf-8")

    with pytest.raises(Exception):  # noqa: B017 - only the file's survival matters here
        add_story()

    assert stories_path.read_text(encoding="utf-8") == "[{not json"


# --- Crash safety --------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="Bug: saves truncate the file in place, so a failed write corrupts it")
def test_failed_save_leaves_previous_file_intact(stories_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = add_story(title="Existing")
    before = stories_path.read_bytes()

    def dump_then_fail(obj: Any, fp: Any, **kwargs: Any) -> None:
        fp.write('[{"id": "partial')
        raise OSError("disk full")

    monkeypatch.setattr(storage.json, "dump", dump_then_fail)
    with pytest.raises(OSError, match="disk full"):
        add_story(title="Never saved")
    monkeypatch.undo()

    assert stories_path.read_bytes() == before
    assert [s.id for s in storage.list_stories()] == [existing.id]


# --- Concurrent writers --------------------------------------------------------

# Each writer waits at a start barrier, then pauses for a second after reading the file.
# Without a lock both writers read the same list, and the second save erases the first.
# With a lock the pause happens while holding it, so the writers run one after the other.
_WRITER = """
import json, pathlib, sys, time
sys.path.insert(0, sys.argv[1])
from stories import storage
from tests.helpers import add_story

ready, go, title = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4]
real_load = json.load

def slow_load(*args, **kwargs):
    data = real_load(*args, **kwargs)
    time.sleep(1.0)
    return data

json.load = slow_load
ready.touch()
while not go.exists():
    time.sleep(0.01)
add_story(title=title)
"""


@pytest.mark.xfail(strict=True, reason="Bug: no cross-process lock, so concurrent writers lose updates")
def test_concurrent_writers_keep_every_story(stories_path: Path, tmp_path: Path) -> None:
    add_story(title="Seed")  # the file must exist so each writer goes through json.load
    go = tmp_path / "go"
    writers = []
    for title in ("Writer A", "Writer B"):
        ready = tmp_path / f"ready-{title}"
        proc = subprocess.Popen([sys.executable, "-c", _WRITER, str(ROOT), str(ready), str(go), title])
        writers.append((proc, ready))

    deadline = time.monotonic() + 30
    while not all(ready.exists() for _, ready in writers):
        assert time.monotonic() < deadline, "writers never became ready"
        time.sleep(0.01)
    go.touch()
    for proc, _ in writers:
        assert proc.wait(timeout=30) == 0

    assert sorted(s.title for s in storage.list_stories()) == ["Seed", "Writer A", "Writer B"]
