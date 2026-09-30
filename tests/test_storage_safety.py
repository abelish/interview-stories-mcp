"""Storage must survive bad records, failed saves, and concurrent writers without losing stories."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
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


def test_record_with_unknown_key_loads(stories_path: Path) -> None:
    _write_records(stories_path, [_record(notes="added by hand")])

    assert [s.id for s in storage.list_stories()] == ["rec"]


def test_record_missing_a_star_l_part_loads_with_it_empty(stories_path: Path) -> None:
    record = _record()
    del record["task"]
    _write_records(stories_path, [record])

    story = storage.get_story("rec")
    assert story is not None
    assert story.task == ""


# --- Malformed files ----------------------------------------------------------


def test_malformed_file_raises_clear_error(stories_path: Path) -> None:
    stories_path.write_text("[{not json", encoding="utf-8")

    with pytest.raises(storage.StoriesFileError, match="not valid JSON") as excinfo:
        storage.list_stories()
    assert str(stories_path) in str(excinfo.value)


def test_malformed_file_is_never_overwritten(stories_path: Path) -> None:
    stories_path.write_text("[{not json", encoding="utf-8")

    with pytest.raises(Exception):  # noqa: B017 - only the file's survival matters here
        add_story()

    assert stories_path.read_text(encoding="utf-8") == "[{not json"


# --- Crash safety --------------------------------------------------------------


def test_failed_save_leaves_previous_file_intact(stories_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = add_story(title="Existing")
    before = stories_path.read_bytes()

    def dump_then_fail(obj: Any, fp: Any, **kwargs: Any) -> None:
        fp.write('[{"id": "partial')
        raise OSError("disk full")

    with monkeypatch.context() as m, pytest.raises(OSError, match="disk full"):
        m.setattr(storage.json, "dump", dump_then_fail)
        add_story(title="Never saved")

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


def test_unknown_keys_survive_an_update(stories_path: Path) -> None:
    _write_records(stories_path, [_record(notes="added by hand")])

    storage.update_story("rec", title="Renamed")

    saved = json.loads(stories_path.read_text(encoding="utf-8"))[0]
    assert saved["notes"] == "added by hand"
    assert saved["title"] == "Renamed"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"id": "not-a-list"}', "must contain a JSON array of story objects"),
        ('["not-an-object"]', "must contain a JSON array of story objects"),
        ('[{"title": "no id"}]', r"The story at position 0 .* has no id"),
    ],
)
def test_wrong_shape_raises_clear_error(stories_path: Path, content: str, message: str) -> None:
    stories_path.write_text(content, encoding="utf-8")

    with pytest.raises(storage.StoriesFileError, match=message):
        storage.list_stories()
    with pytest.raises(storage.StoriesFileError):
        add_story()
    assert stories_path.read_text(encoding="utf-8") == content


def test_saved_file_uses_lf_line_endings(stories_path: Path) -> None:
    add_story()

    content = stories_path.read_bytes()
    assert b"\n" in content
    assert b"\r\n" not in content


def _leftover_files(directory: Path) -> set[str]:
    # The lock file is expected: filelock removes it on Windows but leaves it in place on POSIX.
    return {p.name for p in directory.iterdir()} - {"stories.json", "stories.json.lock"}


def test_failed_save_leaves_no_temp_files(stories_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    add_story()

    with monkeypatch.context() as m, pytest.raises(OSError):
        m.setattr(storage.json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        add_story()

    assert _leftover_files(stories_path.parent) == set()


def test_successful_save_leaves_no_temp_files(stories_path: Path) -> None:
    add_story()
    storage.list_stories()

    assert _leftover_files(stories_path.parent) == set()


# --- Replace retries (Windows: OneDrive, antivirus, or an editor holding the file) ---


def _flaky_replace(monkeypatch: pytest.MonkeyPatch, failures: int) -> list[int]:
    calls: list[int] = []
    real_replace = storage.os.replace

    def replace(src: Any, dst: Any) -> None:
        calls.append(1)
        if len(calls) <= failures:
            raise PermissionError("file in use")
        real_replace(src, dst)

    monkeypatch.setattr(storage.os, "replace", replace)
    monkeypatch.setattr(storage, "REPLACE_BACKOFF_SECONDS", 0)
    return calls


def test_save_retries_while_file_is_briefly_locked(stories_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        calls = _flaky_replace(m, failures=storage.REPLACE_ATTEMPTS - 1)
        story = add_story()

    assert len(calls) == storage.REPLACE_ATTEMPTS
    assert [s.id for s in storage.list_stories()] == [story.id]


def test_save_gives_up_when_file_stays_locked(stories_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = add_story(title="Existing")
    before = stories_path.read_bytes()

    with monkeypatch.context() as m:
        calls = _flaky_replace(m, failures=storage.REPLACE_ATTEMPTS)
        with pytest.raises(PermissionError):
            add_story(title="Never saved")

    assert len(calls) == storage.REPLACE_ATTEMPTS
    assert stories_path.read_bytes() == before
    assert [s.id for s in storage.list_stories()] == [existing.id]
    assert _leftover_files(stories_path.parent) == set()


# --- Lock timeout ---------------------------------------------------------------


def test_lock_timeout_raises_clear_error(stories_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from filelock import FileLock

    add_story()
    monkeypatch.setattr(storage, "LOCK_TIMEOUT_SECONDS", 0.1)
    # filelock locks are reentrant within one thread, so hold it from another thread.
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with FileLock(f"{stories_path}.lock"):
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(10)
        with pytest.raises(storage.StoriesFileError, match=r"Timed out .* waiting for another process"):
            storage.list_stories()
    finally:
        release.set()
        holder.join()
