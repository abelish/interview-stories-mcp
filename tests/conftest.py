from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from stories import storage


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} if directory.exists() else {}


@pytest.fixture(autouse=True, scope="session")
def real_data_untouched() -> Iterator[None]:
    """Fail the run if any test created, changed, or deleted anything in the real data directory."""
    data_dir = storage.DEFAULT_DATA_PATH.parent
    before = _snapshot(data_dir)
    yield
    assert _snapshot(data_dir) == before, f"Tests modified the real data directory {data_dir}"


@pytest.fixture(autouse=True)
def stories_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point storage at a temp file for every test so real data is never touched.

    Patch other things with monkeypatch.context(), not monkeypatch.undo(), which would also undo this.
    """
    path = tmp_path / "stories.json"
    monkeypatch.setenv("STORIES_PATH", str(path))
    return path


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
