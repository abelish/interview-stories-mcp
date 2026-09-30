from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def stories_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point storage at a temp file for every test so real data is never touched."""
    path = tmp_path / "stories.json"
    monkeypatch.setenv("STORIES_PATH", str(path))
    return path


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
