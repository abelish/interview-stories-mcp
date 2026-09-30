from __future__ import annotations

from typing import Any

from stories import storage

STORY_FIELDS: dict[str, Any] = {
    "title": "Missed launch",
    "tags": ["failure", "ownership"],
    "situation": "Team of five, launch slipped a week.",
    "task": "I owned the release plan.",
    "action": "I reset expectations with stakeholders and cut scope.",
    "result": "Shipped two weeks later with no further slips.",
    "learning": "Flag schedule risk the day it appears, not the day it lands.",
}


def add_story(**overrides: Any) -> storage.Story:
    """Add a valid story, overriding any fields given."""
    return storage.add_story(**(STORY_FIELDS | overrides))
