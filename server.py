from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from stories import storage
from stories.storage import Story, StoryError

mcp = MCPServer("interview-stories")

F = TypeVar("F", bound=Callable[..., Any])

TITLE = "A short, memorable name for the story."
TAGS = (
    'The interview scenarios the story covers, e.g. ["conflict", "leadership", "failure", "ambiguity"]. '
    'Tags are stored lowercase with hyphens, so "Conflict Resolution" becomes "conflict-resolution".'
)
SITUATION = "The context: the team, the project, and what was at stake."
TASK = "What you specifically were responsible for."
ACTION = "The concrete steps you took, in first person."
RESULT = "The outcome, measurable where possible."
LEARNING = (
    "What you took away from the experience and what you'd do differently next time. "
    "Keep this distinct from the result."
)


@dataclass
class StorySummary:
    id: str
    title: str
    tags: list[str]
    needs_learning: Annotated[bool, Field(description="True when the story has no learning yet.")]


def tool(fn: F) -> F:
    """Register fn as an MCP tool.

    Publishes its docstring without source indentation, and reports storage problems (like a
    blank field or a malformed stories file) to the client instead of as a generic crash.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except StoryError as exc:
            raise ToolError(str(exc)) from exc

    mcp.tool(description=inspect.cleandoc(fn.__doc__ or ""))(wrapper)
    return fn


def _not_found(story_id: str) -> ToolError:
    # ToolError messages reach the client. Any other exception is reported as a generic crash.
    return ToolError(f"No story with id {story_id!r}")


@tool
def list_stories() -> list[StorySummary]:
    """List all saved interview stories with id, title, and tags (not the full STAR-L text).

    needs_learning is true for stories that are missing their learning.
    """
    return [
        StorySummary(id=s.id, title=s.title, tags=s.tags, needs_learning=not s.learning.strip())
        for s in storage.list_stories()
    ]


@tool
def get_story(story_id: str) -> Story:
    """Get the full STAR-L (situation/task/action/result/learning) text of one story by id."""
    story = storage.get_story(story_id)
    if story is None:
        raise _not_found(story_id)
    return story


@tool
def search_stories(query: str) -> list[Story]:
    """Search stories by keyword across title, tags, and STAR-L text.

    Useful for finding a story that fits a scenario like 'conflict with a peer' or 'missed deadline'.
    """
    return storage.search_stories(query)


@tool
def add_story(
    title: Annotated[str, Field(description=TITLE, min_length=1)],
    tags: Annotated[list[str], Field(description=TAGS, min_length=1)],
    situation: Annotated[str, Field(description=SITUATION, min_length=1)],
    task: Annotated[str, Field(description=TASK, min_length=1)],
    action: Annotated[str, Field(description=ACTION, min_length=1)],
    result: Annotated[str, Field(description=RESULT, min_length=1)],
    learning: Annotated[str, Field(description=LEARNING, min_length=1)],
) -> Story:
    """Save a new interview story in STAR-L format: situation, task, action, result, learning."""
    return storage.add_story(title, tags, situation, task, action, result, learning)


@tool
def update_story(
    story_id: str,
    title: Annotated[str | None, Field(description=TITLE, min_length=1)] = None,
    tags: Annotated[list[str] | None, Field(description=TAGS, min_length=1)] = None,
    situation: Annotated[str | None, Field(description=SITUATION, min_length=1)] = None,
    task: Annotated[str | None, Field(description=TASK, min_length=1)] = None,
    action: Annotated[str | None, Field(description=ACTION, min_length=1)] = None,
    result: Annotated[str | None, Field(description=RESULT, min_length=1)] = None,
    learning: Annotated[str | None, Field(description=LEARNING, min_length=1)] = None,
) -> Story:
    """Update one or more STAR-L fields on an existing story.

    Omit any field you don't want to change. Fields can't be set to blank.
    """
    story = storage.update_story(
        story_id,
        title=title,
        tags=tags,
        situation=situation,
        task=task,
        action=action,
        result=result,
        learning=learning,
    )
    if story is None:
        raise _not_found(story_id)
    return story


@tool
def delete_story(story_id: str) -> str:
    """Delete a story by id."""
    if not storage.delete_story(story_id):
        raise _not_found(story_id)
    return f"Deleted story {story_id}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
