from __future__ import annotations

from dataclasses import dataclass

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from stories import storage
from stories.storage import Story

mcp = MCPServer("interview-stories")


@dataclass
class StorySummary:
    id: str
    title: str
    tags: list[str]


def _not_found(story_id: str) -> ToolError:
    # ToolError messages reach the client. Any other exception is reported as a generic crash.
    return ToolError(f"No story with id {story_id!r}")


@mcp.tool()
def list_stories() -> list[StorySummary]:
    """List all saved interview stories with id, title, and tags (not the full STAR text)."""
    return [StorySummary(id=s.id, title=s.title, tags=s.tags) for s in storage.list_stories()]


@mcp.tool()
def get_story(story_id: str) -> Story:
    """Get the full STAR (situation/task/action/result) text of one story by id."""
    story = storage.get_story(story_id)
    if story is None:
        raise _not_found(story_id)
    return story


@mcp.tool()
def search_stories(query: str) -> list[Story]:
    """Search stories by keyword across title, tags, and STAR text.

    Useful for finding a story that fits a scenario like 'conflict with a peer' or 'missed deadline'.
    """
    return storage.search_stories(query)


@mcp.tool()
def add_story(
    title: str,
    tags: list[str],
    situation: str,
    task: str,
    action: str,
    result: str,
) -> Story:
    """Save a new interview story in STAR format.

    Tags should name the scenarios it covers, e.g. ["conflict", "leadership", "failure", "ambiguity"].
    """
    return storage.add_story(title, tags, situation, task, action, result)


@mcp.tool()
def update_story(
    story_id: str,
    title: str | None = None,
    tags: list[str] | None = None,
    situation: str | None = None,
    task: str | None = None,
    action: str | None = None,
    result: str | None = None,
) -> Story:
    """Update one or more fields on an existing story. Omit any field you don't want to change."""
    story = storage.update_story(
        story_id,
        title=title,
        tags=tags,
        situation=situation,
        task=task,
        action=action,
        result=result,
    )
    if story is None:
        raise _not_found(story_id)
    return story


@mcp.tool()
def delete_story(story_id: str) -> str:
    """Delete a story by id."""
    if not storage.delete_story(story_id):
        raise _not_found(story_id)
    return f"Deleted story {story_id}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
