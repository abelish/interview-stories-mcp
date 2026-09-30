from __future__ import annotations

import dataclasses

from mcp.server.mcpserver import MCPServer

from stories import storage

mcp = MCPServer("interview-stories")


@mcp.tool()
def list_stories() -> list[dict]:
    """List all saved interview stories with id, title, and tags (not the full STAR text)."""
    return [{"id": s.id, "title": s.title, "tags": s.tags} for s in storage.list_stories()]


@mcp.tool()
def get_story(story_id: str) -> dict:
    """Get the full STAR (situation/task/action/result) text of one story by id."""
    story = storage.get_story(story_id)
    if story is None:
        raise ValueError(f"No story with id {story_id!r}")
    return dataclasses.asdict(story)


@mcp.tool()
def search_stories(query: str) -> list[dict]:
    """Search stories by keyword across title, tags, and STAR text. Useful for finding a story that fits a scenario like 'conflict with a peer' or 'missed deadline'."""
    return [dataclasses.asdict(s) for s in storage.search_stories(query)]


@mcp.tool()
def add_story(
    title: str,
    tags: list[str],
    situation: str,
    task: str,
    action: str,
    result: str,
) -> dict:
    """Save a new interview story in STAR format. Tags should name the scenarios it covers, e.g. ["conflict", "leadership", "failure", "ambiguity"]."""
    story = storage.add_story(title, tags, situation, task, action, result)
    return dataclasses.asdict(story)


@mcp.tool()
def update_story(
    story_id: str,
    title: str | None = None,
    tags: list[str] | None = None,
    situation: str | None = None,
    task: str | None = None,
    action: str | None = None,
    result: str | None = None,
) -> dict:
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
        raise ValueError(f"No story with id {story_id!r}")
    return dataclasses.asdict(story)


@mcp.tool()
def delete_story(story_id: str) -> str:
    """Delete a story by id."""
    if not storage.delete_story(story_id):
        raise ValueError(f"No story with id {story_id!r}")
    return f"Deleted story {story_id}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
