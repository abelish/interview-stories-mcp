from __future__ import annotations

import functools
import inspect
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from stories import search, storage
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
NEW_LEARNING = (
    LEARNING + " Use only what the user actually said. If they haven't said what they learned, leave "
    "this empty rather than inventing one: the story is saved with needs_learning, and you should "
    "ask them for it."
)
QUERY = (
    "An interview question or scenario, as-is or as keywords, e.g. "
    "'Tell me about a time you disagreed with your manager' or 'conflict manager'."
)
LIMIT = "The most stories to return, best first."
SUMMARY_SENTENCE_CHARS = 200


@dataclass
class StorySummary:
    id: str
    title: str
    tags: list[str]
    summary: Annotated[str, Field(description="The first sentence of the situation and of the result.")]
    needs_learning: Annotated[bool, Field(description="True when the story has no learning yet.")]


_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _first_sentence(text: str) -> str:
    sentence = _SENTENCE_END.split(text.strip(), maxsplit=1)[0]
    if len(sentence) <= SUMMARY_SENTENCE_CHARS:
        return sentence
    return sentence[: SUMMARY_SENTENCE_CHARS - 3].rstrip() + "..."


def summarize(story: Story) -> StorySummary:
    parts = [_first_sentence(story.situation), _first_sentence(story.result)]
    return StorySummary(
        id=story.id,
        title=story.title,
        tags=story.tags,
        summary=" ".join(p for p in parts if p),
        needs_learning=not story.learning.strip(),
    )


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
    """List every saved interview story with its id, title, tags, and a one-line summary.

    Use this to browse or to choose between stories yourself. It doesn't include the full STAR-L
    text, which get_story returns. needs_learning is true for stories missing their learning.
    """
    return [summarize(s) for s in storage.list_stories()]


@tool
def get_story(story_id: str) -> Story:
    """Get the full STAR-L (situation/task/action/result/learning) text of one story by id."""
    story = storage.get_story(story_id)
    if story is None:
        raise _not_found(story_id)
    return story


@tool
def search_stories(
    query: Annotated[str, Field(description=QUERY, min_length=1)],
    limit: Annotated[int, Field(description=LIMIT, ge=1, le=20)] = search.DEFAULT_LIMIT,
) -> list[Story]:
    """Find the stories that best fit an interview question or scenario, best first, with full STAR-L text.

    Matches meaning as well as keywords, so the interview question can be passed as-is. It returns
    the closest stories even when none is a good fit, so check that a result actually answers the
    question before using it.
    """
    return search.search_stories(query, limit=limit)


@tool
def add_story(
    title: Annotated[str, Field(description=TITLE, min_length=1)],
    tags: Annotated[list[str], Field(description=TAGS, min_length=1)],
    situation: Annotated[str, Field(description=SITUATION, min_length=1)],
    task: Annotated[str, Field(description=TASK, min_length=1)],
    action: Annotated[str, Field(description=ACTION, min_length=1)],
    result: Annotated[str, Field(description=RESULT, min_length=1)],
    learning: Annotated[str, Field(description=NEW_LEARNING)] = "",
) -> Story:
    """Save a new interview story in STAR-L format: situation, task, action, result, learning.

    Save as soon as the user has told the story, even if they haven't given a learning yet. Leave
    learning empty in that case and ask for it, so the story isn't lost and nothing is made up.
    """
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
    # The model download logs every HTTP request at INFO. Keep stderr (Claude's MCP log) readable.
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Load the embedding model in the background so the first search doesn't wait for it.
    threading.Thread(target=search.warm_up, name="embedding-warm-up", daemon=True).start()
    mcp.run()


if __name__ == "__main__":
    main()
