"""End-to-end tests: launch server.py as a subprocess and drive it over stdio like a real MCP client."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp_types import CallToolResult, TextContent

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client(stories_path: Path) -> AsyncIterator[Client]:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "server.py")],
        env={"STORIES_PATH": str(stories_path)},
        cwd=ROOT,
    )
    async with Client(params, read_timeout_seconds=30) as c:
        yield c


def _data(result: CallToolResult) -> Any:
    assert not result.is_error, _text(result)
    assert result.structured_content is not None
    content = result.structured_content
    # Non-object return values are wrapped as {"result": ...}.
    return content["result"] if set(content) == {"result"} else content


def _text(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


STORY_ARGS = {
    "title": "Missed launch",
    "tags": ["failure", "ownership"],
    "situation": "Team of five, launch slipped a week.",
    "task": "I owned the release plan.",
    "action": "I reset expectations with stakeholders and cut scope.",
    "result": "Shipped two weeks later with no further slips.",
}


async def test_exposes_expected_tools(client: Client) -> None:
    tools = await client.list_tools()

    assert {t.name for t in tools.tools} == {
        "list_stories",
        "get_story",
        "search_stories",
        "add_story",
        "update_story",
        "delete_story",
    }


async def test_story_tools_publish_typed_output_schemas(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    story_fields = {"id", "title", "tags", "situation", "task", "action", "result", "created_at", "updated_at"}

    for name in ("get_story", "add_story", "update_story"):
        schema = tools[name].output_schema
        assert schema is not None, name
        assert set(schema["properties"]) == story_fields, name

    list_schema = tools["list_stories"].output_schema
    assert list_schema is not None
    assert list_schema["properties"]["result"]["type"] == "array"


async def test_crud_round_trip(client: Client, stories_path: Path) -> None:
    added = _data(await client.call_tool("add_story", STORY_ARGS))
    story_id = added["id"]
    assert stories_path.exists()

    listed = _data(await client.call_tool("list_stories"))
    assert listed == [{"id": story_id, "title": "Missed launch", "tags": ["failure", "ownership"]}]

    fetched = _data(await client.call_tool("get_story", {"story_id": story_id}))
    assert fetched == added

    updated = _data(await client.call_tool("update_story", {"story_id": story_id, "title": "Renamed"}))
    assert updated["title"] == "Renamed"
    assert updated["situation"] == STORY_ARGS["situation"]

    found = _data(await client.call_tool("search_stories", {"query": "stakeholders"}))
    assert [s["id"] for s in found] == [story_id]

    deleted = await client.call_tool("delete_story", {"story_id": story_id})
    assert not deleted.is_error
    assert _data(await client.call_tool("list_stories")) == []


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("get_story", {"story_id": "missing"}),
        ("update_story", {"story_id": "missing", "title": "x"}),
        ("delete_story", {"story_id": "missing"}),
    ],
)
async def test_unknown_id_is_a_tool_error(client: Client, tool: str, args: dict[str, Any]) -> None:
    result = await client.call_tool(tool, args)

    assert result.is_error
    assert "No story with id 'missing'" in _text(result)
