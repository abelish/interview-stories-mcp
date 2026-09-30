"""End-to-end tests: launch server.py as a subprocess and drive it over stdio like a real MCP client."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp_types import CallToolResult, TextContent

from tests.helpers import STORY_FIELDS

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


STORY_ARGS = STORY_FIELDS

STAR_L_FIELDS = ("situation", "task", "action", "result", "learning")


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
    story_fields = {"id", "title", "tags", *STAR_L_FIELDS, "created_at", "updated_at"}

    for name in ("get_story", "add_story", "update_story"):
        schema = tools[name].output_schema
        assert schema is not None, name
        assert set(schema["properties"]) == story_fields, name

    list_schema = tools["list_stories"].output_schema
    assert list_schema is not None
    assert list_schema["properties"]["result"]["type"] == "array"


async def test_add_story_requires_and_describes_every_star_l_part(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    add_schema = tools["add_story"].input_schema
    update_schema = tools["update_story"].input_schema

    assert set(STAR_L_FIELDS) <= set(add_schema["required"])
    for field in STAR_L_FIELDS:
        assert add_schema["properties"][field]["description"], field
        assert update_schema["properties"][field]["description"], field
        assert field not in update_schema.get("required", []), field


async def test_tool_descriptions_say_star_l(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}

    for name in ("list_stories", "get_story", "search_stories", "add_story", "update_story"):
        description = tools[name].description or ""
        assert "STAR-L" in description, name


async def test_missing_learning_is_flagged_and_can_be_filled(client: Client, stories_path: Path) -> None:
    legacy = {k: v for k, v in STORY_ARGS.items() if k != "learning"}
    legacy |= {"id": "legacy", "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00"}
    stories_path.write_text(json.dumps([legacy]), encoding="utf-8")

    listed = _data(await client.call_tool("list_stories"))
    assert listed[0]["needs_learning"] is True
    assert _data(await client.call_tool("get_story", {"story_id": "legacy"}))["learning"] == ""

    await client.call_tool("update_story", {"story_id": "legacy", "learning": "Now filled in."})
    assert _data(await client.call_tool("list_stories"))[0]["needs_learning"] is False


async def test_crud_round_trip(client: Client, stories_path: Path) -> None:
    added = _data(await client.call_tool("add_story", STORY_ARGS))
    story_id = added["id"]
    assert stories_path.exists()

    listed = _data(await client.call_tool("list_stories"))
    assert listed == [
        {"id": story_id, "title": "Missed launch", "tags": ["failure", "ownership"], "needs_learning": False}
    ]

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


async def test_tool_descriptions_have_no_source_indentation(client: Client) -> None:
    for tool in (await client.list_tools()).tools:
        description = tool.description or ""
        assert description == description.strip(), tool.name
        assert "\n    " not in description, tool.name


async def test_needs_learning_is_described_in_output_schema(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    schema = tools["list_stories"].output_schema
    assert schema is not None

    assert schema["$defs"]["StorySummary"]["properties"]["needs_learning"]["description"]


# --- Known bugs, reproduced through the MCP protocol ----------------------------


@pytest.mark.xfail(strict=True, reason="Bug: search only matches the whole query as one exact substring")
async def test_natural_language_search_finds_story(client: Client) -> None:
    args = STORY_ARGS | {"tags": ["conflict"], "action": "I sat down with my peer to work it out."}
    added = _data(await client.call_tool("add_story", args))

    found = _data(await client.call_tool("search_stories", {"query": "conflict with a peer"}))

    assert [s["id"] for s in found] == [added["id"]]


@pytest.mark.xfail(strict=True, reason="Bug: one record with an unknown key breaks every tool")
async def test_record_with_unknown_key_does_not_break_listing(client: Client, stories_path: Path) -> None:
    record = STORY_ARGS | {"id": "rec", "notes": "added by hand", "created_at": "x", "updated_at": "x"}
    stories_path.write_text(json.dumps([record]), encoding="utf-8")

    listed = _data(await client.call_tool("list_stories"))

    assert [s["id"] for s in listed] == ["rec"]


@pytest.mark.xfail(strict=True, reason="Bug: malformed JSON reaches the client as a generic crash")
async def test_malformed_file_is_reported_clearly(client: Client, stories_path: Path) -> None:
    stories_path.write_text("[{not json", encoding="utf-8")

    result = await client.call_tool("list_stories")

    assert result.is_error
    assert "not valid JSON" in _text(result)
    assert str(stories_path) in _text(result)
