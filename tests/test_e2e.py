"""End-to-end tests: launch server.py as a subprocess and drive it over stdio like a real MCP client."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp_types import CallToolResult, TextContent

from tests.helpers import STORY_FIELDS

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.anyio


def _server_params(stories_path: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "server.py")],
        env={"STORIES_PATH": str(stories_path)},
        cwd=ROOT,
    )


@pytest.fixture
async def client(stories_path: Path) -> AsyncIterator[Client]:
    async with Client(_server_params(stories_path), read_timeout_seconds=30) as c:
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
        {
            "id": story_id,
            "title": "Missed launch",
            "tags": ["failure", "ownership"],
            "summary": "Team of five, launch slipped a week. Shipped two weeks later with no further slips.",
            "needs_learning": False,
        }
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


# --- Robustness and known bugs, through the MCP protocol ----------------------


async def test_natural_language_search_ranks_the_right_story_first(client: Client) -> None:
    peer = _data(
        await client.call_tool(
            "add_story",
            STORY_ARGS
            | {
                "title": "Settling an API design dispute",
                "tags": ["collaboration"],
                "situation": "Another senior engineer and I had opposite views on the design.",
            },
        )
    )
    await client.call_tool(
        "add_story",
        STORY_ARGS | {"title": "Faster builds", "tags": ["initiative"], "situation": "CI took forty minutes."},
    )

    found = _data(
        await client.call_tool("search_stories", {"query": "Tell me about a time you disagreed with a coworker."})
    )

    assert found[0]["id"] == peer["id"]
    assert set(found[0]) >= set(STAR_L_FIELDS)


async def test_search_limit_is_bounded_and_respected(client: Client) -> None:
    for i in range(3):
        await client.call_tool("add_story", STORY_ARGS | {"title": f"Story {i}"})
    tools = {t.name: t for t in (await client.list_tools()).tools}
    limit_schema = tools["search_stories"].input_schema["properties"]["limit"]

    assert (limit_schema["minimum"], limit_schema["maximum"], limit_schema["default"]) == (1, 20, 5)
    assert len(_data(await client.call_tool("search_stories", {"query": "launch", "limit": 2}))) == 2
    result = await client.call_tool("search_stories", {"query": "   "})
    assert result.is_error
    assert "query can't be blank" in _text(result)


async def test_record_with_unknown_key_does_not_break_listing(client: Client, stories_path: Path) -> None:
    record = STORY_ARGS | {"id": "rec", "notes": "added by hand", "created_at": "x", "updated_at": "x"}
    stories_path.write_text(json.dumps([record]), encoding="utf-8")

    listed = _data(await client.call_tool("list_stories"))

    assert [s["id"] for s in listed] == ["rec"]


async def test_malformed_file_is_reported_clearly(client: Client, stories_path: Path) -> None:
    stories_path.write_text("[{not json", encoding="utf-8")

    result = await client.call_tool("list_stories")

    assert result.is_error
    assert "not valid JSON" in _text(result)
    assert str(stories_path) in _text(result)


async def test_unknown_keys_survive_an_update_through_mcp(client: Client, stories_path: Path) -> None:
    record = STORY_ARGS | {"id": "rec", "notes": "added by hand", "created_at": "x", "updated_at": "x"}
    stories_path.write_text(json.dumps([record]), encoding="utf-8")

    updated = _data(await client.call_tool("update_story", {"story_id": "rec", "title": "Renamed"}))

    assert updated["title"] == "Renamed"
    assert "notes" not in updated
    assert json.loads(stories_path.read_text(encoding="utf-8"))[0]["notes"] == "added by hand"


async def test_two_servers_writing_at_once_keep_every_story(stories_path: Path) -> None:
    """Like Claude Code and Claude Desktop each running the server against the same file."""
    per_server = 15

    async def add_many(server: str) -> None:
        async with Client(_server_params(stories_path), read_timeout_seconds=30) as c:
            for i in range(per_server):
                result = await c.call_tool("add_story", STORY_ARGS | {"title": f"{server} {i}"})
                assert not result.is_error, _text(result)

    async with anyio.create_task_group() as tg:
        tg.start_soon(add_many, "A")
        tg.start_soon(add_many, "B")

    titles = {s["title"] for s in json.loads(stories_path.read_text(encoding="utf-8"))}
    assert titles == {f"{server} {i}" for server in "AB" for i in range(per_server)}


# --- Validation through the MCP protocol ------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("add_story", STORY_ARGS | {"task": ""}, "String should have at least 1 character"),
        ("add_story", STORY_ARGS | {"task": "   "}, "task can't be blank"),
        ("add_story", STORY_ARGS | {"tags": []}, "List should have at least 1 item"),
        ("add_story", STORY_ARGS | {"tags": [" ", "_"]}, "at least one tag"),
        ("update_story", {"story_id": "any"}, "Nothing to update"),
        ("update_story", {"story_id": "any", "learning": "  "}, "learning can't be blank"),
    ],
)
async def test_invalid_input_is_a_clear_tool_error(
    client: Client, stories_path: Path, tool: str, args: dict[str, Any], message: str
) -> None:
    result = await client.call_tool(tool, args)

    assert result.is_error
    assert message in _text(result)
    assert not stories_path.exists()


async def test_text_fields_publish_min_length(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    add_props = tools["add_story"].input_schema["properties"]

    for field in (*STAR_L_FIELDS, "title"):
        assert add_props[field]["minLength"] == 1, field
    assert add_props["tags"]["minItems"] == 1


async def test_tags_are_normalized_through_mcp(client: Client) -> None:
    added = _data(await client.call_tool("add_story", STORY_ARGS | {"tags": ["Team Conflict", "team_conflict"]}))

    assert added["tags"] == ["team-conflict"]
