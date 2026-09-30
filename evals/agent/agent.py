"""Run one conversation: Claude with this project's real MCP server as its tools.

The server is launched over stdio exactly as Claude Code or Desktop would launch it, and its tool
definitions are passed to the model unchanged, so the eval measures the tools as they ship.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anyio
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp_types import CallToolResult, TextContent

ROOT = Path(__file__).resolve().parents[2]
SERVER_PATH = ROOT / "server.py"

# Claude Code and Desktop add their own system prompts, which can't be reproduced here. This is a
# deliberately neutral stand-in, so results reflect the tool definitions rather than prompt tuning.
SYSTEM_PROMPT = (
    "You are helping the user prepare for job interviews. They keep a bank of interview stories in "
    "STAR-L format, which you can read and edit with the tools provided."
)
MAX_TURNS = 12
MAX_TOKENS = 16_000
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


class ModelMismatchError(Exception):
    """The response came from a different model than the one requested."""


@dataclass
class ModelTurn:
    """What one model call returned, in a provider-neutral shape the loop can act on."""

    content: list[dict[str, Any]]  # content blocks as plain dicts, echoed back unchanged
    stop_reason: str
    model: str
    usage: dict[str, int]
    retries: int = 0


class Model(Protocol):
    async def respond(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> ModelTurn: ...


@dataclass
class Conversation:
    """Everything a grader or a human reviewer needs about one trial."""

    trace: list[dict[str, Any]] = field(default_factory=list)  # report transcript format
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {name, input, is_error}
    final_text: str = ""
    stop_reason: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=lambda: dict.fromkeys(USAGE_FIELDS, 0))
    turns: int = 0
    retries: int = 0
    latency_s: float = 0.0


def _result_text(result: CallToolResult) -> str:
    if result.structured_content is not None:
        content = result.structured_content
        data = content["result"] if set(content) == {"result"} else content
        return json.dumps(data, ensure_ascii=False)
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


def server_params(stories_path: Path) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        env={"STORIES_PATH": str(stories_path)},
        cwd=ROOT,
    )


async def converse(model: Model, user_message: str, stories_path: Path) -> Conversation:
    """Send one user message and let the model use the tools until it stops."""
    convo = Conversation()
    convo.trace.append({"role": "system", "content": SYSTEM_PROMPT})
    convo.trace.append({"role": "user", "content": user_message})
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    start = time.perf_counter()

    async with Client(server_params(stories_path), read_timeout_seconds=60) as mcp:
        listed = await mcp.list_tools()
        tools = [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema} for t in listed.tools
        ]

        while convo.turns < MAX_TURNS:
            turn = await model.respond(SYSTEM_PROMPT, tools, messages)
            convo.turns += 1
            convo.retries += turn.retries
            convo.model = turn.model
            convo.stop_reason = turn.stop_reason
            for key in USAGE_FIELDS:
                convo.usage[key] += turn.usage.get(key, 0)
            # Echo the assistant content back unchanged, thinking blocks included.
            messages.append({"role": "assistant", "content": turn.content})

            thinking = "\n".join(b.get("thinking", "") for b in turn.content if b["type"] == "thinking").strip()
            text = "\n".join(b["text"] for b in turn.content if b["type"] == "text").strip()
            tool_uses = [b for b in turn.content if b["type"] == "tool_use"]
            if text:
                convo.trace.append(
                    {"role": "assistant", "content": text, **({"thinking": thinking} if thinking else {})}
                )
                convo.final_text = text
                thinking = ""

            if turn.stop_reason != "tool_use" or not tool_uses:
                break

            results = []
            for block in tool_uses:
                call = await mcp.call_tool(block["name"], block["input"])
                result_text = _result_text(call)
                convo.tool_calls.append({"name": block["name"], "input": block["input"], "is_error": call.is_error})
                convo.trace.append(
                    {
                        "role": "tool_call",
                        "name": block["name"],
                        "content": json.dumps(block["input"], indent=2, ensure_ascii=False),
                        **({"thinking": thinking} if thinking else {}),
                    }
                )
                thinking = ""
                convo.trace.append({"role": "tool_result", "content": result_text})
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": result_text,
                        "is_error": call.is_error,
                    }
                )
            # All results for one assistant turn go back in a single user message.
            messages.append({"role": "user", "content": results})

    convo.latency_s = time.perf_counter() - start
    return convo


# --- The real model -------------------------------------------------------------------------------

MAX_ATTEMPTS = 6
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 60.0


async def with_backoff(call: Callable[[], Awaitable[Any]]) -> tuple[Any, int]:
    """Retry transient API errors with jittered exponential backoff. Returns (result, retries)."""
    import anthropic

    transient = (
        anthropic.RateLimitError,
        anthropic.OverloadedError,
        anthropic.InternalServerError,
        anthropic.ServiceUnavailableError,
        anthropic.APIConnectionError,
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            return await call(), attempt
        except transient:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            delay = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * 2**attempt)
            await anyio.sleep(random.uniform(0, delay))
    raise AssertionError("unreachable")


class AnthropicModel:
    """Claude via the Messages API. Retries are ours (SDK retries off) so they can be counted."""

    def __init__(self, model: str, effort: str) -> None:
        import anthropic

        self.model = model
        self.effort = effort
        self._client = anthropic.AsyncAnthropic(max_retries=0)

    async def respond(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> ModelTurn:
        response, retries = await with_backoff(
            lambda: self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=tools,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                output_config={"effort": self.effort},  # type: ignore[typeddict-item]
            )
        )
        if response.model != self.model:
            raise ModelMismatchError(f"requested {self.model}, served by {response.model}")
        usage = response.usage
        return ModelTurn(
            content=[block.model_dump(exclude_none=True) for block in response.content],
            stop_reason=response.stop_reason or "",
            model=response.model,
            usage={key: getattr(usage, key, 0) or 0 for key in USAGE_FIELDS},
            retries=retries,
        )
