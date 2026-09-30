"""Agent eval cases: loading, the story stores they start from, and the human-readable review file.

uv run python -m evals.agent.cases    # regenerate evals/agent/cases.md from cases.json
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from evals.retrieval import CORPUS_PATH

AGENT_DIR = Path(__file__).resolve().parent
CASES_PATH = AGENT_DIR / "cases.json"
CASES_MD_PATH = AGENT_DIR / "cases.md"

Store = Literal["empty", "corpus", "corpus-missing-learning"]
SCENARIOS = ("capture", "find", "gaps", "no-fit")

# The stories whose learning is removed for the "gaps" scenario.
MISSING_LEARNING_IDS = ("mentoring-junior", "learned-rust-fast", "faster-ci")

STORE_DESCRIPTIONS: dict[str, str] = {
    "empty": "no stories",
    "corpus": "the 21 synthetic eval stories",
    "corpus-missing-learning": "the 21 stories, with learning removed from 3",
}


@dataclass(frozen=True)
class Case:
    id: str
    tags: tuple[str, ...]
    store: Store
    message: str
    expect: dict[str, Any]

    @property
    def scenario(self) -> str:
        return self.tags[0]


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Case(id=c["id"], tags=tuple(c["tags"]), store=c["store"], message=c["message"], expect=c["expect"]) for c in raw
    ]


def store_records(store: Store) -> list[dict[str, Any]]:
    """The stories a case starts with. A fresh copy every time, so no case can affect another."""
    if store == "empty":
        return []
    records = copy.deepcopy(json.loads(CORPUS_PATH.read_text(encoding="utf-8")))
    if store == "corpus-missing-learning":
        for record in records:
            if record["id"] in MISSING_LEARNING_IDS:
                record["learning"] = ""
    return records


def expected_summary(case: Case) -> str:
    e = case.expect
    if "saves" in e:
        if not e["saves"]:
            return "does **not** save yet, asks first"
        if e.get("learning_empty"):
            return "saves the story with the learning left empty, and asks for it"
        return "saves exactly one complete STAR-L story"
    if "missing" in e:
        if e["updates"]:
            return "updates only " + ", ".join(f"`{k}`" for k in e["updates"]) + " with the given learning"
        return "names the 3 incomplete stories, changes nothing without your input"
    if e["best"] is None:
        return "says no story fits well"
    alt = ", ".join(f"`{a}`" for a in e["acceptable"]) or "none"
    return f"recommends `{e['best']}` (also OK: {alt})"


def to_markdown(cases: list[Case]) -> str:
    titles = {s["id"]: s["title"] for s in store_records("corpus")}
    lines = [
        "# Agent eval cases",
        "",
        "Each case sends one user message to Claude, with this project's MCP tools connected to a real",
        "server seeded with the listed store. Generated from `cases.json` by",
        "`uv run python -m evals.agent.cases`, so edit that file, not this one.",
        "",
        "| id | scenario | store | expected |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| `{c.id}` | {' / '.join(c.tags)} | {STORE_DESCRIPTIONS[c.store]} | {expected_summary(c)} |" for c in cases
    ]
    lines.append("")
    for c in cases:
        lines += [
            f"## `{c.id}`",
            "",
            f"**Store:** {STORE_DESCRIPTIONS[c.store]}. **Expected:** {expected_summary(c)}.",
            "",
        ]
        if c.expect.get("best"):
            lines += [f"Best story: *{titles[c.expect['best']]}*", ""]
        lines += ["````text", c.message, "````", ""]
    return "\n".join(lines)


def main() -> None:
    cases = load_cases()
    CASES_MD_PATH.write_text(to_markdown(cases), encoding="utf-8", newline="\n")
    print(f"Wrote {len(cases)} cases to {CASES_MD_PATH}")


if __name__ == "__main__":
    main()
