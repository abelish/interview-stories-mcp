"""The agent eval harness, driven by a scripted model and a fake judge, so no API calls are made.

These check the plumbing the real runs depend on: an obviously correct run must pass, a do-nothing run
must fail, unwanted writes must be caught, and failures that aren't the model's must never be scored.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import pytest
from pydantic import BaseModel

from evals.agent import agent, grade, runner
from evals.agent.agent import Conversation, ModelTurn, converse
from evals.agent.cases import MISSING_LEARNING_IDS, Case, load_cases, store_records, to_markdown
from evals.agent.grade import (
    CaptureVerdict,
    DuplicateVerdict,
    GapsVerdict,
    NoFitVerdict,
    RecommendationVerdict,
    diff_stores,
)

pytestmark = pytest.mark.anyio

CASES = {c.id: c for c in load_cases()}


# --- Fakes -----------------------------------------------------------------------------------------------


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def tool(name: str, tool_input: dict[str, Any], tool_id: str = "t1") -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


class ScriptedModel:
    """Replays a fixed list of assistant turns, recording what it was sent."""

    def __init__(self, *turns: list[dict[str, Any]], model: str = "claude-opus-5-5") -> None:
        self.turns = list(turns)
        self.model = model
        self.received: list[list[dict[str, Any]]] = []

    async def respond(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> ModelTurn:
        self.received.append(json.loads(json.dumps(messages)))
        content = self.turns.pop(0) if self.turns else [text("")]
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return ModelTurn(
            content=content,
            stop_reason=stop,
            model=self.model,
            usage={"input_tokens": 100, "output_tokens": 10},
            retries=0,
        )


class FakeJudge:
    model = "claude-sonnet-5-5"

    def __init__(self, decide: Callable[[type[BaseModel], str], BaseModel]) -> None:
        self.decide = decide
        self.prompts: list[str] = []

    async def verdict(self, schema: type[BaseModel], prompt: str) -> tuple[BaseModel, dict[str, int]]:
        self.prompts.append(prompt)
        return self.decide(schema, prompt), {"input_tokens": 50, "output_tokens": 5}


def approving_judge(
    recommend: str = "none", identified: list[str] | None = None, said_no_good_fit: bool = True
) -> FakeJudge:
    def decide(schema: type[BaseModel], prompt: str) -> BaseModel:
        if schema is CaptureVerdict:
            return CaptureVerdict(reasoning="ok", parts_placed=True, faithful=True, asked_for_learning=True)
        if schema is DuplicateVerdict:
            return DuplicateVerdict(reasoning="ok", flagged_existing=True)
        if schema is RecommendationVerdict:
            return RecommendationVerdict(reasoning="ok", recommended_story_id=recommend)
        if schema is NoFitVerdict:
            return NoFitVerdict(reasoning="ok", recommended_story_id=recommend, said_no_good_fit=said_no_good_fit)
        return GapsVerdict(reasoning="ok", identified_story_ids=identified or [])

    return FakeJudge(decide)


async def run_case(case: Case, model: ScriptedModel, judge: FakeJudge, tmp_path: Path) -> grade.Grade:
    path = tmp_path / "stories.json"
    before = store_records(case.store)
    path.write_text(json.dumps(before), encoding="utf-8")
    convo = await converse(model, case.message, path)
    after = json.loads(path.read_text(encoding="utf-8"))
    return await grade.grade(case, convo, before, after, judge)


STORY = {
    "title": "Cutting on-call noise",
    "tags": ["initiative"],
    "situation": "On-call was paging five times a night.",
    "task": "I took on fixing alert noise.",
    "action": "I analysed three months of pages and rewrote the four noisiest checks.",
    "result": "Pages dropped from 40 to under 10 a week.",
    "learning": "Alert fatigue is a people problem as much as a technical one.",
}
# A different story, so the server doesn't refuse it as a duplicate of STORY.
OTHER_STORY = {
    "title": "Onboarding a new hire remotely",
    "tags": ["mentoring"],
    "situation": "A new engineer joined the team during a hiring freeze, fully remote.",
    "task": "I volunteered to be their onboarding buddy.",
    "action": "I paired with them daily for two weeks and wrote a setup guide as we went.",
    "result": "They shipped their first feature in their third week.",
    "learning": "Writing things down while onboarding someone helps everyone who comes after.",
}


# --- Cases file ------------------------------------------------------------------------------------------


def test_cases_are_well_formed() -> None:
    corpus_ids = {s["id"] for s in store_records("corpus")}
    assert len(CASES) == 20
    for case in CASES.values():
        assert case.scenario in {"capture", "find", "gaps", "no-fit"}, case.id
        if case.expect.get("best"):
            assert {case.expect["best"], *case.expect["acceptable"]} <= corpus_ids, case.id
        for story_id in case.expect.get("missing", []):
            assert story_id in MISSING_LEARNING_IDS, case.id


def test_stores_are_fresh_copies() -> None:
    a, b = store_records("corpus"), store_records("corpus")
    a[0]["title"] = "mutated"
    assert b[0]["title"] != "mutated"
    assert store_records("empty") == []
    missing = {s["id"] for s in store_records("corpus-missing-learning") if not s["learning"]}
    assert missing == set(MISSING_LEARNING_IDS)


def test_cases_md_is_up_to_date() -> None:
    committed = (runner.AGENT_DIR / "cases.md").read_text(encoding="utf-8")
    assert committed == to_markdown(list(CASES.values())), "run: uv run python -m evals.agent.cases"


# --- Diffing ----------------------------------------------------------------------------------------------


def test_diff_stores_reports_adds_removes_and_changed_fields_ignoring_updated_at() -> None:
    before = [{"id": "a", "title": "A", "updated_at": "1"}, {"id": "b", "title": "B"}]
    after = [{"id": "a", "title": "A2", "updated_at": "2"}, {"id": "c", "title": "C"}]

    diff = diff_stores(before, after)

    assert [s["id"] for s in diff.added] == ["c"]
    assert diff.removed == ["b"]
    assert diff.changed == {"a": ["title"]}
    assert not diff.unchanged
    assert diff_stores(before, [{**before[0], "updated_at": "9"}, before[1]]).unchanged


# --- The conversation loop, against the real server ---------------------------------------------------------


async def test_converse_runs_tools_on_the_real_server_and_records_a_trace(tmp_path: Path) -> None:
    path = tmp_path / "stories.json"
    path.write_text("[]", encoding="utf-8")
    model = ScriptedModel([text("Saving it."), tool("add_story", STORY)], [text("Saved!")])

    convo = await converse(model, "save my story", path)

    assert [s["title"] for s in json.loads(path.read_text(encoding="utf-8"))] == [STORY["title"]]
    assert [c["name"] for c in convo.tool_calls] == ["add_story"]
    assert convo.final_text == "Saved!"
    assert convo.turns == 2
    assert convo.usage["input_tokens"] == 200
    assert [t["role"] for t in convo.trace] == ["system", "user", "assistant", "tool_call", "tool_result", "assistant"]
    # The tool result went back to the model in the same shape the API expects.
    [result] = model.received[1][-1]["content"]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1" and not result["is_error"]
    assert json.loads(result["content"])["title"] == STORY["title"]


async def test_converse_passes_tool_errors_back_to_the_model(tmp_path: Path) -> None:
    path = tmp_path / "stories.json"
    path.write_text("[]", encoding="utf-8")
    model = ScriptedModel([tool("get_story", {"story_id": "nope"})], [text("Not found.")])

    convo = await converse(model, "get it", path)

    assert convo.tool_calls[0]["is_error"] is True
    assert "No story with id 'nope'" in model.received[1][-1]["content"][0]["content"]


async def test_converse_stops_at_the_turn_limit(tmp_path: Path) -> None:
    path = tmp_path / "stories.json"
    path.write_text("[]", encoding="utf-8")
    model = ScriptedModel(*[[tool("list_stories", {})] for _ in range(agent.MAX_TURNS + 3)])

    convo = await converse(model, "loop forever", path)

    assert convo.turns == agent.MAX_TURNS
    assert convo.stop_reason == "tool_use"


# --- Grading: oracle passes, null fails, unwanted writes are caught ------------------------------------------


async def test_capture_oracle_passes_and_null_fails(tmp_path: Path) -> None:
    case = CASES["capture-terse"]
    oracle = await run_case(
        case, ScriptedModel([tool("add_story", STORY)], [text("Saved.")]), approving_judge(), tmp_path
    )
    null = await run_case(case, ScriptedModel([text("")]), approving_judge(), tmp_path)

    assert oracle.scores == {"pass": 1.0, "safe_writes": 1.0}
    # Not saving is a wrong answer, not an unsafe write.
    assert null.scores == {"pass": 0.0, "safe_writes": 1.0}
    assert "FAIL saved_exactly_one_story" in null.explanation["pass"]


async def test_capture_saving_extra_stories_is_an_unsafe_write(tmp_path: Path) -> None:
    model = ScriptedModel(
        [tool("add_story", STORY, "t1"), tool("add_story", OTHER_STORY, "t2")],
        [text("Saved.")],
    )

    judged = await run_case(CASES["capture-terse"], model, approving_judge(), tmp_path)

    assert judged.scores == {"pass": 0.0, "safe_writes": 0.0}
    assert "FAIL no_unwanted_writes: added 2" in judged.explanation["safe_writes"]


async def test_capture_without_learning_fails_if_a_learning_is_saved(tmp_path: Path) -> None:
    case = CASES["capture-no-learning"]
    draft = {k: v for k, v in STORY.items() if k != "learning"}
    good = await run_case(
        case, ScriptedModel([tool("add_story", draft)], [text("What did you learn?")]), approving_judge(), tmp_path
    )
    invented = await run_case(
        case, ScriptedModel([tool("add_story", STORY)], [text("Saved.")]), approving_judge(), tmp_path
    )

    assert good.scores == {"pass": 1.0, "safe_writes": 1.0}
    assert invented.scores == {"pass": 0.0, "safe_writes": 0.0}


async def test_capture_duplicate_fails_if_a_copy_is_saved(tmp_path: Path) -> None:
    case = CASES["capture-duplicate"]
    asked = await run_case(
        case, ScriptedModel([text("You already have this story. Update it?")]), approving_judge(), tmp_path
    )
    copied = await run_case(
        case, ScriptedModel([tool("add_story", STORY)], [text("Saved.")]), approving_judge(), tmp_path
    )

    assert asked.scores["pass"] == 1.0
    assert copied.scores == {"pass": 0.0, "safe_writes": 0.0}


@pytest.mark.parametrize(
    ("recommend", "passed"),
    [
        ("pushback-on-roadmap", 1.0),
        ("disagree-and-commit", 1.0),
        ("missed-launch", 0.0),
        ("none", 0.0),
        ("made-up-id", 0.0),
    ],
)
async def test_find_passes_on_best_or_acceptable_story(recommend: str, passed: float, tmp_path: Path) -> None:
    judged = await run_case(
        CASES["find-disagree-boss"],
        ScriptedModel([text("Use this one.")]),
        approving_judge(recommend=recommend),
        tmp_path,
    )

    assert judged.scores == {"pass": passed, "safe_writes": 1.0}


async def test_find_catches_unwanted_writes(tmp_path: Path) -> None:
    model = ScriptedModel([tool("delete_story", {"story_id": "missed-launch"})], [text("Use the roadmap one.")])

    judged = await run_case(CASES["find-disagree-boss"], model, approving_judge("pushback-on-roadmap"), tmp_path)

    assert judged.scores == {"pass": 0.0, "safe_writes": 0.0}
    assert "removed missed-launch" in judged.explanation["safe_writes"]


async def test_no_fit_passes_only_when_nothing_is_recommended(tmp_path: Path) -> None:
    case = CASES["none-budget"]
    admitted = await run_case(case, ScriptedModel([text("None fits.")]), approving_judge("none"), tmp_path)
    stretched = await run_case(case, ScriptedModel([text("Use this.")]), approving_judge("faster-ci"), tmp_path)
    silent = await run_case(case, ScriptedModel([text("")]), approving_judge(said_no_good_fit=False), tmp_path)

    assert admitted.scores["pass"] == 1.0
    assert stretched.scores["pass"] == 0.0
    # Recommending nothing isn't enough: the reply has to tell the user nothing fits.
    assert silent.scores["pass"] == 0.0
    assert "FAIL said_no_good_fit" in silent.explanation["pass"]


async def test_gaps_requires_exactly_the_missing_stories(tmp_path: Path) -> None:
    case = CASES["gaps-which"]
    exact = await run_case(
        case,
        ScriptedModel([text("Three are incomplete.")]),
        approving_judge(identified=list(MISSING_LEARNING_IDS)),
        tmp_path,
    )
    partial = await run_case(
        case, ScriptedModel([text("One is incomplete.")]), approving_judge(identified=["faster-ci"]), tmp_path
    )

    assert exact.scores["pass"] == 1.0
    assert partial.scores["pass"] == 0.0


async def test_gaps_fails_when_it_invents_learnings(tmp_path: Path) -> None:
    model = ScriptedModel([tool("update_story", {"story_id": "faster-ci", "learning": "Invented."})], [text("Done.")])

    judged = await run_case(CASES["gaps-fill"], model, approving_judge(identified=list(MISSING_LEARNING_IDS)), tmp_path)

    assert judged.scores == {"pass": 0.0, "safe_writes": 0.0}


async def test_gaps_user_provided_learning_must_update_only_that_story(tmp_path: Path) -> None:
    case = CASES["gaps-user-provides"]
    learning = "Learning fast comes from a tight feedback loop and a friendly expert."
    right = ScriptedModel(
        [tool("update_story", {"story_id": "learned-rust-fast", "learning": learning})], [text("Done.")]
    )
    wrong = ScriptedModel([tool("update_story", {"story_id": "faster-ci", "learning": learning})], [text("Done.")])

    assert (await run_case(case, right, approving_judge(), tmp_path)).scores == {"pass": 1.0, "safe_writes": 1.0}
    assert (await run_case(case, wrong, approving_judge(), tmp_path)).scores == {"pass": 0.0, "safe_writes": 0.0}


async def test_judge_sees_the_conversation_as_data(tmp_path: Path) -> None:
    judge = approving_judge("pushback-on-roadmap")

    await run_case(CASES["find-disagree-boss"], ScriptedModel([text("Use the roadmap story.")]), judge, tmp_path)

    [prompt] = judge.prompts
    assert "<conversation>" in prompt and "Use the roadmap story." in prompt
    assert "- pushback-on-roadmap: Pushing back on my manager's quarterly roadmap" in prompt


# --- The runner: rows, errors, resume, and the harness gate -------------------------------------------------


@pytest.fixture
def flow_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    flow = tmp_path / "flow"
    monkeypatch.setattr(runner, "FLOW_DIR", flow)
    monkeypatch.setattr(runner, "STATE_PATH", flow / "_state.json")
    return flow


async def test_run_trial_writes_a_graded_row_and_trace(flow_dir: Path) -> None:
    out = runner.Outputs(flow_dir / "baseline")

    await runner.run_trial(
        CASES["capture-terse"],
        0,
        ScriptedModel([tool("add_story", STORY)], [text("Saved.")]),
        approving_judge(),
        out,
        30,
        {"effort": "medium"},
    )

    [row] = [json.loads(line) for line in out.results.read_text(encoding="utf-8").splitlines()]
    assert row["prompt_id"] == "capture-terse" and row["rep"] == 0 and row["status"] == "ok"
    assert row["grade"] == {"pass": 1.0, "safe_writes": 1.0}
    assert row["model"] == "claude-opus-5-5" and row["judge_model"] == "claude-sonnet-5-5"
    assert row["usage"]["input_tokens"] == 200 and row["tool_calls"] == 1 and row["latency_s"] > 0
    assert json.loads(out.trace("capture-terse", 0).read_text(encoding="utf-8"))[0]["role"] == "system"
    assert not out.errors.exists()


async def test_run_trial_records_timeouts_as_errors_not_scores(flow_dir: Path) -> None:
    class Slow(ScriptedModel):
        async def respond(self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> ModelTurn:
            await anyio.sleep(10)
            raise AssertionError("unreachable")

    out = runner.Outputs(flow_dir / "baseline")

    await runner.run_trial(CASES["capture-terse"], 0, Slow(), approving_judge(), out, 2, {})

    assert not out.results.exists()
    [error] = [json.loads(line) for line in out.errors.read_text(encoding="utf-8").splitlines()]
    assert error["class"] == "timeout" and error["prompt_id"] == "capture-terse"


async def test_run_trial_marks_turn_limit_as_ungraded(flow_dir: Path) -> None:
    out = runner.Outputs(flow_dir / "baseline")
    model = ScriptedModel(*[[tool("list_stories", {})] for _ in range(agent.MAX_TURNS)])

    await runner.run_trial(CASES["find-coworker"], 0, model, approving_judge(), out, 60, {})

    [row] = [json.loads(line) for line in out.results.read_text(encoding="utf-8").splitlines()]
    assert row["status"] == "turn_limit" and "grade" not in row


async def test_run_all_resumes_without_redoing_finished_trials(flow_dir: Path) -> None:
    out = runner.Outputs(flow_dir / "baseline")
    case = CASES["none-budget"]

    def fresh() -> ScriptedModel:
        return ScriptedModel([text("None fits.")])

    await runner.run_all([case], 2, fresh(), approving_judge(), out, 1, 60, {})
    first = out.results.read_text(encoding="utf-8")
    await runner.run_all([case], 2, fresh(), approving_judge(), out, 1, 60, {})

    assert out.results.read_text(encoding="utf-8") == first
    assert out.done() == {("none-budget", 0), ("none-budget", 1)}


def test_harness_gate_blocks_until_approved_and_after_changes(flow_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert runner.main(["--cases", "none-budget"]) == 2

    assert runner.main(["--approve-harness"]) == 0
    state = json.loads((flow_dir / "_state.json").read_text(encoding="utf-8"))
    assert state["harness_sha"] == runner.harness_sha()
    assert [m["id"] for m in state["metrics"]] == ["pass", "safe_writes"]

    monkeypatch.setattr(runner, "harness_sha", lambda: "changed")
    assert runner.main(["--cases", "none-budget"]) == 2


def test_variant_names_must_match_what_the_report_reads(flow_dir: Path) -> None:
    with pytest.raises(SystemExit):
        runner.main(["--variant", "better-prompt", "--summary"])


# --- Summary and cost -------------------------------------------------------------------------------------


def test_cost_prices_every_usage_field() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
    }
    # Opus 5.5: $4 in, $20 out, cache write 1.25x in, cache read 0.1x in.
    assert runner.cost_usd("claude-opus-5-5", usage) == pytest.approx(4 + 20 + 5 + 0.4)
    assert runner.cost_usd(None, usage) == 0.0
    with pytest.raises(KeyError):
        runner.cost_usd("unpriced-model", usage)


def test_wilson_interval() -> None:
    assert runner.wilson(0, 0) == (0.0, 1.0)
    lo, hi = runner.wilson(10, 20)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.4, abs=0.05)


def test_summary_reports_pass_rates_by_scenario_and_cost(flow_dir: Path) -> None:
    out = runner.Outputs(flow_dir / "baseline")
    rows = [
        {
            "prompt_id": "a",
            "rep": 0,
            "tags": ["find"],
            "status": "ok",
            "grade": {"pass": 1.0, "safe_writes": 1.0},
            "model": "claude-opus-5-5",
            "usage": {"input_tokens": 10_000, "output_tokens": 1_000},
        },
        {
            "prompt_id": "b",
            "rep": 0,
            "tags": ["find"],
            "status": "ok",
            "grade": {"pass": 0.0, "safe_writes": 1.0},
            "model": "claude-opus-5-5",
            "usage": {"input_tokens": 10_000, "output_tokens": 1_000},
        },
        {"prompt_id": "c", "rep": 0, "tags": ["capture"], "status": "truncated"},
    ]
    for row in rows:
        out.append(out.results, row)

    summary = runner.summarize(out)

    assert "3 trials recorded (2 graded, 1 not gradable)" in summary
    assert "find     pass 1/2 (50%" in summary
    assert "Cost $0.12" in summary


def test_trace_roles_are_what_the_report_renders() -> None:
    allowed = {"system", "user", "assistant", "tool_call", "tool_result"}
    convo = Conversation(trace=[{"role": "system", "content": ""}])
    assert {t["role"] for t in convo.trace} <= allowed
