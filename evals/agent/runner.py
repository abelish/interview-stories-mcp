"""Agent eval runner: Claude using this project's MCP tools, graded on what it did to the story bank.

    uv run python -m evals.agent.runner --approve-harness          # after reviewing the harness (yours to run)
    uv run python -m evals.agent.runner --cases capture-terse,find-coworker   # a pilot on some cases
    uv run python -m evals.agent.runner                            # every case, resuming where it left off
    uv run python -m evals.agent.runner --summary                  # re-print the summary of what's on disk

Needs ANTHROPIC_API_KEY. Results go to .claude/hillclimb/agent-tools/<variant>/ as results.jsonl,
traces/<case>_rep<k>.json, and errors.jsonl for attempts that failed without a gradable outcome.
Build the HTML report with the claude-api skill's report builder (the command is printed at the end).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio

from evals.agent import agent, grade
from evals.agent.agent import AnthropicModel, Model, ModelMismatchError, converse
from evals.agent.cases import AGENT_DIR, Case, load_cases, store_records
from evals.agent.grade import AnthropicJudge, Judge
from evals.retrieval import CORPUS_PATH

ROOT = agent.ROOT
FLOW_DIR = ROOT / ".claude" / "hillclimb" / "agent-tools"
STATE_PATH = FLOW_DIR / "_state.json"

# The files whose contents define how cases are run and graded. Changing any of them changes what the
# numbers mean, so the runner refuses to run until the harness is re-approved.
HARNESS_FILES = (
    AGENT_DIR / "agent.py",
    AGENT_DIR / "grade.py",
    AGENT_DIR / "runner.py",
    AGENT_DIR / "cases.py",
    AGENT_DIR / "cases.json",
    CORPUS_PATH,
)

STATE: dict[str, Any] = {
    "metrics": [
        {"id": "pass", "label": "Pass", "kind": "binary"},
        {"id": "safe_writes", "label": "Safe writes", "kind": "binary"},
    ],
    "perf_fields": [
        {"id": "in_tokens", "label": "In tokens"},
        {"id": "out_tokens", "label": "Out tokens"},
        {"id": "tool_calls", "label": "Tool calls"},
        {"id": "latency_s", "label": "Latency", "unit": "s"},
    ],
    "prices": {"claude-opus-5-5": {"in": 4.0, "out": 20.0}, "claude-sonnet-5-5": {"in": 2.0, "out": 10.0}},
}
# $ per million tokens. Cache writes cost 1.25x input and cache reads 0.1x input.
PRICES = {model: (p["in"], p["out"]) for model, p in STATE["prices"].items()}


def harness_sha() -> str:
    digest = hashlib.sha256()
    for path in HARNESS_FILES:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).digest())
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}


def approve_harness() -> None:
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    state = read_state() | STATE | {"harness_sha": harness_sha()}
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Approved harness {state['harness_sha'][:12]}")


# --- Cost ----------------------------------------------------------------------------------------------


def cost_usd(model: str | None, usage: dict[str, int] | None) -> float:
    if not model or not usage:
        return 0.0
    price_in, price_out = PRICES[model]
    tokens_in = (
        usage.get("input_tokens", 0)
        + 1.25 * usage.get("cache_creation_input_tokens", 0)
        + 0.1 * usage.get("cache_read_input_tokens", 0)
    )
    return (tokens_in * price_in + usage.get("output_tokens", 0) * price_out) / 1_000_000


def row_cost(row: dict[str, Any]) -> float:
    return cost_usd(row.get("model"), row.get("usage")) + cost_usd(row.get("judge_model"), row.get("judge_usage"))


# --- Running one trial -----------------------------------------------------------------------------------


@dataclass
class Outputs:
    variant_dir: Path

    @property
    def results(self) -> Path:
        return self.variant_dir / "results.jsonl"

    @property
    def errors(self) -> Path:
        return self.variant_dir / "errors.jsonl"

    def trace(self, case_id: str, rep: int) -> Path:
        return self.variant_dir / "traces" / f"{case_id}_rep{rep}.json"

    def done(self) -> set[tuple[str, int]]:
        return {(r["prompt_id"], r["rep"]) for r in read_jsonl(self.results)}

    def append(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def run_trial(
    case: Case, rep: int, model: Model, judge: Judge, out: Outputs, timeout_s: float, meta: dict[str, Any]
) -> None:
    """Run and grade one (case, rep). Writes a results row, or an errors row if nothing was gradable."""
    with tempfile.TemporaryDirectory() as tmp:
        stories_path = Path(tmp) / "stories.json"
        before = store_records(case.store)
        stories_path.write_text(json.dumps(before), encoding="utf-8")

        def error(failure: str, detail: str, convo: agent.Conversation | None = None) -> None:
            out.append(
                out.errors,
                {"prompt_id": case.id, "rep": rep, "class": failure, "detail": detail}
                | ({"model": convo.model, "usage": convo.usage, "retries": convo.retries} if convo else {}),
            )
            print(f"  ERROR {case.id} rep{rep} [{failure}] {detail}", flush=True)

        try:
            with anyio.fail_after(timeout_s):
                convo = await converse(model, case.message, stories_path)
        except TimeoutError:
            return error("timeout", f"no result within {timeout_s:g}s")
        except ModelMismatchError as exc:
            return error("model_mismatch", str(exc))
        except Exception as exc:
            return error("harness_or_api_error", f"{type(exc).__name__}: {exc}")

        after = json.loads(stories_path.read_text(encoding="utf-8")) if stories_path.exists() else []

    status = "ok"
    if convo.stop_reason == "max_tokens":
        status = "truncated"
    elif convo.stop_reason == "tool_use":
        status = "turn_limit"

    row: dict[str, Any] = {
        "prompt_id": case.id,
        "rep": rep,
        "prompt": case.message,
        "tags": list(case.tags),
        "stop_reason": convo.stop_reason,
        "status": status,
        "model": convo.model,
        "usage": convo.usage,
        "in_tokens": convo.usage["input_tokens"],
        "out_tokens": convo.usage["output_tokens"],
        "tool_calls": len(convo.tool_calls),
        "latency_s": round(convo.latency_s, 2),
        "meta": meta | {"turns": convo.turns, "retries": convo.retries, "refusal": convo.stop_reason == "refusal"},
    }
    if status == "ok":
        try:
            graded = await grade.grade(case, convo, before, after, judge)
        except Exception as exc:
            return error("judge_error", f"{type(exc).__name__}: {exc}", convo)
        row |= {
            "grade": graded.scores,
            "explanation": graded.explanation,
            "judge_model": graded.judge_model,
            "judge_usage": graded.judge_usage,
        }
        row["meta"]["checks"] = [{"name": c.name, "passed": c.passed, "reason": c.reason} for c in graded.checks]

    trace = out.trace(case.id, rep)
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text(json.dumps(convo.trace, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    out.append(out.results, row)
    verdict = "PASS" if row.get("grade", {}).get("pass") else "FAIL" if status == "ok" else status.upper()
    print(f"  {verdict:<10} {case.id} rep{rep}  ${row_cost(row):.3f}  {convo.latency_s:.1f}s", flush=True)


async def run_all(
    cases: Sequence[Case],
    reps: int,
    model: Model,
    judge: Judge,
    out: Outputs,
    concurrency: int,
    timeout_s: float,
    meta: dict[str, Any],
) -> None:
    done = out.done()
    todo = [(c, r) for c in cases for r in range(reps) if (c.id, r) not in done]
    if len(todo) < len(cases) * reps:
        print(f"Resuming: {len(cases) * reps - len(todo)} already done, {len(todo)} to run")
    limiter = anyio.CapacityLimiter(concurrency)

    async def one(case: Case, rep: int) -> None:
        async with limiter:
            await run_trial(case, rep, model, judge, out, timeout_s, meta)

    async with anyio.create_task_group() as tg:
        for case, rep in todo:
            tg.start_soon(one, case, rep)


# --- Summary -------------------------------------------------------------------------------------------


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a pass rate."""
    if n == 0:
        return (0.0, 1.0)
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(out: Outputs) -> str:
    rows = read_jsonl(out.results)
    errors = read_jsonl(out.errors)
    graded = [r for r in rows if r["status"] == "ok"]
    lines = [
        f"{len(rows)} trials recorded ({len(graded)} graded, {len(rows) - len(graded)} not gradable), "
        f"{len(errors)} failed attempts in errors.jsonl"
    ]
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in graded:
        by_scenario[r["tags"][0]].append(r)
    for label, group in [("all", graded), *sorted(by_scenario.items())]:
        n = len(group)
        passed = sum(r["grade"]["pass"] for r in group)
        safe = sum(r["grade"]["safe_writes"] for r in group)
        lo, hi = wilson(int(passed), n)
        lines.append(
            f"  {label:<8} pass {passed:.0f}/{n} ({passed / max(n, 1):.0%}, 95% CI {lo:.0%}-{hi:.0%})"
            f"   safe writes {safe:.0f}/{n}"
        )
    spent = sum(row_cost(r) for r in rows) + sum(cost_usd(e.get("model"), e.get("usage")) for e in errors)
    agent_cost = sum(cost_usd(r.get("model"), r.get("usage")) for r in rows)
    judge_cost = sum(cost_usd(r.get("judge_model"), r.get("judge_usage")) for r in rows)
    per_trial = sorted(row_cost(r) for r in rows)
    if per_trial:
        lines.append(
            f"Cost ${spent:.2f} (agent ${agent_cost:.2f}, judge ${judge_cost:.2f}); per trial min ${per_trial[0]:.3f} "
            f"median ${per_trial[len(per_trial) // 2]:.3f} max ${per_trial[-1]:.3f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="baseline", help="output directory name: baseline, v1, v2, ...")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--judge-model", default="claude-sonnet-5-5")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--approve-harness", action="store_true", help="record the current harness as reviewed")
    parser.add_argument("--summary", action="store_true", help="print the summary of existing results and exit")
    args = parser.parse_args(argv)

    if args.variant != "baseline" and not (args.variant.startswith("v") and args.variant[1:].isdigit()):
        parser.error("--variant must be 'baseline' or v<N>, which is what the report builder reads")
    out = Outputs(FLOW_DIR / args.variant)
    if args.approve_harness:
        approve_harness()
        return 0
    if args.summary:
        print(summarize(out))
        return 0

    current = harness_sha()
    if read_state().get("harness_sha") != current:
        print(
            "The eval harness has changed since it was last approved (or was never approved). Review the "
            "changes, then run with --approve-harness.",
            file=sys.stderr,
        )
        return 2

    cases = load_cases()
    if args.cases:
        wanted = args.cases.split(",")
        unknown = set(wanted) - {c.id for c in cases}
        if unknown:
            parser.error(f"unknown case ids: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c.id in wanted]

    meta = {"effort": args.effort, "harness_sha": current[:12]}
    print(f"Running {len(cases)} cases x {args.reps} reps on {args.model} (judge {args.judge_model})")
    start = time.perf_counter()
    anyio.run(
        run_all,
        cases,
        args.reps,
        AnthropicModel(args.model, args.effort),
        AnthropicJudge(args.judge_model),
        out,
        args.concurrency,
        args.timeout_s,
        meta,
    )
    print(f"\nDone in {time.perf_counter() - start:.0f}s\n{summarize(out)}")
    print(f"\nReport: node <claude-api skill dir>/shared/evals/report/build-report-lite.mjs {FLOW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
