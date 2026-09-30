"""Retrieval eval: does search rank the right story first for real interview questions?

Runs every labeled query in evals/queries.json against the synthetic stories in evals/corpus.json,
in two forms: the raw interview question (as Claude might pass it straight through) and a keyword
rewrite (as Claude might send after reformulating). No LLM is involved, so results are deterministic.

    uv run python -m evals.retrieval              # summary table
    uv run python -m evals.retrieval --verbose    # plus every query where the best story wasn't ranked first
    uv run python -m evals.retrieval --update-thresholds   # ratchet the minimums in thresholds.json up
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stories import storage
from stories.storage import Story

EVALS_DIR = Path(__file__).resolve().parent
CORPUS_PATH = EVALS_DIR / "corpus.json"
QUERIES_PATH = EVALS_DIR / "queries.json"
THRESHOLDS_PATH = EVALS_DIR / "retrieval_thresholds.json"

FORMS = ("question", "keywords")
TOP_K = 3

SearchFn = Callable[[str], Sequence[Story]]


@dataclass(frozen=True)
class Query:
    id: str
    question: str
    keywords: str
    best: str
    acceptable: tuple[str, ...]

    def text(self, form: str) -> str:
        return self.question if form == "question" else self.keywords


@dataclass(frozen=True)
class QueryResult:
    query: Query
    form: str
    ranked_ids: tuple[str, ...]

    @property
    def best_rank(self) -> int | None:
        """1-based rank of the best story, or None if search didn't return it."""
        try:
            return self.ranked_ids.index(self.query.best) + 1
        except ValueError:
            return None

    @property
    def relevant_in_top_k(self) -> bool:
        relevant = {self.query.best, *self.query.acceptable}
        return any(story_id in relevant for story_id in self.ranked_ids[:TOP_K])


@dataclass(frozen=True)
class Metrics:
    """All values are fractions of queries, from 0 to 1. Higher is better, except empty."""

    top1: float
    """The best story is ranked first."""
    top3: float
    """The best story is in the top 3."""
    mrr: float
    """Mean reciprocal rank of the best story (1 for first, 1/2 for second, ..., 0 if missing)."""
    relevant_top3: float
    """The best story or an acceptable alternative is in the top 3."""
    empty: float
    """Search returned nothing at all. Lower is better."""


HIGHER_IS_BETTER = {"top1": True, "top3": True, "mrr": True, "relevant_top3": True, "empty": False}


def load_corpus() -> list[dict[str, Any]]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def load_queries() -> list[Query]:
    raw = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    return [
        Query(
            id=q["id"],
            question=q["question"],
            keywords=q["keywords"],
            best=q["best"],
            acceptable=tuple(q["acceptable"]),
        )
        for q in raw
    ]


@contextmanager
def isolated_store(records: list[dict[str, Any]]) -> Generator[Path, None, None]:
    """Point storage at a temp file holding records, restoring STORIES_PATH afterwards."""
    previous = os.environ.get("STORIES_PATH")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stories.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        os.environ["STORIES_PATH"] = str(path)
        try:
            yield path
        finally:
            if previous is None:
                del os.environ["STORIES_PATH"]
            else:
                os.environ["STORIES_PATH"] = previous


def run(search: SearchFn = storage.search_stories) -> list[QueryResult]:
    """Run every query in both forms against the corpus using the given search function."""
    queries = load_queries()
    with isolated_store(load_corpus()):
        return [
            QueryResult(query=q, form=form, ranked_ids=tuple(s.id for s in search(q.text(form))))
            for form in FORMS
            for q in queries
        ]


def summarize(results: Sequence[QueryResult]) -> Metrics:
    n = len(results)
    if n == 0:
        raise ValueError("No results to summarize")
    ranks = [r.best_rank for r in results]
    return Metrics(
        top1=sum(rank == 1 for rank in ranks) / n,
        top3=sum(rank is not None and rank <= TOP_K for rank in ranks) / n,
        mrr=sum(1 / rank for rank in ranks if rank is not None) / n,
        relevant_top3=sum(r.relevant_in_top_k for r in results) / n,
        empty=sum(not r.ranked_ids for r in results) / n,
    )


def summarize_by_form(results: Sequence[QueryResult]) -> dict[str, Metrics]:
    return {form: summarize([r for r in results if r.form == form]) for form in FORMS}


def load_thresholds() -> dict[str, dict[str, float]]:
    return json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))


def check_thresholds(by_form: dict[str, Metrics], thresholds: dict[str, dict[str, float]]) -> list[str]:
    """Return a description of every metric that is worse than its threshold."""
    failures = []
    for form, limits in thresholds.items():
        actual = asdict(by_form[form])
        for metric, limit in limits.items():
            value = actual[metric]
            worse = value < limit - 1e-9 if HIGHER_IS_BETTER[metric] else value > limit + 1e-9
            if worse:
                failures.append(f"{form}.{metric} = {value:.3f}, threshold {limit:.3f}")
    return failures


def thresholds_from(by_form: dict[str, Metrics]) -> dict[str, dict[str, float]]:
    """Current results as thresholds, rounded conservatively to 3 places so they're stable."""

    def conservative(metric: str, value: float) -> float:
        rounded = math.floor(value * 1000) / 1000 if HIGHER_IS_BETTER[metric] else math.ceil(value * 1000) / 1000
        return round(rounded, 3)

    return {form: {k: conservative(k, v) for k, v in asdict(m).items()} for form, m in by_form.items()}


def format_report(results: Sequence[QueryResult], verbose: bool = False) -> str:
    by_form = summarize_by_form(results)
    queries = len(results) // len(FORMS)
    lines = [
        f"Retrieval eval: {queries} queries x {len(FORMS)} forms",
        "",
        f"{'form':<10} {'top1':>6} {'top3':>6} {'mrr':>6} {'rel@3':>6} {'empty':>6}",
    ]
    for form, m in by_form.items():
        lines.append(f"{form:<10} {m.top1:>6.3f} {m.top3:>6.3f} {m.mrr:>6.3f} {m.relevant_top3:>6.3f} {m.empty:>6.3f}")
    if verbose:
        misses = [r for r in results if r.best_rank != 1]
        lines += ["", f"Best story not ranked first ({len(misses)}):"]
        for r in misses:
            got = ", ".join(r.ranked_ids[:TOP_K]) or "(nothing)"
            rank = r.best_rank or "-"
            lines.append(f"  [{r.form}] {r.query.id}: best={r.query.best} rank={rank} top3={got}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="list queries where the best story wasn't first")
    parser.add_argument(
        "--update-thresholds",
        action="store_true",
        help="save the current results as the new minimums (only if none got worse)",
    )
    args = parser.parse_args(argv)

    results = run()
    by_form = summarize_by_form(results)
    print(format_report(results, verbose=args.verbose))

    failures = check_thresholds(by_form, load_thresholds())
    if failures:
        print("\nWorse than thresholds:\n  " + "\n  ".join(failures))
        return 1
    if args.update_thresholds:
        THRESHOLDS_PATH.write_text(
            json.dumps(thresholds_from(by_form), indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"\nUpdated {THRESHOLDS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
