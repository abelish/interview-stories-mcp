"""Retrieval eval: does search rank the right story first for real interview questions?

Runs every labeled query in evals/queries.json against the synthetic stories in evals/corpus.json,
in two forms: the raw interview question (as Claude might pass it straight through) and a keyword
rewrite (as Claude might send after reformulating). No LLM is involved, so results are deterministic.

    uv run python -m evals.retrieval                    # hybrid search, the default the server uses
    uv run python -m evals.retrieval --mode keyword     # the keyword-only fallback
    uv run python -m evals.retrieval --verbose          # plus every query whose best story wasn't first
    uv run python -m evals.retrieval --update-thresholds   # ratchet this mode's minimums up
    uv run python -m evals.retrieval --personal         # your own questions against your real stories

--personal reads data/eval_queries.json (gitignored, same format as evals/queries.json, with
"keywords" optional) and runs it against a temp copy of your stories. It reports but never gates.
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
from typing import Any, get_args

from stories import search, storage
from stories.storage import Story

EVALS_DIR = Path(__file__).resolve().parent
CORPUS_PATH = EVALS_DIR / "corpus.json"
QUERIES_PATH = EVALS_DIR / "queries.json"
THRESHOLDS_PATH = EVALS_DIR / "retrieval_thresholds.json"
PERSONAL_QUERIES_PATH = storage.DEFAULT_DATA_PATH.parent / "eval_queries.json"

FORMS = ("question", "keywords")
GATED_MODES: tuple[search.Mode, ...] = ("hybrid", "keyword")
TOP_K = 3
# Rank every story, so MRR sees the full ranking rather than the server's default limit.
EVAL_LIMIT = 10_000

SearchFn = Callable[[str], Sequence[Story]]


@dataclass(frozen=True)
class Query:
    id: str
    question: str
    keywords: str | None
    best: str
    acceptable: tuple[str, ...]

    def text(self, form: str) -> str | None:
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


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_queries(path: Path = QUERIES_PATH) -> list[Query]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Query(
            id=q["id"],
            question=q["question"],
            keywords=q.get("keywords"),
            best=q["best"],
            acceptable=tuple(q.get("acceptable", [])),
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


def searcher(mode: search.Mode) -> SearchFn:
    return lambda query: search.search_stories(query, limit=EVAL_LIMIT, mode=mode)


def run(
    search_fn: SearchFn,
    corpus: list[dict[str, Any]] | None = None,
    queries: list[Query] | None = None,
) -> list[QueryResult]:
    """Run every query in each form it has against the corpus, using the given search function."""
    corpus = load_corpus() if corpus is None else corpus
    queries = load_queries() if queries is None else queries
    with isolated_store(corpus):
        return [
            QueryResult(query=q, form=form, ranked_ids=tuple(s.id for s in search_fn(text)))
            for form in FORMS
            for q in queries
            if (text := q.text(form))
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
    """Metrics for each form that has results."""
    by_form = {form: [r for r in results if r.form == form] for form in FORMS}
    return {form: summarize(rs) for form, rs in by_form.items() if rs}


def load_thresholds() -> dict[str, dict[str, dict[str, float]]]:
    """Minimums by mode, then form, then metric."""
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


def format_report(results: Sequence[QueryResult], title: str, verbose: bool = False) -> str:
    by_form = summarize_by_form(results)
    lines = [
        f"{title}: {len(results)} searches",
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


def _run_personal(mode: search.Mode, verbose: bool) -> int:
    stories_path = storage.data_path()
    if not PERSONAL_QUERIES_PATH.exists():
        print(
            f"No personal queries at {PERSONAL_QUERIES_PATH}.\n"
            "Create it in the same format as evals/queries.json, labeling each question with the id of "
            "your story that best answers it. The keywords form is optional."
        )
        return 1
    corpus = load_corpus(stories_path) if stories_path.exists() else []
    queries = load_queries(PERSONAL_QUERIES_PATH)
    unknown = sorted({q.best for q in queries} - {s["id"] for s in corpus})
    if unknown:
        print(f"These labeled story ids aren't in {stories_path}: {', '.join(unknown)}")
        return 1
    results = run(searcher(mode), corpus=corpus, queries=queries)
    print(format_report(results, f"Personal retrieval eval ({mode})", verbose=verbose))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=get_args(search.Mode), default="hybrid", help="search mode to evaluate")
    parser.add_argument("--verbose", action="store_true", help="list queries where the best story wasn't first")
    parser.add_argument(
        "--update-thresholds",
        action="store_true",
        help="save the current results as this mode's new minimums (only if none got worse)",
    )
    parser.add_argument("--personal", action="store_true", help="run data/eval_queries.json on your real stories")
    args = parser.parse_args(argv)

    if args.personal:
        if args.update_thresholds:
            parser.error("--personal results are never used as thresholds")
        return _run_personal(args.mode, args.verbose)

    results = run(searcher(args.mode))
    by_form = summarize_by_form(results)
    print(format_report(results, f"Retrieval eval ({args.mode})", verbose=args.verbose))

    all_thresholds = load_thresholds()
    if args.mode not in all_thresholds:
        print(f"\nNo thresholds for mode {args.mode!r}, so nothing to check.")
        failures = []
    else:
        failures = check_thresholds(by_form, all_thresholds[args.mode])
    if failures:
        print("\nWorse than thresholds:\n  " + "\n  ".join(failures))
        return 1
    if args.update_thresholds:
        all_thresholds[args.mode] = thresholds_from(by_form)
        THRESHOLDS_PATH.write_text(json.dumps(all_thresholds, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"\nUpdated {args.mode} thresholds in {THRESHOLDS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
