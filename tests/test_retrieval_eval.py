from __future__ import annotations

import os
from collections import Counter
from dataclasses import fields

import pytest

from evals import retrieval
from evals.retrieval import Metrics, Query, QueryResult
from stories import storage

# --- The eval dataset is well formed ------------------------------------------------

CORPUS = retrieval.load_corpus()
QUERIES = retrieval.load_queries()
STORY_IDS = {s["id"] for s in CORPUS}


def test_corpus_ids_are_unique() -> None:
    assert len(STORY_IDS) == len(CORPUS)


@pytest.mark.parametrize("record", CORPUS, ids=lambda r: r["id"])
def test_corpus_story_is_a_valid_star_l_story(record: dict) -> None:
    story_fields = {f.name for f in fields(storage.Story)}
    assert set(record) == story_fields
    for field in ("title", "situation", "task", "action", "result", "learning"):
        assert record[field].strip() == record[field] != "", field
    assert record["tags"] == storage.normalize_tags(record["tags"])


def test_query_ids_are_unique() -> None:
    assert len({q.id for q in QUERIES}) == len(QUERIES)


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.id)
def test_query_labels_point_at_real_stories(query: Query) -> None:
    assert query.question.strip()
    assert query.keywords.strip()
    assert query.best in STORY_IDS
    assert set(query.acceptable) <= STORY_IDS
    assert query.best not in query.acceptable
    assert len(set(query.acceptable)) == len(query.acceptable)


def test_every_story_is_the_best_answer_to_at_least_two_queries() -> None:
    best_counts = Counter(q.best for q in QUERIES)

    assert {story_id: best_counts[story_id] for story_id in STORY_IDS if best_counts[story_id] < 2} == {}


# --- Metric math ---------------------------------------------------------------------


def _result(ranked: tuple[str, ...], best: str = "a", acceptable: tuple[str, ...] = ()) -> QueryResult:
    query = Query(id="q", question="q", keywords="q", best=best, acceptable=acceptable)
    return QueryResult(query=query, form="question", ranked_ids=ranked)


def test_best_rank() -> None:
    assert _result(("a", "b")).best_rank == 1
    assert _result(("b", "a")).best_rank == 2
    assert _result(("b",)).best_rank is None
    assert _result(()).best_rank is None


def test_relevant_in_top_k_counts_acceptable_alternatives() -> None:
    assert _result(("x", "y", "alt"), acceptable=("alt",)).relevant_in_top_k
    assert not _result(("x", "y", "z", "alt"), acceptable=("alt",)).relevant_in_top_k


def test_summarize() -> None:
    results = [
        _result(("a", "b")),  # rank 1
        _result(("b", "a")),  # rank 2
        _result(("b", "c", "d", "a")),  # rank 4
        _result(()),  # empty
    ]

    assert retrieval.summarize(results) == Metrics(
        top1=1 / 4,
        top3=2 / 4,
        mrr=(1 + 1 / 2 + 1 / 4) / 4,
        relevant_top3=2 / 4,
        empty=1 / 4,
    )


def test_summarize_rejects_no_results() -> None:
    with pytest.raises(ValueError, match="No results"):
        retrieval.summarize([])


def test_check_thresholds_respects_direction() -> None:
    by_form = {"question": Metrics(top1=0.5, top3=0.8, mrr=0.6, relevant_top3=0.9, empty=0.1)}

    assert retrieval.check_thresholds(by_form, {"question": {"top1": 0.5, "empty": 0.1}}) == []
    assert retrieval.check_thresholds(by_form, {"question": {"top1": 0.6, "empty": 0.05}}) == [
        "question.top1 = 0.500, threshold 0.600",
        "question.empty = 0.100, threshold 0.050",
    ]


def test_thresholds_from_rounds_conservatively() -> None:
    by_form = {"question": Metrics(top1=2 / 3, top3=1.0, mrr=0.0, relevant_top3=0.5, empty=1 / 3)}

    thresholds = retrieval.thresholds_from(by_form)

    assert thresholds == {"question": {"top1": 0.666, "top3": 1.0, "mrr": 0.0, "relevant_top3": 0.5, "empty": 0.334}}
    assert retrieval.check_thresholds(by_form, thresholds) == []


def test_isolated_store_restores_stories_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STORIES_PATH", "original")

    with retrieval.isolated_store([]) as path:
        assert os.environ["STORIES_PATH"] == str(path)
        assert storage.list_stories() == []

    assert os.environ["STORIES_PATH"] == "original"


# --- The gate: search quality must not regress ---------------------------------------


def test_thresholds_cover_every_form_and_metric() -> None:
    metric_names = {f.name for f in fields(Metrics)}

    assert {form: set(limits) for form, limits in retrieval.load_thresholds().items()} == {
        form: metric_names for form in retrieval.FORMS
    }


def test_search_meets_retrieval_thresholds() -> None:
    results = retrieval.run()
    by_form = retrieval.summarize_by_form(results)

    failures = retrieval.check_thresholds(by_form, retrieval.load_thresholds())

    assert failures == [], "\n" + retrieval.format_report(results) + "\n" + "\n".join(failures)
