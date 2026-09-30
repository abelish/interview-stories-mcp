from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import fields
from pathlib import Path

import pytest

from evals import retrieval
from evals.retrieval import Metrics, Query, QueryResult
from stories import storage
from stories.search import Mode

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
    assert query.keywords is not None, "the committed eval set needs both forms"
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


# --- Running the eval ------------------------------------------------------------------


def test_queries_without_keywords_only_run_the_question_form() -> None:
    corpus = [c for c in CORPUS if c["id"] == "mentoring-junior"]
    queries = [Query(id="q", question="Tell me about someone you mentored.", keywords=None, best="mentoring-junior",
                     acceptable=())]  # fmt: skip

    results = retrieval.run(retrieval.searcher("keyword"), corpus=corpus, queries=queries)

    assert [(r.form, r.best_rank) for r in results] == [("question", 1)]
    assert set(retrieval.summarize_by_form(results)) == {"question"}


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_personal_eval_explains_how_to_start_when_no_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(retrieval, "PERSONAL_QUERIES_PATH", tmp_path / "eval_queries.json")

    assert retrieval.main(["--personal", "--mode", "keyword"]) == 1
    assert "No personal queries" in capsys.readouterr().out


def test_personal_eval_rejects_labels_for_unknown_stories(
    tmp_path: Path, stories_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    queries_path = tmp_path / "eval_queries.json"
    _write_json(queries_path, [{"id": "q", "question": "Tell me about a mentee.", "best": "no-such-story"}])
    monkeypatch.setattr(retrieval, "PERSONAL_QUERIES_PATH", queries_path)

    assert retrieval.main(["--personal", "--mode", "keyword"]) == 1
    assert "no-such-story" in capsys.readouterr().out


def test_personal_eval_scores_real_stories_without_changing_them(
    tmp_path: Path, stories_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_json(stories_path, [c for c in CORPUS if c["id"] in {"mentoring-junior", "faster-ci"}])
    before = stories_path.read_bytes()
    queries_path = tmp_path / "eval_queries.json"
    query = {"id": "q", "question": "Tell me about someone you mentored.", "best": "mentoring-junior"}
    _write_json(queries_path, [query])
    monkeypatch.setattr(retrieval, "PERSONAL_QUERIES_PATH", queries_path)

    assert retrieval.main(["--personal", "--mode", "keyword"]) == 0

    out = capsys.readouterr().out
    assert "Personal retrieval eval (keyword): 1 searches" in out
    assert "question    1.000" in out
    assert stories_path.read_bytes() == before


def test_personal_eval_refuses_to_update_thresholds() -> None:
    with pytest.raises(SystemExit):
        retrieval.main(["--personal", "--update-thresholds"])


# --- The gate: search quality must not regress ---------------------------------------


def test_thresholds_cover_every_gated_mode_form_and_metric() -> None:
    metric_names = {f.name for f in fields(Metrics)}
    thresholds = retrieval.load_thresholds()

    assert set(thresholds) == set(retrieval.GATED_MODES)
    for mode, by_form in thresholds.items():
        assert {form: set(limits) for form, limits in by_form.items()} == {
            form: metric_names for form in retrieval.FORMS
        }, mode


@pytest.mark.parametrize("mode", retrieval.GATED_MODES)
def test_search_meets_retrieval_thresholds(mode: Mode) -> None:
    results = retrieval.run(retrieval.searcher(mode))
    by_form = retrieval.summarize_by_form(results)

    failures = retrieval.check_thresholds(by_form, retrieval.load_thresholds()[mode])

    report = retrieval.format_report(results, f"Retrieval eval ({mode})")
    assert failures == [], "\n".join(["", report, *failures])
