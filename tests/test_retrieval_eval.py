import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from finscope.__main__ import _eval_retrieval
from finscope.memory import (
    MIN_SIMILARITY,
    jaccard_similarity,
    overlap_similarity,
    rank,
    words,
)
from finscope.retrieval_eval import (
    RETRIEVAL_CASES_PATH,
    QueryResult,
    RetrievalCases,
    RetrievalCasesError,
    load_retrieval_cases,
    run_retrieval_eval,
    score,
)


def finding(id_: str, what: str) -> dict[str, Any]:
    return {
        "id": id_,
        "company": "ACME",
        "section": "I-1A",
        "what_changed": what,
        "triage_reason": "",
    }


def query(name: str, text: str, relevant: list[str], kind: str = "easy") -> dict[str, Any]:
    return {
        "name": name,
        "company": "ACME",
        "section": "I-1A",
        "text": text,
        "relevant": relevant,
        "kind": kind,
        "why": "made up for a test",
    }


def cases(findings: list[dict[str, Any]], queries: list[dict[str, Any]]) -> RetrievalCases:
    return RetrievalCases.model_validate({"findings": findings, "queries": queries})


# --- the committed case file --------------------------------------------------------------------


def test_the_case_file_loads_with_every_kind_of_query() -> None:
    loaded = load_retrieval_cases(RETRIEVAL_CASES_PATH)

    assert len(loaded.findings) == 16
    kinds = Counter(q.kind for q in loaded.queries)
    assert kinds == {"easy": 4, "paraphrase": 3, "hard-negative": 1, "nothing-relevant": 3}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"relevant": ["nope"]}, "unknown findings: nope"),
        ({"relevant": [], "kind": "easy"}, "only nothing-relevant queries"),
        ({"relevant": ["a"], "kind": "nothing-relevant"}, "only nothing-relevant queries"),
    ],
)
def test_bad_queries_are_rejected(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    path = tmp_path / "cases.json"
    data = {
        "findings": [finding("a", "Tariffs")],
        "queries": [query("q", "Tariffs", ["a"]) | change],
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RetrievalCasesError, match=message):
        load_retrieval_cases(path)


def test_duplicated_finding_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    data = {"findings": [finding("a", "x"), finding("a", "y")], "queries": []}
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RetrievalCasesError, match="duplicated finding ids: a"):
        load_retrieval_cases(path)


# --- the metrics, worked out by hand -------------------------------------------------------------


def test_reciprocal_rank_is_one_over_the_first_relevant_position() -> None:
    result = QueryResult(name="q", kind="easy", relevant=("b",), retrieved=("a", "b", "c"))

    assert (result.hits, result.reciprocal_rank) == (1, 0.5)
    assert (
        QueryResult(name="q", kind="easy", relevant=("z",), retrieved=("a",)).reciprocal_rank == 0
    )


def test_scores_pool_every_query() -> None:
    small = cases(
        [
            finding("tariffs", "Tariffs on imported battery cells."),
            finding("battery-recall", "A recall of battery packs."),
            finding("lawsuit", "A lawsuit over driver assistance."),
        ],
        [
            # Retrieves tariffs (relevant) and battery-recall ("battery"): 1 hit of 2 retrieved.
            query("tariff", "Tariffs on battery cells imported from Asia.", ["tariffs"]),
            # Finds nothing: the relevant lawsuit is missed.
            query("sued", "Customers sued us over lane keeping.", ["lawsuit"], "paraphrase"),
            query("quake", "Earthquakes may damage our plants.", [], "nothing-relevant"),
        ],
    )

    scores = score(small, method="overlap", threshold=0.15)

    retrieved = {r.name: r.retrieved for r in scores.queries}
    assert retrieved == {"tariff": ("tariffs", "battery-recall"), "sued": (), "quake": ()}
    assert scores.recall_at_k == 1 / 2  # 1 of the 2 relevant findings came back
    assert scores.precision_at_k == 1 / 2  # 1 of the 2 retrieved findings was relevant
    assert scores.mrr == (1 + 0) / 2  # first relevant at rank 1 for tariff, none for sued
    assert scores.false_recall == 0


def test_the_sweep_covers_both_methods_and_includes_the_current_setting() -> None:
    loaded = load_retrieval_cases(RETRIEVAL_CASES_PATH)

    result = run_retrieval_eval(
        loaded,
        cases_file=RETRIEVAL_CASES_PATH,
        current_threshold=MIN_SIMILARITY,
        created_at=datetime(2026, 11, 21, tzinfo=UTC),
    )

    assert {s.method for s in result.sweep} == {"overlap", "jaccard"}
    [same] = [s for s in result.sweep if (s.method, s.threshold) == ("overlap", MIN_SIMILARITY)]
    assert same == result.current
    assert result.file_name == "20261121T000000Z_retrieval_known.json"


# --- memory's ranking, which the eval scores ------------------------------------------------------


def test_overlap_ignores_length_and_jaccard_does_not() -> None:
    short, long_ = words("battery tariffs"), words("battery tariffs imported asia gross margin")

    assert overlap_similarity(short, long_) == 1.0
    assert jaccard_similarity(short, long_) == 2 / 6


def test_generic_cost_words_no_longer_make_findings_similar() -> None:
    # The false recall the eval found: "costs" and "new" alone matched these two.
    insurance = "Our insurance costs could rise as we open new stores in coastal regions."
    tariffs = "Acme added a risk that tariffs on imported battery cells could raise its costs."

    assert words(insurance) & words(tariffs) == frozenset()


def test_rank_respects_the_threshold_and_method() -> None:
    loaded = load_retrieval_cases(RETRIEVAL_CASES_PATH)
    memory = tuple(f.as_past_finding() for f in loaded.findings)
    text = "The FCC may decline to renew our television broadcast licenses."

    ranked = rank(memory, text)
    assert ranked[0][1].change_id == "cobalt-fcc-licenses"
    assert all(s >= MIN_SIMILARITY for s, _ in ranked)
    assert rank(memory, text, min_similarity=1.01) == []
    assert (
        rank(memory, text, similarity=jaccard_similarity)[0][1].change_id == "cobalt-fcc-licenses"
    )


# --- CLI -----------------------------------------------------------------------------------------


def test_cli_prints_each_query_and_the_sweep_and_saves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _eval_retrieval(RETRIEVAL_CASES_PATH, tmp_path) == 0

    out = capsys.readouterr().out
    assert "11 queries over 16 made-up findings" in out
    assert f"=== NOW: overlap at {MIN_SIMILARITY:.2f} (memory.MIN_SIMILARITY) ===" in out
    assert "ok   fcc-licenses" in out
    assert "MISS customers-sued-paraphrase" in out  # the known limit of word overlap
    assert "=== SWEEP ===" in out
    [path] = tmp_path.iterdir()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["k"] == 3 and len(saved["sweep"]) == 14


def test_cli_with_a_bad_case_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cases.json"
    path.write_text("{broken", encoding="utf-8")

    assert _eval_retrieval(path, tmp_path / "results") == 1
    assert "is not a valid case file" in capsys.readouterr().err
