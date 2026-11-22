import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from finscope.__main__ import _run_known_synthesis
from finscope.config import Settings
from finscope.known_synthesis import (
    SYNTHESIS_CASES_PATH,
    KnownSynthesisError,
    SynthesisCase,
    build_synthesis_result,
    load_synthesis_cases,
    run_known_synthesis,
    score_summary,
)
from finscope.llm import LLMError, LLMResult
from finscope.synthesis import FilingSummary, Finding, SummaryPoint

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_model="fake-model",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)


def finding(ref: str, what: str) -> Finding:
    return Finding(ref=ref, section_title="S", what_changed=what, why_it_matters="It matters.")


CASE = SynthesisCase(
    name="tariffs",
    findings=(
        finding("I-1A +4", "Tariffs could raise costs."),
        finding("II-7 ~12", "Tariffs cut the margin."),
        finding("I-3 +1", "A patent lawsuit."),
    ),
    together=(("I-1A +4", "II-7 ~12"),),
    apart=(("I-1A +4", "I-3 +1"), ("II-7 ~12", "I-3 +1")),
    why="made up",
)


def summary(*points: tuple[str, ...], dropped: tuple[str, ...] = ()) -> FilingSummary:
    kept = [SummaryPoint(text="t", refs=refs, status="kept", reason="ok") for refs in points]
    gone = [SummaryPoint(text="t", refs=(), status="dropped", reason=r) for r in dropped]
    return FilingSummary(
        points=tuple(kept + gone),
        model="fake",
        prompt_version="v",
        input_tokens=900,
        output_tokens=80,
        check_input_tokens=400 * len(points),
        check_output_tokens=30 * len(points),
    )


def test_the_committed_cases_load() -> None:
    cases = load_synthesis_cases(SYNTHESIS_CASES_PATH)

    assert [case.name for case in cases] == [
        "tariff-risk-and-margin",
        "recall-risk-and-lawsuit",
        "three-unrelated",
        "shared-words-different-matters",
        "recall-across-three-sections",
    ]
    assert sum(len(case.together) for case in cases) == 3
    assert sum(len(case.apart) for case in cases) == 6


def test_a_case_naming_an_unknown_ref_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    bad = CASE.model_dump(mode="json") | {"together": [["I-1A +4", "I-9 +9"]]}
    path.write_text(json.dumps([bad]), encoding="utf-8")

    with pytest.raises(KnownSynthesisError, match="unknown refs: I-9 \\+9"):
        load_synthesis_cases(path)


@pytest.mark.parametrize(
    ("points", "missed", "joined"),
    [
        ([("I-1A +4", "II-7 ~12"), ("I-3 +1",)], (), ()),  # right
        ([("I-1A +4",), ("II-7 ~12",)], (("I-1A +4", "II-7 ~12"),), ()),  # split one matter
        ([("I-1A +4", "II-7 ~12", "I-3 +1")], (), CASE.apart),  # lumped everything together
        ([], (("I-1A +4", "II-7 ~12"),), ()),  # nothing kept
    ],
)
def test_scoring_finds_split_matters_and_wrong_joins(
    points: list[tuple[str, ...]], missed: tuple[Any, ...], joined: tuple[Any, ...]
) -> None:
    assert score_summary(CASE, summary(*points)) == (missed, joined)


def test_dropped_points_never_count_as_joins() -> None:
    lumped = SummaryPoint(
        text="t", refs=("I-1A +4", "I-3 +1"), status="dropped", reason="check: unsupported: x"
    )
    result = summary(("I-1A +4", "II-7 ~12")).model_copy(
        update={"points": (*summary(("I-1A +4", "II-7 ~12")).points, lumped)}
    )

    assert score_summary(CASE, result) == ((), ())


class ScriptedWriter:
    """complete_json for the summary: the next scripted list of points, or an LLMError."""

    def __init__(self, answers: list[list[dict[str, Any]] | LLMError]) -> None:
        self.answers = list(answers)

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        answer = self.answers.pop(0)
        if isinstance(answer, LLMError):
            raise answer
        text = json.dumps({"points": answer})
        return LLMResult(text=text, model="fake-model", input_tokens=900, output_tokens=80)


class AlwaysSupported:
    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        text = json.dumps({"verdict": "supported", "reason": "follows"})
        return LLMResult(text=text, model="fake-judge", input_tokens=400, output_tokens=30)


RIGHT = [
    {"text": "Tariffs raised costs and cut the margin.", "refs": ["I-1A +4", "II-7 ~12"]},
    {"text": "A patent lawsuit.", "refs": ["I-3 +1"]},
]
LUMPED = [{"text": "Tariffs and a lawsuit.", "refs": ["I-1A +4", "II-7 ~12", "I-3 +1"]}]


def test_a_case_passes_only_when_every_run_is_right() -> None:
    writer = ScriptedWriter([RIGHT, LUMPED])

    [result] = run_known_synthesis(
        [CASE], provider=writer, check_provider=AlwaysSupported(), runs=2
    )

    first, second = result.runs
    assert first.passed and not second.passed
    assert second.wrongly_joined == CASE.apart
    assert not result.passed


def test_a_failed_summary_call_fails_the_run_and_counts() -> None:
    writer = ScriptedWriter([LLMError("down")])

    results = run_known_synthesis([CASE], provider=writer, check_provider=AlwaysSupported(), runs=1)

    [run] = results[0].runs
    assert (run.summary, run.error) == (None, "summary call failed: down")
    built = build_synthesis_result(
        results,
        cases_file=SYNTHESIS_CASES_PATH,
        runs=1,
        settings=SETTINGS,
        created_at=datetime(2026, 11, 22, tzinfo=UTC),
    )
    assert (built.passed, built.failures) == (0, 1)


def test_the_result_counts_points_and_drop_reasons() -> None:
    with_bad_ref = [*RIGHT, {"text": "Something else.", "refs": ["I-9 +9"]}]
    results = run_known_synthesis(
        [CASE], provider=ScriptedWriter([with_bad_ref]), check_provider=AlwaysSupported(), runs=1
    )

    built = build_synthesis_result(
        results,
        cases_file=SYNTHESIS_CASES_PATH,
        runs=1,
        settings=SETTINGS,
        created_at=datetime(2026, 11, 22, tzinfo=UTC),
    )

    assert (built.passed, built.points_kept, built.points_dropped) == (1, 2, 1)
    assert built.drop_reasons == {"unknown ref(s)": 1}
    # One summary call at $2 / $8 per 1M, two checks at the same prices (no judge model set).
    assert built.cost_usd == (Decimal(900 + 800) * 2 + Decimal(80 + 60) * 8) / 1_000_000
    assert built.file_name == "20261122T000000Z_synthesis_known.json"


def test_cli_prints_cases_and_saves(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([CASE.model_dump(mode="json")]), encoding="utf-8")

    code = _run_known_synthesis(
        path,
        provider=ScriptedWriter([RIGHT, LUMPED]),
        check_provider=AlwaysSupported(),
        settings=SETTINGS,
        runs=2,
        results_dir=tmp_path / "results",
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "[tariffs] run 1/2: kept [I-1A +4; II-7 ~12], [I-3 +1]" in out
    assert "FAIL  tariffs" in out
    assert "run 2: wrongly joined: I-1A +4; I-3 +1" in out
    assert "passed: 0/1 case(s)" in out
    assert len(list((tmp_path / "results").iterdir())) == 1
