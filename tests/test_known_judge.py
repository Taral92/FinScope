import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from finscope.__main__ import _run_known_judge_eval
from finscope.config import Settings
from finscope.judge import JudgeVerdict
from finscope.known_cases import (
    JUDGE_CASES_PATH,
    JudgeCase,
    KnownCasesError,
    load_judge_cases,
)
from finscope.known_judge import (
    JudgeCaseResult,
    JudgeRun,
    build_judge_result,
    case_sources,
    run_judge_cases,
)
from finscope.llm import LLMError, LLMResult
from finscope.versions import judge_prompt_version

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_model="fake-model",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)
JUDGE_TOKENS = (700, 40)


class FakeJudge:
    """Implements LLMProvider's complete_json: one scripted verdict (or LLMError) per call."""

    def __init__(self, verdicts: list[str | LLMError]) -> None:
        self.verdicts = list(verdicts)
        self.users: list[str] = []

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        self.users.append(user)
        verdict = self.verdicts.pop(0)
        if isinstance(verdict, LLMError):
            raise verdict
        return LLMResult(
            text=json.dumps({"verdict": verdict, "reason": f"JUDGE says {verdict}"}),
            model="fake-judge",
            input_tokens=JUDGE_TOKENS[0],
            output_tokens=JUDGE_TOKENS[1],
        )


def case(name: str, expected: JudgeVerdict, **changes: Any) -> JudgeCase:
    fields: dict[str, Any] = {
        "name": name,
        "kind": "added",
        "old_text": None,
        "new_text": "We may face higher costs.",
        "what_changed": "Acme added a cost risk.",
        "why_it_matters": "Costs may rise.",
        "expected": expected,
        "why": "made up for a test",
    }
    return JudgeCase.model_validate(fields | changes)


def ran(name: str, expected: JudgeVerdict, *verdicts: JudgeVerdict | None) -> JudgeCaseResult:
    """A case result with these verdicts; None is a failed call."""
    runs = tuple(
        JudgeRun(
            verdict=verdict,
            reason=None if verdict is None else "because",
            error="judge call failed: down" if verdict is None else None,
            input_tokens=None if verdict is None else 700,
            output_tokens=None if verdict is None else 40,
        )
        for verdict in verdicts
    )
    return JudgeCaseResult(name=name, expected=expected, runs=runs)


# --- the committed case file --------------------------------------------------------------------


def test_judge_case_file_has_every_verdict() -> None:
    cases = load_judge_cases(JUDGE_CASES_PATH)

    by_verdict: dict[str, set[str]] = {
        "supported": set(),
        "overstated": set(),
        "unsupported": set(),
    }
    for c in cases:
        by_verdict[c.expected].add(c.name)
    assert by_verdict == {
        "supported": {
            "faithful-may-risk",
            "realized-harm-stated",
            "claim-from-tool-result",
            "added-described-as-added",
        },
        "overstated": {
            "may-becomes-will",
            "possibility-as-fact",
            "may-harm-becomes-has-harmed",
        },
        "unsupported": {
            "wrong-attribution",
            "outside-knowledge",
            "contradicts-filing",
            "removed-described-as-added",
            "small-risk-made-big",  # widening the scope reads as a new claim, not a stronger one
        },
    }
    by_name = {c.name: c for c in cases}
    # A pair on one paragraph: the judge must tell them apart by the explanation alone.
    assert by_name["faithful-may-risk"].change() == by_name["may-becomes-will"].change()
    assert [c.name for c in cases if c.extra_sources] == ["claim-from-tool-result"]


def test_judge_cases_may_share_a_change_but_names_stay_unique(tmp_path: Path) -> None:
    path = tmp_path / "judge_cases.json"
    pair = [
        case("a", "supported").model_dump(mode="json"),
        case("b", "overstated").model_dump(mode="json"),
    ]
    path.write_text(json.dumps(pair), encoding="utf-8")
    assert [c.name for c in load_judge_cases(path)] == ["a", "b"]

    path.write_text(json.dumps([pair[0], pair[0]]), encoding="utf-8")
    with pytest.raises(KnownCasesError, match="duplicated names: a"):
        load_judge_cases(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"expected": "wrong"}, "expected"),  # not a judge verdict
        ({"why_it_matters": ""}, "why_it_matters"),
        ({"kind": "removed"}, "needs old_text"),
    ],
)
def test_bad_judge_cases_are_rejected(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    path = tmp_path / "judge_cases.json"
    fields = case("a", "supported").model_dump(mode="json") | changes
    path.write_text(json.dumps([fields]), encoding="utf-8")

    with pytest.raises(KnownCasesError, match=message):
        load_judge_cases(path)


# --- running and scoring ------------------------------------------------------------------------


def test_the_judge_sees_the_paragraph_then_the_extra_sources() -> None:
    with_tool = case("t", "supported", extra_sources=("The Ohio plant supplies all new models.",))
    removed = case(
        "r", "unsupported", kind="removed", old_text="We had one factory.", new_text=None
    )
    judge = FakeJudge(["supported", "unsupported"])

    run_judge_cases([with_tool, removed], provider=judge, settings=SETTINGS, runs=1)

    assert case_sources(with_tool) == [
        "We may face higher costs.",
        "The Ohio plant supplies all new models.",
    ]
    first, second = judge.users
    assert first.index("We may face higher costs.") < first.index("The Ohio plant supplies")
    assert "Sources: 2" in first
    assert "WHAT CHANGED: Acme added a cost risk.\nWHY IT MATTERS: Costs may rise." in first
    assert "Company: Acme Motors" in first
    assert "Change: removed paragraph" in second
    assert "We had one factory." in second


def test_a_failed_call_is_recorded_and_the_next_still_runs() -> None:
    judge = FakeJudge([LLMError("rate limited"), "overstated"])

    [result] = run_judge_cases([case("a", "overstated")], provider=judge, settings=SETTINGS, runs=2)

    assert [run.verdict for run in result.runs] == [None, "overstated"]
    assert result.runs[0].error == "judge call failed: rate limited"
    assert (result.matches, result.passed) == (1, False)


def test_scores_split_gate_mistakes_from_mix_ups() -> None:
    cases = [
        ran("faithful", "supported", "supported", "overstated"),  # one wrongly gated
        ran("may-will", "overstated", "overstated", "unsupported"),  # one mixed up: still gates
        ran("invented", "unsupported", "supported", "unsupported"),  # one let through
        ran("flaky", "unsupported", "unsupported", None),  # one failed call
    ]

    result = build_judge_result(
        cases,
        cases_file=JUDGE_CASES_PATH,
        runs=2,
        settings=SETTINGS,
        created_at=datetime(2026, 11, 20, tzinfo=UTC),
    )

    assert (result.passed, result.total_runs, result.judge_failures) == (0, 8, 1)
    assert (result.gate_right, result.wrongly_gated, result.let_through) == (5, 1, 1)
    assert result.mixed_up == 1
    assert (result.input_tokens, result.output_tokens) == (7 * 700, 7 * 40)
    assert not result.usage_complete
    assert result.judge_version == judge_prompt_version()
    assert result.file_name == "20261120T000000Z_judge_known.json"


# --- CLI end to end, with a fake judge ------------------------------------------------------------


def test_cli_runs_every_committed_case_prints_and_saves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = load_judge_cases(JUDGE_CASES_PATH)
    # Every case right in run 1; in run 2, the first case is wrongly gated.
    verdicts: list[str | LLMError] = []
    for number, c in enumerate(cases):
        verdicts += [c.expected, "overstated" if number == 0 else c.expected]
    judge = FakeJudge(verdicts)

    exit_code = _run_known_judge_eval(
        JUDGE_CASES_PATH, provider=judge, settings=SETTINGS, runs=2, results_dir=tmp_path
    )

    assert exit_code == 0
    [path] = tmp_path.iterdir()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert (saved["passed"], saved["wrongly_gated"], saved["total_runs"]) == (11, 1, 24)
    assert saved["model"] == "fake-model"

    out = capsys.readouterr().out
    assert "12 known judge case(s)" in out and "24 call(s)" in out
    first = cases[0].name
    assert f"FAIL 1/2  {first}" in out
    assert "judge: JUDGE says overstated" in out  # the reason of the run that missed
    assert "passed: 11/12 case(s)" in out
    assert "gate:   23/24 run(s) right (supported vs not)" in out
    assert "wrongly gated: 1 run(s)" in out
    assert "let through:   0 run(s)" in out


def test_cli_fails_when_every_call_failed(tmp_path: Path) -> None:
    path = tmp_path / "judge_cases.json"
    path.write_text(json.dumps([case("a", "supported").model_dump(mode="json")]), "utf-8")
    judge = FakeJudge([LLMError("down")])

    exit_code = _run_known_judge_eval(
        path, provider=judge, settings=SETTINGS, runs=1, results_dir=tmp_path / "results"
    )

    assert exit_code == 1
