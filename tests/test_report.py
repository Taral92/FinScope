import json
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

import finscope.__main__ as cli
from finscope.__main__ import _print_report
from finscope.config import Settings
from finscope.diff import ParagraphChange, SectionDiff
from finscope.edgar import Filing
from finscope.llm import (
    AgentStep,
    ConversationState,
    LLMError,
    LLMResult,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from finscope.report import SectionReport, build_section_report
from finscope.sections import Section
from finscope.triage import TriageError

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)
TRIAGE_TOKENS = (1_000, 200)


def added(text: str) -> ParagraphChange:
    return ParagraphChange(
        kind="added",
        old_text=None,
        new_text=text,
        old_start=None,
        old_end=None,
        new_start=0,
        new_end=len(text),
        similarity=None,
    )


ROBOTAXI = added("Upon launching our Robotaxi service, we entered a new market.")
TARIFFS = added("New tariffs on imported components could raise our costs.")
OPTIMUS = added("We are developing Optimus, a general purpose humanoid robot.")
CEO = added("We depend on the services of our Chief Executive Officer.")
DIFF = SectionDiff(
    part="I",
    item="1A",
    title="RISK FACTORS",
    unchanged=10,
    changes=(ROBOTAXI, TARIFFS, OPTIMUS, CEO),
)
SECTION = Section(part="I", item="1A", title="RISK FACTORS", start=0, end=12, text="Item 1A. Risk")


def ratings(*levels: str) -> dict[str, str]:
    """Triage levels for +1, +2, ... in order."""
    return {f"+{number}": level for number, level in enumerate(levels, start=1)}


class FakeTriage:
    """Implements LLMProvider's complete_json: rates each change as told."""

    def __init__(self, levels: dict[str, str] | None = None, *, text: str | None = None) -> None:
        items = [
            {"id": i, "importance": lvl, "reason": f"reason {i}"}
            for i, lvl in (levels or {}).items()
        ]
        self.text = text if text is not None else json.dumps({"items": items})

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        return LLMResult(
            text=self.text,
            model="fake-model",
            input_tokens=TRIAGE_TOKENS[0],
            output_tokens=TRIAGE_TOKENS[1],
        )


Run = list[AgentStep] | LLMError


class FakeAgent:
    """Implements LLMProvider's start/resume: one scripted run per agent call, in order.

    A run is its steps, or an LLMError that its first model call raises.
    """

    def __init__(self, runs: list[Run]) -> None:
        self.runs = list(runs)
        self.users: list[str] = []  # the user message of every run started
        self._steps: list[AgentStep] = []

    def start(
        self, system: str, user: str, tools: list[ToolSpec]
    ) -> tuple[AgentStep, ConversationState]:
        self.users.append(user)
        run = self.runs.pop(0)
        if isinstance(run, LLMError):
            raise run
        self._steps = list(run)
        return self._steps.pop(0), ConversationState()

    def resume(
        self, state: ConversationState, tool_results: list[ToolResult]
    ) -> tuple[AgentStep, ConversationState]:
        return self._steps.pop(0), state


def answer(
    quote: str, *, what: str = "A new risk was added.", tokens: tuple[int, int] = (300, 50)
) -> list[AgentStep]:
    text = (
        f"WHAT CHANGED: {what}\n"
        "WHY IT MATTERS: It is a new risk for the business.\n"
        "IMPORTANCE: high - new risk\n"
        f'EVIDENCE: "{quote}"'
    )
    return [AgentStep(text=text, tool_calls=(), input_tokens=tokens[0], output_tokens=tokens[1])]


def no_answer(tokens: tuple[int, int] = (120, 30)) -> list[AgentStep]:
    """A step with neither text nor tool calls: the agent loop stops with an AgentError."""
    return [AgentStep(text=None, tool_calls=(), input_tokens=tokens[0], output_tokens=tokens[1])]


def report(
    triage: FakeTriage, agent: FakeAgent, max_explain: int = 5, triage_runs: int = 1
) -> SectionReport:
    return build_section_report(
        DIFF,
        old_section=SECTION,
        new_section=SECTION,
        metrics=(),
        company="TSLA",
        form="10-K",
        section_title="Part I Item 1A. RISK FACTORS",
        triage_provider=triage,
        agent_provider=agent,
        settings=SETTINGS,
        max_explain=max_explain,
        triage_runs=triage_runs,
    )


def test_high_changes_are_explained_and_verified() -> None:
    triage = FakeTriage(ratings("high", "medium", "high", "low"))
    agent = FakeAgent([answer("we entered a new market"), answer("humanoid robot")])

    result = report(triage, agent)

    assert (result.change_count, result.high_count, result.medium_count, result.low_count) == (
        4,
        2,
        1,
        1,
    )
    assert [item.id for item in result.high] == ["+1", "+3"]
    assert [item.triage_reason for item in result.high] == ["reason +1", "reason +3"]
    # Each run got its own paragraph, in diff order.
    assert ROBOTAXI.new_text in agent.users[0]
    assert OPTIMUS.new_text in agent.users[1]
    for item in result.high:
        assert item.error is None
        assert item.explanation is not None
        assert item.verification is not None
        assert item.verification.passed
        assert item.verification.quotes_checked == 1
        assert item.parsed is not None and item.parsed.importance == "high"
    assert result.not_explained == ()
    assert [(m.id, m.triage_reason) for m in result.medium] == [("+2", "reason +2")]


def test_cap_is_respected_and_skipped_ids_are_listed() -> None:
    triage = FakeTriage(ratings("high", "high", "high", "high"))
    agent = FakeAgent([answer("we entered a new market"), answer("imported components")])

    result = report(triage, agent, max_explain=2)

    assert len(agent.users) == 2
    assert [item.id for item in result.high] == ["+1", "+2"]
    assert [(item.id, item.triage_reason) for item in result.not_explained] == [
        ("+3", "reason +3"),
        ("+4", "reason +4"),
    ]
    assert result.high_count == 4


def test_one_agent_failure_does_not_stop_the_others() -> None:
    triage = FakeTriage(ratings("high", "high", "high", "low"))
    agent = FakeAgent([LLMError("server error"), no_answer(), answer("humanoid robot")])

    result = report(triage, agent)

    failed_call, no_text, answered = result.high
    assert failed_call.error == "model call failed: server error"
    assert failed_call.explanation is None and failed_call.verification is None
    assert no_text.error == "agent failed: model returned no answer"
    assert no_text.verification is None and no_text.parsed is None
    assert answered.error is None
    assert answered.verification is not None and answered.verification.passed


def test_a_failed_fact_check_is_reported_as_failed() -> None:
    triage = FakeTriage(ratings("high", "low", "low", "low"))
    made_up = answer("we entered three new markets", what="Robotaxi revenue grew 45%.")
    agent = FakeAgent([made_up])

    [item] = report(triage, agent).high

    assert item.error is None
    assert item.verification is not None
    assert not item.verification.passed
    assert sorted(issue.kind for issue in item.verification.issues) == ["number", "quote"]


@pytest.mark.parametrize("triage_runs", [1, 3])
@pytest.mark.parametrize(
    ("triage", "error"),
    [
        (FakeTriage(ratings("high", "low", "low")), TriageError),  # +4 not rated
        (FakeTriage(text="not json"), LLMError),
    ],
)
def test_triage_failure_fails_the_report(
    triage: FakeTriage, error: type[Exception], triage_runs: int
) -> None:
    agent = FakeAgent([])

    with pytest.raises(error):  # every run fails the same way
        report(triage, agent, triage_runs=triage_runs)

    assert agent.users == []


def test_costs_are_summed_across_all_calls() -> None:
    triage = FakeTriage(ratings("high", "high", "high", "low"))
    agent = FakeAgent(
        [
            answer("we entered a new market", tokens=(300, 50)),
            no_answer(tokens=(120, 30)),  # failed, but its usage is known from the trace
            answer("humanoid robot", tokens=(400, 60)),
        ]
    )

    result = report(triage, agent)

    assert result.input_tokens == 1_000 + 300 + 120 + 400
    assert result.output_tokens == 200 + 50 + 30 + 60
    # 1,820 x $2 / 1M + 340 x $8 / 1M
    assert result.cost_usd == Decimal("0.00364") + Decimal("0.00272")
    assert result.usage_complete


def test_unknown_usage_of_a_failed_call_is_flagged() -> None:
    triage = FakeTriage(ratings("high", "low", "low", "low"))

    result = report(triage, FakeAgent([LLMError("timeout")]))

    assert not result.usage_complete
    assert (result.input_tokens, result.output_tokens) == TRIAGE_TOKENS


def test_cost_is_none_without_prices() -> None:
    triage = FakeTriage(ratings("low", "low", "low", "low"))
    settings = SETTINGS.model_copy(
        update={"llm_input_price_per_mtok": None, "llm_output_price_per_mtok": None}
    )

    result = build_section_report(
        DIFF,
        old_section=SECTION,
        new_section=SECTION,
        metrics=(),
        company="TSLA",
        form="10-K",
        section_title="Part I Item 1A. RISK FACTORS",
        triage_provider=triage,
        agent_provider=FakeAgent([]),
        settings=settings,
    )

    assert result.cost_usd is None


def test_no_high_changes_gives_a_valid_report_without_agent_calls() -> None:
    triage = FakeTriage(ratings("medium", "low", "low", "medium"))
    agent = FakeAgent([])

    result = report(triage, agent)

    assert agent.users == []
    assert result.high == () and result.not_explained == ()
    assert (result.high_count, result.medium_count, result.low_count) == (0, 2, 2)
    assert (result.input_tokens, result.output_tokens) == TRIAGE_TOKENS
    assert result.cost_usd == Decimal("0.0036")  # 1,000 x $2 / 1M + 200 x $8 / 1M
    assert result.usage_complete


def test_tool_outputs_count_as_fact_check_sources() -> None:
    # The quote is only in a tool result, as in the real Lesson 8 answer.
    section = Section(
        part="I",
        item="1A",
        title="RISK FACTORS",
        start=0,
        end=0,
        text="Item 1A. Risk Factors\n\nWe face intense competition in robotics.\n\n"
        + ROBOTAXI.new_text,
    )
    robotaxi = ROBOTAXI.model_copy(update={"new_start": section.text.index("Upon")})
    diff = DIFF.model_copy(update={"changes": (robotaxi,)})
    nearby = ToolCall(
        id="c1", name="get_nearby_paragraphs", arguments={"count": 1}, raw_arguments='{"count": 1}'
    )
    agent = FakeAgent(
        [
            [
                AgentStep(text=None, tool_calls=(nearby,), input_tokens=100, output_tokens=10),
                *answer("intense competition in robotics"),
            ]
        ]
    )

    result = build_section_report(
        diff,
        old_section=section,
        new_section=section,
        metrics=(),
        company="TSLA",
        form="10-K",
        section_title="Part I Item 1A. RISK FACTORS",
        triage_provider=FakeTriage(ratings("high")),
        agent_provider=agent,
        settings=SETTINGS,
    )

    [item] = result.high
    assert item.verification is not None and item.verification.passed
    assert (item.input_tokens, item.output_tokens) == (100 + 300, 10 + 50)


def test_negative_max_explain_is_rejected() -> None:
    with pytest.raises(ValueError, match="0 or more"):
        report(FakeTriage(), FakeAgent([]), max_explain=-1)


def test_printed_report_shows_failures_and_the_cap(capsys: pytest.CaptureFixture[str]) -> None:
    triage = FakeTriage(ratings("high", "high", "high", "medium"))
    agent = FakeAgent([answer("not in the paragraph"), LLMError("timeout")])

    _print_report(report(triage, agent, max_explain=2))

    out = capsys.readouterr().out
    assert "4 change(s): 3 high, 1 medium, 0 low" in out
    assert "FACT-CHECK: FAILED" in out
    assert '  quote: "not in the paragraph" not found in the sources' in out
    assert "AGENT FAILED: model call failed: timeout" in out
    assert "1 more high change(s) not explained: +3 (raise --max-explain)" in out
    assert "[+4] reason +4" in out
    assert "=== LOW (0) ===" in out
    assert "a failed model call isn't included" in out


def test_printed_fact_check_pass_counts_quotes_and_numbers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    triage = FakeTriage(ratings("high", "low", "low", "low"))

    _print_report(report(triage, FakeAgent([answer("we entered a new market")])))

    assert "FACT-CHECK: passed (1 quote, 0 numbers)" in capsys.readouterr().out


class ScriptedTriage:
    """complete_json: one answer per triage run, in order: levels for +1, +2, ... or an error."""

    def __init__(self, runs: list[dict[str, str] | LLMError]) -> None:
        self.runs = list(runs)
        self.calls = 0

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        self.calls += 1
        run = self.runs.pop(0)
        if isinstance(run, LLMError):
            raise run
        return FakeTriage(run).complete_json(system, user, schema_name, schema)


def test_high_in_any_triage_run_reaches_the_agent_with_its_votes() -> None:
    triage = ScriptedTriage(
        [
            ratings("medium", "medium", "high", "low"),
            ratings("high", "medium", "high", "low"),
            ratings("medium", "low", "high", "low"),
        ]
    )
    agent = FakeAgent([answer("we entered a new market"), answer("humanoid robot")])

    result = report(triage, agent, triage_runs=3)

    assert triage.calls == 3
    assert [(item.id, item.triage_votes) for item in result.high] == [
        ("+1", ("medium", "high", "medium")),  # high in only one run: still explained
        ("+3", ("high", "high", "high")),
    ]
    assert [(item.id, item.triage_votes) for item in result.medium] == [
        ("+2", ("medium", "medium", "low"))
    ]
    assert (result.high_count, result.medium_count, result.low_count) == (2, 1, 1)
    assert (result.triage_runs, result.triage_succeeded, result.triage_errors) == (3, 3, ())


def test_votes_are_kept_on_changes_over_the_cap() -> None:
    triage = ScriptedTriage([ratings("high", "high", "low", "low")] * 2)

    result = report(
        triage, FakeAgent([answer("we entered a new market")]), max_explain=1, triage_runs=2
    )

    assert [(item.id, item.triage_votes) for item in result.not_explained] == [
        ("+2", ("high", "high"))
    ]


def test_triage_tokens_are_summed_over_the_runs() -> None:
    triage = ScriptedTriage([ratings("low", "low", "low", "low")] * 3)

    result = report(triage, FakeAgent([]), triage_runs=3)

    assert (result.input_tokens, result.output_tokens) == (3_000, 600)
    assert result.cost_usd == 3 * Decimal("0.0036")
    assert result.usage_complete


def test_a_failed_triage_run_is_recorded_and_the_others_are_used() -> None:
    triage = ScriptedTriage(
        [
            ratings("high", "low", "low", "low"),
            LLMError("timeout"),
            ratings("low", "low", "low", "low"),
        ]
    )

    result = report(triage, FakeAgent([answer("we entered a new market")]), triage_runs=3)

    assert [item.id for item in result.high] == ["+1"]
    assert result.high[0].triage_votes == ("high", "low")
    assert (result.triage_runs, result.triage_succeeded) == (3, 2)
    assert result.triage_errors == ("run 2: timeout",)
    assert not result.usage_complete  # the failed run's tokens aren't known


def test_one_triage_run_reports_as_before(capsys: pytest.CaptureFixture[str]) -> None:
    triage = ScriptedTriage([ratings("high", "medium", "low", "low")])

    result = report(triage, FakeAgent([answer("we entered a new market")]), triage_runs=1)
    _print_report(result)

    assert triage.calls == 1
    assert (result.input_tokens, result.output_tokens) == (1_000 + 300, 200 + 50)
    assert (result.triage_runs, result.triage_succeeded, result.triage_errors) == (1, 1, ())
    assert result.high[0].triage_votes == ("high",)
    out = capsys.readouterr().out
    assert "[+1] triage: reason +1\n" in out
    assert "runs" not in out
    assert "borderline" not in out


def test_printed_votes_and_borderline_marks(capsys: pytest.CaptureFixture[str]) -> None:
    triage = ScriptedTriage(
        [
            ratings("high", "high", "high", "low"),
            ratings("medium", "high", "high", "low"),
            LLMError("timeout"),
        ]
    )
    agent = FakeAgent(
        [
            answer("we entered a new market"),  # agent high, passes: alert
            answer("not in the paragraph"),  # fact-check fails: review
            answer("humanoid robot"),
        ]
    )

    _print_report(report(triage, agent, triage_runs=3))

    out = capsys.readouterr().out
    assert "triage: 2/3 runs succeeded; a change is high if any run rated it high\n" in out
    assert "  FAILED run 3: timeout\n" in out
    assert "[+1] (borderline) A new risk was added.\n" in out
    assert "[+3] A new risk was added.\n" in out  # unanimous: no mark
    assert "[+2] fact-check failed: quote not found | triage: reason +2\n" in out
    assert "[+1] triage: high in 1/2 runs | reason +1\n" in out
    assert "[+2] triage: high in 2/2 runs | reason +2\n" in out


class FakeSec:
    """Stands in for SecClient in main(): two filings, no network."""

    def __enter__(self) -> "FakeSec":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_cik(self, ticker: str) -> str:
        return "0001318605"

    def get_recent_filings(self, cik: str, forms: set[str], limit: int) -> list[Filing]:
        return [
            Filing(
                cik=cik,
                accession_number=f"0001628280-2{year}-000001",
                form="10-K",
                filing_date=date(2020 + year, 1, 29),
                report_date=None,
                primary_document="tsla.htm",
            )
            for year in (6, 5)
        ]


@pytest.mark.parametrize(("flag", "expected"), [([], 1), (["--triage-runs", "3"], 3)])
def test_report_command_passes_triage_runs_through(
    monkeypatch: pytest.MonkeyPatch, flag: list[str], expected: int
) -> None:
    seen: dict[str, Any] = {}

    def fake_report(*args: Any, **kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "SecClient", FakeSec)
    monkeypatch.setattr(cli, "_report", fake_report)

    assert cli.main(["report", "TSLA", "--form", "10-K", "--item", "1A", *flag]) == 0
    assert seen == {"max_explain": 20, "triage_runs": expected}


@pytest.mark.parametrize("command", [["report", "TSLA", "--item", "1A"], ["check"]])
def test_zero_triage_runs_is_rejected(command: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main([*command, "--triage-runs", "0"])
