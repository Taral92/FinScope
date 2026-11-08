import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from finscope.__main__ import _run_agent_eval
from finscope.agent_eval import (
    AgentEvalError,
    Selection,
    agent_prompt_version,
    run_stability,
    select_changes,
    summarize,
)
from finscope.agent_tools import TOOL_SPECS, ExplainContext, build_tools
from finscope.config import Settings
from finscope.diff import ParagraphChange, SectionDiff
from finscope.evals import prompt_version
from finscope.explain import AGENT_SYSTEM_PROMPT
from finscope.llm import (
    AgentStep,
    ConversationState,
    LLMError,
    LLMResult,
    ToolResult,
    ToolSpec,
)
from finscope.report import RatedItem
from finscope.sections import Section

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_model="fake-model",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)


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
DIFF = SectionDiff(
    part="I", item="1A", title="RISK FACTORS", unchanged=10, changes=(ROBOTAXI, TARIFFS, OPTIMUS)
)
SECTION = Section(part="I", item="1A", title="RISK FACTORS", start=0, end=12, text="Item 1A. Risk")


class FakeTriage:
    """complete_json rates +1, +2, ... as told; `calls` counts how often it was asked."""

    def __init__(self, *levels: str) -> None:
        items = [
            {"id": f"+{n}", "importance": level, "reason": f"reason +{n}"}
            for n, level in enumerate(levels, start=1)
        ]
        self.text = json.dumps({"items": items})
        self.calls = 0

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text, model="fake-model", input_tokens=1_000, output_tokens=200)


Run = list[AgentStep] | LLMError


class FakeAgent:
    """start/resume: one scripted run per agent call, in order (an LLMError fails the call)."""

    def __init__(self, runs: list[Run]) -> None:
        self.runs = list(runs)
        self.users: list[str] = []
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


def answer(level: str, quote: str, tokens: tuple[int, int] = (300, 50)) -> list[AgentStep]:
    text = (
        "WHAT CHANGED: A new risk was added.\n"
        "WHY IT MATTERS: It is a new risk for the business.\n"
        f"IMPORTANCE: {level} - new risk\n"
        f'EVIDENCE: "{quote}"'
    )
    return [AgentStep(text=text, tool_calls=(), input_tokens=tokens[0], output_tokens=tokens[1])]


def no_answer() -> list[AgentStep]:
    """Neither text nor tool calls: the agent loop stops with an AgentError."""
    return [AgentStep(text=None, tool_calls=(), input_tokens=120, output_tokens=30)]


ROBOTAXI_QUOTE = "we entered a new market"
TARIFFS_QUOTE = "imported components"


def picked(*ids: str) -> Selection:
    return Selection(
        picked_by="ids",
        changes=tuple(RatedItem(id=i, triage_reason="picked") for i in ids),
        input_tokens=0,
        output_tokens=0,
    )


def stability(selection: Selection, agent: FakeAgent, runs: int = 3) -> Any:
    return run_stability(
        selection,
        DIFF,
        old_section=SECTION,
        new_section=SECTION,
        metrics=(),
        company="TSLA",
        form="10-K",
        section_title="Part I Item 1A. RISK FACTORS",
        agent_provider=agent,
        settings=SETTINGS,
        runs=runs,
    )


def select(ids: list[str] | None, triage: FakeTriage) -> Selection:
    return select_changes(
        DIFF,
        ids=ids,
        company="TSLA",
        form="10-K",
        section_title="Part I Item 1A. RISK FACTORS",
        triage_provider=triage,
        settings=SETTINGS,
    )


def test_stable_and_unstable_changes_are_told_apart() -> None:
    agent = FakeAgent(
        [
            answer("high", ROBOTAXI_QUOTE),
            answer("high", ROBOTAXI_QUOTE),
            answer("high", ROBOTAXI_QUOTE),
            answer("high", TARIFFS_QUOTE),
            answer("medium", TARIFFS_QUOTE),
            answer("Medium", TARIFFS_QUOTE),  # casing drift is the same level
        ]
    )

    steady, wobbly = stability(picked("+1", "+2"), agent)

    assert [run.importance for run in steady.runs] == ["high", "high", "high"]
    assert [run.tier for run in steady.runs] == ["alert", "alert", "alert"]
    assert steady.importance_stable and not steady.tier_flips and steady.stable
    assert [run.importance for run in wobbly.runs] == ["high", "medium", "medium"]
    assert [run.tier for run in wobbly.runs] == ["alert", "digest", "digest"]
    assert not wobbly.importance_stable and wobbly.tier_flips and not wobbly.stable

    summary = summarize((steady, wobbly), picked("+1", "+2"), SETTINGS)
    assert (summary.changes, summary.importance_stable, summary.tier_flips) == (2, 1, 1)
    assert (summary.total_runs, summary.fact_check_failures, summary.agent_failures) == (6, 0, 0)


def test_a_failed_fact_check_makes_the_run_review_so_the_tier_flips() -> None:
    agent = FakeAgent(
        [
            answer("high", ROBOTAXI_QUOTE),
            answer("high", "we entered three new markets"),  # not in the paragraph
        ]
    )

    [change] = stability(picked("+1"), agent, runs=2)

    first, second = change.runs
    assert (first.tier, first.fact_check_passed) == ("alert", True)
    assert (second.tier, second.fact_check_passed) == ("review", False)
    assert second.tier_reason == "fact-check failed: quote not found"
    # Same importance both times, yet a different tier: the decide rules, not the level.
    assert change.importance_stable and change.tier_flips

    summary = summarize([change], picked("+1"), SETTINGS)
    assert (summary.fact_check_failures, summary.total_runs) == (1, 2)


def test_a_failed_run_is_recorded_and_the_others_continue() -> None:
    agent = FakeAgent([LLMError("server error"), answer("high", ROBOTAXI_QUOTE), no_answer()])

    [change] = stability(picked("+1"), agent)

    call_failed, answered, no_text = change.runs
    assert call_failed.item.error == "model call failed: server error"
    assert (call_failed.importance, call_failed.tier, call_failed.fact_check_passed) == (
        None,
        "review",
        None,
    )
    assert (answered.importance, answered.tier) == ("high", "alert")
    assert no_text.item.error == "agent failed: model returned no answer"
    assert no_text.tier == "review"
    # The one level read is "high": importance holds, but failures still flip the tier.
    assert change.importance_stable and change.tier_flips

    summary = summarize([change], picked("+1"), SETTINGS)
    assert (summary.agent_failures, summary.fact_check_failures) == (2, 0)
    assert not summary.usage_complete  # the failed call's usage is unknown


def test_ids_pick_exactly_those_changes_without_triage() -> None:
    triage = FakeTriage("low", "low", "low")
    selection = select(["+3", "+1"], triage)  # not in diff order: kept as given

    assert [change.id for change in selection.changes] == ["+3", "+1"]
    assert triage.calls == 0
    assert (selection.picked_by, selection.input_tokens) == ("ids", 0)

    agent = FakeAgent([answer("high", "humanoid robot"), answer("low", ROBOTAXI_QUOTE)])
    changes = stability(selection, agent, runs=1)
    assert [change.id for change in changes] == ["+3", "+1"]
    assert OPTIMUS.new_text in agent.users[0] and ROBOTAXI.new_text in agent.users[1]


@pytest.mark.parametrize(
    ("ids", "message"),
    [
        (["+1", "~4", "+9"], "unknown change ID(s): ~4, +9"),
        (["+1", "+2", "+1"], "given more than once: +1"),
    ],
)
def test_bad_ids_are_rejected_before_any_call(ids: list[str], message: str) -> None:
    triage = FakeTriage("high", "high", "high")

    with pytest.raises(AgentEvalError, match=re.escape(message)):
        select(ids, triage)

    assert triage.calls == 0


def test_without_ids_triage_runs_once_and_its_high_changes_are_picked() -> None:
    triage = FakeTriage("high", "medium", "high")

    selection = select(None, triage)

    assert triage.calls == 1
    assert [(c.id, c.triage_reason) for c in selection.changes] == [
        ("+1", "reason +1"),
        ("+3", "reason +3"),
    ]
    assert (selection.input_tokens, selection.output_tokens) == (1_000, 200)


def test_costs_are_summed_over_triage_and_every_run() -> None:
    selection = select(None, FakeTriage("high", "low", "low"))
    agent = FakeAgent(
        [
            answer("high", ROBOTAXI_QUOTE, tokens=(300, 50)),
            no_answer(),  # failed, but its usage is known from the trace: 120 / 30
            answer("medium", ROBOTAXI_QUOTE, tokens=(400, 60)),
        ]
    )

    summary = summarize(stability(selection, agent), selection, SETTINGS)

    assert summary.input_tokens == 1_000 + 300 + 120 + 400
    assert summary.output_tokens == 200 + 50 + 30 + 60
    # 1,820 x $2 / 1M + 340 x $8 / 1M
    assert summary.cost_usd == Decimal("0.00364") + Decimal("0.00272")
    assert summary.usage_complete


def test_prompt_version_covers_the_tools_the_agent_really_gets() -> None:
    context = ExplainContext(change=ROBOTAXI, old_section=SECTION, new_section=SECTION, metrics=())
    assert [tool.spec for tool in build_tools(context).values()] == list(TOOL_SPECS)

    without_tools = prompt_version(AGENT_SYSTEM_PROMPT, {"tools": []})
    assert len(agent_prompt_version()) == 12
    assert agent_prompt_version() != without_tools


def test_importance_rubric_changed_the_agent_prompt_version() -> None:
    # The version before the rubric: eval results saved under it are from the old prompt.
    assert agent_prompt_version() != "a2924454daea"


# --- CLI end to end, with fake providers --------------------------------------------------------


def run_cli(
    tmp_path: Path, triage: FakeTriage, agent: FakeAgent, ids: list[str] | None, runs: int = 2
) -> int:
    return _run_agent_eval(
        DIFF,
        old_section=SECTION,
        new_section=SECTION,
        metrics=(),
        company="TSLA",
        form="10-K",
        new_accession="0001628280-26-003952",
        old_accession="0001628280-25-003063",
        triage_provider=triage,
        agent_provider=agent,
        settings=SETTINGS,
        runs=runs,
        ids=ids,
        results_dir=tmp_path / "results",
    )


def test_cli_runs_prints_and_saves_the_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    triage = FakeTriage("high", "high", "low")
    agent = FakeAgent(
        [
            answer("high", ROBOTAXI_QUOTE),
            answer("high", ROBOTAXI_QUOTE),
            answer("high", TARIFFS_QUOTE),
            answer("medium", TARIFFS_QUOTE),
        ]
    )

    exit_code = run_cli(tmp_path, triage, agent, ids=None)

    assert exit_code == 0
    [path] = (tmp_path / "results").iterdir()  # one file, no .partial left behind
    assert path.name.endswith("Z_agent_TSLA_10-K_0001628280-26-003952_I-1A.json")
    saved = json.loads(path.read_text())
    assert saved["model"] == "fake-model"
    assert saved["prompt_version"] == agent_prompt_version()
    assert (saved["runs"], saved["picked_by"]) == (2, "triage")
    assert (saved["new_accession"], saved["old_accession"]) == (
        "0001628280-26-003952",
        "0001628280-25-003063",
    )
    second = saved["changes"][1]
    assert second["id"] == "+2" and second["tier_flips"] is True
    assert [
        (run["importance"], run["tier"], run["fact_check_passed"]) for run in second["runs"]
    ] == [("high", "alert", True), ("medium", "digest", True)]
    assert second["runs"][1]["item"]["explanation"].startswith("WHAT CHANGED:")
    assert saved["summary"]["importance_stable"] == 1
    assert saved["summary"]["tier_flips"] == 1
    assert (saved["summary"]["input_tokens"], saved["summary"]["output_tokens"]) == (2_200, 400)

    out = capsys.readouterr().out
    assert "2 changes (triage high): +1, +2" in out
    assert out.index("4 agent run(s)") < out.index("[+1] run 1/2")  # announced before calling
    assert "[+1] importance: high, high     tier: alert, alert    stable" in out
    assert "[+2] importance: high, medium   tier: alert, digest   UNSTABLE" in out
    assert "stable importance:   1/2 change(s)" in out
    assert "tier flips:          1/2 change(s)" in out
    assert "fact-check failures: 0/4 run(s)" in out
    assert "(triage + 4 agent run(s))" in out


def test_cli_rejects_unknown_ids_without_calling_or_saving(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    triage = FakeTriage("high", "high", "high")
    agent = FakeAgent([])  # no runs: any agent call would fail the test

    exit_code = run_cli(tmp_path, triage, agent, ids=["~11"])

    assert exit_code == 1
    assert "unknown change ID(s): ~11" in capsys.readouterr().err
    assert triage.calls == 0 and agent.users == []
    assert not (tmp_path / "results").exists()


def test_cli_fails_when_every_run_failed(tmp_path: Path) -> None:
    agent = FakeAgent([LLMError("down"), LLMError("down")])

    exit_code = run_cli(tmp_path, FakeTriage(), agent, ids=["+1"])

    assert exit_code == 1
    [path] = (tmp_path / "results").iterdir()  # still saved: failures are results too
    assert json.loads(path.read_text())["summary"]["agent_failures"] == 2
