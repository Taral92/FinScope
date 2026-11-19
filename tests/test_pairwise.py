import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from finscope.__main__ import _run_pairwise_eval
from finscope.agent_tools import ExplainContext
from finscope.config import Settings
from finscope.diff import ParagraphChange, SectionDiff
from finscope.explain import AGENT_SYSTEM_PROMPT
from finscope.llm import (
    AgentStep,
    ConversationState,
    LLMError,
    LLMResult,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from finscope.pairwise import (
    PairResult,
    PairwiseError,
    combine,
    compare_pair,
    judge_prompt_version,
    load_candidate,
    run_pairwise,
    summarize_pairs,
)
from finscope.report import explain_and_verify
from finscope.sections import Section
from finscope.versions import agent_prompt_version

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)
PROMPT_A = "prompt A: the agent's own"
PROMPT_B = "prompt B: the candidate"
NEARBY_TEXT = "We face intense competition in robotics."
ROBOTAXI_TEXT = "Upon launching our Robotaxi service, we entered a new market."
SECTION = Section(
    part="I",
    item="1A",
    title="RISK FACTORS",
    start=0,
    end=0,
    text=f"Item 1A. Risk Factors\n\n{NEARBY_TEXT}\n\n{ROBOTAXI_TEXT}",
)
ROBOTAXI = ParagraphChange(
    kind="added",
    old_text=None,
    new_text=ROBOTAXI_TEXT,
    old_start=None,
    old_end=None,
    new_start=SECTION.text.index("Upon"),
    new_end=len(SECTION.text),
    similarity=None,
)
CONTEXT = ExplainContext(change=ROBOTAXI, old_section=SECTION, new_section=SECTION, metrics=())
AGENT_TOKENS = (300, 50)
JUDGE_TOKENS = (700, 40)
NEARBY_CALL = ToolCall(
    id="c1", name="get_nearby_paragraphs", arguments={"count": 1}, raw_arguments='{"count": 1}'
)


def answer(what: str, quote: str = "entered a new market", importance: str = "medium") -> str:
    return (
        f"WHAT CHANGED: {what}\n"
        "WHY IT MATTERS: The company now runs a Robotaxi service.\n"
        f"IMPORTANCE: {importance} - potential: new market\n"
        f'EVIDENCE: "{quote}"'
    )


Run = list[AgentStep] | LLMError


def steps(text: str, *, nearby: bool = False) -> list[AgentStep]:
    """One run: optionally a get_nearby_paragraphs call first, then the answer."""
    final = AgentStep(
        text=text, tool_calls=(), input_tokens=AGENT_TOKENS[0], output_tokens=AGENT_TOKENS[1]
    )
    if not nearby:
        return [final]
    call = AgentStep(text=None, tool_calls=(NEARBY_CALL,), input_tokens=100, output_tokens=10)
    return [call, final]


class FakeAgent:
    """Implements LLMProvider's start/resume: one scripted run per agent call, in order."""

    def __init__(self, runs: list[Run]) -> None:
        self.runs = list(runs)
        self.systems: list[str] = []  # the system prompt of every run started
        self._steps: list[AgentStep] = []

    def start(
        self, system: str, user: str, tools: list[ToolSpec]
    ) -> tuple[AgentStep, ConversationState]:
        self.systems.append(system)
        run = self.runs.pop(0)
        if isinstance(run, LLMError):
            raise run
        self._steps = list(run)
        return self._steps.pop(0), ConversationState()

    def resume(
        self, state: ConversationState, tool_results: list[ToolResult]
    ) -> tuple[AgentStep, ConversationState]:
        return self._steps.pop(0), state


class FakeJudge:
    """Implements LLMProvider's complete_json: one scripted pick (or LLMError) per call."""

    def __init__(self, picks: list[str | LLMError]) -> None:
        self.picks = list(picks)
        self.users: list[str] = []
        self.systems: list[str] = []

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        self.systems.append(system)
        self.users.append(user)
        pick = self.picks.pop(0)
        if isinstance(pick, LLMError):
            raise pick
        return LLMResult(
            text=json.dumps({"winner": pick, "reason": f"picked {pick}"}),
            model="fake-judge",
            input_tokens=JUDGE_TOKENS[0],
            output_tokens=JUDGE_TOKENS[1],
        )


def compare(agent: FakeAgent, judge: FakeJudge) -> PairResult:
    return compare_pair(
        "+1",
        "reason +1",
        CONTEXT,
        prompt_a=PROMPT_A,
        prompt_b=PROMPT_B,
        company="TSLA",
        form="10-K",
        section_title="Part I Item 1A. RISK FACTORS",
        agent_provider=agent,
        judge_provider=judge,
        settings=SETTINGS,
    )


def test_both_orders_agreeing_decide_and_picks_are_mapped_back_to_a_and_b() -> None:
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B says so."))])
    # Order 1 shows A first, so "2" is B; order 2 shows B first, so "1" is B too.
    judge = FakeJudge(["2", "1"])

    result = compare(agent, judge)

    assert agent.systems == [PROMPT_A, PROMPT_B]
    assert (result.outcome, result.decided_by) == ("b", "judge")
    assert [(call.first, call.pick) for call in result.calls] == [("a", "b"), ("b", "b")]
    first, second = judge.users
    assert first.index("WHAT CHANGED: A says so.") < first.index("WHAT CHANGED: B says so.")
    assert second.index("WHAT CHANGED: B says so.") < second.index("WHAT CHANGED: A says so.")
    assert result.a.parsed is not None and result.a.parsed.what_changed == "A says so."


def test_the_judge_sees_sources_and_all_three_lines_but_not_the_evidence() -> None:
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B says so.", importance="low"))])
    judge = FakeJudge(["tie", "tie"])

    result = compare(agent, judge)

    assert result.outcome == "tie"
    user = judge.users[0]
    assert "Change: added paragraph" in user
    assert ROBOTAXI_TEXT in user
    assert "Explanation 1:\nWHAT CHANGED: A says so.\nWHY IT MATTERS: The company now runs" in user
    assert "IMPORTANCE: medium - potential: new market" in user
    assert "IMPORTANCE: low - potential: new market" in user
    assert "EVIDENCE" not in user


@pytest.mark.parametrize(
    ("picks", "outcome"),
    [
        (["1", "1"], "inconsistent"),  # each call picked whichever came first: position bias
        (["1", "tie"], "inconsistent"),
        (["1", "2"], "a"),  # A first and picked, then A second and picked
        (["tie", "tie"], "tie"),
    ],
)
def test_orders_that_disagree_are_inconsistent(picks: list[str], outcome: str) -> None:
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B says so."))])

    assert compare(agent, FakeJudge(list(picks))).outcome == outcome


def test_combine() -> None:
    assert combine("a", "a") == "a"
    assert combine("tie", "tie") == "tie"
    assert combine("a", "b") == "inconsistent"


def test_only_one_usable_answer_wins_without_a_judge_call() -> None:
    # B's quote isn't in the paragraph: it fails the fact-check.
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B", quote="made up"))])
    judge = FakeJudge([])  # a call would pop from an empty list and fail the test

    result = compare(agent, judge)

    assert (result.outcome, result.decided_by, result.calls) == ("a", "one_usable", ())
    assert result.b.verification is not None and not result.b.verification.passed


def test_an_agent_failure_on_a_lets_b_win() -> None:
    agent = FakeAgent([LLMError("rate limited"), steps(answer("B says so."))])

    result = compare(agent, FakeJudge([]))

    assert (result.outcome, result.decided_by) == ("b", "one_usable")
    assert result.a.error == "model call failed: rate limited"


def test_neither_usable_is_a_tie_without_a_judge_call() -> None:
    agent = FakeAgent([LLMError("timeout"), steps(answer("B", quote="made up"))])

    result = compare(agent, FakeJudge([]))

    assert (result.outcome, result.decided_by, result.calls) == ("tie", "neither_usable", ())


def test_a_failed_judge_call_is_recorded_and_the_second_is_not_made() -> None:
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B says so."))])
    judge = FakeJudge([LLMError("rate limited")])

    result = compare(agent, judge)

    assert (result.outcome, result.calls) == ("failed", ())
    assert result.error == "judge call failed: rate limited"
    assert len(judge.users) == 1


def test_a_failure_in_the_second_order_keeps_the_first_call() -> None:
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B says so."))])

    result = compare(agent, FakeJudge(["1", LLMError("timeout")]))

    assert result.outcome == "failed"
    assert [(call.first, call.pick) for call in result.calls] == [("a", "a")]


def test_tool_results_of_both_sides_are_sources_each_once() -> None:
    # A's quote is only in the nearby paragraph, so A passes only with its tool result.
    a_run = steps(answer("A says so.", quote="intense competition in robotics"), nearby=True)
    both = compare(
        FakeAgent([a_run, steps(answer("B says so."), nearby=True)]), judge := FakeJudge(["1", "2"])
    )

    assert both.outcome == "a"
    assert both.a.verification is not None and both.a.verification.passed
    # Both called the same tool and got the same text: the judge sees it once.
    assert judge.users[0].count(NEARBY_TEXT) == 1
    assert "Sources: 2" in judge.users[0]  # the paragraph and the one tool result

    only_a = FakeJudge(["tie", "tie"])
    compare(FakeAgent([a_run, steps(answer("B says so."))]), only_a)
    assert NEARBY_TEXT in only_a.users[0]  # B didn't look, but A's answer is checked on it


def test_tokens_add_both_agent_runs_and_both_judge_calls() -> None:
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B says so."))])

    result = compare(agent, FakeJudge(["1", "2"]))

    assert result.input_tokens == 2 * AGENT_TOKENS[0] + 2 * JUDGE_TOKENS[0]
    assert result.output_tokens == 2 * AGENT_TOKENS[1] + 2 * JUDGE_TOKENS[1]


def test_run_and_summarize() -> None:
    agent = FakeAgent(
        [
            steps(answer("A1")),
            steps(answer("B1")),  # change 1: judge picks B in both orders
            steps(answer("A2")),
            steps(answer("B2", quote="made up")),  # change 2: only A usable
            steps(answer("A3")),
            steps(answer("B3")),  # change 3: position bias
            LLMError("timeout"),
            steps(answer("B4")),  # change 4: only B usable; A's usage unknown
        ]
    )
    judge = FakeJudge(["2", "1", "1", "1"])
    seen: list[str] = []
    results = run_pairwise(
        [(f"+{n}", f"reason +{n}", CONTEXT) for n in range(1, 5)],
        prompt_a=PROMPT_A,
        prompt_b=PROMPT_B,
        company="TSLA",
        form="10-K",
        section_title="Part I Item 1A. RISK FACTORS",
        agent_provider=agent,
        judge_provider=judge,
        settings=SETTINGS,
        on_result=lambda result: seen.append(result.id),
    )

    assert seen == ["+1", "+2", "+3", "+4"]
    assert [result.outcome for result in results] == ["b", "a", "inconsistent", "b"]
    summary = summarize_pairs(results, SETTINGS, extra_input_tokens=1_000, extra_output_tokens=200)
    assert (summary.a_wins, summary.b_wins, summary.ties) == (1, 2, 0)
    assert (summary.inconsistent, summary.failed, summary.decided_by_judge) == (1, 0, 2)
    # Triage 1,000; 7 agent runs with known usage (+4's A is lost); 4 judge calls.
    assert summary.input_tokens == 1_000 + 7 * 300 + 4 * 700
    assert summary.output_tokens == 200 + 7 * 50 + 4 * 40
    assert summary.cost_usd == (
        Decimal(summary.input_tokens) * 2 / 1_000_000
        + Decimal(summary.output_tokens) * 8 / 1_000_000
    )
    assert not summary.usage_complete


def test_a_failed_judge_call_makes_usage_incomplete() -> None:
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B says so."))])
    failed = compare(agent, FakeJudge([LLMError("rate limited")]))

    summary = summarize_pairs([failed], SETTINGS)

    assert (summary.failed, summary.decided_by_judge) == (1, 1)
    assert not summary.usage_complete


def test_explain_and_verify_uses_the_agent_prompt_unless_given_another() -> None:
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B says so."))])
    kwargs: dict[str, Any] = {
        "company": "TSLA",
        "form": "10-K",
        "section_title": "Part I Item 1A. RISK FACTORS",
        "provider": agent,
        "settings": SETTINGS,
    }

    item, sources = explain_and_verify("+1", "reason", CONTEXT, **kwargs)
    explain_and_verify("+1", "reason", CONTEXT, system_prompt=PROMPT_B, **kwargs)

    assert agent.systems == [AGENT_SYSTEM_PROMPT, PROMPT_B]
    assert sources == [ROBOTAXI_TEXT]
    assert item.verification is not None and item.verification.passed


# --- candidate prompt and CLI end to end, with fake providers ------------------------------------


def test_load_candidate_strips_and_rejects_unusable_files(tmp_path: Path) -> None:
    path = tmp_path / "agent_b.txt"
    path.write_text(f"\n{PROMPT_B}\n\n", encoding="utf-8")
    assert load_candidate(path) == PROMPT_B

    path.write_text("  \n", encoding="utf-8")
    with pytest.raises(PairwiseError, match="is empty"):
        load_candidate(path)
    path.write_text(AGENT_SYSTEM_PROMPT + "\n", encoding="utf-8")
    with pytest.raises(PairwiseError, match="the agent's own prompt"):
        load_candidate(path)
    with pytest.raises(PairwiseError, match="can't read"):
        load_candidate(tmp_path / "missing.txt")


def test_versions_tell_the_prompts_apart() -> None:
    assert agent_prompt_version(AGENT_SYSTEM_PROMPT) == agent_prompt_version()
    assert agent_prompt_version(PROMPT_B) != agent_prompt_version()
    assert len(judge_prompt_version()) == 12


TARIFFS_TEXT = "New tariffs on imported components could raise our costs."
TARIFFS = ROBOTAXI.model_copy(update={"new_text": TARIFFS_TEXT, "new_end": len(TARIFFS_TEXT)})
DIFF = SectionDiff(
    part="I", item="1A", title="RISK FACTORS", unchanged=3, changes=(ROBOTAXI, TARIFFS)
)
TRIAGE_TOKENS = (1_000, 200)


class FakeTriage:
    """Implements LLMProvider's complete_json: rates +1 and +2 as told."""

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
        return LLMResult(
            text=self.text,
            model="fake-model",
            input_tokens=TRIAGE_TOKENS[0],
            output_tokens=TRIAGE_TOKENS[1],
        )


def run_cli(
    tmp_path: Path,
    triage: FakeTriage,
    agent: FakeAgent,
    judge: FakeJudge,
    *,
    ids: list[str] | None = None,
    max_changes: int = 5,
) -> int:
    return _run_pairwise_eval(
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
        judge_provider=judge,
        settings=SETTINGS,
        candidate_file="evals/prompts/agent_b.txt",
        candidate=PROMPT_B,
        ids=ids,
        max_changes=max_changes,
        results_dir=tmp_path / "results",
    )


def test_cli_compares_triage_picks_up_to_max_prints_and_saves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    triage = FakeTriage("high", "high")
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B says so."))])
    judge = FakeJudge(["2", "1"])  # B in both orders

    assert run_cli(tmp_path, triage, agent, judge, max_changes=1) == 0

    assert agent.systems == [AGENT_SYSTEM_PROMPT, PROMPT_B]
    [path] = (tmp_path / "results").iterdir()
    assert path.name.endswith("Z_pairwise_TSLA_10-K_0001628280-26-003952_I-1A.json")
    saved = json.loads(path.read_text())
    assert (saved["prompt_a"], saved["prompt_b"]) == (
        agent_prompt_version(),
        agent_prompt_version(PROMPT_B),
    )
    assert saved["candidate_prompt"] == PROMPT_B
    assert saved["candidate_file"] == "evals/prompts/agent_b.txt"
    assert saved["judge_version"] == judge_prompt_version()
    assert saved["picked_by"] == "triage"
    assert [r["id"] for r in saved["results"]] == ["+1"]
    assert saved["results"][0]["outcome"] == "b"
    # Triage, two agent runs, two judge calls.
    assert saved["summary"]["input_tokens"] == 1_000 + 2 * 300 + 2 * 700

    out = capsys.readouterr().out
    assert "triage rated 2 changes high; comparing the first 1" in out
    assert f"B = evals/prompts/agent_b.txt ({agent_prompt_version(PROMPT_B)})" in out
    assert "=== [+1] B wins (judge, both orders) ===" in out
    assert "A: WHAT CHANGED: A says so.\n   WHY IT MATTERS: The company now runs" in out
    assert "   IMPORTANCE: medium - potential: new market\n" in out
    assert "judge, A first: picked B - picked 2\njudge, B first: picked B - picked 1" in out
    assert "A wins: 0   B wins: 1   ties: 0   inconsistent: 0   failed: 0" in out
    assert "judge asked on 1/1 change(s)" in out
    assert "(1 change: too few for the count to mean much)" in out


def test_cli_with_ids_skips_triage_and_the_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    triage = FakeTriage()
    agent = FakeAgent(
        [
            steps(answer("A says so.", quote="New tariffs")),  # +2 first: the tariffs one
            steps(answer("B", quote="made up")),  # B fails the fact-check
            steps(answer("A again.")),
            steps(answer("B again.")),
        ]
    )
    judge = FakeJudge(["1", "1"])  # +1: each order picks the first answer

    assert run_cli(tmp_path, triage, agent, judge, ids=["+2", "+1"], max_changes=1) == 0

    assert triage.calls == 0
    out = capsys.readouterr().out
    assert "2 changes (--ids): +2, +1" in out
    assert "=== [+2] A wins (only one answer usable) ===" in out
    assert "B: not usable: failed the fact-check (quote)" in out
    assert "=== [+1] INCONSISTENT: each order picked differently (judge, both orders) ===" in out
    assert "judge asked on 1/2 change(s)" in out


def test_cli_rejects_unknown_ids_without_calling_or_saving(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent, judge = FakeAgent([]), FakeJudge([])

    assert run_cli(tmp_path, FakeTriage(), agent, judge, ids=["~9"]) == 1

    assert "unknown change ID(s): ~9" in capsys.readouterr().err
    assert agent.systems == [] and judge.users == []
    assert not (tmp_path / "results").exists()


def test_cli_fails_when_every_judge_call_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = FakeAgent([steps(answer("A says so.")), steps(answer("B says so."))])

    assert run_cli(tmp_path, FakeTriage(), agent, FakeJudge([LLMError("down")]), ids=["+1"]) == 1

    out = capsys.readouterr().out
    assert "=== [+1] FAILED (judge, both orders) ===" in out
    assert "judge: judge call failed: down" in out
    assert "(a failed model call isn't included" in out
    assert len(list((tmp_path / "results").iterdir())) == 1  # still saved, for the record
