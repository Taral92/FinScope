import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import finscope.__main__ as cli
from finscope.agent_eval import agent_prompt_version
from finscope.agent_tools import get_nearby_paragraphs
from finscope.config import Settings
from finscope.diff import diff_sections
from finscope.known_agent import case_context, mismatch_reasons, run_known_cases
from finscope.known_cases import (
    AGENT_CASES_PATH,
    AgentCase,
    KnownCasesError,
    load_agent_cases,
    load_known_cases,
)
from finscope.llm import AgentStep, ConversationState, LLMError, ToolResult, ToolSpec

REPO = Path(__file__).resolve().parent.parent
AGENT_CASES_FILE = REPO / AGENT_CASES_PATH

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_model="fake-model",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)

VIOLENCE = "Protests at our stores escalated to vandalism and arson, damaging several locations."
LAUNCH = "We launched a ride-hailing service in two cities. Regulations may change."
OLD_WORDING = "We utilize third-party logistics providers to deliver our vehicles."
NEW_WORDING = "We use third-party logistics providers to deliver our vehicles."


def case(**fields: Any) -> dict[str, Any]:
    """A valid agent case as JSON-ready data; keyword arguments override fields."""
    return {
        "name": "violence-occurred",
        "kind": "added",
        "old_text": None,
        "new_text": VIOLENCE,
        "expected": "high",
        "why": "It happened.",
    } | fields


CASES = [
    case(),
    case(name="launch-may-face-regulation", new_text=LAUNCH, expected="medium"),
    case(
        name="wording",
        kind="modified",
        old_text=OLD_WORDING,
        new_text=NEW_WORDING,
        expected="low",
    ),
]


def write_cases(tmp_path: Path, cases: Any) -> Path:
    path = tmp_path / "agent_cases.json"
    path.write_text(json.dumps(cases))
    return path


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


def answer(importance: str, quote: str) -> list[AgentStep]:
    """One step with a final answer; `importance` is the IMPORTANCE line after the label."""
    text = (
        "WHAT CHANGED: A paragraph changed.\n"
        "WHY IT MATTERS: It may matter.\n"
        f"IMPORTANCE: {importance}\n"
        f'EVIDENCE: "{quote}"'
    )
    return [AgentStep(text=text, tool_calls=(), input_tokens=300, output_tokens=50)]


# --- the committed case file --------------------------------------------------------------------


def test_agent_case_file_loads_with_the_eleven_cases() -> None:
    cases = load_agent_cases(AGENT_CASES_FILE)

    by_level: dict[str, set[str]] = {"high": set(), "medium": set(), "low": set()}
    for c in cases:
        by_level[c.expected].add(c.name)
    assert by_level == {
        "high": {
            "tariffs-raised-costs",
            "credit-repealed",
            "lawsuit-filed",
            "violence-occurred",
            "recall-done",
        },
        "medium": {
            "launch-may-face-regulation",
            "law-in-effect-may-cost",
            "new-product-may-not-sell",
            "award-may-misdirect",
        },
        "low": {"wording", "reordered-list"},
    }
    assert {c.name for c in cases if c.kind == "modified"} == {
        "tariffs-raised-costs",
        "credit-repealed",
        "law-in-effect-may-cost",
        "wording",
        "reordered-list",
    }


@pytest.mark.parametrize("agent_case", load_agent_cases(AGENT_CASES_FILE), ids=lambda c: c.name)
def test_fake_sections_hold_the_case_where_the_tools_find_it(agent_case: AgentCase) -> None:
    context = case_context(agent_case)

    nearby = get_nearby_paragraphs(context, 3)  # raises if the paragraph isn't at its offset

    assert nearby.count("(none)") == 2  # nothing before, nothing after
    assert "offset 23" in nearby  # just after "Item 1A. Risk Factors" and a blank line
    # The fake sections diff into exactly this change, offsets included.
    assert diff_sections(context.old_section, context.new_section).changes == (context.change,)
    assert context.metrics == ()


# --- loading ------------------------------------------------------------------------------------


def test_medium_is_an_answer_for_agent_cases_but_not_for_triage_cases(tmp_path: Path) -> None:
    path = write_cases(tmp_path, [case(expected="medium")])

    assert load_agent_cases(path)[0].expected == "medium"
    with pytest.raises(KnownCasesError, match="expected"):
        load_known_cases(path)


def test_a_text_that_is_not_one_paragraph_is_rejected(tmp_path: Path) -> None:
    path = write_cases(tmp_path, [case(new_text="First paragraph.\n\nSecond paragraph.")])

    with pytest.raises(KnownCasesError, match="exactly one paragraph"):
        load_agent_cases(path)


def test_agent_cases_get_the_same_checks_as_triage_cases(tmp_path: Path) -> None:
    with pytest.raises(KnownCasesError, match="duplicated names"):
        load_agent_cases(write_cases(tmp_path, [case(), case(new_text="Another paragraph.")]))
    with pytest.raises(KnownCasesError, match="similar"):
        load_agent_cases(
            write_cases(tmp_path, [case(kind="modified", old_text="Short.", new_text=LAUNCH)])
        )


# --- PASS/FAIL ----------------------------------------------------------------------------------


def run_cases(agent: FakeAgent, runs: int = 2) -> Any:
    cases = [AgentCase.model_validate(c) for c in CASES]
    return run_known_cases(cases, provider=agent, settings=SETTINGS, runs=runs)


def test_pass_needs_every_run_to_give_the_expected_importance() -> None:
    agent = FakeAgent(
        [
            answer("high - realized: stores damaged", "vandalism and arson"),
            answer("high - realized: stores damaged", "vandalism and arson"),
            answer("medium - potential: rules may change", "Regulations may change"),
            answer("high - realized: service launched", "ride-hailing service"),
            answer("low - wording: synonym", "third-party logistics"),
            answer("low - wording: synonym", "third-party logistics"),
        ]
    )

    violence, launch, wording = run_cases(agent)

    assert (violence.passed, violence.ratings) == (True, ("high", "high"))
    assert (launch.passed, launch.matches, launch.ratings) == (False, 1, ("medium", "high"))
    assert wording.passed
    assert mismatch_reasons(launch) == ["realized: service launched"]  # only the missed run
    assert mismatch_reasons(violence) == []
    # The agent saw each case as a real change of a made-up company's risk factors.
    assert "Company: Acme Motors" in agent.users[0]
    assert "Section: Part I Item 1A. Risk Factors" in agent.users[0]
    assert "Change: modified" in agent.users[4]


def test_failed_runs_fail_the_case_and_say_why() -> None:
    no_level = [AgentStep(text="I am not sure.", tool_calls=(), input_tokens=100, output_tokens=10)]
    agent = FakeAgent(
        [
            LLMError("down"),
            no_level,
            answer("high - realized: same reason", "vandalism and arson"),
            answer("high - realized: same reason", "ride-hailing service"),
            answer("low - wording", "third-party logistics"),
            answer("low - wording", "third-party logistics"),
        ]
    )

    violence, launch, _ = run_cases(agent)

    assert violence.ratings == (None, None) and not violence.passed
    assert mismatch_reasons(violence) == [
        "model call failed: down",
        "no readable IMPORTANCE line",
    ]
    assert mismatch_reasons(launch) == ["realized: same reason"]  # shown once


# --- CLI end to end, with a fake agent ----------------------------------------------------------


def test_cli_prints_pass_fail_with_reasons_and_saves_the_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_cases(tmp_path, CASES)
    agent = FakeAgent(
        [
            answer("high - realized: stores damaged", "vandalism and arson"),
            answer("high - realized: stores damaged", "vandalism and arson"),
            answer("high - realized: service launched", "ride-hailing service"),
            answer("high - realized: service launched", "ride-hailing service"),
            answer("low - wording: synonym", "third-party logistics"),
            answer("low - wording: synonym", "a quote that is not there"),  # fact-check fails
        ]
    )
    results = tmp_path / "results"

    exit_code = cli._run_known_agent_eval(
        path, provider=agent, settings=SETTINGS, runs=2, results_dir=results
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Running the agent 2 time(s) per case: 6 agent run(s)." in out
    assert out.index("6 agent run(s)") < out.index("[violence-occurred] run 1/2: high")
    assert (
        "PASS 2/2  violence-occurred          expected high    got high, high\n"
        "FAIL 0/2  launch-may-face-regulation expected medium  got high, high\n"
        "          reason: realized: service launched\n"
        "PASS 2/2  wording                    expected low     got low, low\n"
    ) in out
    assert "passed: 2/3 case(s)" in out
    assert "fact-check failures: 1/6 run(s)" in out  # reported, but doesn't fail the case
    assert f"agent prompt version {agent_prompt_version()}" in out
    assert "tokens: 1,800 input, 300 output" in out
    assert "cost:   $0.0060" in out  # 1,800 x $2/M + 300 x $8/M

    [saved] = results.iterdir()  # one file, no .partial left behind
    assert saved.name.endswith("Z_agent_known.json")
    assert f"saved:  {saved}" in out
    data = json.loads(saved.read_text())
    assert data["prompt_version"] == agent_prompt_version()
    assert (data["model"], data["runs"], data["passed"]) == ("fake-model", 2, 2)
    assert data["cases_file"] == str(path)
    launch = data["cases"][1]
    assert (launch["name"], launch["expected"], launch["ratings"]) == (
        "launch-may-face-regulation",
        "medium",
        ["high", "high"],
    )
    assert launch["runs"][0]["parsed"]["importance_reason"] == "realized: service launched"
    assert (data["fact_check_failures"], data["agent_failures"]) == (1, 0)


def test_cli_reports_a_bad_case_file_without_calling_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = FakeAgent([])  # no runs: any agent call would fail the test

    exit_code = cli._run_known_agent_eval(
        write_cases(tmp_path, []),
        provider=agent,
        settings=SETTINGS,
        runs=2,
        results_dir=tmp_path / "results",
    )

    assert exit_code == 1
    assert "no cases" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_cli_fails_when_every_run_failed(tmp_path: Path) -> None:
    path = write_cases(tmp_path, [case()])
    agent = FakeAgent([LLMError("down"), LLMError("down")])

    exit_code = cli._run_known_agent_eval(
        path, provider=agent, settings=SETTINGS, runs=2, results_dir=tmp_path / "results"
    )

    assert exit_code == 1
    [saved] = (tmp_path / "results").iterdir()  # still saved: failures are results too
    assert json.loads(saved.read_text())["agent_failures"] == 2


def test_eval_known_agent_needs_no_sec_and_runs_twice_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_sec() -> None:
        raise AssertionError("eval-known-agent must not use the SEC client")

    monkeypatch.setattr(cli, "SecClient", no_sec)
    monkeypatch.setattr(cli, "_eval_known_agent", lambda runs: runs)

    assert cli.main(["eval-known-agent"]) == 2
    assert cli.main(["eval-known-agent", "--runs", "3"]) == 3
