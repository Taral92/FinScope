"""Known-answer eval for the explain agent: made-up changes whose right IMPORTANCE is obvious.

eval-known does this for triage. Here the agent's own level is scored, because it decides
alert (high) versus digest (medium). Most cases sit on the realized-vs-potential line: some
say a harm already happened (high), others that something launched or took effect but the
harm only may come (medium).

Each case gets its own tiny fake section (known_cases.known_sections), so the agent's tools
work as on a real filing, and the report's own explain_and_check runs unchanged on it: the
same agent, prompt and fact-check. No prompts of its own.
"""

from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from finscope.agent_eval import agent_prompt_version
from finscope.agent_tools import ExplainContext
from finscope.config import Settings
from finscope.evals import Importance, write_json_atomic
from finscope.known_cases import (
    COMPANY,
    FORM,
    ITEM,
    PART,
    TITLE,
    AgentCase,
    CaseResult,
    known_sections,
)
from finscope.llm import LLMProvider, cost_for_tokens
from finscope.report import HighItem, explain_and_check

SECTION_TITLE = f"Part {PART} Item {ITEM}. {TITLE}"  # as _diff_label would write it
# explain_and_check records why a change was picked; here nothing was triaged.
PICKED_REASON = "known case (triage not run)"


class AgentCaseResult(CaseResult):
    """One case across runs: CaseResult's PASS/FAIL, plus every run's full answer."""

    runs: tuple[HighItem, ...]  # one per run; .id is the case name


class KnownAgentEvalResult(BaseModel):
    """Everything about one known-answer agent eval, saved to evals/results/."""

    model_config = ConfigDict(frozen=True)

    created_at: datetime  # UTC
    cases_file: str
    company: str
    model: str  # the model that was requested
    prompt_version: str  # agent_prompt_version()
    runs: int  # per case
    cases: tuple[AgentCaseResult, ...]  # in file order
    passed: int  # cases
    total_runs: int
    fact_check_failures: int  # runs that answered but failed the fact-check
    agent_failures: int  # runs with no answer: the agent or a model call failed
    input_tokens: int  # every agent run, failed ones included where known
    output_tokens: int
    cost_usd: Decimal | None  # None when no prices are configured
    usage_complete: bool  # False when a failed call's usage is unknown (totals undercount)

    @property
    def file_name(self) -> str:
        return f"{self.created_at:%Y%m%dT%H%M%SZ}_agent_known.json"


def case_context(case: AgentCase) -> ExplainContext:
    """What the agent's tools see for one case: its fake sections and no XBRL metrics."""
    known = known_sections(case)
    return ExplainContext(
        change=known.change,
        old_section=known.old_section,
        new_section=known.new_section,
        metrics=(),
    )


def run_known_cases(
    cases: Sequence[AgentCase],
    *,
    provider: LLMProvider,
    settings: Settings,
    runs: int,
    on_run: Callable[[str, int, HighItem], None] | None = None,
) -> tuple[AgentCaseResult, ...]:
    """Explain and fact-check every case `runs` times; a failed run is recorded, not raised.

    `on_run(case_name, run_number, item)` is called after every run, e.g. to print progress.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    results = []
    for case in cases:
        context = case_context(case)
        items = []
        for number in range(1, runs + 1):
            item = explain_and_check(
                case.name,
                PICKED_REASON,
                context,
                company=COMPANY,
                form=FORM,
                section_title=SECTION_TITLE,
                provider=provider,
                settings=settings,
            )
            items.append(item)
            if on_run is not None:
                on_run(case.name, number, item)
        results.append(
            AgentCaseResult(
                name=case.name,
                expected=case.expected,
                ratings=tuple(_importance(item) for item in items),
                runs=tuple(items),
            )
        )
    return tuple(results)


def build_result(
    cases: Sequence[AgentCaseResult],
    *,
    cases_file: Path,
    runs: int,
    settings: Settings,
    created_at: datetime,
) -> KnownAgentEvalResult:
    items = [item for case in cases for item in case.runs]
    input_tokens = sum(item.input_tokens or 0 for item in items)
    output_tokens = sum(item.output_tokens or 0 for item in items)
    return KnownAgentEvalResult(
        created_at=created_at,
        cases_file=str(cases_file),
        company=COMPANY,
        model=settings.llm_model,
        prompt_version=agent_prompt_version(),
        runs=runs,
        cases=tuple(cases),
        passed=sum(case.passed for case in cases),
        total_runs=len(items),
        fact_check_failures=sum(
            1 for item in items if item.verification is not None and not item.verification.passed
        ),
        agent_failures=sum(1 for item in items if item.error is not None),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        # Price is per token, so the cost of the total is the sum of each call's cost.
        cost_usd=cost_for_tokens(input_tokens, output_tokens, settings),
        usage_complete=all(item.input_tokens is not None for item in items),
    )


def mismatch_reasons(case: AgentCaseResult) -> list[str]:
    """Why each run that missed the expected level said what it said; repeats shown once."""
    reasons: list[str] = []
    for rating, item in zip(case.ratings, case.runs, strict=True):
        if rating == case.expected:
            continue
        if item.error is not None:
            reason = item.error
        elif rating is None:
            reason = "no readable IMPORTANCE line"
        else:
            reason = (item.parsed.importance_reason if item.parsed else None) or "(none given)"
        if reason not in reasons:
            reasons.append(reason)
    return reasons


def save_known_agent_eval(result: KnownAgentEvalResult, directory: Path) -> Path:
    path = directory / result.file_name
    write_json_atomic(path, result)
    return path


def _importance(item: HighItem) -> Importance | None:
    return item.parsed.importance if item.parsed is not None else None
