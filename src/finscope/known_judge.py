"""Known-answer eval for the judge: made-up explanations whose right verdict is obvious.

The judge gates alerts (decide.py), so two of its mistakes cost something: calling a faithful
explanation overstated or unsupported sends a good alert to review ("wrongly gated"), and
calling an overstated or unsupported one supported lets it through ("let through"). Each case
pairs a made-up change with an explanation and the one right verdict; the real
judge.judge_explanation runs on it unchanged, seeing the paragraph(s) and any extra sources
exactly as it sees a real answer's paragraph and tool results. No prompts of its own.

A case passes only on its exact verdict. Overstated versus unsupported is counted separately
("mixed up"): both gate, so for the alert decision that mistake costs nothing.
"""

from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from finscope.config import Settings
from finscope.evals import write_json_atomic
from finscope.judge import JudgeVerdict, judge_explanation
from finscope.known_cases import COMPANY, FORM, SECTION_TITLE, JudgeCase
from finscope.llm import LLMError, LLMProvider, cost_for_tokens
from finscope.verify import ParsedExplanation, sources_for
from finscope.versions import judge_prompt_version


class JudgeRun(BaseModel):
    """One judge call on one case: its verdict, or why it failed."""

    model_config = ConfigDict(frozen=True)

    verdict: JudgeVerdict | None  # None when the call failed
    reason: str | None
    error: str | None
    input_tokens: int | None  # None when the call failed: its usage is lost with it
    output_tokens: int | None


class JudgeCaseResult(BaseModel):
    """One case across runs. It passes only if every run gave exactly the expected verdict."""

    model_config = ConfigDict(frozen=True)

    name: str
    expected: JudgeVerdict
    runs: tuple[JudgeRun, ...]

    @property
    def matches(self) -> int:
        return sum(1 for run in self.runs if run.verdict == self.expected)

    @property
    def passed(self) -> bool:
        return bool(self.runs) and self.matches == len(self.runs)


class KnownJudgeEvalResult(BaseModel):
    """Everything about one known-answer judge eval, saved to evals/results/."""

    model_config = ConfigDict(frozen=True)

    created_at: datetime  # UTC
    cases_file: str
    company: str
    model: str  # the model that was requested
    judge_version: str  # versions.judge_prompt_version()
    runs: int  # per case
    cases: tuple[JudgeCaseResult, ...]  # in file order
    passed: int  # cases
    total_runs: int
    judge_failures: int  # runs whose call failed; left out of the gate counts below
    gate_right: int  # runs where "gates or not" matched the expected verdict
    wrongly_gated: int  # expected supported, got overstated or unsupported
    let_through: int  # expected overstated or unsupported, got supported
    mixed_up: int  # overstated for unsupported or the reverse: both gate
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None  # None when no prices are configured
    usage_complete: bool  # False when a failed call's usage is unknown (totals undercount)

    @property
    def file_name(self) -> str:
        return f"{self.created_at:%Y%m%dT%H%M%SZ}_judge_known.json"


def case_explanation(case: JudgeCase) -> ParsedExplanation:
    """The case's explanation as the judge receives an agent's: only WHAT and WHY are read."""
    return ParsedExplanation(
        what_changed=case.what_changed,
        why_it_matters=case.why_it_matters,
        importance=None,
        importance_reason=None,
        evidence=(),
        issues=(),
    )


def case_sources(case: JudgeCase) -> list[str]:
    """The paragraph(s), then the extra sources: the order verify.sources_for gives real ones."""
    return sources_for(case.change(), list(case.extra_sources))


def run_judge_cases(
    cases: Sequence[JudgeCase],
    *,
    provider: LLMProvider,
    settings: Settings,
    runs: int,
    on_run: Callable[[str, int, JudgeRun], None] | None = None,
) -> tuple[JudgeCaseResult, ...]:
    """Judge every case `runs` times. A failed call is recorded and the next one still runs.

    `on_run(case name, run number, run)` is called after every call, e.g. to print progress.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    results = []
    for case in cases:
        case_runs = []
        for number in range(1, runs + 1):
            run = _judge_once(case, provider=provider, settings=settings)
            case_runs.append(run)
            if on_run is not None:
                on_run(case.name, number, run)
        results.append(
            JudgeCaseResult(name=case.name, expected=case.expected, runs=tuple(case_runs))
        )
    return tuple(results)


def build_judge_result(
    cases: Sequence[JudgeCaseResult],
    *,
    cases_file: Path,
    runs: int,
    settings: Settings,
    created_at: datetime,
) -> KnownJudgeEvalResult:
    pairs = [(case.expected, run) for case in cases for run in case.runs]
    answered = [(expected, run.verdict) for expected, run in pairs if run.verdict is not None]
    input_tokens = sum(run.input_tokens or 0 for _, run in pairs)
    output_tokens = sum(run.output_tokens or 0 for _, run in pairs)
    return KnownJudgeEvalResult(
        created_at=created_at,
        cases_file=str(cases_file),
        company=COMPANY,
        model=settings.llm_model,
        judge_version=judge_prompt_version(),
        runs=runs,
        cases=tuple(cases),
        passed=sum(case.passed for case in cases),
        total_runs=len(pairs),
        judge_failures=len(pairs) - len(answered),
        gate_right=sum(1 for e, v in answered if _gates(e) == _gates(v)),
        wrongly_gated=sum(1 for e, v in answered if not _gates(e) and _gates(v)),
        let_through=sum(1 for e, v in answered if _gates(e) and not _gates(v)),
        mixed_up=sum(1 for e, v in answered if _gates(e) and _gates(v) and e != v),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        # Price is per token, so the cost of the total is the sum of each call's cost.
        cost_usd=cost_for_tokens(input_tokens, output_tokens, settings),
        usage_complete=all(run.input_tokens is not None for _, run in pairs),
    )


def save_known_judge_eval(result: KnownJudgeEvalResult, directory: Path) -> Path:
    path = directory / result.file_name
    write_json_atomic(path, result)
    return path


def _gates(verdict: JudgeVerdict) -> bool:
    """Whether this verdict sends a high answer to review (decide.py)."""
    return verdict != "supported"


def _judge_once(case: JudgeCase, *, provider: LLMProvider, settings: Settings) -> JudgeRun:
    try:
        judgment = judge_explanation(
            case_explanation(case),
            case_sources(case),
            kind=case.kind,
            company=COMPANY,
            form=FORM,
            section_title=SECTION_TITLE,
            provider=provider,
            settings=settings,
        )
    except LLMError as error:
        return JudgeRun(
            verdict=None,
            reason=None,
            error=f"judge call failed: {error}",
            input_tokens=None,
            output_tokens=None,
        )
    return JudgeRun(
        verdict=judgment.verdict,
        reason=judgment.reason,
        error=None,
        input_tokens=judgment.input_tokens,
        output_tokens=judgment.output_tokens,
    )
