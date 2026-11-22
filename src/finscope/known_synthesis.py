"""Known-answer eval for the filing summary: does it join the right findings, and only those?

Each case (evals/known/synthesis_cases.json) is a made-up filing's verified findings, with the
groups that are one matter ("together": every ref of a group must be cited by a single kept
point) and the pairs that are not ("apart": no kept point may cite both). The real
synthesis.summarize_filing runs on each case, the summary call and the point checks, so its
guardrails are scored too: how often a point is dropped, and why.

A case passes when every run kept every "together" group in one point and joined no "apart"
pair. No prompts of its own.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from finscope.config import Settings
from finscope.evals import write_json_atomic
from finscope.llm import LLMError, LLMProvider, cost_for_tokens, judge_cost_for_tokens
from finscope.synthesis import FilingSummary, Finding, summarize_filing, summary_prompt_version

SYNTHESIS_CASES_PATH = Path("evals") / "known" / "synthesis_cases.json"
COMPANY = "Acme Motors"


class KnownSynthesisError(Exception):
    """The case file can't be used; the message says why."""


class SynthesisCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    findings: tuple[Finding, ...]
    together: tuple[tuple[str, ...], ...]  # each: refs that are one matter
    apart: tuple[tuple[str, str], ...]  # each: two refs that are different matters
    why: str = Field(min_length=1)


class SynthesisRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: FilingSummary | None  # None when the summary call failed
    error: str | None
    missed: tuple[tuple[str, ...], ...]  # "together" groups no single kept point covers
    wrongly_joined: tuple[tuple[str, str], ...]  # "apart" pairs a kept point joins

    @property
    def passed(self) -> bool:
        return self.summary is not None and not self.missed and not self.wrongly_joined


class SynthesisCaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    runs: tuple[SynthesisRun, ...]

    @property
    def passed(self) -> bool:
        return all(run.passed for run in self.runs)


class KnownSynthesisResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at: datetime
    cases_file: str
    model: str  # the summary's model
    prompt_version: str  # synthesis.summary_prompt_version()
    runs: int
    cases: tuple[SynthesisCaseResult, ...]
    passed: int  # cases
    points_kept: int  # over every run
    points_dropped: int
    drop_reasons: dict[str, int]  # e.g. {"check: overstated": 1, "unknown ref(s)": 1}
    failures: int  # runs whose summary call failed
    cost_usd: Decimal | None

    @property
    def file_name(self) -> str:
        return f"{self.created_at:%Y%m%dT%H%M%SZ}_synthesis_known.json"


_CASES = TypeAdapter(tuple[SynthesisCase, ...])


def load_synthesis_cases(path: Path) -> tuple[SynthesisCase, ...]:
    """The cases, or a KnownSynthesisError: bad JSON, or a group naming an unknown ref."""
    try:
        cases = _CASES.validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise KnownSynthesisError(f"can't read {path}: {error}") from error
    except ValidationError as error:
        raise KnownSynthesisError(f"{path} is not a valid case file:\n{error}") from error
    problems = []
    if duplicates := [n for n, c in Counter(case.name for case in cases).items() if c > 1]:
        problems.append(f"duplicated names: {', '.join(duplicates)}")
    for case in cases:
        refs = {finding.ref for finding in case.findings}
        named = {ref for group in case.together + case.apart for ref in group}
        if unknown := sorted(named - refs):
            problems.append(f"case {case.name!r} names unknown refs: {', '.join(unknown)}")
    if problems:
        raise KnownSynthesisError(f"{path}: " + "; ".join(problems))
    return cases


def score_summary(
    case: SynthesisCase, summary: FilingSummary
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, str], ...]]:
    """(together groups no kept point covers, apart pairs a kept point joins)."""
    kept = [set(point.refs) for point in summary.kept]
    missed = tuple(group for group in case.together if not any(set(group) <= p for p in kept))
    joined = tuple(pair for pair in case.apart if any(set(pair) <= p for p in kept))
    return missed, joined


def run_known_synthesis(
    cases: Sequence[SynthesisCase],
    *,
    provider: LLMProvider,
    check_provider: LLMProvider,
    runs: int,
    on_run: Callable[[str, int, SynthesisRun], None] | None = None,
) -> tuple[SynthesisCaseResult, ...]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    results = []
    for case in cases:
        case_runs = []
        for number in range(1, runs + 1):
            try:
                summary = summarize_filing(
                    case.findings,
                    company=COMPANY,
                    form="10-K",
                    provider=provider,
                    check_provider=check_provider,
                )
            except LLMError as error:
                run = SynthesisRun(
                    summary=None,
                    error=f"summary call failed: {error}",
                    missed=case.together,
                    wrongly_joined=(),
                )
            else:
                missed, joined = score_summary(case, summary)
                run = SynthesisRun(
                    summary=summary, error=None, missed=missed, wrongly_joined=joined
                )
            case_runs.append(run)
            if on_run is not None:
                on_run(case.name, number, run)
        results.append(SynthesisCaseResult(name=case.name, runs=tuple(case_runs)))
    return tuple(results)


def build_synthesis_result(
    cases: Sequence[SynthesisCaseResult],
    *,
    cases_file: Path,
    runs: int,
    settings: Settings,
    created_at: datetime,
) -> KnownSynthesisResult:
    summaries = [run.summary for case in cases for run in case.runs if run.summary is not None]
    points = [point for summary in summaries for point in summary.points]
    reasons = Counter(_reason_kind(p.reason) for p in points if p.status == "dropped")
    written = cost_for_tokens(
        sum(s.input_tokens for s in summaries), sum(s.output_tokens for s in summaries), settings
    )
    checked = judge_cost_for_tokens(
        sum(s.check_input_tokens for s in summaries),
        sum(s.check_output_tokens for s in summaries),
        settings,
    )
    return KnownSynthesisResult(
        created_at=created_at,
        cases_file=str(cases_file),
        model=settings.llm_model,
        prompt_version=summary_prompt_version(),
        runs=runs,
        cases=tuple(cases),
        passed=sum(case.passed for case in cases),
        points_kept=sum(1 for p in points if p.status == "kept"),
        points_dropped=sum(1 for p in points if p.status == "dropped"),
        drop_reasons=dict(reasons),
        failures=sum(1 for case in cases for run in case.runs if run.summary is None),
        cost_usd=None if written is None or checked is None else written + checked,
    )


def save_known_synthesis(result: KnownSynthesisResult, directory: Path) -> Path:
    path = directory / result.file_name
    write_json_atomic(path, result)
    return path


def _reason_kind(reason: str) -> str:
    """The kind of drop, without its details: 'check: overstated', 'unknown ref(s)', ..."""
    if reason.startswith("check: "):
        return ":".join(reason.split(":")[:2])
    return reason.split(":")[0]
