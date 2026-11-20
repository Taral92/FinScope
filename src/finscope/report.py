"""One run over a section: triage every change, explain the high ones, fact-check each answer.

Triage runs `triage_runs` times and a change counts as high if any run rated it high, so a
borderline change still reaches the agent; each item keeps every run's rating (its votes).

Plain orchestration code: no prompts of its own. It chains triage_section,
explain_change_with_agent and the fact-checker, and returns plain data a later alert step
can use.
"""

from collections.abc import Sequence
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from finscope.agent import AgentError
from finscope.agent_tools import ExplainContext
from finscope.config import Settings
from finscope.diff import SectionDiff, changes_by_id
from finscope.explain import AGENT_SYSTEM_PROMPT, explain_change_with_agent
from finscope.judge import Judgment, judge_explanation
from finscope.llm import LLMError, LLMProvider, split_cost
from finscope.sections import Section
from finscope.triage import Importance, triage_section_votes
from finscope.verify import (
    ParsedExplanation,
    Verification,
    parse_explanation,
    sources_for,
    verify_explanation,
)
from finscope.versions import PromptVersions, current_versions
from finscope.xbrl import MetricValue


class HighItem(BaseModel):
    """One high change the agent was asked to explain: its answer and fact-check, or an error.

    An answer that doesn't follow the format is not an error: it is kept, and the problems
    show up as format issues in a failed verification.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    triage_reason: str
    explanation: str | None  # None when the agent failed
    error: str | None  # why the agent failed; None when it answered
    parsed: ParsedExplanation | None  # the answer split into its labeled lines; None on failure
    verification: Verification | None  # None when there is no answer to check
    # Every model call of this run, the judge's included; None when a call failed and its
    # usage isn't known.
    input_tokens: int | None
    output_tokens: int | None
    # Every triage run's rating of this change; empty when it wasn't picked by triage.
    triage_votes: tuple[Importance, ...] = ()
    # The judge's verdict on the answer. None when there was nothing to judge (no answer, or
    # the fact-check already failed), when no judge was given, or when the judge call failed:
    # then judge_error says why. Reports saved before the judge existed load with None.
    judgment: Judgment | None = None
    judge_error: str | None = None


class RatedItem(BaseModel):
    """A change with only its triage rating: a medium one, or a high one over the cap."""

    model_config = ConfigDict(frozen=True)

    id: str
    triage_reason: str
    triage_votes: tuple[Importance, ...] = ()  # every triage run's rating; empty if not triaged


class SectionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    company: str
    form: str
    section_title: str
    change_count: int
    high_count: int
    medium_count: int
    low_count: int
    high: tuple[HighItem, ...]  # explained (or attempted) high changes, in diff order
    not_explained: tuple[RatedItem, ...]  # high changes over the max_explain cap, in diff order
    medium: tuple[RatedItem, ...]
    # Defaults describe a report saved before several triage runs existed: one run, it worked.
    triage_runs: int = 1  # triage runs attempted
    triage_succeeded: int = 1
    triage_errors: tuple[str, ...] = ()  # one per failed triage run
    # Totals over every call: each triage run plus each agent run, failed runs included.
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None  # None when no prices are configured
    usage_complete: bool  # False when a failed call's usage is unknown (totals undercount)
    # The model and prompts that made this report; None for reports saved before it was kept.
    prompt_versions: PromptVersions | None = None


def build_section_report(
    diff: SectionDiff,
    *,
    old_section: Section,
    new_section: Section,
    metrics: tuple[MetricValue, ...],
    company: str,
    form: str,
    section_title: str,
    triage_provider: LLMProvider,
    agent_provider: LLMProvider,
    settings: Settings,
    max_explain: int = 20,
    triage_runs: int = 1,
    judge_provider: LLMProvider | None = None,
) -> SectionReport:
    """Triage the section, then explain and fact-check up to `max_explain` high changes.

    Triage runs `triage_runs` times; a change's rating is the highest any run gave. A failed
    triage run is recorded and skipped; if every run fails, its error (LLMError, TriageError)
    is raised: without ratings there is nothing to report on. A failed agent run is recorded
    on its item and the next change still runs. With a `judge_provider`, every answer that
    passes the fact-check is also judged (see judge.py).
    """
    if max_explain < 0:
        raise ValueError("max_explain must be 0 or more")
    triage = triage_section_votes(
        diff,
        company=company,
        form=form,
        section_title=section_title,
        provider=triage_provider,
        settings=settings,
        runs=triage_runs,
    )
    changes = changes_by_id(diff)
    high = [item for item in triage.items if item.importance == "high"]
    medium = [item for item in triage.items if item.importance == "medium"]
    low_count = sum(1 for item in triage.items if item.importance == "low")

    explained = []
    for item in high[:max_explain]:
        context = ExplainContext(
            change=changes[item.id],
            old_section=old_section,
            new_section=new_section,
            metrics=metrics,
        )
        explained.append(
            explain_and_check(
                item.id,
                item.reason,
                context,
                company=company,
                form=form,
                section_title=section_title,
                provider=agent_provider,
                settings=settings,
                judge_provider=judge_provider,
            ).model_copy(update={"triage_votes": item.votes})
        )

    input_tokens = triage.input_tokens + sum(item.input_tokens or 0 for item in explained)
    output_tokens = triage.output_tokens + sum(item.output_tokens or 0 for item in explained)
    return SectionReport(
        company=company,
        form=form,
        section_title=section_title,
        change_count=len(triage.items),
        high_count=len(high),
        medium_count=len(medium),
        low_count=low_count,
        high=tuple(explained),
        not_explained=tuple(
            RatedItem(id=item.id, triage_reason=item.reason, triage_votes=item.votes)
            for item in high[max_explain:]
        ),
        medium=tuple(
            RatedItem(id=item.id, triage_reason=item.reason, triage_votes=item.votes)
            for item in medium
        ),
        triage_runs=triage.runs,
        triage_succeeded=triage.succeeded,
        triage_errors=triage.errors,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        # Price is per token, so the cost of the total is the sum of each call's cost; the
        # judge's share is priced separately, since it may run on another model.
        cost_usd=split_cost(
            input_tokens,
            output_tokens,
            judge_input_tokens=judge_tokens(explained)[0],
            judge_output_tokens=judge_tokens(explained)[1],
            settings=settings,
        ),
        # A failed triage run's usage is lost with it, like a failed agent call's.
        usage_complete=triage.succeeded == triage.runs
        and all(item.input_tokens is not None for item in explained),
        prompt_versions=current_versions(settings, judged=judge_provider is not None),
    )


def judge_tokens(items: Sequence[HighItem]) -> tuple[int, int]:
    """Input and output tokens the judge used over these items: part of their own totals."""
    judged = [item.judgment for item in items if item.judgment is not None]
    return (
        sum(judgment.input_tokens for judgment in judged),
        sum(judgment.output_tokens for judgment in judged),
    )


def explain_and_check(
    change_id: str,
    triage_reason: str,
    context: ExplainContext,
    *,
    company: str,
    form: str,
    section_title: str,
    provider: LLMProvider,
    settings: Settings,
    judge_provider: LLMProvider | None = None,
) -> HighItem:
    """One agent run on one change, fact-checked. Agent and model-call failures are recorded.

    Used by build_section_report for each high change, and by the agent eval for each run.
    With a `judge_provider`, an answer that passes the fact-check is judged too; a failed
    judge call is recorded on the item, never fatal.
    """
    item, sources = explain_and_verify(
        change_id,
        triage_reason,
        context,
        company=company,
        form=form,
        section_title=section_title,
        provider=provider,
        settings=settings,
    )
    # Judged only when the answer could become an alert or a digest entry: a failed
    # fact-check already sends it to review, so a judge call would be money for nothing.
    parsed, verification = item.parsed, item.verification
    if judge_provider is None or parsed is None or verification is None:
        return item
    if not verification.passed or not parsed.what_changed or not parsed.why_it_matters:
        return item
    try:
        judgment = judge_explanation(
            parsed,
            sources,
            kind=context.change.kind,
            company=company,
            form=form,
            section_title=section_title,
            provider=judge_provider,
            settings=settings,
        )
    except LLMError as error:
        return item.model_copy(update={"judge_error": f"judge call failed: {error}"})
    assert item.input_tokens is not None and item.output_tokens is not None  # it answered
    return item.model_copy(
        update={
            "judgment": judgment,
            "input_tokens": item.input_tokens + judgment.input_tokens,
            "output_tokens": item.output_tokens + judgment.output_tokens,
        }
    )


def explain_and_verify(
    change_id: str,
    triage_reason: str,
    context: ExplainContext,
    *,
    company: str,
    form: str,
    section_title: str,
    provider: LLMProvider,
    settings: Settings,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
) -> tuple[HighItem, list[str]]:
    """explain_and_check without the judge: the item, and the sources its answer was checked
    against (the paragraph(s) alone when the agent failed), for a judge to see the same.

    `system_prompt` replaces the agent's prompt, as in explain_change_with_agent.
    """
    try:
        explanation = explain_change_with_agent(
            context,
            company=company,
            form=form,
            section_title=section_title,
            provider=provider,
            settings=settings,
            system_prompt=system_prompt,
        )
    except AgentError as error:
        # The trace says what the steps so far used, so this run's usage is still known.
        failed = HighItem(
            id=change_id,
            triage_reason=triage_reason,
            explanation=None,
            error=f"agent failed: {error}",
            parsed=None,
            verification=None,
            input_tokens=sum(step.input_tokens for step in error.trace),
            output_tokens=sum(step.output_tokens for step in error.trace),
        )
        return failed, sources_for(context.change)
    except LLMError as error:
        # A model call failed inside the loop: what the earlier steps used is lost with it.
        failed = HighItem(
            id=change_id,
            triage_reason=triage_reason,
            explanation=None,
            error=f"model call failed: {error}",
            parsed=None,
            verification=None,
            input_tokens=None,
            output_tokens=None,
        )
        return failed, sources_for(context.change)

    run = explanation.run
    parsed = parse_explanation(run.text, expect_evidence=True)
    sources = sources_for(context.change, run.tool_outputs)
    verification = verify_explanation(parsed, sources)
    item = HighItem(
        id=change_id,
        triage_reason=triage_reason,
        explanation=run.text,
        error=None,
        parsed=parsed,
        verification=verification,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
    )
    return item, sources
