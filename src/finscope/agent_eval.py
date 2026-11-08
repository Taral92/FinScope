"""Agent stability eval: explain the same changes several times and see if the answer holds.

The agent's IMPORTANCE decides the tier (decide.py), so a change rated high in one run and
medium in the next could alert today and only reach the digest tomorrow. This runs the
report's own explain-and-fact-check code several times per change, puts every run through
the decide rules, and counts how often importance and tier change. No prompts of its own.
"""

from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from finscope.agent_tools import TOOL_SPECS, ExplainContext
from finscope.config import Settings
from finscope.decide import Tier, decide_item
from finscope.diff import SectionDiff, changes_by_id
from finscope.evals import Importance, prompt_version, write_json_atomic
from finscope.explain import AGENT_SYSTEM_PROMPT
from finscope.llm import LLMProvider, cost_for_tokens
from finscope.report import HighItem, RatedItem, explain_and_check
from finscope.sections import Section
from finscope.triage import triage_section
from finscope.xbrl import MetricValue

PICKED_REASON = "picked with --ids (triage not run)"


class AgentEvalError(Exception):
    """The eval can't start, e.g. an --ids entry that isn't a change in the diff."""


class Selection(BaseModel):
    """The changes to run the agent on, and what choosing them cost."""

    model_config = ConfigDict(frozen=True)

    picked_by: Literal["triage", "ids"]
    changes: tuple[RatedItem, ...]
    input_tokens: int  # the one triage call; 0 with --ids
    output_tokens: int


class StabilityRun(BaseModel):
    """One agent run on one change: what it said and the tier the decide rules gave it."""

    model_config = ConfigDict(frozen=True)

    number: int  # 1-based
    importance: Importance | None  # None when the agent failed or gave no readable level
    tier: Tier
    tier_reason: str
    fact_check_passed: bool | None  # None when there was no answer to check
    item: HighItem  # the full answer, parse and fact-check, as a report would hold it


class ChangeStability(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    triage_reason: str
    runs: tuple[StabilityRun, ...]
    # Every run that gave an IMPORTANCE gave the same one. Failed runs are left out here;
    # they are counted as agent failures instead.
    importance_stable: bool
    # Not every run landed in the same tier. Every run counts: a failed one is review.
    tier_flips: bool

    @property
    def stable(self) -> bool:
        return self.importance_stable and not self.tier_flips


class StabilitySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    changes: int
    importance_stable: int  # changes
    tier_flips: int  # changes
    total_runs: int
    fact_check_failures: int  # runs that answered but failed the fact-check
    agent_failures: int  # runs with no answer: the agent or a model call failed
    # Triage (if it ran) plus every agent run, failed ones included.
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None  # None when no prices are configured
    usage_complete: bool  # False when a failed call's usage is unknown (totals undercount)


class AgentEvalResult(BaseModel):
    """Everything about one agent eval, saved to evals/results/ next to the triage evals."""

    model_config = ConfigDict(frozen=True)

    created_at: datetime  # UTC
    company: str
    form: str
    part: str | None
    item: str
    new_accession: str
    old_accession: str
    model: str  # the model that was requested
    prompt_version: str  # agent_prompt_version()
    runs: int  # per change
    picked_by: Literal["triage", "ids"]
    changes: tuple[ChangeStability, ...]
    summary: StabilitySummary

    @property
    def file_name(self) -> str:
        section = f"{self.part}-{self.item}" if self.part else self.item
        return (
            f"{self.created_at:%Y%m%dT%H%M%SZ}_agent_{self.company}_{self.form}_"
            f"{self.new_accession}_{section}.json"
        )


def agent_prompt_version() -> str:
    """Like the triage prompt version, over the agent's system prompt and its tool specs."""
    tools = {"tools": [spec.model_dump(mode="json") for spec in TOOL_SPECS]}
    return prompt_version(AGENT_SYSTEM_PROMPT, tools)


def select_changes(
    diff: SectionDiff,
    *,
    ids: Sequence[str] | None,
    company: str,
    form: str,
    section_title: str,
    triage_provider: LLMProvider,
    settings: Settings,
) -> Selection:
    """The changes to evaluate: exactly `ids` (no triage call), or triage's high ones.

    Unknown or repeated IDs raise AgentEvalError before anything is called. A failed triage
    raises LLMError or TriageError, as in the report.
    """
    if ids is not None:
        known = changes_by_id(diff)
        unknown = [change_id for change_id in ids if change_id not in known]
        if unknown:
            raise AgentEvalError(
                f"unknown change ID(s): {', '.join(unknown)}; `diff` lists this section's IDs"
            )
        repeated = sorted({change_id for change_id in ids if ids.count(change_id) > 1})
        if repeated:
            raise AgentEvalError(f"change ID(s) given more than once: {', '.join(repeated)}")
        return Selection(
            picked_by="ids",
            changes=tuple(
                RatedItem(id=change_id, triage_reason=PICKED_REASON) for change_id in ids
            ),
            input_tokens=0,
            output_tokens=0,
        )

    triage = triage_section(
        diff,
        company=company,
        form=form,
        section_title=section_title,
        provider=triage_provider,
        settings=settings,
    )
    return Selection(
        picked_by="triage",
        changes=tuple(
            RatedItem(id=item.id, triage_reason=item.reason)
            for item in triage.items
            if item.importance == "high"
        ),
        input_tokens=triage.input_tokens,
        output_tokens=triage.output_tokens,
    )


def run_stability(
    selection: Selection,
    diff: SectionDiff,
    *,
    old_section: Section,
    new_section: Section,
    metrics: tuple[MetricValue, ...],
    company: str,
    form: str,
    section_title: str,
    agent_provider: LLMProvider,
    settings: Settings,
    runs: int,
    on_run: Callable[[str, StabilityRun], None] | None = None,
) -> tuple[ChangeStability, ...]:
    """Explain and fact-check each selected change `runs` times; a failed run is recorded.

    `on_run(change_id, run)` is called after every run, e.g. to print progress.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    changes = changes_by_id(diff)
    results = []
    for picked in selection.changes:
        context = ExplainContext(
            change=changes[picked.id],
            old_section=old_section,
            new_section=new_section,
            metrics=metrics,
        )
        change_runs = []
        for number in range(1, runs + 1):
            item = explain_and_check(
                picked.id,
                picked.triage_reason,
                context,
                company=company,
                form=form,
                section_title=section_title,
                provider=agent_provider,
                settings=settings,
            )
            run = _stability_run(number, item)
            change_runs.append(run)
            if on_run is not None:
                on_run(picked.id, run)
        results.append(_change_stability(picked, change_runs))
    return tuple(results)


def summarize(
    changes: Sequence[ChangeStability], selection: Selection, settings: Settings
) -> StabilitySummary:
    items = [run.item for change in changes for run in change.runs]
    input_tokens = selection.input_tokens + sum(item.input_tokens or 0 for item in items)
    output_tokens = selection.output_tokens + sum(item.output_tokens or 0 for item in items)
    return StabilitySummary(
        changes=len(changes),
        importance_stable=sum(change.importance_stable for change in changes),
        tier_flips=sum(change.tier_flips for change in changes),
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


def save_agent_eval(result: AgentEvalResult, directory: Path) -> Path:
    path = directory / result.file_name
    write_json_atomic(path, result)
    return path


def _stability_run(number: int, item: HighItem) -> StabilityRun:
    entry = decide_item(item)  # the same rules the report's decision uses
    return StabilityRun(
        number=number,
        importance=item.parsed.importance if item.parsed is not None else None,
        tier=entry.tier,
        tier_reason=entry.reason,
        fact_check_passed=item.verification.passed if item.verification is not None else None,
        item=item,
    )


def _change_stability(picked: RatedItem, runs: list[StabilityRun]) -> ChangeStability:
    importances = {run.importance for run in runs if run.importance is not None}
    return ChangeStability(
        id=picked.id,
        triage_reason=picked.triage_reason,
        runs=tuple(runs),
        importance_stable=len(importances) <= 1,
        tier_flips=len({run.tier for run in runs}) > 1,
    )
