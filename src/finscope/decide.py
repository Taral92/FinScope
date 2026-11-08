"""Decide what to do with each high change of a section report. Plain code, no LLM.

Every change triage rated high lands in exactly one tier:
- review: anything we can't trust. The agent failed, the fact-check failed, the answer had no
  readable IMPORTANCE line, or the change was never explained (over the max_explain cap).
- alert: the agent also rated it high and its answer passed the fact-check.
- digest: the answer passed the fact-check, but the agent rated it medium or low.

Review wins over everything, then alert; otherwise digest. Only an alert carries the agent's
text: a digest or review entry shows the triage reason and why it landed there, so a claim
that wasn't verified (or wasn't rated high by the agent) is never passed on as a finding.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from finscope.report import HighItem, SectionReport
from finscope.triage import Importance

Tier = Literal["alert", "digest", "review"]

# Fixed words for each kind of fact-check issue. The issue messages themselves quote the
# agent's unverified text, so they stay in the detailed report and out of the decision.
_ISSUE_SUMMARY = {
    "format": "answer not in the expected format",
    "quote": "quote not found",
    "number": "number not found",
}


class DecisionEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tier: Tier
    reason: str  # why it landed in this tier, in plain words
    triage_reason: str
    triage_votes: tuple[Importance, ...] = ()  # every triage run's rating, copied from the report
    # From the agent's answer, for an alert only; None in every digest and review entry.
    what_changed: str | None = None
    why_it_matters: str | None = None


class Decision(BaseModel):
    """One entry per high change of the report, in diff order."""

    model_config = ConfigDict(frozen=True)

    entries: tuple[DecisionEntry, ...]

    def tier(self, tier: Tier) -> tuple[DecisionEntry, ...]:
        return tuple(entry for entry in self.entries if entry.tier == tier)


def decide(report: SectionReport) -> Decision:
    """Put every high change into alert, digest or review. None is ever left out."""
    entries = [decide_item(item) for item in report.high]
    entries += [
        DecisionEntry(
            id=item.id,
            tier="review",
            reason="not explained: over the cap",
            triage_reason=item.triage_reason,
            triage_votes=item.triage_votes,
        )
        for item in report.not_explained
    ]
    return Decision(entries=tuple(entries))


def decide_item(item: HighItem) -> DecisionEntry:
    """The tier of one explained change. decide() uses it, and so does the agent eval."""

    def review(reason: str) -> DecisionEntry:
        return DecisionEntry(
            id=item.id,
            tier="review",
            reason=reason,
            triage_reason=item.triage_reason,
            triage_votes=item.triage_votes,
        )

    if item.error is not None:
        return review(item.error)
    parsed, check = item.parsed, item.verification
    if parsed is None or check is None:  # an answer is always parsed and checked
        return review("answer not checked")
    if parsed.importance is None:
        return review("no readable IMPORTANCE line")
    if not check.passed:
        return review(_fact_check_failure(check.issues[0].kind, len(check.issues)))
    if not parsed.what_changed or not parsed.why_it_matters:
        return review("answer has an empty WHAT CHANGED or WHY IT MATTERS")

    if parsed.importance == "high":
        return DecisionEntry(
            id=item.id,
            tier="alert",
            reason="agent rated high, fact-check passed",
            triage_reason=item.triage_reason,
            triage_votes=item.triage_votes,
            what_changed=parsed.what_changed,
            why_it_matters=parsed.why_it_matters,
        )
    return DecisionEntry(
        id=item.id,
        tier="digest",
        reason=f"agent rated {parsed.importance}",
        triage_reason=item.triage_reason,
        triage_votes=item.triage_votes,
    )


def _fact_check_failure(first_kind: str, issue_count: int) -> str:
    reason = f"fact-check failed: {_ISSUE_SUMMARY[first_kind]}"
    more = issue_count - 1
    if more:
        reason += f" (+{more} more issue{'' if more == 1 else 's'})"
    return reason
