"""Build the agent's memory from files FinScope already writes. Plain code, no LLM.

Every saved report (data/reports/) adds its explained changes; your feedback (data/feedback/)
adds your verdict and note to the ones you rated. Only answers that passed the fact-check are
remembered: a claim that failed it never became a finding, so it isn't one now.
"""

from pathlib import Path

from finscope.detect import CheckResult
from finscope.feedback import Feedback, load_feedback, load_report
from finscope.memory import PastFinding


def load_memory(
    reports: Path, feedback: Path, *, exclude_accession: str | None = None
) -> tuple[PastFinding, ...]:
    """Every remembered finding, oldest filing first.

    `exclude_accession` leaves out the filing being explained: a re-run must not recall its
    own earlier answers, or your verdicts on them. A report or feedback file that can't be
    read is a FeedbackError, as in eval-alerts: memory is never silently partial.
    """
    findings: list[PastFinding] = []
    for path in sorted(reports.glob("*.json")):
        result = load_report(path)
        if result.new_filing.accession_number == exclude_accession:
            continue
        answers_path = feedback / path.name
        answers = load_feedback(answers_path) if answers_path.exists() else ()
        findings += _findings(result, answers)
    return tuple(sorted(findings, key=lambda f: (f.filing_date, f.company, f.change_id)))


def _findings(result: CheckResult, answers: tuple[Feedback, ...]) -> list[PastFinding]:
    verdicts = {answer.id: answer for answer in answers}
    tiers = {entry.id: entry.tier for entry in result.decision.entries}
    entry = result.entry
    section = f"{entry.part}-{entry.item}" if entry.part else entry.item
    found = []
    for item in result.report.high:
        parsed, check = item.parsed, item.verification
        if parsed is None or check is None or not check.passed or not parsed.what_changed:
            continue
        answer = verdicts.get(item.id)
        found.append(
            PastFinding(
                company=entry.ticker,
                form=entry.form,
                section=section,
                filing_date=result.new_filing.filing_date,
                accession=result.new_filing.accession_number,
                change_id=item.id,
                what_changed=parsed.what_changed,
                triage_reason=item.triage_reason,
                importance=parsed.importance,
                tier=tiers.get(item.id, "review"),
                verdict=answer.verdict if answer else None,
                note=answer.note if answer else None,
            )
        )
    return found
