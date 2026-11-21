import json
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from finscope.decide import Decision, DecisionEntry
from finscope.detect import CheckResult, WatchEntry
from finscope.edgar import Filing
from finscope.llm import LLMError, LLMResult
from finscope.notify import format_filing_message
from finscope.report import SectionReport
from finscope.synthesis import (
    MAX_POINTS,
    Finding,
    findings_of,
    needs_summary,
    summarize_filing,
    summary_prompt_version,
)

TARIFFS = Finding(
    ref="I-1A +4",
    section_title="Part I Item 1A. Risk Factors",
    what_changed="Acme added a risk that tariffs on battery cells could raise its costs.",
    why_it_matters="Tariffs of 25% now apply to its imported cells.",
)
MARGIN = Finding(
    ref="II-7 ~12",
    section_title="Part II Item 7. MD&A",
    what_changed="Acme says tariffs reduced its gross margin to 17.1%.",
    why_it_matters="The tariff risk has already hurt profitability.",
)
LAWSUIT = Finding(
    ref="I-3 +1",
    section_title="Part I Item 3. Legal Proceedings",
    what_changed="Acme added a class action over driver-assistance claims.",
    why_it_matters="It may lead to damages.",
)


class FakeWriter:
    """complete_json for the summary call: one scripted list of points, or an LLMError."""

    def __init__(self, points: list[dict[str, Any]] | LLMError) -> None:
        self.points = points
        self.users: list[str] = []

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        self.users.append(user)
        if isinstance(self.points, LLMError):
            raise self.points
        text = json.dumps({"points": self.points})
        return LLMResult(text=text, model="fake-writer", input_tokens=900, output_tokens=80)


class FakeChecker:
    """complete_json for the point checks: one scripted verdict (or LLMError) per call."""

    def __init__(self, verdicts: list[str | LLMError]) -> None:
        self.verdicts = list(verdicts)
        self.users: list[str] = []

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        self.users.append(user)
        verdict = self.verdicts.pop(0)
        if isinstance(verdict, LLMError):
            raise verdict
        text = json.dumps({"verdict": verdict, "reason": f"{verdict} because"})
        return LLMResult(text=text, model="fake-judge", input_tokens=400, output_tokens=30)


def point(text: str, *refs: str) -> dict[str, Any]:
    return {"text": text, "refs": list(refs)}


def summarize(writer: FakeWriter, checker: FakeChecker) -> Any:
    return summarize_filing(
        [TARIFFS, MARGIN, LAWSUIT],
        company="ACME",
        form="10-K",
        provider=writer,
        check_provider=checker,
    )


def test_points_that_pass_the_code_checks_and_the_check_call_are_kept() -> None:
    writer = FakeWriter(
        [
            point(
                "A new tariff risk has already cut gross margin to 17.1%.", "I-1A +4", "II-7 ~12"
            ),
            point("A class action over driver assistance may lead to damages.", "I-3 +1"),
        ]
    )
    checker = FakeChecker(["supported", "supported"])

    summary = summarize(writer, checker)

    assert [p.status for p in summary.points] == ["kept", "kept"]
    assert summary.kept[0].refs == ("I-1A +4", "II-7 ~12")
    # The writer sees every finding with its ref; each check sees only the findings cited.
    [written] = writer.users
    assert 'ref="I-1A +4"' in written and 'ref="I-3 +1"' in written
    first_check, second_check = checker.users
    assert "17.1%" in first_check and "class action" not in first_check
    assert "class action" in second_check and "17.1%" not in second_check
    assert (summary.input_tokens, summary.check_input_tokens) == (900, 800)
    assert summary.prompt_version == summary_prompt_version()


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (point("Tariffs matter.", "I-1A +9"), "unknown ref(s): I-1A +9"),
        (point("Tariffs matter."), "cites no finding"),
        (point("   ", "I-1A +4"), "empty text"),
        # 30% is in no cited finding; 25% and 17.1% would have been fine.
        (point("Tariffs of 30% hit margins.", "I-1A +4", "II-7 ~12"), "number not in the cited"),
    ],
)
def test_plain_code_drops_bad_points_without_a_check_call(bad: dict[str, Any], reason: str) -> None:
    checker = FakeChecker([])  # a check call would pop from an empty list and fail the test

    summary = summarize(FakeWriter([bad]), checker)

    [dropped] = summary.points
    assert dropped.status == "dropped" and dropped.reason.startswith(reason)
    assert summary.kept == ()


def test_only_the_first_three_points_are_considered() -> None:
    # Words, not numbers: "Point 1" would be dropped for citing a number no finding has.
    four = [
        point(f"The {n} point about tariffs.", "I-1A +4")
        for n in ("first", "second", "third", "fourth")
    ]

    summary = summarize(FakeWriter(four), FakeChecker(["supported"] * MAX_POINTS))

    assert [p.status for p in summary.points] == ["kept", "kept", "kept", "dropped"]
    assert summary.points[3].reason == f"more than {MAX_POINTS} points"


def test_a_point_the_check_rejects_is_dropped_with_its_verdict() -> None:
    writer = FakeWriter([point("Tariffs will destroy Acme's margin.", "I-1A +4", "II-7 ~12")])

    summary = summarize(writer, FakeChecker(["overstated"]))

    [dropped] = summary.points
    assert (dropped.status, dropped.reason) == ("dropped", "check: overstated: overstated because")


def test_a_failed_check_drops_that_point_and_the_next_still_runs() -> None:
    writer = FakeWriter(
        [point("Tariffs cut margin.", "I-1A +4", "II-7 ~12"), point("A class action.", "I-3 +1")]
    )

    summary = summarize(writer, FakeChecker([LLMError("timeout"), "supported"]))

    assert [p.status for p in summary.points] == ["dropped", "kept"]
    assert summary.points[0].reason == "check failed: timeout"


def test_a_failed_summary_call_is_the_callers() -> None:
    with pytest.raises(LLMError, match="rate limited"):
        summarize(FakeWriter(LLMError("rate limited")), FakeChecker([]))


# --- which filings get a summary, and how it reads ------------------------------------------------


def alert(change_id: str) -> DecisionEntry:
    return DecisionEntry(
        id=change_id,
        tier="alert",
        reason="agent rated high, fact-check passed",
        triage_reason="t",
        what_changed=f"Change {change_id}.",
        why_it_matters=f"Why {change_id}.",
    )


def result(item: str, *entries: DecisionEntry) -> CheckResult:
    report = SectionReport(
        company="ACME",
        form="10-K",
        section_title=f"Part I Item {item}. Title",
        change_count=len(entries),
        high_count=len(entries),
        medium_count=0,
        low_count=0,
        high=(),
        not_explained=(),
        medium=(),
        input_tokens=0,
        output_tokens=0,
        cost_usd=Decimal("0"),
        usage_complete=True,
    )
    filing = Filing(
        cik="0000000001",
        accession_number="0000000001-26-000001",
        form="10-K",
        filing_date=date(2026, 1, 29),
        report_date=None,
        primary_document="x.htm",
    )
    return CheckResult(
        checked_at=datetime(2026, 11, 21, tzinfo=UTC),
        entry=WatchEntry(ticker="ACME", form="10-K", item=item, part="I"),
        old_filing=filing,
        new_filing=filing,
        report=report,
        decision=Decision(entries=entries),
    )


def test_only_alerts_from_two_or_more_sections_need_a_summary() -> None:
    two_sections = findings_of([result("1A", alert("+1")), result("3", alert("+2"))])
    one_section = findings_of([result("1A", alert("+1"), alert("+2"))])

    assert [f.ref for f in two_sections] == ["I-1A +1", "I-3 +2"]
    assert needs_summary(two_sections)
    assert not needs_summary(one_section)
    assert not needs_summary([])


def test_the_message_shows_kept_points_first_and_never_dropped_ones() -> None:
    results = [result("1A", alert("+1")), result("3", alert("+2"))]
    writer = FakeWriter(
        [point("Kept point.", "I-1A +1", "I-3 +2"), point("Rejected point.", "I-3 +2")]
    )
    findings = findings_of(results)
    summary = summarize_filing(
        findings,
        company="ACME",
        form="10-K",
        provider=writer,
        check_provider=FakeChecker(["supported", "unsupported"]),
    )

    message = format_filing_message(results, summary=summary)

    assert message is not None
    blocks = message.split("\n\n")
    assert blocks[1] == "SUMMARY\n1. Kept point. [I-1A +1; I-3 +2]"
    assert "Rejected point" not in message
    without = format_filing_message(results)
    assert without is not None and "SUMMARY" not in without
