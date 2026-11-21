from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, get_args

import pytest

from finscope.__main__ import _memory
from finscope.agent_tools import TOOL_SPECS, ExplainContext, build_tools, recall_past_findings
from finscope.config import Settings
from finscope.decide import Tier, decide
from finscope.detect import CheckResult, WatchEntry, reports_dir, save_check_result
from finscope.diff import ParagraphChange
from finscope.edgar import Filing
from finscope.feedback import Feedback, Verdict, feedback_dir, save_feedback
from finscope.llm import AgentStep, ConversationState, ToolCall, ToolResult, ToolSpec
from finscope.memory import MAX_TEXT_CHARS, PastFinding, format_recall, recall
from finscope.memory import Tier as MemoryTier
from finscope.memory import Verdict as MemoryVerdict
from finscope.memory_store import load_memory
from finscope.report import HighItem, SectionReport, explain_and_verify
from finscope.sections import Section
from finscope.verify import parse_explanation, verify_explanation

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)
PACKAGING = "New packaging and sustainability reporting rules increase our compliance costs."
STRATEGY = "Our restructuring plan closes stores and changes how results compare across periods."


def answer(what: str, quote: str, importance: str = "high") -> str:
    return (
        f"WHAT CHANGED: {what}\n"
        "WHY IT MATTERS: It could matter.\n"
        f"IMPORTANCE: {importance} - potential: a reason\n"
        f'EVIDENCE: "{quote}"'
    )


def item(change_id: str, source: str, what: str, quote: str, importance: str = "high") -> HighItem:
    text = answer(what, quote, importance)
    parsed = parse_explanation(text, expect_evidence=True)
    return HighItem(
        id=change_id,
        triage_reason=f"triage {change_id}",
        explanation=text,
        error=None,
        parsed=parsed,
        verification=verify_explanation(parsed, [source]),
        input_tokens=300,
        output_tokens=50,
    )


def saved_report(data_dir: Path, ticker: str, accession: str, day: date, *items: HighItem) -> str:
    """A check result as `check` saves it; returns its file name."""
    report = SectionReport(
        company=ticker,
        form="10-K",
        section_title="Part I Item 1A. Risk Factors",
        change_count=len(items),
        high_count=len(items),
        medium_count=0,
        low_count=0,
        high=items,
        not_explained=(),
        medium=(),
        input_tokens=0,
        output_tokens=0,
        cost_usd=None,
        usage_complete=True,
    )
    filing = Filing(
        cik="0000000001",
        accession_number=accession,
        form="10-K",
        filing_date=day,
        report_date=None,
        primary_document="x.htm",
    )
    result = CheckResult(
        checked_at=datetime(2026, 11, 20, tzinfo=UTC),
        entry=WatchEntry(ticker=ticker, form="10-K", item="1A", part="I"),
        old_filing=filing,
        new_filing=filing,
        report=report,
        decision=decide(report),
    )
    save_check_result(result, reports_dir(data_dir))
    return result.file_name


def rate(data_dir: Path, report: str, *answers: tuple[str, Verdict, str | None]) -> None:
    feedback = [
        Feedback(
            report=report,
            checked_at=datetime(2026, 11, 20, tzinfo=UTC),
            id=change_id,
            tier="alert",
            verdict=verdict,
            answered_at=datetime(2026, 11, 21, tzinfo=UTC),
            note=note,
        )
        for change_id, verdict, note in answers
    ]
    save_feedback(feedback, feedback_dir(data_dir) / report)


def finding(company: str, change_id: str, what: str, **fields: Any) -> PastFinding:
    values: dict[str, Any] = {
        "company": company,
        "form": "10-K",
        "section": "I-1A",
        "filing_date": date(2025, 11, 1),
        "accession": f"acc-{company}",
        "change_id": change_id,
        "what_changed": what,
        "triage_reason": "",
        "importance": "high",
        "tier": "alert",
        "verdict": None,
        "note": None,
    }
    return PastFinding(**(values | fields))


def test_memorys_copies_of_tier_and_verdict_match_the_originals() -> None:
    assert get_args(MemoryTier) == get_args(Tier)
    assert get_args(MemoryVerdict) == get_args(Verdict)


# --- building memory from saved reports and feedback ---------------------------------------------


def test_memory_holds_checked_answers_with_your_verdicts(tmp_path: Path) -> None:
    sbux = saved_report(
        tmp_path,
        "SBUX",
        "acc-sbux-26",
        date(2026, 11, 12),
        item("+5", PACKAGING, "Starbucks added packaging rules.", "packaging and sustainability"),
        item("+6", PACKAGING, "A made-up claim.", "not in the paragraph"),  # failed fact-check
    )
    rate(tmp_path, sbux, ("+5", "not_useful", "routine compliance"))
    saved_report(
        tmp_path,
        "DIS",
        "acc-dis-26",
        date(2026, 11, 17),
        item("+3", STRATEGY, "Disney added restructuring.", "restructuring plan", "medium"),
    )

    memory = load_memory(reports_dir(tmp_path), feedback_dir(tmp_path))

    assert [(f.company, f.change_id) for f in memory] == [("SBUX", "+5"), ("DIS", "+3")]
    sbux_5, dis_3 = memory
    assert (sbux_5.verdict, sbux_5.note, sbux_5.tier) == (
        "not_useful",
        "routine compliance",
        "alert",
    )
    assert (sbux_5.section, sbux_5.accession) == ("I-1A", "acc-sbux-26")
    assert (dis_3.verdict, dis_3.tier, dis_3.importance) == (None, "digest", "medium")


def test_the_filing_being_explained_is_never_in_its_own_memory(tmp_path: Path) -> None:
    saved_report(
        tmp_path,
        "SBUX",
        "acc-sbux-26",
        date(2026, 11, 12),
        item("+5", PACKAGING, "Starbucks added packaging rules.", "packaging and sustainability"),
    )

    memory = load_memory(
        reports_dir(tmp_path), feedback_dir(tmp_path), exclude_accession="acc-sbux-26"
    )

    assert memory == ()


def test_the_cli_loads_memory_without_the_new_filing_and_says_how_much(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sbux = saved_report(
        tmp_path,
        "SBUX",
        "acc-sbux-25",
        date(2025, 11, 14),
        item("+5", PACKAGING, "Starbucks added packaging rules.", "packaging and sustainability"),
    )
    rate(tmp_path, sbux, ("+5", "useful", None))
    saved_report(
        tmp_path,
        "SBUX",
        "acc-sbux-26",
        date(2026, 11, 12),
        item("+1", STRATEGY, "Starbucks added restructuring.", "restructuring plan"),
    )
    new = Filing(
        cik="0000000001",
        accession_number="acc-sbux-26",
        form="10-K",
        filing_date=date(2026, 11, 12),
        report_date=None,
        primary_document="x.htm",
    )

    memory = _memory(tmp_path, new)

    assert [(f.accession, f.change_id) for f in memory] == [("acc-sbux-25", "+5")]
    assert "memory: 1 earlier finding, 1 rated by you" in capsys.readouterr().out


# --- recall ---------------------------------------------------------------------------------------

PAST = (
    finding("SBUX", "+5", "Starbucks added packaging and sustainability reporting rules."),
    finding("SBUX", "+1", "Starbucks added a store restructuring plan."),
    finding(
        "DIS",
        "+9",
        "Disney added packaging and reporting rules for its consumer products.",
        verdict="not_useful",
        note="routine compliance",
    ),
    finding("DIS", "+2", "Disney added new packaging rules."),  # not rated: not a preference
    finding("TSLA", "~1", "Tesla expanded its tariff risk.", verdict="useful"),  # unrelated
)


def test_recall_finds_this_companys_history_and_rated_alerts_elsewhere() -> None:
    history, rated = recall(PAST, PACKAGING, company="SBUX", section="I-1A")

    assert [(f.company, f.change_id) for f in history] == [("SBUX", "+5")]
    assert [(f.company, f.change_id) for f in rated] == [("DIS", "+9")]  # rated, and similar


def test_recall_keeps_another_sections_findings_out_of_the_history() -> None:
    other_section = finding("SBUX", "+5", "Starbucks added packaging rules.", section="I-7")

    history, _ = recall((other_section,), PACKAGING, company="SBUX", section="I-1A")

    assert history == []


def test_nothing_similar_recalls_nothing() -> None:
    assert recall(
        PAST, "Earthquakes may damage our data centers.", company="X", section="I-1A"
    ) == (
        [],
        [],
    )


def test_the_answer_marks_memory_as_not_filing_text_and_shows_verdicts() -> None:
    long_what = "Starbucks added " + "packaging " * 100
    history = [finding("SBUX", "+5", long_what)]
    rated = [PAST[2]]

    text = format_recall(history, rated, company="SBUX", section="I-1A")

    assert text.startswith("FinScope's own earlier findings (its past wording, not filing text")
    assert "never quote it as evidence" in text
    assert "Earlier filings of SBUX, section I-1A:" in text
    assert (
        '- DIS 10-K filed 2025-11-01 [+9]: rated high, alert, user said not_useful ("routine'
        in text
    )
    history_line = text.split("\n")[2]
    assert history_line.endswith("…") and len(history_line) < MAX_TEXT_CHARS + 80
    empty = format_recall([], [], company="SBUX", section="I-1A")
    assert empty.count("(none similar)") == 2


# --- the tool, and the fact-check refusing memory as a source ------------------------------------

SECTION = Section(part="I", item="1A", title="Risk Factors", start=0, end=0, text=PACKAGING)
CHANGE = ParagraphChange(
    kind="added",
    old_text=None,
    new_text=PACKAGING,
    old_start=None,
    old_end=None,
    new_start=0,
    new_end=len(PACKAGING),
    similarity=None,
)


def context(memory: tuple[PastFinding, ...]) -> ExplainContext:
    return ExplainContext(
        change=CHANGE,
        old_section=SECTION,
        new_section=SECTION,
        metrics=(),
        memory=memory,
        company="SBUX",
        section="I-1A",
    )


def test_the_agent_has_the_memory_tool_and_it_answers_from_the_context() -> None:
    tools = build_tools(context(PAST))

    assert "recall_past_findings" in tools
    assert [tool.spec for tool in tools.values()] == list(TOOL_SPECS)
    text = recall_past_findings(context(PAST))
    assert "[+5]" in text and "[+9]" in text
    assert "(none similar)" in recall_past_findings(context(()))  # no memory: an honest empty


class ToolThenAnswer:
    """Implements LLMProvider's start/resume: calls recall_past_findings, then answers."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.tool_results: list[ToolResult] = []

    def start(
        self, system: str, user: str, tools: list[ToolSpec]
    ) -> tuple[AgentStep, ConversationState]:
        call = ToolCall(id="c1", name="recall_past_findings", arguments={}, raw_arguments="{}")
        return (
            AgentStep(text=None, tool_calls=(call,), input_tokens=100, output_tokens=10),
            ConversationState(),
        )

    def resume(
        self, state: ConversationState, tool_results: list[ToolResult]
    ) -> tuple[AgentStep, ConversationState]:
        self.tool_results += tool_results
        return AgentStep(text=self.text, tool_calls=(), input_tokens=300, output_tokens=50), state


def explain(provider: ToolThenAnswer) -> tuple[HighItem, list[str]]:
    return explain_and_verify(
        "+1",
        "triage +1",
        context(PAST),
        company="SBUX",
        form="10-K",
        section_title="Part I Item 1A. Risk Factors",
        provider=provider,
        settings=SETTINGS,
    )


def test_a_quote_found_only_in_memory_fails_the_fact_check() -> None:
    # "routine compliance" is the user's note, in the memory tool's answer, not in the filing.
    provider = ToolThenAnswer(answer("Starbucks added packaging rules.", "routine compliance"))

    item, sources = explain(provider)

    [result] = provider.tool_results
    assert "routine compliance" in result.output  # the agent did see it
    assert sources == [PACKAGING]  # but memory is not a source
    assert item.verification is not None and not item.verification.passed


def test_a_quote_from_the_filing_still_passes_after_recalling() -> None:
    provider = ToolThenAnswer(
        answer("Starbucks added packaging rules.", "increase our compliance costs")
    )

    item, _ = explain(provider)

    assert item.verification is not None and item.verification.passed
