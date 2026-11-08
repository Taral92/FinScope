from decimal import Decimal

import pytest

from finscope.__main__ import _print_report
from finscope.decide import Decision, DecisionEntry, decide
from finscope.report import HighItem, RatedItem, SectionReport
from finscope.verify import parse_explanation, verify_explanation

SOURCE = "Upon launching our Robotaxi service, we entered a new market."
# Words that appear only in the agent's answers, so a test can see if any of it leaked.
AGENT_WORDS = "AGENT-ONLY"


def answer_text(importance: str | None = "high", quote: str = "we entered a new market") -> str:
    lines = [
        f"WHAT CHANGED: {AGENT_WORDS} Robotaxi risk added.",
        f"WHY IT MATTERS: {AGENT_WORDS} a new business line brings new risks.",
    ]
    if importance is not None:
        lines.append(f"IMPORTANCE: {importance} - {AGENT_WORDS} reason")
    lines.append(f'EVIDENCE: "{quote}"')
    return "\n".join(lines)


def answered(change_id: str, text: str) -> HighItem:
    """A high change the agent answered, parsed and fact-checked as report.py does."""
    parsed = parse_explanation(text, expect_evidence=True)
    return HighItem(
        id=change_id,
        triage_reason=f"triage reason {change_id}",
        explanation=text,
        error=None,
        parsed=parsed,
        verification=verify_explanation(parsed, [SOURCE]),
        input_tokens=300,
        output_tokens=50,
    )


def failed(change_id: str, error: str) -> HighItem:
    return HighItem(
        id=change_id,
        triage_reason=f"triage reason {change_id}",
        explanation=None,
        error=error,
        parsed=None,
        verification=None,
        input_tokens=None,
        output_tokens=None,
    )


def section_report(high: list[HighItem], not_explained: tuple[str, ...] = ()) -> SectionReport:
    over_cap = tuple(RatedItem(id=i, triage_reason=f"triage reason {i}") for i in not_explained)
    return SectionReport(
        company="TSLA",
        form="10-K",
        section_title="Part I Item 1A. RISK FACTORS",
        change_count=len(high) + len(over_cap) + 1,
        high_count=len(high) + len(over_cap),
        medium_count=1,
        low_count=0,
        high=tuple(high),
        not_explained=over_cap,
        medium=(RatedItem(id="+9", triage_reason="triage reason +9"),),
        triage_runs=1,
        triage_succeeded=1,
        triage_errors=(),
        input_tokens=1_000,
        output_tokens=200,
        cost_usd=Decimal("0.003"),
        usage_complete=True,
    )


def only_entry(decision: Decision) -> DecisionEntry:
    [entry] = decision.entries
    return entry


def test_agent_high_and_passed_fact_check_is_an_alert() -> None:
    entry = only_entry(decide(section_report([answered("~14", answer_text("high"))])))

    assert entry.id == "~14"
    assert entry.tier == "alert"
    assert entry.reason == "agent rated high, fact-check passed"
    assert entry.triage_reason == "triage reason ~14"
    assert entry.what_changed == f"{AGENT_WORDS} Robotaxi risk added."
    assert entry.why_it_matters == f"{AGENT_WORDS} a new business line brings new risks."


@pytest.mark.parametrize("level", ["medium", "low", "Medium"])
def test_agent_rating_lower_with_passed_fact_check_is_a_digest(level: str) -> None:
    entry = only_entry(decide(section_report([answered("+3", answer_text(level))])))

    assert entry.tier == "digest"
    assert entry.reason == f"agent rated {level.lower()}"
    assert entry.triage_reason == "triage reason +3"


@pytest.mark.parametrize(
    ("item", "reason"),
    [
        # The agent said high, but a quote isn't in the sources: review wins over alert.
        (
            answered("~11", answer_text("high", quote="not in the paragraph")),
            "fact-check failed: quote not found",
        ),
        # Medium with a failed check is review too, not digest.
        (
            answered("~11", answer_text("medium", quote="not in the paragraph")),
            "fact-check failed: quote not found",
        ),
        (answered("~11", answer_text(None)), "no readable IMPORTANCE line"),
        (answered("~11", answer_text("critical")), "no readable IMPORTANCE line"),
        (failed("~11", "agent failed: step limit reached"), "agent failed: step limit reached"),
        (failed("~11", "model call failed: timeout"), "model call failed: timeout"),
    ],
)
def test_anything_untrusted_is_review(item: HighItem, reason: str) -> None:
    entry = only_entry(decide(section_report([item])))

    assert entry.tier == "review"
    assert entry.reason == reason
    assert entry.triage_reason == "triage reason ~11"


def test_every_issue_is_counted_in_the_review_reason() -> None:
    text = answer_text("high", quote="not in the paragraph").replace(
        "Robotaxi risk added.", "Robotaxi revenue grew 45%."
    )

    entry = only_entry(decide(section_report([answered("~11", text)])))

    assert entry.reason == "fact-check failed: quote not found (+1 more issue)"


def test_changes_over_the_cap_are_review() -> None:
    entry = only_entry(decide(section_report([], not_explained=("+7",))))

    assert entry.tier == "review"
    assert entry.reason == "not explained: over the cap"
    assert entry.triage_reason == "triage reason +7"


def test_no_high_change_is_dropped_and_diff_order_is_kept() -> None:
    report = section_report(
        [
            answered("+1", answer_text("high")),
            answered("+2", answer_text("medium")),
            answered("~3", answer_text("high", quote="made up")),
            failed("~4", "model call failed: timeout"),
        ],
        not_explained=("~5", "~6"),
    )

    decision = decide(report)

    assert len(decision.entries) == report.high_count
    assert [entry.id for entry in decision.entries] == ["+1", "+2", "~3", "~4", "~5", "~6"]
    assert [entry.id for entry in decision.tier("alert")] == ["+1"]
    assert [entry.id for entry in decision.tier("digest")] == ["+2"]
    assert [entry.id for entry in decision.tier("review")] == ["~3", "~4", "~5", "~6"]


def test_digest_and_review_never_carry_the_agents_text() -> None:
    report = section_report(
        [
            answered("+2", answer_text("medium")),
            answered("~3", answer_text("high", quote=f"{AGENT_WORDS} made-up quote")),
            answered("~4", answer_text(f"{AGENT_WORDS}")),  # unreadable IMPORTANCE
        ]
    )

    decision = decide(report)

    assert [entry.tier for entry in decision.entries] == ["digest", "review", "review"]
    for entry in decision.entries:
        assert entry.what_changed is None and entry.why_it_matters is None
        assert AGENT_WORDS not in entry.model_dump_json()


def test_zero_high_changes_gives_an_empty_decision() -> None:
    assert decide(section_report([])).entries == ()


def test_printed_decision_comes_before_the_details(capsys: pytest.CaptureFixture[str]) -> None:
    report = section_report(
        [answered("~14", answer_text("high")), answered("+3", answer_text("medium"))]
    )

    _print_report(report)

    out = capsys.readouterr().out
    assert (
        "=== ALERT (1) ===\n"
        f"[~14] {AGENT_WORDS} Robotaxi risk added.\n"
        f"      WHY IT MATTERS: {AGENT_WORDS} a new business line brings new risks.\n"
    ) in out
    assert (
        "=== DIGEST (1): agent rated lower than triage ===\n"
        "[+3] agent rated medium | triage: triage reason +3\n"
    ) in out
    assert "=== REVIEW (0): read these yourself ===\n(none)\n" in out
    assert out.index("=== ALERT") < out.index("=== REVIEW") < out.index("=== HIGH")


def test_printed_empty_tiers_say_none(capsys: pytest.CaptureFixture[str]) -> None:
    _print_report(section_report([], not_explained=("+7",)))

    out = capsys.readouterr().out
    assert "=== ALERT (0) ===\n(none)\n" in out
    assert "=== DIGEST (0): agent rated lower than triage ===\n(none)\n" in out
    assert (
        "=== REVIEW (1): read these yourself ===\n"
        "[+7] not explained: over the cap | triage: triage reason +7\n"
    ) in out
