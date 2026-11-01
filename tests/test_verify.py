from pathlib import Path

from finscope.diff import ParagraphChange
from finscope.verify import Issue, parse_explanation, sources_for, verify_explanation

GOOD_ANSWER = (
    "WHAT CHANGED: Tesla added a risk factor about its Robotaxi service.\n"
    "WHY IT MATTERS: Revenue of $28,046M now depends partly on a new, unproven business.\n"
    "IMPORTANCE: Medium - a new business line with adoption risk\n"
    # One straight-quoted and one curly-quoted piece of evidence.
    'EVIDENCE: "Upon launching our Robotaxi service in June 2025" ; '
    "\u201camid growing competition\u201d"
)

SOURCE = (
    "Upon launching our Robotaxi service in June 2025, we also entered into the autonomous "
    "ride-hailing service market, and have plans to mass produce Cybercab, a purpose-built "
    "Robotaxi product. Our success will depend on the acceptance and adoption by consumers "
    "of autonomous driving solutions, and Robotaxi as a preferable option, amid growing "
    "competition."
)
METRICS = "Revenue | 2026-07-01 to 2026-09-30 | $28,046M | prior $28,095M | -0.2%"


def verify(answer: str, sources: list[str]) -> list[Issue]:
    return list(verify_explanation(parse_explanation(answer), sources).issues)


# --- Parsing ------------------------------------------------------------------------------------


def test_good_answer_is_parsed() -> None:
    parsed = parse_explanation(GOOD_ANSWER)

    assert parsed.what_changed == "Tesla added a risk factor about its Robotaxi service."
    assert parsed.why_it_matters is not None and parsed.why_it_matters.startswith("Revenue")
    assert parsed.importance == "medium"  # "Medium" normalized
    assert parsed.importance_reason == "a new business line with adoption risk"
    # Straight and curly quotes both work.
    assert parsed.evidence == (
        "Upon launching our Robotaxi service in June 2025",
        "amid growing competition",
    )
    assert parsed.issues == ()


def test_missing_section_is_a_format_issue() -> None:
    answer = "\n".join(line for line in GOOD_ANSWER.splitlines() if "WHY IT MATTERS" not in line)

    parsed = parse_explanation(answer)

    assert parsed.why_it_matters is None
    assert parsed.issues == (Issue(kind="format", message="missing WHY IT MATTERS line"),)


def test_evidence_is_optional_for_the_single_call_format() -> None:
    answer = "\n".join(line for line in GOOD_ANSWER.splitlines() if "EVIDENCE" not in line)

    assert parse_explanation(answer, expect_evidence=False).issues == ()
    assert parse_explanation(answer).issues == (
        Issue(kind="format", message="missing EVIDENCE line"),
    )


def test_unknown_importance_level_is_an_issue() -> None:
    parsed = parse_explanation(GOOD_ANSWER.replace("Medium -", "Urgent -"))

    assert parsed.importance is None
    assert [issue.kind for issue in parsed.issues] == ["format"]
    assert "unknown IMPORTANCE level" in parsed.issues[0].message


def test_importance_with_an_em_dash_and_capitals() -> None:
    parsed = parse_explanation(GOOD_ANSWER.replace("Medium -", "HIGH \u2014"))

    assert (parsed.importance, parsed.issues) == ("high", ())


def test_evidence_line_without_quotes_is_an_issue() -> None:
    parsed = parse_explanation(GOOD_ANSWER.split("EVIDENCE:")[0] + "EVIDENCE: none given")

    assert parsed.evidence == ()
    assert parsed.issues == (Issue(kind="format", message="EVIDENCE line has no quoted text"),)


def test_a_label_can_continue_on_the_next_line() -> None:
    parsed = parse_explanation(GOOD_ANSWER.replace("depends partly", "depends\n  partly"))

    assert parsed.why_it_matters is not None and "depends partly on" in parsed.why_it_matters


# --- Quotes -------------------------------------------------------------------------------------


def quoted(*quotes: str) -> str:
    evidence = " ; ".join(f'"{quote}"' for quote in quotes)
    return GOOD_ANSWER.split("EVIDENCE:")[0] + f"EVIDENCE: {evidence}"


def test_exact_quotes_pass() -> None:
    assert verify(GOOD_ANSWER, [SOURCE, METRICS]) == []


def test_made_up_quote_fails() -> None:
    issues = verify(quoted("Robotaxi revenue will double next year"), [SOURCE, METRICS])

    assert issues == [
        Issue(
            kind="quote",
            message='"Robotaxi revenue will double next year" not found in the sources',
        )
    ]


def test_whitespace_and_typography_differences_still_pass() -> None:
    source = SOURCE.replace("amid growing", "amid\n  growing").replace(
        "ride-hailing", "ride\u2013hailing"
    )

    assert (
        verify(quoted("amid growing competition", "autonomous ride-hailing"), [source, METRICS])
        == []
    )


def test_quotes_are_case_sensitive() -> None:
    issues = verify(quoted("upon launching our robotaxi service"), [SOURCE, METRICS])

    assert [issue.kind for issue in issues] == ["quote"]


def test_ellipsis_pieces_must_all_be_in_one_source() -> None:
    passes = quoted("Upon launching our Robotaxi service ... amid growing competition")
    split_across_sources = quoted("Upon launching our Robotaxi \u2026 prior $28,095M")

    assert verify(passes, [SOURCE, METRICS]) == []
    assert [i.kind for i in verify(split_across_sources, [SOURCE, METRICS])] == ["quote"]


# --- Numbers ------------------------------------------------------------------------------------


def with_claim(why: str) -> str:
    return GOOD_ANSWER.replace(
        "Revenue of $28,046M now depends partly on a new, unproven business.", why
    )


def test_number_in_a_source_passes() -> None:
    check = verify_explanation(parse_explanation(GOOD_ANSWER), [SOURCE, METRICS])

    assert check.passed
    assert check.numbers_checked == 1  # $28,046M


def test_invented_number_fails() -> None:
    issues = verify(with_claim("Robotaxi could add 31% to revenue."), [SOURCE, METRICS])

    assert issues == [Issue(kind="number", message="31% not found in the sources")]


def test_formatted_number_matches_plain_digits_in_a_source() -> None:
    assert verify(with_claim("Revenue was $28,046M."), [SOURCE, "Total revenues | 28,046"]) == []


def test_identifiers_are_not_treated_as_numbers() -> None:
    check = verify_explanation(
        parse_explanation(with_claim("The 10-K adds this to Item 1A for Q3, in 2025.")),
        [SOURCE],
    )

    assert check.passed
    assert check.numbers_checked == 1  # only 2025 (from "June 2025" in the source)


def test_differently_written_number_is_flagged() -> None:
    # Known limit: a correct rounding in other words is still flagged (safe, not silent).
    issues = verify(with_claim("Revenue was $28.0 billion."), [SOURCE, METRICS])

    assert issues == [Issue(kind="number", message="$28.0 not found in the sources")]


# --- Sources ------------------------------------------------------------------------------------


def test_sources_are_the_paragraphs_plus_tool_outputs() -> None:
    modified = ParagraphChange(
        kind="modified",
        old_text="old paragraph",
        new_text="new paragraph",
        old_start=1,
        old_end=14,
        new_start=1,
        new_end=14,
        similarity=0.7,
    )

    assert sources_for(modified, ["tool output"]) == [
        "old paragraph",
        "new paragraph",
        "tool output",
    ]
    assert sources_for(modified) == ["old paragraph", "new paragraph"]


# --- A real answer --------------------------------------------------------------------------------

# Verbatim output of `explain TSLA --form 10-K --item 1A --kind added --n 4` (single call,
# gpt-6.1-terra, 2026-11-01). That format has no EVIDENCE line.
REAL_ANSWER = (
    "WHAT CHANGED: Tesla added that business growth depends on its ability to develop and "
    "commercialize Bots, including Optimus.\n"
    "WHY IT MATTERS: The filing describes Bots as a nascent industry that has yet to develop "
    "commercially, introducing uncertainty around this growth dependency.\n"
    "IMPORTANCE: medium - New commercialization risk to growth"
)
# What the model saw: the added paragraph, as stored in the cached 10-K text.
REAL_PARAGRAPH = (
    "Growth of our business is also dependent upon our ability to develop and commercialize "
    "Bots, including Optimus, which is in a nascent industry that has yet to develop "
    "commercially."
)


def test_real_single_call_answer_passes() -> None:
    parsed = parse_explanation(REAL_ANSWER, expect_evidence=False)

    check = verify_explanation(parsed, [REAL_PARAGRAPH])

    assert (parsed.importance, parsed.importance_reason) == (
        "medium",
        "New commercialization risk to growth",
    )
    assert check.passed


def test_real_answer_with_an_invented_number_fails() -> None:
    tampered = REAL_ANSWER.replace("introducing uncertainty", "putting 40% of growth at risk")

    check = verify_explanation(parse_explanation(tampered, expect_evidence=False), [REAL_PARAGRAPH])

    assert check.issues == (Issue(kind="number", message="40% not found in the sources"),)


# Verbatim output of the same command with --agent (2 steps, 2026-11-01).
REAL_AGENT_ANSWER = (
    "WHAT CHANGED: Tesla added a risk disclosure linking business growth to developing and "
    "commercializing Bots, including Optimus, in a nascent industry.\n"
    "WHY IT MATTERS: Tesla says it has not yet commercialized Bots and cannot predict how "
    "demand will develop. The surrounding disclosure says delays or failure in the program "
    "could harm its business and financial condition.\n"
    "IMPORTANCE: medium - uncertain commercialization and future demand\n"
    'EVIDENCE: "Growth of our business is also dependent upon our ability to develop and '
    'commercialize Bots"; "We have yet to commercialize Bots and cannot predict how demand '
    'for Bots will develop"'
)
# The get_nearby_paragraphs(count=1) result the agent received, rebuilt from the cached 10-K
# (1,738 chars, as in the run's trace).
NEARBY = (Path(__file__).parent / "fixtures" / "tsla_10k_1a_added4_nearby.txt").read_text(
    encoding="utf-8"
)


def test_real_agent_answer_passes_against_what_it_saw() -> None:
    check = verify_explanation(parse_explanation(REAL_AGENT_ANSWER), [REAL_PARAGRAPH, NEARBY])

    assert check.passed
    assert (check.quotes_checked, check.numbers_checked) == (2, 0)


def test_real_agent_answer_needs_the_tool_output_as_a_source() -> None:
    # The second quote comes from a neighbouring paragraph the agent fetched with a tool.
    check = verify_explanation(parse_explanation(REAL_AGENT_ANSWER), [REAL_PARAGRAPH])

    assert [issue.kind for issue in check.issues] == ["quote"]
    assert "We have yet to commercialize Bots" in check.issues[0].message
