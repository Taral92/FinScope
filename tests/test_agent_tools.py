from datetime import date
from decimal import Decimal

from finscope.agent_tools import (
    ExplainContext,
    NearbyParagraphsArgs,
    build_tools,
    get_key_metrics,
    get_nearby_paragraphs,
)
from finscope.diff import diff_sections
from finscope.sections import split_sections
from finscope.xbrl import Fact, MetricValue

P1 = "Risks Related to Our Business"
P2 = "Our growth depends on new products."
P3 = "We launched a Robotaxi service in June 2025."  # the added paragraph
P4 = "Autonomous driving faces regulatory approval."
P5 = "Competition in robotics is increasing."


def filing(paragraphs: list[str]) -> str:
    return "FORM 10-K\n\nITEM 1A. RISK FACTORS\n\n" + "\n\n".join(paragraphs) + "\n"


OLD_TEXT = filing([P1, P2, P4, P5])
NEW_TEXT = filing([P1, P2, P3, P4, P5])
OLD, NEW = split_sections(OLD_TEXT)[0], split_sections(NEW_TEXT)[0]
[ADDED] = diff_sections(OLD, NEW).changes
[REMOVED] = diff_sections(NEW, OLD).changes  # the same paragraph, seen as removed


def fact(start: date, end: date, value: str) -> Fact:
    return Fact(
        concept="Revenues",
        unit="USD",
        value=Decimal(value),
        start=start,
        end=end,
        fiscal_year=2026,
        fiscal_period="Q3",
        form="10-Q",
        accession_number="0001628280-26-066775",
        filed=date(2026, 10, 22),
    )


REVENUE = MetricValue(
    name="Revenue",
    current=fact(date(2026, 7, 1), date(2026, 9, 30), "28046000000"),
    prior=fact(date(2025, 7, 1), date(2025, 9, 30), "28095000000"),
    change_pct=(Decimal(28_046) - 28_095) / 28_095 * 100,
)
CONTEXT = ExplainContext(change=ADDED, old_section=OLD, new_section=NEW, metrics=(REVENUE,))


def test_nearby_paragraphs_are_the_neighbours_with_offsets() -> None:
    output = get_nearby_paragraphs(CONTEXT, 1)

    assert f'offset="{NEW_TEXT.index(P2):,}"' in output
    assert f'offset="{NEW_TEXT.index(P4):,}"' in output
    assert P2 in output and P4 in output
    assert P1 not in output and P5 not in output
    assert P3 not in output  # the changed paragraph itself is not repeated
    assert output.index(P2) < output.index("After the changed paragraph") < output.index(P4)


def test_count_widens_the_window_and_stops_at_section_edges() -> None:
    output = get_nearby_paragraphs(CONTEXT, 3)

    assert all(p in output for p in (P1, P2, P4, P5))  # only 2 exist on each side


def test_removed_change_looks_in_the_old_filing() -> None:
    removed_context = ExplainContext(change=REMOVED, old_section=NEW, new_section=OLD, metrics=())

    output = get_nearby_paragraphs(removed_context, 1)

    assert 'source="previous filing"' in output
    assert P2 in output and P4 in output


def test_paragraphs_are_wrapped_as_filing_text() -> None:
    output = get_nearby_paragraphs(CONTEXT, 1)

    assert output.count("<filing_text") == output.count("</filing_text>") == 2


def test_metrics_lines_show_xbrl_values() -> None:
    assert get_key_metrics(CONTEXT) == (
        "Revenue | 2026-07-01 to 2026-09-30 | $28,046M | prior $28,095M | -0.2%"
    )


def test_no_metrics_says_so() -> None:
    empty = ExplainContext(change=ADDED, old_section=OLD, new_section=NEW, metrics=())

    assert get_key_metrics(empty) == "No XBRL metrics are available for this filing."


def test_tools_are_wired_and_their_schemas_follow_strict_rules() -> None:
    tools = build_tools(CONTEXT)

    nearby = tools["get_nearby_paragraphs"].function(NearbyParagraphsArgs(count=1))
    assert nearby == get_nearby_paragraphs(CONTEXT, 1)
    for tool in tools.values():
        schema = tool.spec.parameters
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])
