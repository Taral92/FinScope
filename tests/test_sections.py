import pytest

from finscope.sections import split_sections

# A miniature 10-Q in the format html_to_text produces: a table of contents (rows end in
# page numbers), then the real headings. Both parts contain an Item 1 and an Item 2.
FAKE_10Q = """\
TESLA, INC.
FORM 10-Q
INDEX
PART I. FINANCIAL INFORMATION
Item 1. | Financial Statements | 4
Item 2. | Management's Discussion and Analysis | 20
PART II. OTHER INFORMATION
Item 1. | Legal Proceedings | 30
Item 1A. | Risk Factors | 31
Item 2. | Unregistered Sales of Equity Securities | 32

PART I. FINANCIAL INFORMATION

ITEM 1. FINANCIAL STATEMENTS

Total revenues | 28,095 | 25,182

ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue grew. As described in Part I, Item 1A, Risk Factors, demand may still fall.
Item 1A of our annual report lists further risks.

PART II. OTHER INFORMATION

ITEM 1. LEGAL PROCEEDINGS

None.

ITEM 1A. RISK FACTORS

There have been no material changes to our risk factors.

ITEM 2. UNREGISTERED SALES OF EQUITY SECURITIES

None.
"""

FAKE_8K = """\
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
FORM 8-K
Item 2.02 Results of Operations and Financial Condition.
On October 22, 2026, Tesla released its financial results for the third quarter.
Item 9.01 Financial Statements and Exhibits.
(d) Exhibits
99.1 | Press release dated October 22, 2026
"""


def test_10q_items_are_found_with_their_parts() -> None:
    sections = split_sections(FAKE_10Q)

    assert [(s.part, s.item, s.title) for s in sections] == [
        ("I", "1", "FINANCIAL STATEMENTS"),
        ("I", "2", "MANAGEMENT'S DISCUSSION AND ANALYSIS"),
        ("II", "1", "LEGAL PROCEEDINGS"),
        ("II", "1A", "RISK FACTORS"),
        ("II", "2", "UNREGISTERED SALES OF EQUITY SECURITIES"),
    ]


def test_table_of_contents_is_skipped() -> None:
    sections = split_sections(FAKE_10Q)

    assert sections[0].text.startswith("ITEM 1. FINANCIAL STATEMENTS\n")
    assert all("|" not in s.title for s in sections)


def test_mid_sentence_mentions_are_not_sections() -> None:
    mda = split_sections(FAKE_10Q)[1]

    # Both cross-references stay inside Item 2 instead of starting new sections.
    assert "As described in Part I, Item 1A, Risk Factors" in mda.text
    assert "Item 1A of our annual report" in mda.text
    assert mda.text.endswith("lists further risks.")


def test_8k_decimal_items_have_no_part() -> None:
    sections = split_sections(FAKE_8K)

    assert [(s.part, s.item, s.title) for s in sections] == [
        (None, "2.02", "Results of Operations and Financial Condition"),
        (None, "9.01", "Financial Statements and Exhibits"),
    ]
    assert "released its financial results" in sections[0].text


def test_duplicate_heading_keeps_the_longest_section() -> None:
    # A table of contents *without* page numbers isn't caught by the TOC rule.
    text = (
        "Item 1. | Business\nItem 2. | Properties\n\n"
        "ITEM 1. BUSINESS\nWe make cars.\n\nITEM 2. PROPERTIES\nWe own factories.\n"
    )

    sections = split_sections(text)

    assert [s.text for s in sections] == [
        "ITEM 1. BUSINESS\nWe make cars.",
        "ITEM 2. PROPERTIES\nWe own factories.",
    ]


@pytest.mark.parametrize("text", [FAKE_10Q, FAKE_8K], ids=["10-Q", "8-K"])
def test_offsets_point_at_the_section_text(text: str) -> None:
    sections = split_sections(text)

    assert sections
    for section in sections:
        assert text[section.start : section.end] == section.text
    starts = [s.start for s in sections]
    assert starts == sorted(starts)
