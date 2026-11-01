from finscope.diff import diff_sections, word_diff
from finscope.sections import Section, split_sections

UNCHANGED = "We design, develop, manufacture and sell electric vehicles."
MOVED = "Our products depend on lithium-ion battery cells from a small number of suppliers."
EDIT_OLD = "Demand for our vehicles may decline in 2025."
EDIT_NEW = "Demand for our vehicles may decline sharply in 2026."
REMOVED = "We are dependent on the continued services of our Chief Executive Officer."
ADDED = "New tariffs on imported components could raise our production costs."
REPEATED = "None."  # short answers like this repeat across Items in real filings
PAGE_NUMBER_OLD, PAGE_NUMBER_NEW = "27", "28"  # page breaks shift between years

# The paragraph order differs (MOVED jumps to the top), REPEATED drops from 2 copies to 1.
OLD_PARAGRAPHS = [UNCHANGED, MOVED, EDIT_OLD, REPEATED, PAGE_NUMBER_OLD, REMOVED, REPEATED]
NEW_PARAGRAPHS = [MOVED, UNCHANGED, EDIT_NEW, REPEATED, PAGE_NUMBER_NEW, ADDED]


def filing(paragraphs: list[str], cover: str) -> tuple[str, Section]:
    """A full filing text with a cover page before the section, so offsets aren't zero-based."""
    text = f"{cover}\n\nITEM 1A. RISK FACTORS\n\n" + "\n\n".join(paragraphs) + "\n"
    return text, split_sections(text)[0]


OLD_TEXT, OLD = filing(OLD_PARAGRAPHS, "FORM 10-K for fiscal 2024")
NEW_TEXT, NEW = filing(NEW_PARAGRAPHS, "FORM 10-K for fiscal year 2025")


def test_unchanged_and_moved_paragraphs_are_not_reported() -> None:
    diff = diff_sections(OLD, NEW)

    reported = {change.old_text for change in diff.changes} | {c.new_text for c in diff.changes}
    assert UNCHANGED not in reported
    assert MOVED not in reported


def test_small_edit_is_modified_with_high_similarity() -> None:
    [modified] = diff_sections(OLD, NEW).of_kind("modified")

    assert (modified.old_text, modified.new_text) == (EDIT_OLD, EDIT_NEW)
    assert modified.similarity is not None
    assert modified.similarity > 0.8


def test_word_diff_marks_the_edit_inline() -> None:
    assert word_diff(EDIT_OLD, EDIT_NEW) == (
        "Demand for our vehicles may decline {+sharply+} in [-2025.-]{+2026.+}"
    )


def test_new_paragraph_is_added_and_deleted_one_is_removed() -> None:
    diff = diff_sections(OLD, NEW)

    assert [c.new_text for c in diff.of_kind("added")] == [ADDED]
    removed = [c.old_text for c in diff.of_kind("removed")]
    assert REMOVED in removed


def test_duplicate_paragraphs_are_counted() -> None:
    diff = diff_sections(OLD, NEW)

    # UNCHANGED, MOVED and one REPEATED match; the second REPEATED copy is gone.
    assert diff.unchanged == 3
    assert sorted(c.old_text or "" for c in diff.of_kind("removed")) == sorted([REMOVED, REPEATED])


def test_page_numbers_are_ignored() -> None:
    diff = diff_sections(OLD, NEW)

    texts = {c.old_text for c in diff.changes} | {c.new_text for c in diff.changes}
    assert PAGE_NUMBER_OLD not in texts
    assert PAGE_NUMBER_NEW not in texts


def test_typography_differences_are_not_changes() -> None:
    _, old = filing(["Tesla\u2019s \u201cMaster Plan\u201d \u2014 part one."], "COVER")
    _, new = filing(['Tesla\'s "Master Plan" - part one.'], "COVER")

    diff = diff_sections(old, new)

    assert (diff.unchanged, diff.changes) == (1, ())


def test_offsets_point_at_the_paragraph_in_the_full_text() -> None:
    diff = diff_sections(OLD, NEW)

    # None of these paragraphs is split by a page break, so start..end is exactly the text.
    # (For rejoined paragraphs see test_rejoined_paragraph_spans_first_to_last_piece.)
    assert diff.changes
    for change in diff.changes:
        if change.old_text is not None:
            assert change.old_start is not None and change.old_end is not None
            assert OLD_TEXT[change.old_start : change.old_end] == change.old_text
        if change.new_text is not None:
            assert change.new_start is not None and change.new_end is not None
            assert NEW_TEXT[change.new_start : change.new_end] == change.new_text


# A sentence split across page breaks, as html_to_text renders a printed filing.
PIECE_1 = "Our factories may be damaged by earthquakes, floods or other"
PIECE_2 = "climate-related events, and our suppliers may be unable to"
PIECE_3 = "deliver components, which could halt production."
WHOLE = f"{PIECE_1} {PIECE_2} {PIECE_3}"
_, EMPTY = filing([], "COVER")  # a section with no paragraphs: everything is added


def test_paragraph_split_by_a_page_break_is_unchanged() -> None:
    _, old = filing([f"{PIECE_1} {PIECE_2}", "23", "Table of Contents", PIECE_3], "COVER")
    _, new = filing([WHOLE], "COVER")

    diff = diff_sections(old, new)

    assert (diff.unchanged, diff.changes) == (1, ())


def test_three_pieces_merge_into_one_paragraph() -> None:
    _, new = filing([PIECE_1, "23", "Table of Contents", PIECE_2, "24", PIECE_3], "COVER")

    [added] = diff_sections(EMPTY, new).changes

    assert added.new_text == WHOLE


def test_rejoined_paragraph_spans_first_to_last_piece() -> None:
    text, new = filing([PIECE_1, "23", "Table of Contents", PIECE_2, "24", PIECE_3], "COVER")

    [added] = diff_sections(EMPTY, new).changes

    assert added.new_start == text.index(PIECE_1)
    assert added.new_end == text.index(PIECE_3) + len(PIECE_3)


def test_complete_paragraphs_are_not_merged() -> None:
    paragraphs = [
        "Risks Related to Our Business",  # heading: no period, but the next starts uppercase
        "We may be unable to grow.",
        "Competition may increase.",
    ]
    _, new = filing(paragraphs, "COVER")

    added = [change.new_text for change in diff_sections(EMPTY, new).changes]

    assert added == paragraphs


def test_list_items_ending_with_semicolons_are_not_merged() -> None:
    paragraphs = [
        "We depend on:",
        "suppliers of battery cells;",
        "suppliers of semiconductors;",
        "logistics providers.",
    ]
    _, new = filing(paragraphs, "COVER")

    added = [change.new_text for change in diff_sections(EMPTY, new).changes]

    assert added == paragraphs
