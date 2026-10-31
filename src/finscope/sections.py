"""Split a filing's plain text (from `text.html_to_text`) into its standard Items."""

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

# "PART I", "PART II. OTHER INFORMATION", "Part IV - Exhibits". A title needs a separator
# (period, colon, hyphen, en or em dash), so "Part I, Item 1A, ..." doesn't match.
_PART = re.compile(r"PART\s+(IV|I{1,3})\b\s*(?:[.:\-\u2013\u2014]\s*(.*))?", re.IGNORECASE)
# "ITEM 1A. RISK FACTORS", "Item 7. MD&A", "Item 2.02 Results of Operations" (8-K style).
_ITEM = re.compile(
    r"ITEM\s+(\d{1,2}(?:\.\d{2})?[A-Z]?)\b\s*[.:\-\u2013\u2014]?\s*(.*)", re.IGNORECASE
)
# A table-of-contents row ends in a page-number cell: "Item 1A. | Risk Factors | 41".
_TOC_ROW = re.compile(r"\|\s*\d{1,3}\s*$")
# Headings are short; a longer line starting with "Item" is a sentence.
_MAX_HEADING_CHARS = 200


class Section(BaseModel):
    """One Item of a filing, with its position in the full text."""

    model_config = ConfigDict(frozen=True)

    part: str | None  # "I", "II", ...; None for forms without parts (e.g. 8-K)
    item: str  # "1A", "7", "2.02"
    title: str
    start: int  # character offsets into the full text: text[start:end] == self.text
    end: int
    text: str


@dataclass(frozen=True)
class _Heading:
    start: int
    part: str | None
    item: str | None  # None for a PART heading
    title: str


def split_sections(text: str) -> list[Section]:
    """Return the filing's Items in document order."""
    headings = _find_headings(text)

    # A (part, item) can still appear more than once, e.g. a table of contents without page
    # numbers, or a cross-reference that happens to sit on its own line. Those produce tiny
    # sections, while the real heading is followed by the actual content, so the longest wins.
    longest: dict[tuple[str | None, str], Section] = {}
    for section in _cut(text, headings):
        key = (section.part, section.item)
        if key not in longest or len(section.text) > len(longest[key].text):
            longest[key] = section

    # Cut again with only the winners, so a discarded heading doesn't leave a gap in the
    # section that contains it.
    kept = {section.start for section in longest.values()}
    return _cut(text, [h for h in headings if h.item is None or h.start in kept])


def _find_headings(text: str) -> list[_Heading]:
    headings = []
    part = None
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        start, offset = offset, offset + len(raw_line)
        line = raw_line.strip()
        if not line or len(line) > _MAX_HEADING_CHARS or _TOC_ROW.search(line):
            continue
        if match := _PART.fullmatch(line):
            part = match[1].upper()
            headings.append(_Heading(start, part, None, _clean_title(match[2] or "")))
        elif (match := _ITEM.fullmatch(line)) and not match[2][:1].islower():
            # A lowercase title ("Item 1A of our annual report ...") is prose, not a heading.
            headings.append(_Heading(start, part, match[1].upper(), _clean_title(match[2])))
    return headings


def _cut(text: str, headings: list[_Heading]) -> list[Section]:
    """Turn headings into sections; each runs until the next heading of any kind."""
    ends = [heading.start for heading in headings[1:]] + [len(text)]
    sections = []
    for heading, next_start in zip(headings, ends, strict=True):
        if heading.item is None:
            continue
        end = heading.start + len(text[heading.start : next_start].rstrip())
        sections.append(
            Section(
                part=heading.part,
                item=heading.item,
                title=heading.title,
                start=heading.start,
                end=end,
                text=text[heading.start : end],
            )
        )
    return sections


def _clean_title(title: str) -> str:
    """Drop "|" separators, a trailing page number and a trailing period from a title."""
    cells = [cell.strip() for cell in title.split("|") if cell.strip()]
    if len(cells) > 1 and cells[-1].isdigit():
        cells.pop()  # trailing page number
    return " ".join(cells).rstrip(" .")
