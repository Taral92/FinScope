"""Turn SEC filing HTML into plain, readable text."""

import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from bs4.element import NavigableString

# Elements that start a new line when rendered. (Tables are handled separately.)
_BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "title"]
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
# \s also matches non-breaking spaces (\xa0, from &nbsp;), which SEC filings use heavily.
_WHITESPACE = re.compile(r"\s+")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
# Typographic variants that mean the same thing: curly quotes, en/em dashes, minus sign.
_TYPOGRAPHY = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


def html_to_text(html: str) -> str:
    """Strip markup and hidden content, keeping paragraphs and table rows on their own lines."""
    with warnings.catch_warnings():
        # Modern filings are inline XBRL, i.e. XHTML, so bs4 suggests an XML parser. The
        # forgiving HTML parser handles them fine and also handles older, non-XML filings.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")

    # 1. Remove content a reader never sees: code, styling, and hidden inline-XBRL data.
    for element in soup.find_all(["script", "style", "ix:header"]):
        element.decompose()
    for element in soup.find_all(style=_HIDDEN_STYLE):
        if not element.decomposed:  # may already be gone with a hidden parent
            element.decompose()

    # 2. Collapse whitespace inside text. A newline in HTML source is just a space, so
    #    this must happen before step 3 adds the newlines we actually want.
    for string in soup.find_all(string=True):
        if type(string) is NavigableString:  # skip comments, doctype, etc.
            string.replace_with(_WHITESPACE.sub(" ", string))

    # 3. Rebuild the layout: one line per table row, blank line around each block.
    for table in soup.find_all("table"):
        rows = []
        for row in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            cells = [cell for cell in cells if cell]  # filings pad tables with empty cells
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n\n" + "\n".join(rows) + "\n\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(_BLOCK_TAGS):
        block.insert_before("\n\n")
        block.insert_after("\n\n")

    # 4. Tidy lines: trim each one and allow at most one blank line in a row.
    lines = (_WHITESPACE.sub(" ", line).strip() for line in soup.get_text().split("\n"))
    return _EXTRA_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def unify_typography(text: str) -> str:
    """Map curly quotes, en/em dashes and the minus sign to plain ASCII, for comparisons."""
    return text.translate(_TYPOGRAPHY)
