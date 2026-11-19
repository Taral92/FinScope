"""An MCP server that lets any MCP client (Claude Code, Claude Desktop, ...) read FinScope's data.

MCP (Model Context Protocol) is the standard way for an AI client to discover tools in another
program and call them with JSON arguments. This server exposes the plain-code data layer:
filings, sections, diffs, change context and XBRL numbers. No model runs here, nothing is
written, and every tool answers in text. Start it with `python -m finscope mcp`.

The server speaks over stdin/stdout, so nothing in this module may print to stdout.
"""

from typing import Annotated

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from finscope.agent_tools import ExplainContext, format_metrics, get_nearby_paragraphs
from finscope.config import get_settings
from finscope.diff import (
    SectionDiff,
    change_text,
    changes_by_id,
    compact_word_diff,
    diff_sections,
    numbered_changes,
    word_diff,
)
from finscope.edgar import Filing, SecClient, TickerNotFoundError
from finscope.filings import load_filing_text, load_metrics
from finscope.sections import Section, SectionLookupError, find_section, split_sections

SERVER_NAME = "finscope"
MAX_OUTPUT_CHARS = 40_000  # one answer must not flood the client's context
SNIPPET_CHARS = 300  # per change in diff_section; get_change_context has the full text

Ticker = Annotated[str, Field(description="Stock ticker, e.g. TSLA or AAPL.", min_length=1)]
Form = Annotated[str, Field(description="SEC form type: 10-K (annual) or 10-Q (quarterly).")]
Item = Annotated[str, Field(description="Item number inside the filing, e.g. 1A (Risk Factors).")]
Part = Annotated[
    str | None,
    Field(description="Part the Item is in (I, II, ...). Only needed when the Item repeats."),
]
Count = Annotated[int, Field(ge=1, le=3, description="Paragraphs on each side, 1 to 3.")]
ChangeId = Annotated[str, Field(description="A change id from diff_section, e.g. +4 or ~19.")]


def build_server(sec: SecClient) -> MCPServer:
    """The server with its five read-only tools, all answering from `sec` and its disk cache."""
    server = MCPServer(
        SERVER_NAME,
        instructions=(
            "Read-only access to SEC filings as FinScope sees them. Start with diff_section to "
            "see what changed between a company's two latest filings of a form, then "
            "get_change_context for one change's full text and surroundings. Financial numbers "
            "come from get_key_metrics (SEC XBRL data); never infer them from the text."
        ),
    )

    @server.tool()
    def list_filings(
        ticker: Ticker,
        form: Form = "10-K",
        limit: Annotated[int, Field(ge=1, le=20)] = 5,
    ) -> str:
        """Recent filings of one form for a company, newest first, with dates and accessions."""
        filings = _filings(sec, ticker, form, limit)
        lines = [f"{ticker.upper()} {form}: {len(filings)} filing(s), newest first"]
        lines += [
            f"{filing.filing_date.isoformat()}  accession {filing.accession_number}  "
            f"period {filing.report_date.isoformat() if filing.report_date else 'n/a'}"
            for filing in filings
        ]
        return "\n".join(lines)

    @server.tool()
    def get_section(
        ticker: Ticker, form: Form = "10-K", item: Item = "1A", part: Part = None
    ) -> str:
        """The text of one Item in the company's latest filing of that form (long; capped)."""
        [filing] = _filings(sec, ticker, form, 1)
        section = _section(sec, filing, item, part)
        header = f"{_filed(ticker, form, filing)}: {_title(section)}"
        return _cap(f"{header}\n\n{section.text}")

    @server.tool()
    def diff_section(
        ticker: Ticker, form: Form = "10-K", item: Item = "1A", part: Part = None
    ) -> str:
        """Every paragraph that changed in one Item between the two latest filings of a form.

        Each change has an id: +n added, -n removed, ~n modified (least similar first). Modified
        paragraphs are shown as compact word diffs with [-old-] and {+new+}. Texts are cut to a
        snippet; pass an id to get_change_context for the whole paragraph.
        """
        old, new, diff = _diff(sec, ticker, form, item, part)
        numbered = numbered_changes(diff)
        lines = [
            f"{ticker.upper()} {form} {_diff_title(diff)}",
            f"new: filed {new.filing_date.isoformat()} ({new.accession_number})",
            f"old: filed {old.filing_date.isoformat()} ({old.accession_number})",
            f"{diff.unchanged} unchanged, {len(numbered['added'])} added, "
            f"{len(numbered['removed'])} removed, {len(numbered['modified'])} modified",
        ]
        for kind, prefix in (("added", "+"), ("removed", "-"), ("modified", "~")):
            for n, change in enumerate(numbered[kind], start=1):
                if kind == "modified":
                    assert change.old_text is not None and change.new_text is not None
                    body = compact_word_diff(change.old_text, change.new_text)
                else:
                    body = change_text(change)
                lines.append(f"\n[{prefix}{n}] {kind}\n{_snippet(body)}")
        return _cap("\n".join(lines))

    @server.tool()
    def get_change_context(
        ticker: Ticker,
        change_id: ChangeId,
        form: Form = "10-K",
        item: Item = "1A",
        part: Part = None,
        count: Count = 1,
    ) -> str:
        """One change in full (old and new text, word diff) plus the paragraphs around it."""
        _, _, diff = _diff(sec, ticker, form, item, part)
        change = changes_by_id(diff).get(change_id)
        if change is None:
            raise ToolError(
                f"no change {change_id!r} in {_diff_title(diff)}; ids come from diff_section"
            )
        parts = [f"[{change_id}] {change.kind}"]
        if change.kind == "modified":
            assert change.old_text is not None and change.new_text is not None
            parts += [
                f"OLD:\n{change.old_text}",
                f"NEW:\n{change.new_text}",
                f"WORD DIFF:\n{word_diff(change.old_text, change.new_text)}",
            ]
        else:
            parts.append(f"TEXT:\n{change_text(change)}")
        old_section, new_section = _sections(sec, *_pair(sec, ticker, form), item, part)
        context = ExplainContext(
            change=change, old_section=old_section, new_section=new_section, metrics=()
        )
        parts.append(get_nearby_paragraphs(context, count))
        return _cap("\n\n".join(parts))

    @server.tool()
    def get_key_metrics(ticker: Ticker, form: Form = "10-K") -> str:
        """Key financial numbers of the latest filing from SEC XBRL data, with prior-year values.

        Metric | period | value | prior | change. These are the only reliable numbers.
        """
        [filing] = _filings(sec, ticker, form, 1)
        header = f"{_filed(ticker, form, filing)} ({filing.accession_number})"
        return f"{header}\n{format_metrics(tuple(load_metrics(sec, filing)))}"

    return server


def run_server() -> None:
    """Serve over stdin/stdout until the client disconnects."""
    with SecClient(get_settings()) as sec:
        build_server(sec).run("stdio")


# --- Lookups: every failure a client can cause becomes a ToolError the model can read --------


def _filings(sec: SecClient, ticker: str, form: str, limit: int) -> list[Filing]:
    try:
        cik = sec.get_cik(ticker)
        filings = sec.get_recent_filings(cik, forms={form}, limit=limit)
    except TickerNotFoundError as error:
        raise ToolError(str(error)) from None
    except httpx.HTTPError as error:
        raise ToolError(f"SEC request failed: {error}") from None
    if not filings:
        raise ToolError(f"no recent {form} filing for {ticker.upper()}")
    return filings


def _pair(sec: SecClient, ticker: str, form: str) -> tuple[Filing, Filing]:
    """(old, new): the two latest filings of the form."""
    filings = _filings(sec, ticker, form, 2)
    if len(filings) < 2:
        raise ToolError(f"need two {form} filings for {ticker.upper()} to compare; found one")
    return filings[1], filings[0]


def _section(sec: SecClient, filing: Filing, item: str, part: str | None) -> Section:
    try:
        return find_section(split_sections(_text(sec, filing)), item, part)
    except SectionLookupError as error:
        raise ToolError(str(error)) from None


def _sections(
    sec: SecClient, old: Filing, new: Filing, item: str, part: str | None
) -> tuple[Section, Section]:
    return _section(sec, old, item, part), _section(sec, new, item, part)


def _diff(
    sec: SecClient, ticker: str, form: str, item: str, part: str | None
) -> tuple[Filing, Filing, SectionDiff]:
    old, new = _pair(sec, ticker, form)
    return old, new, diff_sections(*_sections(sec, old, new, item, part))


def _text(sec: SecClient, filing: Filing) -> str:
    try:
        return load_filing_text(sec, filing)
    except httpx.HTTPError as error:
        raise ToolError(f"SEC request failed: {error}") from None


# --- Formatting --------------------------------------------------------------------------------


def _filed(ticker: str, form: str, filing: Filing) -> str:
    return f"{ticker.upper()} {form} filed {filing.filing_date.isoformat()}"


def _title(section: Section) -> str:
    label = f"Part {section.part} Item {section.item}" if section.part else f"Item {section.item}"
    return f"{label}. {section.title}"


def _diff_title(diff: SectionDiff) -> str:
    label = f"Part {diff.part} Item {diff.item}" if diff.part else f"Item {diff.item}"
    return f"{label}. {diff.title}"


def _snippet(text: str) -> str:
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + " …"


def _cap(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    cut = len(text) - MAX_OUTPUT_CHARS
    return text[:MAX_OUTPUT_CHARS] + f"\n\n[output cut here: {cut:,} more characters]"
