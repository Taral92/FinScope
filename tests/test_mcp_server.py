"""The MCP server is exercised through the SDK's in-memory client: real protocol, no subprocess."""

import asyncio
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp.client.client import Client
from mcp.server.mcpserver import MCPServer

from finscope.config import Settings
from finscope.edgar import SecClient
from finscope.mcp_server import MAX_OUTPUT_CHARS, SNIPPET_CHARS, build_server

SETTINGS = Settings(_env_file=None, sec_user_agent="FinScope Test test@example.com")
CIK = "0001318605"

BASE = (
    "Our vehicles may be subject to recalls, which could harm our reputation.",
    "We depend on a limited number of suppliers for key battery cells and raw materials.",
    "Changes in trade policy and tariffs could increase our costs significantly.",
)
NEW_RISK = "Our Optimus humanoid robot program may never reach volume production or profitability."
MODIFIED_OLD = "We rely on third-party vendors for cloud hosting of our connectivity services."
MODIFIED_NEW = (
    "We rely on a single third-party vendor for cloud hosting of our connectivity services."
)


def filing_html(*paragraphs: str) -> str:
    body = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    return (
        f"<html><body><p>PART I</p><p>Item 1A. Risk Factors</p>{body}"
        "<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>"
        "<p>PART II</p><p>Item 7. Management's Discussion</p><p>Revenue grew.</p></body></html>"
    )


OLD = ("0001628280-25-000002", date(2025, 1, 30), filing_html(*BASE, MODIFIED_OLD))
NEW = ("0001628280-26-000003", date(2026, 1, 29), filing_html(*BASE, MODIFIED_NEW, NEW_RISK))


class FakeEdgar:
    """Stand-in for SEC's servers: one company, 10-K filings newest first, no XBRL facts."""

    def __init__(self, *filings: tuple[str, date, str]) -> None:
        self.filings = list(filings)
        self.facts: dict[str, Any] = {"facts": {}}
        self.missing_documents: set[str] = set()  # accessions whose document URL 404s

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "https://www.sec.gov/files/company_tickers.json":
            return httpx.Response(200, json={"0": {"cik_str": int(CIK), "ticker": "TSLA"}})
        if url == f"https://data.sec.gov/submissions/CIK{CIK}.json":
            return httpx.Response(200, json=self._submissions())
        if url == f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json":
            return httpx.Response(200, json=self.facts)
        for accession, _, html in self.filings:
            if url.endswith(f"/{accession}.htm") and accession not in self.missing_documents:
                return httpx.Response(200, text=html)
        return httpx.Response(404)

    def _submissions(self) -> dict[str, Any]:
        return {
            "filings": {
                "recent": {
                    "accessionNumber": [accession for accession, _, _ in self.filings],
                    "form": ["10-K" for _ in self.filings],
                    "filingDate": [day.isoformat() for _, day, _ in self.filings],
                    "reportDate": ["" for _ in self.filings],
                    "primaryDocument": [f"{accession}.htm" for accession, _, _ in self.filings],
                }
            }
        }


@pytest.fixture
def edgar() -> FakeEdgar:
    return FakeEdgar(NEW, OLD)


@pytest.fixture
def server(edgar: FakeEdgar, tmp_path: Path) -> Iterator[MCPServer]:
    settings = SETTINGS.model_copy(update={"data_dir": tmp_path})
    with SecClient(settings=settings, transport=httpx.MockTransport(edgar.handle)) as sec:
        yield build_server(sec)


def call(server: MCPServer, tool: str, **arguments: Any) -> tuple[bool, str]:
    """(is_error, text) from one tool call, the way a client sees it."""

    async def run() -> tuple[bool, str]:
        async with Client(server) as client:
            result = await client.call_tool(tool, arguments)
            [content] = result.content
            assert content.type == "text"
            return bool(result.is_error), content.text

    return asyncio.run(run())


def ok(server: MCPServer, tool: str, **arguments: Any) -> str:
    is_error, text = call(server, tool, **arguments)
    assert not is_error, text
    return text


def error(server: MCPServer, tool: str, **arguments: Any) -> str:
    is_error, text = call(server, tool, **arguments)
    assert is_error, text
    return text


# --- Discovery ----------------------------------------------------------------------------------


def test_client_discovers_five_read_only_tools_with_schemas(server: MCPServer) -> None:
    async def run() -> dict[str, Any]:
        async with Client(server) as client:
            return {tool.name: tool for tool in (await client.list_tools()).tools}

    tools = asyncio.run(run())

    assert set(tools) == {
        "list_filings",
        "get_section",
        "diff_section",
        "get_change_context",
        "get_key_metrics",
    }
    schema = tools["get_change_context"].input_schema
    assert set(schema["required"]) == {"ticker", "change_id"}
    assert schema["properties"]["count"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 3,
        "default": 1,
        "title": "Count",
        "description": "Paragraphs on each side, 1 to 3.",
    }
    for tool in tools.values():
        assert tool.description  # the client's model has nothing else to go on


# --- Tools --------------------------------------------------------------------------------------


def test_list_filings_newest_first(server: MCPServer) -> None:
    text = ok(server, "list_filings", ticker="tsla")

    assert text.splitlines() == [
        "TSLA 10-K: 2 filing(s), newest first",
        "2026-01-29  accession 0001628280-26-000003  period n/a",
        "2025-01-30  accession 0001628280-25-000002  period n/a",
    ]


def test_get_section_returns_the_latest_filings_item(server: MCPServer) -> None:
    text = ok(server, "get_section", ticker="TSLA", item="1a")

    assert text.startswith("TSLA 10-K filed 2026-01-29: Part I Item 1A. Risk Factors\n\n")
    assert NEW_RISK in text
    assert "Unresolved Staff Comments" not in text  # the next Item is not included


def test_diff_section_numbers_every_change(server: MCPServer) -> None:
    text = ok(server, "diff_section", ticker="TSLA")

    assert text.splitlines()[:4] == [
        "TSLA 10-K Part I Item 1A. Risk Factors",
        "new: filed 2026-01-29 (0001628280-26-000003)",
        "old: filed 2025-01-30 (0001628280-25-000002)",
        "3 unchanged, 1 added, 0 removed, 1 modified",
    ]
    assert f"[+1] added\n{NEW_RISK}" in text
    # Modified paragraphs come as one compact word diff, not as two full texts.
    assert "[~1] modified\nWe rely on {+a single+} third-party [-vendors-]{+vendor+} for" in text
    assert MODIFIED_OLD not in text
    assert MODIFIED_NEW not in text


def test_diff_section_cuts_long_paragraphs_to_a_snippet(
    server: MCPServer, edgar: FakeEdgar
) -> None:
    long_paragraph = "Risk " * 200
    edgar.filings[0] = (NEW[0], NEW[1], filing_html(*BASE, MODIFIED_NEW, long_paragraph))

    text = ok(server, "diff_section", ticker="TSLA")

    [added] = [line for line in text.splitlines() if line.startswith("Risk ")]
    assert len(added) == SNIPPET_CHARS + 2  # the snippet plus " …"
    assert added.endswith(" …")


def test_get_change_context_has_the_full_change_and_its_neighbours(server: MCPServer) -> None:
    text = ok(server, "get_change_context", ticker="TSLA", change_id="~1", count=1)

    assert text.startswith("[~1] modified\n\nOLD:\n" + MODIFIED_OLD)
    assert "NEW:\n" + MODIFIED_NEW in text
    assert "WORD DIFF:\n" in text
    assert "Before the changed paragraph" in text
    assert BASE[2] in text  # the paragraph just before, from the new filing
    assert "After the changed paragraph" in text
    assert NEW_RISK in text  # the paragraph just after


def test_get_key_metrics_reports_when_there_are_none(server: MCPServer) -> None:
    text = ok(server, "get_key_metrics", ticker="TSLA")

    assert text == (
        "TSLA 10-K filed 2026-01-29 (0001628280-26-000003)\n"
        "No XBRL metrics are available for this filing."
    )


# --- Failures the client can cause come back as readable errors, never as crashes --------------


def test_unknown_ticker_is_a_tool_error(server: MCPServer) -> None:
    assert "ticker 'NOPE' not found" in error(server, "list_filings", ticker="NOPE")


def test_missing_item_lists_what_exists(server: MCPServer) -> None:
    text = error(server, "get_section", ticker="TSLA", item="9Z")

    assert text.endswith("Item 9Z not found. Available: I 1A, I 1B, II 7")


def test_unknown_change_id_is_a_tool_error(server: MCPServer) -> None:
    text = error(server, "get_change_context", ticker="TSLA", change_id="+9")

    assert "no change '+9' in Part I Item 1A. Risk Factors" in text


def test_one_filing_cannot_be_diffed(server: MCPServer, edgar: FakeEdgar) -> None:
    del edgar.filings[1]

    text = error(server, "diff_section", ticker="TSLA")

    assert "need two 10-K filings for TSLA to compare" in text


def test_out_of_range_count_is_rejected_before_the_tool_runs(server: MCPServer) -> None:
    text = error(server, "get_change_context", ticker="TSLA", change_id="~1", count=4)

    assert "less than or equal to 3" in text


def test_no_filing_of_that_form_is_a_tool_error(server: MCPServer, edgar: FakeEdgar) -> None:
    edgar.filings.clear()

    assert "no recent 10-K filing for TSLA" in error(server, "diff_section", ticker="TSLA")


def test_sec_http_failure_is_a_tool_error(server: MCPServer, edgar: FakeEdgar) -> None:
    edgar.missing_documents.add(NEW[0])  # listed in the submissions, but the document URL 404s

    text = error(server, "get_section", ticker="TSLA")

    assert "SEC request failed: Client error '404 Not Found'" in text


# --- Output cap ---------------------------------------------------------------------------------


def test_huge_sections_are_cut_with_a_note(server: MCPServer, edgar: FakeEdgar) -> None:
    huge = " ".join(f"Paragraph {n} of a very long risk factor section." for n in range(2_000))
    edgar.filings[0] = (NEW[0], NEW[1], filing_html(huge))

    text = ok(server, "get_section", ticker="TSLA")

    assert len(text) < MAX_OUTPUT_CHARS + 100
    assert "[output cut here:" in text.splitlines()[-1]
