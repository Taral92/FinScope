from collections.abc import Iterator
from datetime import date

import httpx
import pytest

from finscope.config import Settings
from finscope.edgar import Filing, SecClient, TickerNotFoundError

USER_AGENT = "FinScope Test test@example.com"

FAKE_TICKERS = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1318605, "ticker": "TSLA", "title": "Tesla, Inc."},
}

# Same column-oriented shape as the real submissions endpoint, newest filing first.
FAKE_SUBMISSIONS = {
    "cik": "1318605",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0001318605-25-000003",
                "0001628280-25-000002",
                "0001318605-25-000001",
            ],
            "form": ["4", "10-Q", "8-K"],
            "filingDate": ["2025-10-24", "2025-10-23", "2025-10-22"],
            "reportDate": ["", "2025-09-30", "2025-10-22"],
            "primaryDocument": ["form4.xml", "tsla-20250930.htm", "tsla-8k.htm"],
        }
    },
}


def fake_sec(request: httpx.Request) -> httpx.Response:
    """Stand-in for SEC's servers: answers the two endpoints we use, 404 for anything else."""
    if request.url == "https://www.sec.gov/files/company_tickers.json":
        return httpx.Response(200, json=FAKE_TICKERS)
    if request.url == "https://data.sec.gov/submissions/CIK0001318605.json":
        return httpx.Response(200, json=FAKE_SUBMISSIONS)
    return httpx.Response(404)


@pytest.fixture
def requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def sec(requests: list[httpx.Request]) -> Iterator[SecClient]:
    def record_and_answer(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return fake_sec(request)

    settings = Settings(_env_file=None, sec_user_agent=USER_AGENT, sec_max_requests_per_second=10)
    with SecClient(settings=settings, transport=httpx.MockTransport(record_and_answer)) as client:
        yield client


def test_ticker_resolves_to_zero_padded_cik(sec: SecClient) -> None:
    assert sec.get_cik("TSLA") == "0001318605"


def test_ticker_lookup_is_case_insensitive(sec: SecClient) -> None:
    assert sec.get_cik("aapl") == "0000320193"


def test_unknown_ticker_raises(sec: SecClient) -> None:
    with pytest.raises(TickerNotFoundError, match="NOPE"):
        sec.get_cik("NOPE")


def test_column_arrays_become_filing_rows(sec: SecClient) -> None:
    filings = sec.get_recent_filings("0001318605")

    assert filings == [
        Filing(
            cik="0001318605",
            accession_number="0001318605-25-000003",
            form="4",
            filing_date=date(2025, 10, 24),
            report_date=None,  # "" in the JSON
            primary_document="form4.xml",
        ),
        Filing(
            cik="0001318605",
            accession_number="0001628280-25-000002",
            form="10-Q",
            filing_date=date(2025, 10, 23),
            report_date=date(2025, 9, 30),
            primary_document="tsla-20250930.htm",
        ),
        Filing(
            cik="0001318605",
            accession_number="0001318605-25-000001",
            form="8-K",
            filing_date=date(2025, 10, 22),
            report_date=date(2025, 10, 22),
            primary_document="tsla-8k.htm",
        ),
    ]


def test_form_filter(sec: SecClient) -> None:
    filings = sec.get_recent_filings("0001318605", forms={"10-Q", "8-K"})

    assert [filing.form for filing in filings] == ["10-Q", "8-K"]


def test_limit_keeps_newest(sec: SecClient) -> None:
    filings = sec.get_recent_filings("0001318605", limit=2)

    assert [filing.filing_date for filing in filings] == [date(2025, 10, 24), date(2025, 10, 23)]


def test_url_is_built_from_cik_accession_and_document() -> None:
    filing = Filing(
        cik="0001318605",
        accession_number="0001628280-25-045968",
        form="10-Q",
        filing_date=date(2025, 10, 23),
        report_date=date(2025, 9, 30),
        primary_document="tsla-20250930.htm",
    )

    assert filing.url == (
        "https://www.sec.gov/Archives/edgar/data/1318605/000162828025045968/tsla-20250930.htm"
    )


def test_user_agent_header_is_sent(sec: SecClient, requests: list[httpx.Request]) -> None:
    sec.get_cik("TSLA")

    assert requests[0].headers["User-Agent"] == USER_AGENT


def test_http_errors_raise(sec: SecClient) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        sec.get_recent_filings("0000000001")  # fake_sec answers 404


@pytest.mark.integration
def test_real_sec_tesla() -> None:
    """Hits the real SEC. Needs FINSCOPE_SEC_USER_AGENT; run with `uv run pytest -m integration`."""
    with SecClient() as sec:
        cik = sec.get_cik("TSLA")
        filings = sec.get_recent_filings(cik, forms={"10-K"}, limit=1)

    assert cik == "0001318605"
    assert len(filings) == 1
    assert filings[0].form == "10-K"
