"""A small client for SEC EDGAR: company lookups, filing lists, and filing downloads."""

import time
from datetime import date
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import BaseModel, ConfigDict

from finscope.config import Settings, get_settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


class TickerNotFoundError(LookupError):
    """Raised when a ticker isn't in SEC's company ticker list."""

    def __init__(self, ticker: str) -> None:
        super().__init__(f"ticker {ticker!r} not found in SEC's company ticker list")
        self.ticker = ticker


class Filing(BaseModel):
    """One filing from a company's EDGAR submission history."""

    model_config = ConfigDict(frozen=True)

    cik: str  # zero-padded to 10 digits
    accession_number: str  # e.g. "0001628280-25-045968"
    form: str  # e.g. "10-K", "10-Q", "8-K"
    filing_date: date
    report_date: date | None  # period the filing covers; not every form has one
    primary_document: str  # main file inside the filing, e.g. "tsla-20250930.htm"

    @property
    def url(self) -> str:
        """Link to the filing's primary document in the EDGAR archives."""
        return ARCHIVES_URL.format(
            cik=int(self.cik),  # the archive path uses the CIK without leading zeros
            accession=self.accession_number.replace("-", ""),
            document=self.primary_document,
        )


def filing_path(data_dir: Path, filing: Filing, suffix: str) -> Path:
    """Where a filing's local files live, e.g. data/filings/0001318605/0001628280-25-045968.html."""
    return data_dir / "filings" / filing.cik / f"{filing.accession_number}{suffix}"


class SecClient:
    """Synchronous EDGAR client that identifies itself and stays under SEC's rate limit."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        settings = settings or get_settings()
        self._min_interval = 1.0 / settings.sec_max_requests_per_second
        self._last_request_at: float | None = None
        self._data_dir = settings.data_dir
        self._client = httpx.Client(
            headers={"User-Agent": settings.sec_user_agent},
            timeout=30.0,
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_cik(self, ticker: str) -> str:
        """Look up a ticker's CIK, zero-padded to 10 digits."""
        # Shape: {"0": {"cik_str": 1318605, "ticker": "TSLA", "title": "Tesla, Inc."}, ...}
        companies = self._get_json(TICKERS_URL)
        wanted = ticker.strip().upper()
        for company in companies.values():
            if company["ticker"].upper() == wanted:
                return str(company["cik_str"]).zfill(10)
        raise TickerNotFoundError(ticker)

    def get_recent_filings(
        self,
        cik: str,
        forms: set[str] | None = None,
        limit: int = 10,
    ) -> list[Filing]:
        """Return up to `limit` of the company's newest filings, optionally only `forms`."""
        cik = cik.zfill(10)
        data = self._get_json(SUBMISSIONS_URL.format(cik=cik))

        # EDGAR stores recent filings column-wise: one array per field, where index i
        # in every array describes the same filing. zip() turns the columns into rows.
        recent = data["filings"]["recent"]
        rows = zip(
            recent["accessionNumber"],
            recent["form"],
            recent["filingDate"],
            recent["reportDate"],
            recent["primaryDocument"],
            strict=True,  # mismatched array lengths would silently misalign fields
        )
        filings = [
            Filing(
                cik=cik,
                accession_number=accession_number,
                form=form,
                filing_date=filing_date,
                report_date=report_date or None,  # EDGAR uses "" for "no report date"
                primary_document=primary_document,
            )
            for accession_number, form, filing_date, report_date, primary_document in rows
            if forms is None or form in forms
        ]
        filings.sort(key=lambda filing: filing.filing_date, reverse=True)
        return filings[:limit]

    def get_filing_html(self, filing: Filing) -> str:
        """Return the filing's primary document, downloading it only on the first call."""
        # Caching forever is safe: a published filing never changes. Corrections are filed
        # as new amendments (e.g. 10-Q/A) with their own accession numbers.
        path = filing_path(self._data_dir, filing, ".html")
        if path.exists():
            return path.read_text(encoding="utf-8")

        html = self._get(filing.url).text
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file, then rename: a crash mid-write must not leave a truncated
        # file that the cache would then serve forever.
        partial = path.with_suffix(".partial")
        partial.write_text(html, encoding="utf-8")
        partial.replace(path)
        return html

    def _get_json(self, url: str) -> Any:
        return self._get(url).json()

    def _get(self, url: str) -> httpx.Response:
        """The single place requests are sent: throttled, identified, and checked."""
        self._throttle()
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def _throttle(self) -> None:
        """Sleep just long enough to keep requests at least `_min_interval` seconds apart."""
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()
