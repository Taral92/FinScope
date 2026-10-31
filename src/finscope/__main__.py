"""Command-line entry point.

python -m finscope filings TSLA          list recent 10-K, 10-Q and 8-K filings
python -m finscope read TSLA [--form F]  download the latest filing and convert it to text
"""

import argparse
import sys

from finscope.config import get_settings
from finscope.edgar import SecClient, TickerNotFoundError, filing_path
from finscope.text import html_to_text

LISTED_FORMS = {"10-K", "10-Q", "8-K"}
PREVIEW_LINES = 40


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finscope", description="Explore SEC EDGAR filings.")
    commands = parser.add_subparsers(dest="command", required=True)

    filings_parser = commands.add_parser("filings", help="list recent 10-K, 10-Q and 8-K filings")
    filings_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")

    read_parser = commands.add_parser("read", help="download the latest filing as plain text")
    read_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    read_parser.add_argument("--form", default="10-Q", help="form type to read (default: 10-Q)")

    args = parser.parse_args(argv)

    with SecClient() as sec:
        try:
            cik = sec.get_cik(args.ticker)
        except TickerNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        if args.command == "filings":
            return _list_filings(sec, cik)
        return _read_filing(sec, cik, args.form)


def _list_filings(sec: SecClient, cik: str) -> int:
    filings = sec.get_recent_filings(cik, forms=LISTED_FORMS, limit=10)
    print(f"{'DATE':<10}  {'FORM':<5}  {'ACCESSION NUMBER':<20}  URL")
    for filing in filings:
        print(
            f"{filing.filing_date.isoformat():<10}  {filing.form:<5}  "
            f"{filing.accession_number:<20}  {filing.url}"
        )
    return 0


def _read_filing(sec: SecClient, cik: str, form: str) -> int:
    filings = sec.get_recent_filings(cik, forms={form}, limit=1)
    if not filings:
        print(f"error: no recent {form} filing found", file=sys.stderr)
        return 1
    filing = filings[0]

    text = html_to_text(sec.get_filing_html(filing))
    path = filing_path(get_settings().data_dir, filing, ".txt")
    path.write_text(text, encoding="utf-8")

    lines = text.splitlines()
    print(f"{filing.form} filed {filing.filing_date.isoformat()}  ({filing.accession_number})")
    print(f"Text:  {len(text):,} characters, {len(lines):,} lines")
    print(f"Saved: {path}")
    print()
    for line in [line for line in lines if line][:PREVIEW_LINES]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
