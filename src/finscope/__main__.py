"""Command-line entry point: `python -m finscope TICKER` lists a company's recent filings."""

import argparse
import sys

from finscope.edgar import SecClient, TickerNotFoundError

FORMS = {"10-K", "10-Q", "8-K"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="finscope", description="Show a company's recent 10-K, 10-Q and 8-K filings."
    )
    parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    args = parser.parse_args(argv)

    with SecClient() as sec:
        try:
            cik = sec.get_cik(args.ticker)
        except TickerNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        filings = sec.get_recent_filings(cik, forms=FORMS, limit=10)

    print(f"{'DATE':<10}  {'FORM':<5}  {'ACCESSION NUMBER':<20}  URL")
    for filing in filings:
        print(
            f"{filing.filing_date.isoformat():<10}  {filing.form:<5}  "
            f"{filing.accession_number:<20}  {filing.url}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
