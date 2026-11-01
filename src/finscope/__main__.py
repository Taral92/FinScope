"""Command-line entry point.

python -m finscope filings TSLA                     list recent 10-K, 10-Q and 8-K filings
python -m finscope read TSLA [--form F]             download the latest filing as plain text
python -m finscope sections TSLA [--form F]         list the latest filing's Items
python -m finscope sections TSLA --item 1A [--part I]   print one Item's text
python -m finscope facts TSLA [--form F]            key financial numbers from XBRL
"""

import argparse
import os
import sys
from decimal import Decimal
from pathlib import Path

from finscope.config import get_settings
from finscope.edgar import Filing, SecClient, TickerNotFoundError, filing_path
from finscope.sections import Section, split_sections
from finscope.text import html_to_text
from finscope.xbrl import METRICS, MetricValue, key_metrics, parse_facts

LISTED_FORMS = {"10-K", "10-Q", "8-K"}
PREVIEW_LINES = 40
TITLE_WIDTH = 60


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finscope", description="Explore SEC EDGAR filings.")
    commands = parser.add_subparsers(dest="command", required=True)

    filings_parser = commands.add_parser("filings", help="list recent 10-K, 10-Q and 8-K filings")
    filings_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")

    read_parser = commands.add_parser("read", help="download the latest filing as plain text")
    read_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    read_parser.add_argument("--form", default="10-Q", help="form type to read (default: 10-Q)")

    sections_parser = commands.add_parser("sections", help="split the latest filing into Items")
    sections_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    sections_parser.add_argument("--form", default="10-Q", help="form type (default: 10-Q)")
    sections_parser.add_argument("--item", help="print this Item's full text, e.g. 1A")
    sections_parser.add_argument("--part", help="with --item: which Part, e.g. II")

    facts_parser = commands.add_parser("facts", help="key financial numbers from XBRL data")
    facts_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    facts_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])

    args = parser.parse_args(argv)

    with SecClient() as sec:
        try:
            cik = sec.get_cik(args.ticker)
        except TickerNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        if args.command == "filings":
            return _list_filings(sec, cik)

        filings = sec.get_recent_filings(cik, forms={args.form}, limit=1)
        if not filings:
            print(f"error: no recent {args.form} filing found", file=sys.stderr)
            return 1
        if args.command == "read":
            return _read_filing(sec, filings[0])
        if args.command == "facts":
            return _show_facts(sec, filings[0])
        return _show_sections(sec, filings[0], args.item, args.part)


def _list_filings(sec: SecClient, cik: str) -> int:
    filings = sec.get_recent_filings(cik, forms=LISTED_FORMS, limit=10)
    print(f"{'DATE':<10}  {'FORM':<5}  {'ACCESSION NUMBER':<20}  URL")
    for filing in filings:
        print(
            f"{filing.filing_date.isoformat():<10}  {filing.form:<5}  "
            f"{filing.accession_number:<20}  {filing.url}"
        )
    return 0


def _read_filing(sec: SecClient, filing: Filing) -> int:
    # Always re-convert, so `read` also refreshes the saved text after html_to_text changes.
    text, path = _filing_text(sec, filing, reuse_saved=False)

    lines = text.splitlines()
    print(f"{filing.form} filed {filing.filing_date.isoformat()}  ({filing.accession_number})")
    print(f"Text:  {len(text):,} characters, {len(lines):,} lines")
    print(f"Saved: {path}")
    print()
    for line in [line for line in lines if line][:PREVIEW_LINES]:
        print(line)
    return 0


def _show_sections(sec: SecClient, filing: Filing, item: str | None, part: str | None) -> int:
    text, _ = _filing_text(sec, filing, reuse_saved=True)
    sections = split_sections(text)

    if item is None:
        print(f"{filing.form} filed {filing.filing_date.isoformat()}  ({filing.accession_number})")
        print()
        print(f"{'PART':<4}  {'ITEM':<5}  {'TITLE':<{TITLE_WIDTH}}  {'CHARS':>7}")
        for section in sections:
            print(
                f"{section.part or '-':<4}  {section.item:<5}  "
                f"{_shorten(section.title, TITLE_WIDTH):<{TITLE_WIDTH}}  {len(section.text):>7,}"
            )
        return 0

    matches = [
        section
        for section in sections
        if section.item == item.upper() and (part is None or section.part == part.upper())
    ]
    if not matches:
        available = ", ".join(_label(section) for section in sections)
        print(f"error: Item {item} not found. Available: {available}", file=sys.stderr)
        return 1
    if len(matches) > 1:
        options = " or ".join(f"--part {section.part}" for section in matches)
        print(f"error: Item {item} appears in more than one Part; add {options}", file=sys.stderr)
        return 1
    print(matches[0].text)
    return 0


def _show_facts(sec: SecClient, filing: Filing) -> int:
    concepts = {concept for metric in METRICS for concept in metric.concepts}
    facts = parse_facts(sec.get_company_facts(filing.cik), concepts)
    metrics = key_metrics(facts, filing.accession_number, filing.form)
    found = {value.name: value for value in metrics}

    print(f"{filing.form} filed {filing.filing_date.isoformat()}  ({filing.accession_number})")
    if not found:
        # SEC usually updates companyfacts within minutes of a filing, but not instantly.
        print("No XBRL facts for this filing yet.")
        return 0
    print()
    print(f"{'METRIC':<16}  {'PERIOD':<24}  {'VALUE':>10}  {'PRIOR':>10}  {'CHANGE':>7}  CONCEPT")
    for metric in METRICS:
        value = found.get(metric.name)
        if value is None:
            print(f"{metric.name:<16}  not reported")
            continue
        prior = _format_amount(value.prior.value, value.unit) if value.prior else "-"
        change = f"{value.change_pct:+.1f}%" if value.change_pct is not None else "-"
        print(
            f"{metric.name:<16}  {_format_period(value):<24}  "
            f"{_format_amount(value.current.value, value.unit):>10}  {prior:>10}  {change:>7}  "
            f"{value.concept}"
        )
    return 0


def _format_period(value: MetricValue) -> str:
    current, prior = value.current, value.prior
    if current.start is not None:
        return f"{current.start.isoformat()} to {current.end.isoformat()}"
    # Instants: show the comparison date, since it isn't always exactly a year earlier.
    if prior is not None:
        return f"{current.end.isoformat()} vs {prior.end.isoformat()}"
    return f"at {current.end.isoformat()}"


def _format_amount(value: Decimal, unit: str) -> str:
    sign = "-" if value < 0 else ""
    if unit == "USD":
        return f"{sign}${abs(value) / 1_000_000:,.0f}M"
    if unit == "USD/shares":
        return f"{sign}${abs(value):,.2f}"
    return f"{value:,} {unit}"


def _filing_text(sec: SecClient, filing: Filing, *, reuse_saved: bool) -> tuple[str, Path]:
    """The filing as plain text, from the saved .txt when allowed, else converted and saved."""
    path = filing_path(get_settings().data_dir, filing, ".txt")
    if reuse_saved and path.exists():
        return path.read_text(encoding="utf-8"), path
    text = html_to_text(sec.get_filing_html(filing))
    path.write_text(text, encoding="utf-8")
    return text, path


def _label(section: Section) -> str:
    return f"{section.part} {section.item}" if section.part else section.item


def _shorten(title: str, width: int) -> str:
    return title if len(title) <= width else title[: width - 1] + "…"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # The reader stopped early (e.g. `| head`). Point stdout at devnull so the flush at
        # exit doesn't raise again; recipe from the Python docs (signal module, SIGPIPE).
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(1) from None
