"""Command-line entry point.

python -m finscope filings TSLA                     list recent 10-K, 10-Q and 8-K filings
python -m finscope read TSLA [--form F]             download the latest filing as plain text
python -m finscope sections TSLA [--form F]         list the latest filing's Items
python -m finscope sections TSLA --item 1A [--part I]   print one Item's text
python -m finscope facts TSLA [--form F]            key financial numbers from XBRL
python -m finscope diff TSLA [--form F] [--item 1A] [--part I]   compare with the previous filing
python -m finscope explain TSLA --item 1A --kind added --n 4      AI explanation of one change
python -m finscope explain TSLA --item 1A --kind added --n 4 --agent   same, model uses tools first
"""

import argparse
import os
import sys

from finscope.agent import AgentError, TraceStep
from finscope.agent_tools import ExplainContext
from finscope.config import get_settings
from finscope.diff import ParagraphChange, SectionDiff, diff_sections, word_diff
from finscope.edgar import Filing, SecClient, TickerNotFoundError, filing_path
from finscope.explain import explain_change, explain_change_with_agent
from finscope.filings import load_filing_text
from finscope.llm import LLMError, OpenAIProvider
from finscope.sections import Section, split_sections
from finscope.xbrl import (
    METRICS,
    MetricValue,
    format_amount,
    format_period,
    key_metrics,
    parse_facts,
)

LISTED_FORMS = {"10-K", "10-Q", "8-K"}
PREVIEW_LINES = 40
TITLE_WIDTH = 60
SNIPPET_CHARS = 600


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

    diff_parser = commands.add_parser("diff", help="compare with the previous filing")
    diff_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    diff_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])
    diff_parser.add_argument("--item", help="show paragraph changes for this Item, e.g. 1A")
    diff_parser.add_argument("--part", help="with --item: which Part, e.g. I")

    explain_parser = commands.add_parser("explain", help="AI explanation of one diff change")
    explain_parser.add_argument("ticker", help="stock ticker, e.g. TSLA")
    explain_parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"])
    explain_parser.add_argument("--item", required=True, help="the Item, e.g. 1A")
    explain_parser.add_argument("--part", help="which Part, if the Item exists in several")
    explain_parser.add_argument("--kind", required=True, choices=["added", "removed", "modified"])
    explain_parser.add_argument("--n", type=int, required=True, help="change number from `diff`")
    explain_parser.add_argument(
        "--agent", action="store_true", help="let the model gather context with tools first"
    )

    args = parser.parse_args(argv)

    with SecClient() as sec:
        try:
            cik = sec.get_cik(args.ticker)
        except TickerNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        if args.command == "filings":
            return _list_filings(sec, cik)

        # The form filter is an exact match, so amendments (e.g. 10-K/A) are excluded.
        filings = sec.get_recent_filings(cik, forms={args.form}, limit=2)
        if not filings:
            print(f"error: no recent {args.form} filing found", file=sys.stderr)
            return 1
        if args.command in ("diff", "explain"):
            if len(filings) < 2:
                print(f"error: need two {args.form} filings to compare", file=sys.stderr)
                return 1
            old, new = filings[1], filings[0]
            if args.command == "explain":
                return _explain(
                    sec, args.ticker, old, new, args.item, args.part, args.kind, args.n, args.agent
                )
            return _show_diff(sec, old=old, new=new, item=args.item, part=args.part)
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
    text = load_filing_text(sec, filing, refresh=True)
    path = filing_path(sec.data_dir, filing, ".txt")

    lines = text.splitlines()
    print(f"{filing.form} filed {filing.filing_date.isoformat()}  ({filing.accession_number})")
    print(f"Text:  {len(text):,} characters, {len(lines):,} lines")
    print(f"Saved: {path}")
    print()
    for line in [line for line in lines if line][:PREVIEW_LINES]:
        print(line)
    return 0


def _show_sections(sec: SecClient, filing: Filing, item: str | None, part: str | None) -> int:
    sections = split_sections(load_filing_text(sec, filing))

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

    chosen = _pick_section(sections, item, part)
    if chosen is None:
        return 1
    print(chosen.text)
    return 0


def _pick_section(sections: list[Section], item: str, part: str | None) -> Section | None:
    """The one section matching item (and part); prints why and returns None otherwise."""
    matches = [
        section
        for section in sections
        if section.item == item.upper() and (part is None or section.part == part.upper())
    ]
    if not matches:
        available = ", ".join(_label(section) for section in sections)
        print(f"error: Item {item} not found. Available: {available}", file=sys.stderr)
        return None
    if len(matches) > 1:
        options = " or ".join(f"--part {section.part}" for section in matches)
        print(f"error: Item {item} appears in more than one Part; add {options}", file=sys.stderr)
        return None
    return matches[0]


def _show_facts(sec: SecClient, filing: Filing) -> int:
    found = {value.name: value for value in _load_metrics(sec, filing)}

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
        prior = format_amount(value.prior.value, value.unit) if value.prior else "-"
        change = f"{value.change_pct:+.1f}%" if value.change_pct is not None else "-"
        print(
            f"{metric.name:<16}  {format_period(value):<24}  "
            f"{format_amount(value.current.value, value.unit):>10}  {prior:>10}  {change:>7}  "
            f"{value.concept}"
        )
    return 0


def _load_metrics(sec: SecClient, filing: Filing) -> list[MetricValue]:
    concepts = {concept for metric in METRICS for concept in metric.concepts}
    facts = parse_facts(sec.get_company_facts(filing.cik), concepts)
    return key_metrics(facts, filing.accession_number, filing.form)


def _show_diff(
    sec: SecClient, *, old: Filing, new: Filing, item: str | None, part: str | None
) -> int:
    _print_filing_pair(old, new)
    if item is not None:
        diff = _section_diff(sec, old, new, item, part)
        if diff is None:
            return 1
        _print_section_changes(diff)
        return 0

    old_sections = split_sections(load_filing_text(sec, old))
    new_sections = split_sections(load_filing_text(sec, new))
    old_by_key = {(s.part, s.item): s for s in old_sections}
    new_by_key = {(s.part, s.item): s for s in new_sections}
    print(
        f"{'PART':<4}  {'ITEM':<5}  {'TITLE':<40}  "
        f"{'UNCHANGED':>9}  {'MODIFIED':>8}  {'ADDED':>5}  {'REMOVED':>7}"
    )
    for key, new_section in new_by_key.items():
        if key not in old_by_key:
            continue
        diff = diff_sections(old_by_key[key], new_section)
        print(
            f"{diff.part or '-':<4}  {diff.item:<5}  {_shorten(diff.title, 40):<40}  "
            f"{diff.unchanged:>9}  {len(diff.of_kind('modified')):>8}  "
            f"{len(diff.of_kind('added')):>5}  {len(diff.of_kind('removed')):>7}"
        )
    only_new = [_label(s) for key, s in new_by_key.items() if key not in old_by_key]
    only_old = [_label(s) for key, s in old_by_key.items() if key not in new_by_key]
    if only_new or only_old:
        print()
        print(f"Only in the new filing: {', '.join(only_new) or '-'}")
        print(f"Only in the old filing: {', '.join(only_old) or '-'}")
    return 0


def _explain(
    sec: SecClient,
    ticker: str,
    old: Filing,
    new: Filing,
    item: str,
    part: str | None,
    kind: str,
    number: int,
    agent: bool,
) -> int:
    settings = get_settings()
    try:
        provider = OpenAIProvider(settings)  # fails fast without an API key
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _print_filing_pair(old, new)
    sections = _section_pair(sec, old, new, item, part)
    if sections is None:
        return 1
    old_section, new_section = sections
    diff = diff_sections(old_section, new_section)
    changes = _numbered(diff)[kind]
    if not 1 <= number <= len(changes):
        print(
            f"error: --n {number} is out of range; {_diff_label(diff)} has "
            f"{len(changes)} {kind} change(s)",
            file=sys.stderr,
        )
        return 1
    change = changes[number - 1]
    print(f"{_diff_label(diff)}, {kind} #{number}:")
    print(_snippet(_change_text(change)))
    print()

    if agent:
        context = ExplainContext(
            change=change,
            old_section=old_section,
            new_section=new_section,
            metrics=tuple(_load_metrics(sec, new)),  # loaded now: tools never touch the network
        )
        return _explain_with_agent(context, ticker, new, _diff_label(diff), provider)

    try:
        explanation = explain_change(
            change,
            company=ticker.upper(),
            form=new.form,
            section_title=_diff_label(diff),
            provider=provider,
            settings=settings,
        )
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    result = explanation.result
    cost = "unknown (no prices configured)"
    if explanation.cost_usd is not None:
        cost = f"${explanation.cost_usd:.4f}"
    print(result.text)
    print()
    print(f"model:      {result.model}")
    print(f"tokens:     {result.input_tokens:,} input, {result.output_tokens:,} output")
    print(f"cost:       {cost}")
    print(f"request id: {result.request_id or '-'}")
    return 0


def _explain_with_agent(
    context: ExplainContext, ticker: str, filing: Filing, section: str, provider: OpenAIProvider
) -> int:
    settings = get_settings()
    try:
        explanation = explain_change_with_agent(
            context,
            company=ticker.upper(),
            form=filing.form,
            section_title=section,
            provider=provider,
            settings=settings,
        )
    except AgentError as error:
        _print_trace(error.trace)
        print(f"error: {error}", file=sys.stderr)
        return 1
    except LLMError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    run = explanation.run
    _print_trace(run.trace)
    print()
    print(run.text)
    print()
    cost = "unknown (no prices configured)"
    if explanation.cost_usd is not None:
        cost = f"${explanation.cost_usd:.4f}"
    print(f"steps:  {len(run.trace)}")
    print(f"tokens: {run.input_tokens:,} input, {run.output_tokens:,} output")
    print(f"cost:   {cost}")
    return 0


def _print_trace(trace: tuple[TraceStep, ...]) -> None:
    print("TRACE")
    for step in trace:
        calls = "; ".join(
            f"{call.name} {call.arguments} -> "
            + (f"ERROR: {call.error}" if call.is_error else f"{call.result_chars:,} chars")
            for call in step.tool_calls
        )
        action = calls if step.tool_calls else "answer"
        print(f"  {step.number}  {action}  ({step.input_tokens:,} in / {step.output_tokens:,} out)")


def _print_filing_pair(old: Filing, new: Filing) -> None:
    print(f"{new.form}  new: {new.filing_date.isoformat()} ({new.accession_number})")
    print(f"{' ' * len(new.form)}  old: {old.filing_date.isoformat()} ({old.accession_number})")
    print()


def _section_diff(
    sec: SecClient, old: Filing, new: Filing, item: str, part: str | None
) -> SectionDiff | None:
    sections = _section_pair(sec, old, new, item, part)
    return None if sections is None else diff_sections(*sections)


def _section_pair(
    sec: SecClient, old: Filing, new: Filing, item: str, part: str | None
) -> tuple[Section, Section] | None:
    """The same Item in the old and the new filing (errors are printed by _pick_section)."""
    old_section = _pick_section(split_sections(load_filing_text(sec, old)), item, part)
    new_section = _pick_section(split_sections(load_filing_text(sec, new)), item, part)
    if old_section is None or new_section is None:
        return None
    return old_section, new_section


def _numbered(diff: SectionDiff) -> dict[str, list[ParagraphChange]]:
    """Changes per kind in display order; `explain --n` counts in this same order."""
    return {
        "added": diff.of_kind("added"),
        "removed": diff.of_kind("removed"),
        "modified": sorted(diff.of_kind("modified"), key=lambda change: change.similarity or 0),
    }


def _print_section_changes(diff: SectionDiff) -> None:
    numbered = _numbered(diff)
    added, removed, modified = numbered["added"], numbered["removed"], numbered["modified"]
    print(
        f"{_diff_label(diff)}: {diff.unchanged} unchanged, {len(modified)} modified, "
        f"{len(added)} added, {len(removed)} removed"
    )

    print(f"\n=== ADDED ({len(added)}) ===")
    for n, change in enumerate(added, start=1):
        print(f"\n[+{n}] new @ {change.new_start:,}")
        print(_snippet(_change_text(change)))

    print(f"\n=== REMOVED ({len(removed)}) ===")
    for n, change in enumerate(removed, start=1):
        print(f"\n[-{n}] old @ {change.old_start:,}")
        print(_snippet(_change_text(change)))

    print(f"\n=== MODIFIED ({len(modified)}), least similar first ===")
    for n, change in enumerate(modified, start=1):
        print(
            f"\n[~{n}] {change.similarity or 0:.2f} similar  "
            f"old @ {change.old_start:,}  new @ {change.new_start:,}"
        )
        print(_snippet(_change_text(change)))


def _diff_label(diff: SectionDiff) -> str:
    label = f"Part {diff.part} Item {diff.item}" if diff.part else f"Item {diff.item}"
    return f"{label}. {diff.title}"


def _change_text(change: ParagraphChange) -> str:
    """What to show for a change: the paragraph, or the word diff for a modified one."""
    if change.kind == "modified":
        return word_diff(change.old_text or "", change.new_text or "")
    return change.new_text or change.old_text or ""


def _snippet(text: str) -> str:
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + " …"


def _label(section: Section) -> str:
    return f"{section.part} {section.item}" if section.part else section.item


def _shorten(title: str, width: int) -> str:
    return title if len(title) <= width else title[: width - 1] + "…"


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.stdout.flush()  # surface a closed pipe here, not during interpreter shutdown
    except BrokenPipeError:
        # The reader stopped early (e.g. `| head`). Point stdout at devnull so the flush at
        # exit doesn't raise again; recipe from the Python docs (signal module, SIGPIPE).
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        exit_code = 1
    raise SystemExit(exit_code)
