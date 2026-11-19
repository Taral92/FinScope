"""Load a filing's text (from the on-disk caches) and its key XBRL numbers."""

from finscope.edgar import Filing, SecClient, filing_path
from finscope.text import html_to_text
from finscope.xbrl import METRICS, MetricValue, key_metrics, parse_facts


def load_metrics(sec: SecClient, filing: Filing) -> list[MetricValue]:
    """The filing's key financial numbers from SEC XBRL data (never from a model)."""
    concepts = {concept for metric in METRICS for concept in metric.concepts}
    facts = parse_facts(sec.get_company_facts(filing.cik), concepts)
    return key_metrics(facts, filing.accession_number, filing.form)


def load_filing_text(sec: SecClient, filing: Filing, *, refresh: bool = False) -> str:
    """The filing as plain text: the saved .txt if present, else converted from HTML and saved.

    `refresh=True` re-converts even when a .txt exists, e.g. after html_to_text improves.
    """
    path = filing_path(sec.data_dir, filing, ".txt")
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")

    text = html_to_text(sec.get_filing_html(filing))  # HTML comes from its own cache
    # Same temp-file-then-rename as the HTML cache, so a crash never leaves half a file.
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)
    return text
