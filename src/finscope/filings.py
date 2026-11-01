"""Load a filing as plain text, using the on-disk caches."""

from finscope.edgar import Filing, SecClient, filing_path
from finscope.text import html_to_text


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
