"""Detect: which watched sections have a new filing since the last check. Plain code, no LLM.

The watchlist (watchlist.json, committed) says what to watch: a company, a form and one Item.
The state file (data/state/seen.json, not committed) remembers, per watchlist entry, the
newest filing already processed. plan_entry() compares EDGAR's two newest filings with that
memory and says what to do; it only decides, so it is easy to test and to dry-run.
"""

from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from finscope.decide import Decision
from finscope.edgar import Filing
from finscope.report import SectionReport

# Relative to the working directory, like KNOWN_CASES_PATH. Committed with the code.
WATCHLIST_PATH = Path("watchlist.json")


class WatchlistError(Exception):
    """The watchlist can't be used: missing, not JSON, a bad entry, or a duplicate."""


class StateError(Exception):
    """The state file exists but can't be read; never overwritten, so nothing is forgotten."""


class CheckError(Exception):
    """One watchlist entry can't be checked, e.g. EDGAR lists too few filings to compare."""


class WatchEntry(BaseModel):
    """One section to watch, e.g. TSLA 10-K Part I Item 1A."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    form: Literal["10-K", "10-Q"]
    item: str = Field(min_length=1)
    part: str | None = None  # needed only when the Item exists in several Parts

    @field_validator("ticker", "item", "part")
    @classmethod
    def _normalize(cls, value: str | None) -> str | None:
        # " tsla " and "TSLA" are the same entry, so they must also give the same key.
        if value is None:
            return None
        value = value.strip().upper()
        if not value:
            raise ValueError("must not be blank")
        return value

    @property
    def key(self) -> str:
        """Where this entry's memory lives in the state file, e.g. "TSLA|10-K|I|1A"."""
        return f"{self.ticker}|{self.form}|{self.part or '-'}|{self.item}"

    @property
    def label(self) -> str:
        """e.g. "TSLA 10-K Part I Item 1A"."""
        section = f"Part {self.part} Item {self.item}" if self.part else f"Item {self.item}"
        return f"{self.ticker} {self.form} {section}"


class SeenFiling(BaseModel):
    """The newest filing already processed for one watchlist entry."""

    model_config = ConfigDict(frozen=True)

    accession_number: str
    filing_date: date
    processed_at: datetime


class Plan(BaseModel):
    """What one check does for one entry.

    - start: first sight; remember `new` as the starting point, no AI calls.
    - unchanged: `new` is the remembered filing; nothing to do.
    - compare: run the report on `old` -> `new`, then remember `new`.
    """

    model_config = ConfigDict(frozen=True)

    action: Literal["start", "unchanged", "compare"]
    new: Filing
    old: Filing | None  # set only for compare
    # compare only: the remembered filing is older than `old`, so filings in between are
    # skipped (only the newest pair is compared).
    skipped: bool = False


class CheckResult(BaseModel):
    """What a check saves for one compared pair: the full report and its decision."""

    model_config = ConfigDict(frozen=True)

    checked_at: datetime
    entry: WatchEntry
    old_filing: Filing
    new_filing: Filing
    report: SectionReport
    decision: Decision

    @property
    def file_name(self) -> str:
        """e.g. TSLA_10-K_0001628280-26-003952_I-1A.json, like the labels files."""
        entry = self.entry
        section = f"{entry.part}-{entry.item}" if entry.part else entry.item
        return f"{entry.ticker}_{entry.form}_{self.new_filing.accession_number}_{section}.json"


_WATCHLIST = TypeAdapter(tuple[WatchEntry, ...])
_STATE = TypeAdapter(dict[str, SeenFiling])


def load_watchlist(path: Path) -> tuple[WatchEntry, ...]:
    """The entries in the file, or a WatchlistError saying exactly what's wrong with it."""
    try:
        entries = _WATCHLIST.validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise WatchlistError(f"can't read {path}: {error}") from error
    except ValidationError as error:  # bad JSON or a bad entry; the message says where
        raise WatchlistError(f"{path} is not a valid watchlist:\n{error}") from error

    if not entries:
        raise WatchlistError(f"{path}: no entries")
    keys = Counter(entry.key for entry in entries)
    if duplicates := [key for key, count in keys.items() if count > 1]:
        raise WatchlistError(f"{path}: duplicated entries: {', '.join(duplicates)}")
    return entries


def state_path(data_dir: Path) -> Path:
    return data_dir / "state" / "seen.json"


def reports_dir(data_dir: Path) -> Path:
    return data_dir / "reports"


def load_state(path: Path) -> dict[str, SeenFiling]:
    """Entry key -> last processed filing. No file yet means nothing has been seen."""
    if not path.exists():
        return {}
    try:
        return _STATE.validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise StateError(f"can't use {path}; fix or delete it:\n{error}") from error


def save_state(state: dict[str, SeenFiling], path: Path) -> None:
    _write_atomic(path, _STATE.dump_json(state, indent=2).decode() + "\n")


def save_check_result(result: CheckResult, directory: Path) -> Path:
    path = directory / result.file_name
    _write_atomic(path, result.model_dump_json(indent=2) + "\n")
    return path


def plan_entry(
    filings: Sequence[Filing], seen: SeenFiling | None, *, include_unseen: bool = False
) -> Plan:
    """Decide what to do from EDGAR's newest filings (newest first) and what was seen last.

    `include_unseen=True` compares on first sight instead of only remembering the start.
    """
    if not filings:
        raise CheckError("EDGAR lists no filing of this form")
    newest = filings[0]
    if seen is not None and seen.accession_number == newest.accession_number:
        return Plan(action="unchanged", new=newest, old=None)
    if seen is None and not include_unseen:
        return Plan(action="start", new=newest, old=None)
    if len(filings) < 2:
        raise CheckError("EDGAR lists only one filing of this form; need two to compare")
    previous = filings[1]
    return Plan(
        action="compare",
        new=newest,
        old=previous,
        skipped=seen is not None and seen.accession_number != previous.accession_number,
    )


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same temp-file-then-rename as the filing cache, so a crash never leaves half a file.
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)
