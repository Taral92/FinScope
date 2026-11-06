"""Human labels for triage evals: a person rates each change; the ratings are ground truth.

Plain code, no LLM. One JSON file per section under evals/labels/, committed with the code,
so an eval always scores against the same answers.
"""

import hashlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from finscope.diff import ParagraphChange, SectionDiff, changes_by_id
from finscope.text import unify_typography
from finscope.triage import block_text

# Relative to the working directory, like Settings.data_dir. Unlike data/, this is committed.
LABELS_DIR = Path("evals") / "labels"

Label = Literal["high", "medium", "low", "unsure"]
KEYS: dict[str, Label] = {"h": "high", "m": "medium", "l": "low", "?": "unsure"}

GUIDE = """\
Rate each change as an investor would:
  h  high    I'd want an alert
  m  medium  worth a line in a summary
  l  low     noise
  ?  unsure  can't decide
Other keys: n = add a note before rating, f = full old and new paragraphs,
            b = back one change, q = save and quit. Every answer is saved right away."""
PROMPT = "[h/m/l/?  n f b q] > "


class LabelsError(Exception):
    """A label file no longer matches the section's current diff."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("labels don't match the current diff: " + "; ".join(problems))
        self.problems = problems


class ChangeLabel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str  # as `diff` and `triage` print it, e.g. "~24"
    fingerprint: str  # of the change's text; catches IDs that point at another change
    label: Label
    note: str | None = None


class SectionLabels(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    company: str
    form: str
    part: str | None
    item: str
    new_accession: str
    old_accession: str
    labels: tuple[ChangeLabel, ...]  # in diff order; changes not rated yet are absent

    @property
    def file_name(self) -> str:
        """e.g. TSLA_10-K_0001628280-26-003952_I-1A.json"""
        section = f"{self.part}-{self.item}" if self.part else self.item
        return f"{self.company}_{self.form}_{self.new_accession}_{section}.json"


def fingerprint(change: ParagraphChange) -> str:
    """First 12 hex characters of sha256 over the change's normalized old and new text."""
    old = " ".join(unify_typography(change.old_text or "").split())
    new = " ".join(unify_typography(change.new_text or "").split())
    # Normalized text has no newlines, so "\n" keeps ("ab", "c") apart from ("a", "bc").
    return hashlib.sha256(f"{old}\n{new}".encode()).hexdigest()[:12]


def save_labels(labels: SectionLabels, directory: Path) -> Path:
    path = directory / labels.file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same temp-file-then-rename as the filing cache, so a crash never leaves half a file.
    partial = path.with_name(path.name + ".partial")
    partial.write_text(labels.model_dump_json(indent=2) + "\n", encoding="utf-8")
    partial.replace(path)
    return path


def load_labels(path: Path, diff: SectionDiff) -> SectionLabels:
    """The saved labels, checked against the current diff: LabelsError if anything is off.

    Never repaired silently: a label whose change can't be found is an error, not skipped.
    """
    labels = SectionLabels.model_validate_json(path.read_text(encoding="utf-8"))
    changes = changes_by_id(diff)
    problems = []
    if (labels.part, labels.item) != (diff.part, diff.item):
        problems.append(f"file is for {labels.part} {labels.item}, diff is {diff.part} {diff.item}")
    counts = Counter(label.id for label in labels.labels)
    if duplicates := [change_id for change_id, count in counts.items() if count > 1]:
        problems.append(f"duplicated: {', '.join(duplicates)}")
    if unknown := [label.id for label in labels.labels if label.id not in changes]:
        problems.append(f"unknown: {', '.join(unknown)}")
    if changed := [
        label.id
        for label in labels.labels
        if label.id in changes and label.fingerprint != fingerprint(changes[label.id])
    ]:
        # The ID now points at a different change, e.g. because the diff code changed.
        problems.append(f"fingerprint mismatch: {', '.join(changed)}")
    if problems:
        raise LabelsError(problems)
    return labels


def label_changes(
    diff: SectionDiff,
    labels: SectionLabels,
    directory: Path,
    *,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> SectionLabels:
    """Ask for a label per change, one at a time, saving after every answer.

    Starts at the first unlabeled change, so a second run resumes. `ask` and `show` stand in
    for input and print, so tests can script a session. Never shows a model rating: the
    labels must be the person's own judgement.
    """
    changes = changes_by_id(diff)
    ids = list(changes)
    done = {label.id: label for label in labels.labels}
    path = directory / labels.file_name

    def save() -> None:
        nonlocal labels
        ordered = tuple(done[change_id] for change_id in ids if change_id in done)
        labels = labels.model_copy(update={"labels": ordered})
        save_labels(labels, directory)

    show(GUIDE)
    index = next((i for i, change_id in enumerate(ids) if change_id not in done), len(ids))
    shown, note = -1, None
    while index < len(ids):
        change_id = ids[index]
        change = changes[change_id]
        if shown != index:  # a new change: print it once, not again after n or f
            _show_change(show, index, ids, change, done.get(change_id))
            shown = index
            previous = done.get(change_id)
            note = previous.note if previous else None

        key = ask(PROMPT).strip().lower()
        if key in KEYS:
            done[change_id] = ChangeLabel(
                id=change_id, fingerprint=fingerprint(change), label=KEYS[key], note=note
            )
            save()
            index += 1
        elif key == "n":
            note = ask("note: ").strip() or None
            show("Note kept. Now rate this change.")
        elif key == "f":
            show(f"--- old ---\n{change.old_text or '(none: added)'}")
            show(f"--- new ---\n{change.new_text or '(none: removed)'}")
        elif key == "b":
            if index == 0:
                show("Already at the first change.")
            else:
                index -= 1
        elif key == "q":
            save()
            break
        else:
            show(f"Unknown key {key!r}. {PROMPT.strip(' >')}")

    counts = Counter(label.label for label in labels.labels)
    show(f"\nLabeled {len(labels.labels)}/{len(ids)} change(s):")
    show("  " + "  ".join(f"{name} {counts[name]}" for name in KEYS.values()))
    show(f"Saved: {path}")
    return labels


def _show_change(
    show: Callable[[str], None],
    index: int,
    ids: list[str],
    change: ParagraphChange,
    previous: ChangeLabel | None,
) -> None:
    yours = f"  (your label: {previous.label})" if previous else ""
    show(f"\n[{index + 1}/{len(ids)}] {ids[index]}  {change.kind}{yours}")
    show(block_text(change))
