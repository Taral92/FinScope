from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest

from finscope.diff import ParagraphChange, SectionDiff
from finscope.labels import (
    ChangeLabel,
    LabelsError,
    SectionLabels,
    fingerprint,
    label_changes,
    load_labels,
    save_labels,
)


def change(
    kind: Literal["added", "removed", "modified"],
    old: str | None,
    new: str | None,
    similarity: float | None = None,
) -> ParagraphChange:
    return ParagraphChange(
        kind=kind,
        old_text=old,
        new_text=new,
        old_start=0 if old else None,
        old_end=len(old) if old else None,
        new_start=0 if new else None,
        new_end=len(new) if new else None,
        similarity=similarity,
    )


ROBOTAXI = change("added", None, "Upon launching our Robotaxi service, we entered a new market.")
CEO = change("removed", "We depend on the services of our Chief Executive Officer.", None)
DEMAND = change(
    "modified",
    "Demand for our vehicles may decline in 2025.",
    "Demand for our vehicles may decline sharply in 2026.",
    similarity=0.7,
)
DIFF = SectionDiff(
    part="I", item="1A", title="RISK FACTORS", unchanged=10, changes=(ROBOTAXI, CEO, DEMAND)
)
IDS = ["+1", "-1", "~1"]  # diff order: added, removed, modified

EMPTY = SectionLabels(
    company="TSLA",
    form="10-K",
    part="I",
    item="1A",
    new_accession="0001628280-26-003952",
    old_accession="0001628280-25-003063",
    labels=(),
)


def labeled(
    *items: tuple[str, ParagraphChange, Literal["high", "medium", "low", "unsure"]],
) -> SectionLabels:
    labels = tuple(
        ChangeLabel(id=change_id, fingerprint=fingerprint(c), label=label)
        for change_id, c, label in items
    )
    return EMPTY.model_copy(update={"labels": labels})


def scripted(*answers: str) -> tuple[Callable[[str], str], list[str]]:
    """An `ask` that replays answers in order, and a list collecting everything shown."""
    queue = list(answers)
    shown: list[str] = []

    def ask(prompt: str) -> str:
        shown.append(prompt)
        return queue.pop(0)

    return ask, shown


def test_file_name_names_company_form_filing_and_section() -> None:
    assert EMPTY.file_name == "TSLA_10-K_0001628280-26-003952_I-1A.json"


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    labels = labeled(("+1", ROBOTAXI, "high"), ("~1", DEMAND, "low"))
    with_note = labels.labels[0].model_copy(update={"note": "new market"})
    labels = labels.model_copy(update={"labels": (with_note, labels.labels[1])})

    path = save_labels(labels, tmp_path)

    assert path == tmp_path / EMPTY.file_name
    assert load_labels(path, DIFF) == labels


def test_atomic_write_leaves_no_partial_file(tmp_path: Path) -> None:
    save_labels(labeled(("+1", ROBOTAXI, "high")), tmp_path / "evals" / "labels")

    files = [p.name for p in (tmp_path / "evals" / "labels").iterdir()]
    assert files == [EMPTY.file_name]


def test_fingerprint_ignores_whitespace_and_typography() -> None:
    spaced = change(
        "added", None, "Upon launching our  Robotaxi\nservice, we entered a new market."
    )
    straight = change("added", None, "Robotaxi's launch.")
    curly = change("added", None, "Robotaxi\u2019s launch.")

    assert fingerprint(spaced) == fingerprint(ROBOTAXI)
    assert fingerprint(curly) == fingerprint(straight)
    assert fingerprint(CEO) != fingerprint(ROBOTAXI)
    assert len(fingerprint(CEO)) == 12


def test_fingerprint_mismatch_raises(tmp_path: Path) -> None:
    # Labeled when +1 was Robotaxi; now the diff code numbers a different paragraph +1.
    path = save_labels(labeled(("+1", ROBOTAXI, "high"), ("-1", CEO, "low")), tmp_path)
    other = change("added", None, "New tariffs on imported components could raise our costs.")
    shifted = DIFF.model_copy(update={"changes": (other, CEO, DEMAND)})

    with pytest.raises(LabelsError) as error:
        load_labels(path, shifted)

    assert error.value.problems == ["fingerprint mismatch: +1"]


def test_unknown_id_raises(tmp_path: Path) -> None:
    path = save_labels(labeled(("+1", ROBOTAXI, "high"), ("+2", CEO, "low")), tmp_path)

    with pytest.raises(LabelsError, match=r"unknown: \+2"):
        load_labels(path, DIFF)


def test_session_h_m_back_l_q_saves_the_right_labels(tmp_path: Path) -> None:
    ask, _ = scripted("h", "m", "b", "l", "q")

    result = label_changes(DIFF, EMPTY, tmp_path, ask=ask, show=lambda _: None)

    saved = load_labels(tmp_path / EMPTY.file_name, DIFF)
    assert saved == result
    # +1 high; -1 was medium, then "back" and re-rated low; ~1 never reached.
    assert [(label.id, label.label) for label in saved.labels] == [("+1", "high"), ("-1", "low")]


def test_every_answer_is_saved_before_the_next_question(tmp_path: Path) -> None:
    path = tmp_path / EMPTY.file_name

    def ask(prompt: str) -> str:
        if path.exists():  # the second question: the first answer must already be on disk
            assert [label.id for label in load_labels(path, DIFF).labels] == ["+1"]
            raise KeyboardInterrupt  # like Ctrl+C
        return "h"

    with pytest.raises(KeyboardInterrupt):
        label_changes(DIFF, EMPTY, tmp_path, ask=ask, show=lambda _: None)

    assert [label.label for label in load_labels(path, DIFF).labels] == ["high"]


def test_resume_starts_at_the_first_unlabeled_change(tmp_path: Path) -> None:
    ask, _ = scripted("m")
    shown: list[str] = []

    result = label_changes(
        DIFF,
        labeled(("+1", ROBOTAXI, "high"), ("-1", CEO, "low")),
        tmp_path,
        ask=ask,
        show=shown.append,
    )

    assert [(label.id, label.label) for label in result.labels] == [
        ("+1", "high"),
        ("-1", "low"),
        ("~1", "medium"),
    ]
    assert "\n[3/3] ~1  modified" in shown  # the first change shown is the third one
    assert not any("[1/3]" in line or "[2/3]" in line for line in shown)


def test_note_is_attached_to_the_next_rating(tmp_path: Path) -> None:
    ask, _ = scripted("n", "new business line", "h", "q")

    result = label_changes(DIFF, EMPTY, tmp_path, ask=ask, show=lambda _: None)

    [label] = result.labels
    assert (label.id, label.label, label.note) == ("+1", "high", "new business line")


def test_shows_the_text_triage_sends_and_the_counts(tmp_path: Path) -> None:
    ask, _ = scripted("h", "?", "f", "l")
    shown: list[str] = []

    label_changes(DIFF, EMPTY, tmp_path, ask=ask, show=shown.append)

    output = "\n".join(shown)
    assert "Upon launching our Robotaxi service" in output  # added: the paragraph
    assert "{+sharply+}" in output  # modified: the compact word diff
    assert "--- old ---\nDemand for our vehicles may decline in 2025." in output  # from f
    assert "high 1  medium 0  low 1  unsure 1" in output  # the end-of-session counts
    assert f"Saved: {tmp_path / EMPTY.file_name}" in output
