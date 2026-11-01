"""Compare one section across two filings, paragraph by paragraph. Plain code, no AI."""

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, ConfigDict

from finscope.sections import Section

# Below this word-level similarity, two paragraphs count as unrelated (removed + added).
MODIFIED_THRESHOLD = 0.6

# Typographic variants that shouldn't count as edits: curly quotes, en/em dashes, minus.
_UNIFY = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)
# A paragraph is a run of non-blank lines.
_PARAGRAPH = re.compile(r"\S.*(?:\n.*\S.*)*")
# Page furniture repeated at every page break: page numbers and the "back to contents" link.
# Page numbers shift whenever earlier pages grow, which would show up as fake changes.
_PAGE_FURNITURE = re.compile(r"\d{1,3}|Table of Contents", re.IGNORECASE)


class ParagraphChange(BaseModel):
    """One paragraph that differs between the old and new section."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["added", "removed", "modified"]
    old_text: str | None
    new_text: str | None
    old_start: int | None  # offset into the old filing's full text
    new_start: int | None  # offset into the new filing's full text
    similarity: float | None  # 0-1, only for "modified"


class SectionDiff(BaseModel):
    """How one section changed: an unchanged-paragraph count plus every change."""

    model_config = ConfigDict(frozen=True)

    part: str | None
    item: str
    title: str
    unchanged: int
    changes: tuple[ParagraphChange, ...]

    def of_kind(self, kind: str) -> list[ParagraphChange]:
        return [change for change in self.changes if change.kind == kind]


@dataclass(frozen=True)
class _Paragraph:
    text: str  # exactly as in the filing
    start: int  # absolute offset into the filing's full text
    words: tuple[str, ...]  # normalized, for comparison only

    @property
    def key(self) -> str:
        return " ".join(self.words)


def diff_sections(old: Section, new: Section) -> SectionDiff:
    old_paragraphs = _paragraphs(old)
    new_paragraphs = _paragraphs(new)

    # 1. Unchanged: identical paragraphs, wherever they are. Counting handles repeats: if a
    #    paragraph appears twice before and once now, one copy is unchanged, one removed.
    shared = Counter(p.key for p in old_paragraphs) & Counter(p.key for p in new_paragraphs)
    old_left = _without(old_paragraphs, shared)
    new_left = _without(new_paragraphs, shared)

    # 2. Modified: score every remaining (old, new) pair, then pair the most similar first.
    #    Each paragraph is used at most once.
    scored = []
    for new_p in new_left:
        for old_p in old_left:
            similarity = _similarity(old_p, new_p)
            if similarity >= MODIFIED_THRESHOLD:
                scored.append((similarity, old_p, new_p))
    scored.sort(key=lambda candidate: candidate[0], reverse=True)

    changes = []
    paired_old: set[_Paragraph] = set()
    paired_new: set[_Paragraph] = set()
    for similarity, old_p, new_p in scored:
        if old_p in paired_old or new_p in paired_new:
            continue
        paired_old.add(old_p)
        paired_new.add(new_p)
        changes.append(_change("modified", old_p, new_p, similarity))

    # 3. Whatever is still unpaired was added or removed.
    changes += [_change("added", None, p, None) for p in new_left if p not in paired_new]
    changes += [_change("removed", p, None, None) for p in old_left if p not in paired_old]

    return SectionDiff(
        part=new.part,
        item=new.item,
        title=new.title,
        unchanged=sum(shared.values()),
        changes=tuple(changes),
    )


def word_diff(old: str, new: str) -> str:
    """Inline word diff in git's style: [-removed words-]{+added words+}."""
    old_words, new_words = old.split(), new.split()
    matcher = SequenceMatcher(
        None, [_normalize(w) for w in old_words], [_normalize(w) for w in new_words], autojunk=False
    )
    out = []
    for op, old_lo, old_hi, new_lo, new_hi in matcher.get_opcodes():
        if op == "equal":
            out.append(" ".join(new_words[new_lo:new_hi]))
            continue
        removed = " ".join(old_words[old_lo:old_hi])  # empty for "insert"
        added = " ".join(new_words[new_lo:new_hi])  # empty for "delete"
        out.append((f"[-{removed}-]" if removed else "") + (f"{{+{added}+}}" if added else ""))
    return " ".join(out)


def _paragraphs(section: Section) -> list[_Paragraph]:
    """Paragraphs of the section body (heading and page furniture skipped), with offsets."""
    heading_end = section.text.find("\n")
    if heading_end == -1:
        return []  # heading only, no body
    return [
        _Paragraph(
            text=match.group(),
            start=section.start + match.start(),  # match.start() already counts from 0
            words=tuple(_normalize(match.group()).split()),
        )
        for match in _PARAGRAPH.finditer(section.text, heading_end)
        if not _PAGE_FURNITURE.fullmatch(match.group().strip())
    ]


def _normalize(text: str) -> str:
    return text.translate(_UNIFY)


def _without(paragraphs: list[_Paragraph], shared: Counter[str]) -> list[_Paragraph]:
    remaining = shared.copy()
    left = []
    for paragraph in paragraphs:
        if remaining[paragraph.key] > 0:
            remaining[paragraph.key] -= 1
        else:
            left.append(paragraph)
    return left


def _similarity(old: _Paragraph, new: _Paragraph) -> float:
    matcher = SequenceMatcher(None, old.words, new.words, autojunk=False)
    # real_quick_ratio and quick_ratio are cheap upper bounds of ratio: if even they are below
    # the threshold, the expensive exact ratio can't reach it, so skip computing it.
    if matcher.real_quick_ratio() < MODIFIED_THRESHOLD:
        return 0.0
    if matcher.quick_ratio() < MODIFIED_THRESHOLD:
        return 0.0
    return matcher.ratio()


def _change(
    kind: Literal["added", "removed", "modified"],
    old: _Paragraph | None,
    new: _Paragraph | None,
    similarity: float | None,
) -> ParagraphChange:
    return ParagraphChange(
        kind=kind,
        old_text=old.text if old else None,
        new_text=new.text if new else None,
        old_start=old.start if old else None,
        new_start=new.start if new else None,
        similarity=similarity,
    )
