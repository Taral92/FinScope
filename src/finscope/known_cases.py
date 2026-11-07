"""Known-answer triage cases: made-up changes whose right rating is obvious by design.

Plain code, no LLM. Unlike human labels, nobody has to rate anything: each case says what the
answer must be ("a going-concern warning is high", "a typo fix is low"). The cases are about a
made-up company, Acme Motors, so the model can't lean on what it knows about a real one.

The cases are turned into a fake SectionDiff, so triage_section runs on them unchanged.
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from finscope.diff import (
    MODIFIED_THRESHOLD,
    ParagraphChange,
    SectionDiff,
    changes_by_id,
    similarity,
)
from finscope.evals import Importance, Ratings

# Relative to the working directory, like LABELS_DIR. Committed with the code.
KNOWN_CASES_PATH = Path("evals") / "known" / "triage_cases.json"

# What the fake filing claims to be; the company is made up on purpose.
COMPANY = "Acme Motors"
FORM = "10-K"
PART = "I"
ITEM = "1A"
TITLE = "Risk Factors"


class KnownCasesError(Exception):
    """The case file can't be used: missing, not JSON, or a case that doesn't make sense."""


class KnownCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)  # e.g. "going-concern"
    kind: Literal["added", "removed", "modified"]
    old_text: str | None  # null for an added case
    new_text: str | None  # null for a removed case
    expected: Literal["high", "low"]  # the only right answer
    why: str = Field(min_length=1)  # why that answer is obvious

    @model_validator(mode="after")
    def _texts_fit_kind(self) -> Self:
        for field, text, needed in [
            ("old_text", self.old_text, self.kind != "added"),
            ("new_text", self.new_text, self.kind != "removed"),
        ]:
            if needed and not (text and text.strip()):
                raise ValueError(f"case {self.name!r} ({self.kind}) needs {field}")
            if not needed and text is not None:
                raise ValueError(f"case {self.name!r} ({self.kind}) needs {field}: null")
        if self.kind == "modified":
            # Only a pair the real diff would also call "modified" is a realistic test.
            score = similarity(self.old_text or "", self.new_text or "")
            if score == 1:
                raise ValueError(
                    f"case {self.name!r}: old_text and new_text have the same words, "
                    "so diff.py would count it as unchanged"
                )
            if score < MODIFIED_THRESHOLD:
                raise ValueError(
                    f"case {self.name!r}: old and new text are only {score:.2f} similar "
                    f"(below {MODIFIED_THRESHOLD}), so diff.py would show it as removed + added"
                )
        return self

    def change(self) -> ParagraphChange:
        """The change as diff_sections would produce it, minus the offsets (no real filing)."""
        return ParagraphChange(
            kind=self.kind,
            old_text=self.old_text,
            new_text=self.new_text,
            old_start=None,
            old_end=None,
            new_start=None,
            new_end=None,
            similarity=(
                similarity(self.old_text or "", self.new_text or "")
                if self.kind == "modified"
                else None
            ),
        )


class KnownDiff(BaseModel):
    """The cases as one fake section, plus which change ID is which case."""

    model_config = ConfigDict(frozen=True)

    cases: tuple[KnownCase, ...]  # in file order
    diff: SectionDiff
    ids: dict[str, str]  # case name -> change ID (e.g. "~3"), in file order

    @property
    def names(self) -> dict[str, str]:
        """Change ID -> case name."""
        return {change_id: name for name, change_id in self.ids.items()}

    @property
    def expected(self) -> dict[str, str]:
        """Change ID -> expected rating: what score_triage takes as the human labels."""
        return {self.ids[case.name]: case.expected for case in self.cases}


class CaseResult(BaseModel):
    """One case across runs. It passes only if every run gave exactly the expected rating."""

    model_config = ConfigDict(frozen=True)

    name: str
    expected: Importance
    ratings: tuple[Importance | None, ...]  # one per run; None = that run failed

    @property
    def matches(self) -> int:
        return self.ratings.count(self.expected)

    @property
    def passed(self) -> bool:
        return bool(self.ratings) and self.matches == len(self.ratings)


_CASES = TypeAdapter(tuple[KnownCase, ...])


def load_known_cases(path: Path) -> tuple[KnownCase, ...]:
    """The cases in the file, or a KnownCasesError saying exactly what's wrong with it."""
    try:
        cases = _CASES.validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise KnownCasesError(f"can't read {path}: {error}") from error
    except ValidationError as error:  # bad JSON or a bad case; the message says where
        raise KnownCasesError(f"{path} is not a valid case file:\n{error}") from error

    problems = []
    if not cases:
        problems.append("no cases")
    names = Counter(case.name for case in cases)
    if duplicates := [name for name, count in names.items() if count > 1]:
        problems.append(f"duplicated names: {', '.join(duplicates)}")
    # Two identical changes couldn't be told apart when mapping change IDs back to names.
    texts = Counter((case.kind, case.old_text, case.new_text) for case in cases)
    if same := [case.name for case in cases if texts[case.kind, case.old_text, case.new_text] > 1]:
        problems.append(f"same change in several cases: {', '.join(same)}")
    if problems:
        raise KnownCasesError(f"{path}: " + "; ".join(problems))
    return cases


def build_known_diff(cases: Sequence[KnownCase]) -> KnownDiff:
    """One fake section holding every case, numbered exactly as `diff` and `triage` number it."""
    diff = SectionDiff(
        part=PART,
        item=ITEM,
        title=TITLE,
        unchanged=0,
        changes=tuple(case.change() for case in cases),
    )
    # changes_by_id decides the IDs (added, removed, then modified least similar first), so
    # look each case up by its change rather than repeating that ordering here.
    id_of = {change: change_id for change_id, change in changes_by_id(diff).items()}
    return KnownDiff(
        cases=tuple(cases), diff=diff, ids={case.name: id_of[case.change()] for case in cases}
    )


def case_results(known: KnownDiff, runs: Sequence[Ratings | None]) -> list[CaseResult]:
    """Each case's rating in every run (None = failed run), in file order."""
    return [
        CaseResult(
            name=case.name,
            expected=case.expected,
            ratings=tuple(run[known.ids[case.name]] if run is not None else None for run in runs),
        )
        for case in known.cases
    ]
