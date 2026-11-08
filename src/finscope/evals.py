"""Score triage runs against human labels. Plain code: no LLM, and this module never imports
openai (or anything that does), so scoring can't call or cost anything.

The labels say what a person thinks each change is worth; each run says what the model thought.
Comparing them gives agreement, recall and precision on "high", critical misses (a human-high
the model called low) and stability (did repeated runs give the same answer?).
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# Relative to the working directory, like LABELS_DIR. Committed: the history of baselines.
RESULTS_DIR = Path("evals") / "results"

Importance = Literal["low", "medium", "high"]
LEVELS: tuple[Importance, ...] = ("high", "medium", "low")
Ratings = Mapping[str, Importance]  # one triage run: change ID -> importance


class Coverage(BaseModel):
    """How many of the section's changes can be scored."""

    model_config = ConfigDict(frozen=True)

    changes: int  # all changes in the section's diff
    scored: int  # human label high, medium or low
    unsure: int  # labeled "unsure": excluded from scoring
    unlabeled: int  # not labeled yet: excluded from scoring

    def describe(self) -> str:
        labeled = self.scored + self.unsure
        return f"{labeled}/{self.changes} labeled, {self.unsure} unsure ({self.scored} scored)"


class RunScore(BaseModel):
    """One run against the labels. Shares are 0-1; None where the denominator would be 0."""

    model_config = ConfigDict(frozen=True)

    agreement: float | None  # of scored changes, how many the model rated like the human
    high_recall: float | None  # of human-high, how many the model rated high
    high_precision: float | None  # of model-high (scored changes only), how many are human-high
    critical_misses: tuple[str, ...]  # human high, model low
    confusion: dict[str, dict[str, int]]  # human level -> model level -> count


class Spread(BaseModel):
    """One metric across runs; None when no run had a value for it."""

    model_config = ConfigDict(frozen=True)

    mean: float | None
    min: float | None
    max: float | None


class ChangeRatings(BaseModel):
    """One change: the human label and what each run said (None = that run failed)."""

    model_config = ConfigDict(frozen=True)

    id: str
    human: str | None  # high/medium/low/unsure, or None if unlabeled
    ratings: tuple[Importance | None, ...]


class EvalScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    coverage: Coverage
    runs: tuple[RunScore | None, ...]  # one per run, None for a failed run
    agreement: Spread
    high_recall: Spread
    high_precision: Spread
    confusion: dict[str, dict[str, int]]  # summed over successful runs
    critical_misses: tuple[ChangeRatings, ...]  # a critical miss in at least one run
    unstable: tuple[ChangeRatings, ...]  # successful runs didn't all agree


class RatedChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    importance: Importance
    reason: str


class RunRecord(BaseModel):
    """One triage run as it happened, including a failure: a failed run is a result too."""

    model_config = ConfigDict(frozen=True)

    number: int  # 1-based
    error: str | None  # None when the run succeeded
    model: str | None
    items: tuple[RatedChange, ...]  # empty for a failed run
    input_tokens: int  # 0 for a failed run: its usage isn't known
    output_tokens: int
    cost_usd: Decimal | None

    @property
    def ratings(self) -> dict[str, Importance]:
        return {item.id: item.importance for item in self.items}


class EvalResult(BaseModel):
    """Everything about one eval, saved to evals/results/ so baselines can be compared later."""

    model_config = ConfigDict(frozen=True)

    created_at: datetime  # UTC
    company: str
    form: str
    part: str | None
    item: str
    new_accession: str
    old_accession: str
    labels_file: str
    model: str  # the model that was requested; each run records what answered
    prompt_version: str
    runs: tuple[RunRecord, ...]
    score: EvalScore
    input_tokens: int  # successful runs only
    output_tokens: int
    cost_usd: Decimal | None

    def file_name(self, labels_stem: str) -> str:
        return f"{self.created_at:%Y%m%dT%H%M%SZ}_{labels_stem}.json"


def prompt_version(system_prompt: str, schema: Mapping[str, Any]) -> str:
    """First 12 hex characters of sha256 over the prompt and the schema.

    Changes whenever either changes, so results from different prompts are never mixed up.
    """
    payload = system_prompt + json.dumps(schema, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def coverage_of(change_ids: Sequence[str], human: Mapping[str, str]) -> Coverage:
    """`human` maps labeled change IDs to high/medium/low/unsure."""
    scored = len(_gold(change_ids, human))
    unsure = sum(1 for change_id in change_ids if human.get(change_id) == "unsure")
    return Coverage(
        changes=len(change_ids),
        scored=scored,
        unsure=unsure,
        unlabeled=len(change_ids) - scored - unsure,
    )


def score_run(gold: Mapping[str, Importance], ratings: Ratings) -> RunScore:
    """Score one run on the scored changes (`gold`: ID -> human high/medium/low)."""
    pairs = [(human, ratings[change_id]) for change_id, human in gold.items()]
    model_for_human_high = [model for human, model in pairs if human == "high"]
    human_for_model_high = [human for human, model in pairs if model == "high"]
    confusion: dict[str, dict[str, int]] = {human: dict.fromkeys(LEVELS, 0) for human in LEVELS}
    for human, model in pairs:
        confusion[human][model] += 1
    return RunScore(
        agreement=_share(sum(human == model for human, model in pairs), len(pairs)),
        high_recall=_share(model_for_human_high.count("high"), len(model_for_human_high)),
        high_precision=_share(human_for_model_high.count("high"), len(human_for_model_high)),
        critical_misses=tuple(
            change_id
            for change_id, human in gold.items()
            if human == "high" and ratings[change_id] == "low"
        ),
        confusion=confusion,
    )


def score_triage(
    change_ids: Sequence[str], human: Mapping[str, str], runs: Sequence[Ratings | None]
) -> EvalScore:
    """Score every run (None = failed) against the human labels, then compare the runs."""
    gold = _gold(change_ids, human)
    scores = [score_run(gold, run) if run is not None else None for run in runs]
    done = [score for score in scores if score is not None]

    confusion: dict[str, dict[str, int]] = {row: dict.fromkeys(LEVELS, 0) for row in LEVELS}
    for score in done:
        for row in LEVELS:
            for column in LEVELS:
                confusion[row][column] += score.confusion[row][column]

    def ratings_of(change_id: str) -> ChangeRatings:
        return ChangeRatings(
            id=change_id,
            human=human.get(change_id),
            ratings=tuple(run.get(change_id) if run is not None else None for run in runs),
        )

    missed = {change_id for score in done for change_id in score.critical_misses}
    successful = [run for run in runs if run is not None]
    return EvalScore(
        coverage=coverage_of(change_ids, human),
        runs=tuple(scores),
        agreement=_spread([score.agreement for score in done]),
        high_recall=_spread([score.high_recall for score in done]),
        high_precision=_spread([score.high_precision for score in done]),
        confusion=confusion,
        critical_misses=tuple(ratings_of(i) for i in change_ids if i in missed),
        # Stability is about the model, so every change counts here, labeled or not.
        unstable=tuple(
            ratings_of(change_id)
            for change_id in change_ids
            if len({run.get(change_id) for run in successful}) > 1
        ),
    )


def save_result(result: EvalResult, directory: Path, labels_stem: str) -> Path:
    path = directory / result.file_name(labels_stem)
    write_json_atomic(path, result)
    return path


def write_json_atomic(path: Path, result: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same temp-file-then-rename as the filing cache, so a crash never leaves half a file.
    partial = path.with_name(path.name + ".partial")
    partial.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def _gold(change_ids: Sequence[str], human: Mapping[str, str]) -> dict[str, Importance]:
    """The scored changes: those the human rated high, medium or low (not unsure)."""
    gold: dict[str, Importance] = {}
    for change_id in change_ids:
        for level in LEVELS:
            if human.get(change_id) == level:
                gold[change_id] = level
    return gold


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None  # never divide by zero


def _spread(values: list[float | None]) -> Spread:
    present = [value for value in values if value is not None]
    if not present:
        return Spread(mean=None, min=None, max=None)
    return Spread(mean=fmean(present), min=min(present), max=max(present))
