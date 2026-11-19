"""Score the alerts against your feedback: how many were worth it? Plain code, no LLM.

Reads every file in data/feedback/ (one per rated report) and counts, per report and in
total, how the alerts were rated. The two numbers that matter:
- alert precision: useful alerts / rated alerts. Low means the system alerts too eagerly.
- wrong rate: wrong alerts / rated alerts. These passed the fact-check and still misled you,
  which a quote-and-number check can't catch: the explanation itself is the problem.
Missed alerts (digest entries you wanted as alerts) are listed too; they are the recall side.

The judge (judge.py) is measured here as well: for every rated alert whose report has a judge
verdict, does the judge agree with you? You said "wrong" and the judge said overstated or
unsupported: agreement. You said useful or not_useful (a faithful explanation, whatever its
importance) and the judge said supported: agreement. Anything else is a disagreement, listed
by id so it can be looked at.

Reports also record the model and prompt versions that made them (versions.py), so the same
numbers are given per version: the start of an A/B comparison between prompts. With a handful
of rated alerts per version the difference is noise; the alert counts are printed for that.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from finscope.detect import CheckResult
from finscope.evals import write_json_atomic
from finscope.feedback import Feedback, Verdict, load_feedback, load_report
from finscope.judge import JudgeVerdict, Judgment
from finscope.versions import PromptVersions

# report file name -> change id -> the judge's verdict on it (None: not judged)
Judgments = Mapping[str, Mapping[str, Judgment | None]]
# report file name -> the versions that made it (None: saved before versions were kept)
Versions = Mapping[str, PromptVersions | None]


class JudgedAlert(BaseModel):
    """One rated alert next to the judge's verdict on the same explanation."""

    model_config = ConfigDict(frozen=True)

    id: str
    human: Verdict
    judge: JudgeVerdict | None  # None: the report predates the judge or its call failed
    judge_reason: str | None

    @property
    def agrees(self) -> bool | None:
        """None when there is no judge verdict to compare with."""
        if self.judge is None:
            return None
        return judge_agrees(self.human, self.judge)


def judge_agrees(human: Verdict, judge: JudgeVerdict) -> bool:
    """ "wrong" means the explanation misled you; supported means the judge found it faithful."""
    return (human == "wrong") == (judge != "supported")


class ReportScore(BaseModel):
    """Your verdicts on one report, by id, so a bad alert can be looked up again."""

    model_config = ConfigDict(frozen=True)

    report: str
    checked_at: datetime
    useful: tuple[str, ...]  # alert ids
    not_useful: tuple[str, ...]
    wrong: tuple[str, ...]
    reviews_rated: int
    reviews_useful: int
    missed: tuple[str, ...]  # digest ids you'd have wanted as alerts
    notes: dict[str, str] = {}  # id -> your note, for every answer that has one
    judged: tuple[JudgedAlert, ...] = ()  # every rated alert, with the judge's verdict if any
    versions: PromptVersions | None = None  # None: unknown (old report, or not on disk)

    @property
    def judge_compared(self) -> int:
        return sum(1 for alert in self.judged if alert.agrees is not None)

    @property
    def judge_agreed(self) -> int:
        return sum(1 for alert in self.judged if alert.agrees)

    @property
    def disagreements(self) -> tuple[JudgedAlert, ...]:
        return tuple(alert for alert in self.judged if alert.agrees is False)

    @property
    def alerts_rated(self) -> int:
        return len(self.useful) + len(self.not_useful) + len(self.wrong)

    @property
    def precision(self) -> float | None:
        """useful / rated; None when no alert was rated."""
        return len(self.useful) / self.alerts_rated if self.alerts_rated else None

    @property
    def wrong_rate(self) -> float | None:
        return len(self.wrong) / self.alerts_rated if self.alerts_rated else None


class VersionScore(BaseModel):
    """The alert numbers over every report made with one set of versions."""

    model_config = ConfigDict(frozen=True)

    versions: PromptVersions | None  # None: the reports whose versions are unknown
    reports: tuple[str, ...]  # oldest check first
    alerts_rated: int
    useful: int
    wrong: int
    missed: int
    judge_compared: int
    judge_agreed: int

    @property
    def precision(self) -> float | None:
        return self.useful / self.alerts_rated if self.alerts_rated else None

    @property
    def wrong_rate(self) -> float | None:
        return self.wrong / self.alerts_rated if self.alerts_rated else None


class AlertEvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at: datetime
    reports: tuple[ReportScore, ...]  # oldest check first
    alerts_rated: int
    useful: int
    not_useful: int
    wrong: int
    reviews_rated: int
    reviews_useful: int
    missed: int
    precision: float | None
    wrong_rate: float | None
    judge_compared: int = 0  # rated alerts that also have a judge verdict
    judge_agreed: int = 0
    judge_agreement: float | None = None  # agreed / compared; None when nothing was compared
    by_version: tuple[VersionScore, ...] = ()  # in the order each set of versions was first used

    @property
    def file_name(self) -> str:
        return f"{self.created_at:%Y%m%dT%H%M%SZ}_alerts.json"


def load_all_feedback(directory: Path) -> list[Feedback]:
    """Every answer in every feedback file; a bad file is a FeedbackError, not skipped."""
    answers: list[Feedback] = []
    for path in sorted(directory.glob("*.json")):
        answers.extend(load_feedback(path))
    return answers


def load_reports(reports: Path, names: Iterable[str]) -> dict[str, CheckResult]:
    """The saved reports with these file names, skipping any no longer on disk (their alerts
    then count as not judged, versions unknown); one that can't be read is a FeedbackError,
    like a bad feedback file."""
    loaded: dict[str, CheckResult] = {}
    for name in sorted(set(names)):
        path = reports / name
        if path.exists():
            loaded[name] = load_report(path)
    return loaded


def judgments_in(results: Mapping[str, CheckResult]) -> dict[str, dict[str, Judgment | None]]:
    """The judge's verdicts in each report, by change id."""
    return {
        name: {item.id: item.judgment for item in result.report.high}
        for name, result in results.items()
    }


def versions_in(results: Mapping[str, CheckResult]) -> dict[str, PromptVersions | None]:
    return {name: result.report.prompt_versions for name, result in results.items()}


def score_reports(
    answers: Sequence[Feedback],
    judgments: Judgments | None = None,
    versions: Versions | None = None,
) -> tuple[ReportScore, ...]:
    """One score per report, oldest check first. `judgments` adds the judge comparison,
    `versions` what made each report."""
    by_report: dict[str, list[Feedback]] = defaultdict(list)
    for answer in answers:
        by_report[answer.report].append(answer)
    scores = [
        _score_report(name, group, (judgments or {}).get(name, {})).model_copy(
            update={"versions": (versions or {}).get(name)}
        )
        for name, group in by_report.items()
    ]
    return tuple(sorted(scores, key=lambda score: (score.checked_at, score.report)))


def score_versions(scores: Sequence[ReportScore]) -> tuple[VersionScore, ...]:
    """The scores grouped by the full set of versions (model and all three prompts), in the
    order each set was first used. `scores` must be oldest check first, as score_reports
    returns them."""
    groups: dict[PromptVersions | None, list[ReportScore]] = defaultdict(list)
    for score in scores:
        groups[score.versions].append(score)
    return tuple(
        VersionScore(
            versions=versions,
            reports=tuple(score.report for score in group),
            alerts_rated=sum(score.alerts_rated for score in group),
            useful=sum(len(score.useful) for score in group),
            wrong=sum(len(score.wrong) for score in group),
            missed=sum(len(score.missed) for score in group),
            judge_compared=sum(score.judge_compared for score in group),
            judge_agreed=sum(score.judge_agreed for score in group),
        )
        for versions, group in groups.items()
    )


def summarize_alerts(scores: Sequence[ReportScore], created_at: datetime) -> AlertEvalResult:
    alerts_rated = sum(score.alerts_rated for score in scores)
    useful = sum(len(score.useful) for score in scores)
    wrong = sum(len(score.wrong) for score in scores)
    compared = sum(score.judge_compared for score in scores)
    agreed = sum(score.judge_agreed for score in scores)
    return AlertEvalResult(
        created_at=created_at,
        reports=tuple(scores),
        alerts_rated=alerts_rated,
        useful=useful,
        not_useful=sum(len(score.not_useful) for score in scores),
        wrong=wrong,
        reviews_rated=sum(score.reviews_rated for score in scores),
        reviews_useful=sum(score.reviews_useful for score in scores),
        missed=sum(len(score.missed) for score in scores),
        precision=useful / alerts_rated if alerts_rated else None,
        wrong_rate=wrong / alerts_rated if alerts_rated else None,
        judge_compared=compared,
        judge_agreed=agreed,
        judge_agreement=agreed / compared if compared else None,
        by_version=score_versions(scores),
    )


def save_alert_eval(result: AlertEvalResult, directory: Path) -> Path:
    path = directory / result.file_name
    write_json_atomic(path, result)
    return path


def _score_report(
    name: str, answers: Sequence[Feedback], judgments: Mapping[str, Judgment | None]
) -> ReportScore:
    alerts = [answer for answer in answers if answer.tier == "alert"]
    reviews = [answer for answer in answers if answer.tier == "review"]
    return ReportScore(
        report=name,
        checked_at=answers[0].checked_at,
        useful=tuple(a.id for a in alerts if a.verdict == "useful"),
        not_useful=tuple(a.id for a in alerts if a.verdict == "not_useful"),
        wrong=tuple(a.id for a in alerts if a.verdict == "wrong"),
        reviews_rated=len(reviews),
        reviews_useful=sum(1 for a in reviews if a.verdict == "useful"),
        # A digest entry gets feedback only when you named it as a missed alert.
        missed=tuple(a.id for a in answers if a.tier == "digest" and a.verdict == "useful"),
        notes={a.id: a.note for a in answers if a.note},
        judged=tuple(_judged(a, judgments.get(a.id)) for a in alerts),
    )


def _judged(answer: Feedback, judgment: Judgment | None) -> JudgedAlert:
    return JudgedAlert(
        id=answer.id,
        human=answer.verdict,
        judge=judgment.verdict if judgment else None,
        judge_reason=judgment.reason if judgment else None,
    )
