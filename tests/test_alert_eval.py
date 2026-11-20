from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

import finscope.__main__ as cli
from finscope.__main__ import _eval_alerts
from finscope.alert_eval import (
    AlertEvalResult,
    judge_agrees,
    judgments_in,
    load_all_feedback,
    load_reports,
    score_reports,
    score_versions,
    summarize_alerts,
    versions_in,
)
from finscope.decide import Decision, Tier
from finscope.detect import CheckResult, WatchEntry, save_check_result
from finscope.edgar import Filing
from finscope.feedback import Feedback, FeedbackError, Verdict, save_feedback
from finscope.judge import JudgeVerdict, Judgment
from finscope.report import HighItem, SectionReport
from finscope.versions import PromptVersions

REPORT_A = "TSLA_10-K_0001628280-26-000003_I-1A.json"  # checked 2026-11-07
REPORT_B = "AAPL_10-K_0000320193-26-000001_I-1A.json"  # checked 2026-11-10
REPORT_C = "NVDA_10-K_0001045810-26-000002_I-1A.json"  # checked 2026-11-12
CHECKED = {
    REPORT_A: datetime(2026, 11, 7, tzinfo=UTC),
    REPORT_B: datetime(2026, 11, 10, tzinfo=UTC),
    REPORT_C: datetime(2026, 11, 12, tzinfo=UTC),
}
ANSWERED = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def answer(
    report: str, change_id: str, tier: Tier, verdict: Verdict, note: str | None = None
) -> Feedback:
    return Feedback(
        report=report,
        checked_at=CHECKED[report],
        id=change_id,
        tier=tier,
        verdict=verdict,
        answered_at=ANSWERED,
        note=note,
    )


A = [
    answer(REPORT_A, "~1", "alert", "useful"),
    answer(REPORT_A, "~5", "alert", "not_useful"),
    answer(REPORT_A, "~19", "alert", "wrong", "says violence occurred"),
    answer(REPORT_A, "+7", "review", "useful"),
    answer(REPORT_A, "+3", "digest", "useful"),  # a missed alert
]
B = [
    answer(REPORT_B, "+1", "alert", "useful"),
    answer(REPORT_B, "+2", "alert", "useful"),
    answer(REPORT_B, "~4", "review", "not_useful"),
]


def write_feedback(directory: Path, *groups: list[Feedback]) -> None:
    for group in groups:
        save_feedback(group, directory / group[0].report)


def test_scores_are_per_report_oldest_check_first() -> None:
    # B is written first, but A was checked earlier, so A comes first.
    scores = score_reports(B + A)

    assert [score.report for score in scores] == [REPORT_A, REPORT_B]
    a, b = scores
    assert (a.useful, a.not_useful, a.wrong) == (("~1",), ("~5",), ("~19",))
    assert (a.alerts_rated, a.precision, a.wrong_rate) == (
        3,
        pytest.approx(1 / 3),
        pytest.approx(1 / 3),
    )
    assert (a.reviews_rated, a.reviews_useful, a.missed) == (1, 1, ("+3",))
    assert (b.useful, b.not_useful, b.wrong) == (("+1", "+2"), (), ())
    assert (b.precision, b.wrong_rate, b.reviews_useful, b.missed) == (1.0, 0.0, 0, ())


def test_summary_sums_every_report() -> None:
    result = summarize_alerts(score_reports(A + B), ANSWERED)

    assert (result.alerts_rated, result.useful, result.not_useful, result.wrong) == (5, 3, 1, 1)
    assert (result.reviews_rated, result.reviews_useful, result.missed) == (2, 1, 1)
    assert result.precision == pytest.approx(3 / 5)
    assert result.wrong_rate == pytest.approx(1 / 5)
    assert result.file_name == "20261114T120000Z_alerts.json"


def test_no_rated_alert_gives_no_precision() -> None:
    only_review = [answer(REPORT_A, "+7", "review", "useful")]

    result = summarize_alerts(score_reports(only_review), ANSWERED)

    assert result.alerts_rated == 0
    assert result.precision is None
    assert result.wrong_rate is None
    assert result.reports[0].precision is None


def test_load_all_feedback_reads_every_file_and_fails_on_a_bad_one(tmp_path: Path) -> None:
    write_feedback(tmp_path, A, B)

    assert len(load_all_feedback(tmp_path)) == len(A) + len(B)

    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(FeedbackError, match=r"bad\.json"):
        load_all_feedback(tmp_path)


def test_command_prints_per_report_and_totals_and_saves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    feedback, results = tmp_path / "feedback", tmp_path / "results"
    write_feedback(feedback, A, B)

    assert _eval_alerts(feedback, tmp_path / "reports", results) == 0

    [saved] = list(results.iterdir())
    assert saved.name.endswith("_alerts.json")
    result = AlertEvalResult.model_validate_json(saved.read_text(encoding="utf-8"))
    assert (result.alerts_rated, result.useful) == (5, 3)
    out = capsys.readouterr().out
    assert "=== PER REPORT (2) ===" in out
    assert f"{REPORT_A}  checked 2026-11-07" in out
    assert "alerts 3: 1 useful, 1 not_useful, 1 wrong   precision 33%" in out
    assert "  wrong:  ~19" in out
    assert "  missed: +3" in out
    assert "  note ~19: says violence occurred" in out
    assert "alert precision:  60%" in out
    assert "wrong:            20%" in out
    assert "reviews useful:   1/2" in out
    assert "missed alerts:    1" in out
    assert f"saved:  {saved}" in out


def test_command_without_feedback_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _eval_alerts(tmp_path / "feedback", tmp_path / "reports", tmp_path / "results") == 1

    assert "no feedback in" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_command_with_a_bad_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "bad.json").write_text("[]x", encoding="utf-8")

    assert _eval_alerts(tmp_path, tmp_path / "reports", tmp_path / "results") == 1

    assert "can't read feedback" in capsys.readouterr().err


def test_eval_alerts_command_uses_the_data_dir_and_results_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def fake_eval(feedback: Path, reports: Path, results_dir: Path) -> int:
        seen.update(feedback=feedback, reports=reports, results_dir=results_dir)
        return 0

    class FakeSettings:
        data_dir = tmp_path

    monkeypatch.setattr(cli, "_eval_alerts", fake_eval)
    monkeypatch.setattr(cli, "get_settings", lambda: FakeSettings())

    assert cli.main(["eval-alerts"]) == 0
    assert seen == {
        "feedback": tmp_path / "feedback",
        "reports": tmp_path / "reports",
        "results_dir": cli.RESULTS_DIR,
    }


# --- the judge against your verdicts ---------------------------------------------------------


def judgment(verdict: JudgeVerdict, reason: str = "because") -> Judgment:
    return Judgment(verdict=verdict, reason=reason, model="fake", input_tokens=1, output_tokens=1)


@pytest.mark.parametrize(
    ("human", "judge", "expected"),
    [
        ("useful", "supported", True),
        ("not_useful", "supported", True),  # importance is your call; the text was faithful
        ("wrong", "supported", False),
        ("wrong", "overstated", True),
        ("wrong", "unsupported", True),
        ("useful", "overstated", False),
        ("not_useful", "unsupported", False),
    ],
)
def test_judge_agrees_when_wrong_matches_not_supported(
    human: Verdict, judge: JudgeVerdict, expected: bool
) -> None:
    assert judge_agrees(human, judge) is expected


def test_scores_compare_each_rated_alert_with_its_judgment() -> None:
    judgments = {
        REPORT_A: {
            "~1": judgment("supported"),
            "~5": judgment("overstated", "says will, filing says may"),
            "~19": None,  # the judge call failed on this one
        }
    }

    [a, b] = score_reports(A + B, judgments)

    assert [(j.id, j.human, j.judge, j.agrees) for j in a.judged] == [
        ("~1", "useful", "supported", True),
        ("~5", "not_useful", "overstated", False),
        ("~19", "wrong", None, None),
    ]
    assert (a.judge_compared, a.judge_agreed) == (2, 1)
    assert [(d.id, d.judge_reason) for d in a.disagreements] == [
        ("~5", "says will, filing says may")
    ]
    # Report B has no judgments at all (it predates the judge): compared with nothing.
    assert [j.judge for j in b.judged] == [None, None]
    assert (b.judge_compared, b.judge_agreed, b.disagreements) == (0, 0, ())

    result = summarize_alerts([a, b], ANSWERED)
    assert (result.judge_compared, result.judge_agreed) == (2, 1)
    assert result.judge_agreement == pytest.approx(0.5)


def test_without_judgments_nothing_is_compared() -> None:
    result = summarize_alerts(score_reports(A + B), ANSWERED)

    assert (result.judge_compared, result.judge_agreed, result.judge_agreement) == (0, 0, None)


def high_item(change_id: str, judged: Judgment | None) -> HighItem:
    return HighItem(
        id=change_id,
        triage_reason="reason",
        explanation="WHAT CHANGED: x\nWHY IT MATTERS: y",
        error=None,
        parsed=None,
        verification=None,
        input_tokens=1,
        output_tokens=1,
        judgment=judged,
    )


def save_report(
    directory: Path, name: str, *items: HighItem, versions: PromptVersions | None = None
) -> None:
    ticker, form, accession, section = name.removesuffix(".json").split("_")
    part, item = section.split("-")
    report = SectionReport(
        company=ticker,
        form=form,
        section_title="Part I Item 1A. RISK FACTORS",
        change_count=len(items),
        high_count=len(items),
        medium_count=0,
        low_count=0,
        high=items,
        not_explained=(),
        medium=(),
        input_tokens=0,
        output_tokens=0,
        cost_usd=None,
        usage_complete=True,
        prompt_versions=versions,
    )
    filing = Filing(
        cik="0000000001",
        accession_number=accession,
        form=form,
        filing_date=date(2026, 1, 29),
        report_date=None,
        primary_document="x.htm",
    )
    result = CheckResult(
        checked_at=CHECKED[name],
        entry=WatchEntry(ticker=ticker, form=form, item=item, part=part),  # type: ignore[arg-type]
        old_filing=filing,
        new_filing=filing,
        report=report,
        decision=Decision(entries=()),
    )
    assert result.file_name == name
    save_check_result(result, directory)


def test_load_reports_reads_the_reports_that_exist(tmp_path: Path) -> None:
    save_report(tmp_path, REPORT_A, high_item("~1", judgment("supported")), high_item("~5", None))

    loaded = judgments_in(load_reports(tmp_path, [REPORT_A, REPORT_B, REPORT_A]))

    assert list(loaded) == [REPORT_A]  # B is not on disk: its alerts are simply not judged
    assert loaded[REPORT_A]["~1"] is not None and loaded[REPORT_A]["~1"].verdict == "supported"
    assert loaded[REPORT_A]["~5"] is None

    (tmp_path / REPORT_B).write_text("{broken", encoding="utf-8")
    with pytest.raises(FeedbackError, match="can't read report"):
        load_reports(tmp_path, [REPORT_B])


def test_command_prints_the_judge_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    feedback, reports, results = tmp_path / "feedback", tmp_path / "reports", tmp_path / "results"
    write_feedback(feedback, A, B)
    save_report(
        reports,
        REPORT_A,
        high_item("~1", judgment("supported")),
        high_item("~5", judgment("supported")),
        high_item("~19", judgment("supported", "follows from the filing")),  # you said wrong
    )

    assert _eval_alerts(feedback, reports, results) == 0

    out = capsys.readouterr().out
    assert "  judge agrees with you on 2/3 alert(s)" in out
    assert (
        "  judge disagrees [~19]: you wrong, judge supported\n    judge: follows from the filing"
        in out
    )
    assert "judge agreement:  67%  (2/3 judged alerts; 2 not judged)" in out
    [saved] = list(results.iterdir())
    result = AlertEvalResult.model_validate_json(saved.read_text(encoding="utf-8"))
    assert (result.judge_compared, result.judge_agreed) == (3, 2)


V1 = PromptVersions(model="m", triage="t1", agent="a1", judge="j1")
V2 = PromptVersions(model="m", triage="t1", agent="a2", judge="j1")  # only the agent changed
C = [
    answer(REPORT_C, "+1", "alert", "wrong", "made up"),
    answer(REPORT_C, "+9", "digest", "useful"),
]


def test_scores_carry_their_report_versions() -> None:
    scores = score_reports(A + B, versions={REPORT_A: None, REPORT_B: V1})

    assert [score.versions for score in scores] == [None, V1]
    assert [score.versions for score in score_reports(A)] == [None]  # no versions given


def test_versions_group_reports_in_the_order_each_was_first_used() -> None:
    scores = score_reports(A + B + C, versions={REPORT_A: V2, REPORT_B: V1, REPORT_C: V2})

    v2, v1 = score_versions(scores)

    assert (v2.versions, v2.reports) == (V2, (REPORT_A, REPORT_C))
    # A: ~1 useful, ~5 not_useful, ~19 wrong; C: +1 wrong. Missed: A's +3 and C's +9.
    assert (v2.alerts_rated, v2.useful, v2.wrong, v2.missed) == (4, 1, 2, 2)
    assert (v2.precision, v2.wrong_rate) == (0.25, 0.5)
    assert (v1.versions, v1.reports) == (V1, (REPORT_B,))
    assert (v1.alerts_rated, v1.useful, v1.wrong, v1.missed) == (2, 2, 0, 0)
    assert (v1.precision, v1.wrong_rate) == (1.0, 0.0)


def test_reports_without_versions_form_one_unknown_group() -> None:
    scores = score_reports(A + B + C, versions={REPORT_C: V1})

    unknown, v1 = score_versions(scores)

    assert (unknown.versions, unknown.reports) == (None, (REPORT_A, REPORT_B))
    assert (v1.versions, v1.reports) == (V1, (REPORT_C,))
    # The groups split the totals: nothing counted twice or lost.
    total = summarize_alerts(scores, ANSWERED)
    assert total.by_version == (unknown, v1)
    assert unknown.alerts_rated + v1.alerts_rated == total.alerts_rated
    assert unknown.useful + v1.useful == total.useful


def test_versions_in_reads_each_saved_report(tmp_path: Path) -> None:
    save_report(tmp_path, REPORT_A, high_item("~1", None))
    save_report(tmp_path, REPORT_B, high_item("+1", None), versions=V1)

    assert versions_in(load_reports(tmp_path, [REPORT_A, REPORT_B])) == {
        REPORT_A: None,
        REPORT_B: V1,
    }


def test_command_prints_the_numbers_per_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    feedback, reports, results = tmp_path / "feedback", tmp_path / "reports", tmp_path / "results"
    write_feedback(feedback, A, B)
    save_report(reports, REPORT_A, high_item("~1", None))  # saved before versions were kept
    save_report(
        reports,
        REPORT_B,
        high_item("+1", judgment("supported")),
        high_item("+2", judgment("supported")),
        versions=V1,
    )

    assert _eval_alerts(feedback, reports, results) == 0

    out = capsys.readouterr().out
    by_version = out[out.index("=== BY VERSION (2) ===") :]
    assert (
        "unknown (saved before versions)\n"
        f"  1 report(s): {REPORT_A}\n"
        "  alerts 3   precision 33%   wrong 33%   missed 1   judge agrees -\n"
    ) in by_version
    assert (
        "model m, triage t1, agent a1, judge j1\n"
        f"  1 report(s): {REPORT_B}\n"
        "  alerts 2   precision 100%   wrong 0%   missed 0   judge agrees 2/2\n"
    ) in by_version
    assert "nothing to compare yet" not in out
    [saved] = list(results.iterdir())
    result = AlertEvalResult.model_validate_json(saved.read_text(encoding="utf-8"))
    assert [group.versions for group in result.by_version] == [None, V1]


def test_command_says_when_there_is_one_version_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    feedback = tmp_path / "feedback"
    write_feedback(feedback, B)

    assert _eval_alerts(feedback, tmp_path / "reports", tmp_path / "results") == 0

    out = capsys.readouterr().out
    assert "=== BY VERSION (1) ===\nunknown (saved before versions)" in out
    assert "one set of versions so far: nothing to compare yet" in out


def test_the_judge_gate_counts_rated_alerts_it_would_send_to_review() -> None:
    judgments = {
        REPORT_A: {
            "~1": judgment("unsupported"),  # you said useful: a cost of the gate
            "~5": judgment("supported"),  # stays an alert
            "~19": judgment("overstated"),  # you said wrong: caught
        }
    }

    scores = score_reports(A + B, judgments)
    result = summarize_alerts(scores, ANSWERED)

    [a, b] = scores
    assert [alert.id for alert in a.gated] == ["~1", "~19"]
    assert b.gated == ()  # not judged: the gate doesn't touch it
    assert (result.gate_demoted, result.gate_caught) == (2, 1)


def test_command_prints_the_judge_gate_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    feedback, reports, results = tmp_path / "feedback", tmp_path / "reports", tmp_path / "results"
    write_feedback(feedback, A, B)
    save_report(
        reports,
        REPORT_A,
        high_item("~1", judgment("supported")),
        high_item("~5", judgment("supported")),
        high_item("~19", judgment("overstated", "says violence caused damage")),
    )

    assert _eval_alerts(feedback, reports, results) == 0

    out = capsys.readouterr().out
    assert (
        "judge gate:       demotes 1/5 rated alerts to review: "
        "1 you marked wrong (caught), 0 useful or not_useful (cost)"
    ) in out
