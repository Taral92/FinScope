import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import finscope.__main__ as cli
from finscope.config import Settings
from finscope.diff import changes_by_id, diff_sections, similarity
from finscope.known_cases import (
    KNOWN_CASES_PATH,
    CaseResult,
    KnownCase,
    KnownCasesError,
    build_known_diff,
    case_results,
    load_known_cases,
)
from finscope.llm import LLMResult
from finscope.sections import split_sections
from finscope.triage import triage_section

REPO = Path(__file__).resolve().parent.parent
CASES_FILE = REPO / KNOWN_CASES_PATH

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_model="fake-model",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)


def case(**fields: Any) -> dict[str, Any]:
    """A valid case as JSON-ready data; keyword arguments override fields."""
    return {
        "name": "going-concern",
        "kind": "added",
        "old_text": None,
        "new_text": "There is substantial doubt about our ability to continue as a going concern.",
        "expected": "high",
        "why": "It may not survive.",
    } | fields


def write_cases(tmp_path: Path, cases: Any) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    return path


# --- the committed case file -----------------------------------------------------------------


def test_case_file_loads_with_the_twelve_cases() -> None:
    cases = load_known_cases(CASES_FILE)

    assert {c.name for c in cases if c.expected == "high"} == {
        "going-concern",
        "tariffs-hit",
        "material-weakness",
        "sec-subpoena",
        "customer-concentration",
        "covenant-breach",
    }
    assert {c.name for c in cases if c.expected == "low"} == {
        "wording",
        "typo",
        "cross-reference",
        "reordered-list",
        "defined-term",
        "boilerplate",
    }
    assert all("Tesla" not in (c.old_text or "") + (c.new_text or "") for c in cases)


def test_fake_diff_matches_what_diff_sections_produces() -> None:
    """Put the cases into two fake filings and diff them for real: same changes, same IDs."""
    cases = load_known_cases(CASES_FILE)
    old = [c.old_text for c in cases if c.old_text]
    new = [c.new_text for c in cases if c.new_text]
    [old_section] = split_sections("ITEM 1A. RISK FACTORS\n\n" + "\n\n".join(old) + "\n")
    [new_section] = split_sections("ITEM 1A. RISK FACTORS\n\n" + "\n\n".join(new) + "\n")
    real = changes_by_id(diff_sections(old_section, new_section))

    known = build_known_diff(cases)
    fake = changes_by_id(known.diff)

    def without_offsets(change: Any) -> tuple[Any, ...]:
        return (change.kind, change.old_text, change.new_text, change.similarity)

    assert {i: without_offsets(c) for i, c in fake.items()} == {
        i: without_offsets(c) for i, c in real.items()
    }
    assert all(c.old_start is None and c.new_start is None for c in known.diff.changes)


# --- loading: a bad file fails clearly -------------------------------------------------------


def test_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(KnownCasesError, match="can't read"):
        load_known_cases(tmp_path / "nope.json")


def test_invalid_json_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text("[{")
    with pytest.raises(KnownCasesError, match="not a valid case file"):
        load_known_cases(path)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (case(expected="medium"), "expected"),
        (case(kind="changed"), "kind"),
        (case(why=""), "why"),
        (case(extra="field"), "extra"),
        (case(old_text="Old text."), "\\(added\\) needs old_text: null"),
        (case(new_text=None), "\\(added\\) needs new_text"),
        (case(kind="removed", old_text=None, new_text=None), "\\(removed\\) needs old_text"),
        (
            case(kind="modified", old_text="We utilize vendors.", new_text="We  utilize vendors."),
            "same words",
        ),
        (
            case(kind="modified", old_text="We utilize vendors.", new_text="Tariffs hurt margin."),
            "removed \\+ added",
        ),
    ],
)
def test_bad_case_is_rejected_with_its_reason(
    tmp_path: Path, bad: dict[str, Any], message: str
) -> None:
    with pytest.raises(KnownCasesError, match=message):
        load_known_cases(write_cases(tmp_path, [bad]))


def test_problem_names_the_case(tmp_path: Path) -> None:
    with pytest.raises(KnownCasesError, match="case 'going-concern'"):
        load_known_cases(write_cases(tmp_path, [case(old_text="Old text.")]))


def test_empty_file_duplicate_names_and_duplicate_changes_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(KnownCasesError, match="no cases"):
        load_known_cases(write_cases(tmp_path, []))
    with pytest.raises(KnownCasesError, match="duplicated names: going-concern"):
        load_known_cases(write_cases(tmp_path, [case(), case(new_text="Another paragraph.")]))
    with pytest.raises(KnownCasesError, match="same change in several cases: a, b"):
        load_known_cases(write_cases(tmp_path, [case(name="a"), case(name="b")]))


# --- the fake diff and the ID -> name mapping ------------------------------------------------


def modified(name: str, old: str, new: str, expected: str = "low") -> KnownCase:
    return KnownCase.model_validate(
        case(name=name, kind="modified", old_text=old, new_text=new, expected=expected)
    )


ADDED = KnownCase.model_validate(case(name="new-risk"))
REMOVED = KnownCase.model_validate(
    case(name="old-risk", kind="removed", old_text="An old risk.", new_text=None, expected="low")
)
SMALL_EDIT = modified(
    "small-edit",
    "We depend on our suppliers to recieve components on time.",
    "We depend on our suppliers to receive components on time.",
)
BIG_EDIT = modified(
    "big-edit",
    "Our credit agreement has covenants. We were in compliance with all of them.",
    "Our credit agreement has covenants. We did not comply with one of them.",
    expected="high",
)
CASES = (SMALL_EDIT, ADDED, BIG_EDIT, REMOVED)


def test_fake_diff_holds_every_case_with_similarity_for_modified_only() -> None:
    diff = build_known_diff(CASES).diff

    assert [c.kind for c in diff.changes] == ["modified", "added", "modified", "removed"]
    assert diff.changes[0].similarity == similarity(
        SMALL_EDIT.old_text or "", SMALL_EDIT.new_text or ""
    )
    assert diff.changes[1].similarity is None
    assert diff.changes[3].similarity is None
    assert {c.old_start for c in diff.changes} == {None}
    assert {c.new_end for c in diff.changes} == {None}


def test_ids_map_back_to_case_names() -> None:
    known = build_known_diff(CASES)

    # Added, removed, then modified least similar first: the big edit before the typo fix.
    assert known.ids == {"small-edit": "~2", "new-risk": "+1", "big-edit": "~1", "old-risk": "-1"}
    assert list(known.names) == ["~2", "+1", "~1", "-1"]  # file order, like ids
    assert known.names["~1"] == "big-edit"
    assert known.expected == {"~2": "low", "+1": "high", "~1": "high", "-1": "low"}
    for change_id, change in changes_by_id(known.diff).items():
        assert change == next(c for c in CASES if c.name == known.names[change_id]).change()


# --- PASS / FAIL -----------------------------------------------------------------------------


def test_pass_needs_every_run_to_match_exactly() -> None:
    assert CaseResult(name="x", expected="high", ratings=("high", "high", "high")).passed
    one_off = CaseResult(name="x", expected="high", ratings=("high", "medium", "high"))
    assert (one_off.passed, one_off.matches) == (False, 2)
    failed_run = CaseResult(name="x", expected="low", ratings=("low", None, "low"))
    assert (failed_run.passed, failed_run.matches) == (False, 2)
    assert not CaseResult(name="x", expected="low", ratings=()).passed


def test_case_results_follow_file_order_and_skip_failed_runs() -> None:
    known = build_known_diff(CASES)
    run = {"~2": "low", "+1": "high", "~1": "medium", "-1": "low"}

    results = case_results(known, [run, None])

    assert [r.name for r in results] == ["small-edit", "new-risk", "big-edit", "old-risk"]
    assert results[2].ratings == ("medium", None)


# --- triage on the fake diff, and the CLI ----------------------------------------------------


class FakeProvider:
    """complete_json returns the next canned answer: one per triage run."""

    def __init__(self, *answers: dict[str, str]) -> None:
        self.answers = list(answers)
        self.users: list[str] = []

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        self.users.append(user)
        items = [
            {"id": change_id, "importance": level, "reason": "Fake."}
            for change_id, level in self.answers.pop(0).items()
        ]
        return LLMResult(
            text=json.dumps({"items": items}),
            model="fake-model",
            input_tokens=1_000,
            output_tokens=200,
        )


def test_triage_runs_unchanged_on_the_fake_diff() -> None:
    known = build_known_diff(CASES)
    provider = FakeProvider({"+1": "high", "-1": "low", "~1": "high", "~2": "low"})

    report = triage_section(
        known.diff,
        company="Acme Motors",
        form="10-K",
        section_title="Part I Item 1A. Risk Factors",
        provider=provider,
        settings=SETTINGS,
    )

    assert [item.id for item in report.items] == ["+1", "-1", "~1", "~2"]
    assert "Company: Acme Motors" in provider.users[0]


def test_cli_prints_pass_fail_per_case_and_the_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_cases(tmp_path, [c.model_dump() for c in CASES])
    right = {"+1": "high", "-1": "low", "~1": "high", "~2": "low"}
    provider = FakeProvider(right, right | {"~1": "medium"}, right | {"+1": "low"})

    exit_code = cli._run_known_eval(path, provider=provider, settings=SETTINGS, runs=3)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Running triage 3 time(s): 3 API call(s)." in out
    assert "PASS 3/3  small-edit  expected low   got low, low, low" in out
    assert "FAIL 2/3  new-risk    expected high  got high, high, low" in out
    assert "FAIL 2/3  big-edit    expected high  got high, medium, high" in out
    assert "passed: 2/4 case(s)" in out
    assert "high recall" in out
    assert "[+1 new-risk] high, high, low   (human: high)" in out  # critical miss, by name
    assert "tokens: 3,000 input, 600 output" in out
    assert list(tmp_path.iterdir()) == [path]  # nothing saved


def test_cli_reports_a_bad_case_file_without_calling_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_cases(tmp_path, [])
    provider = FakeProvider()  # no answers: any call would fail the test

    exit_code = cli._run_known_eval(path, provider=provider, settings=SETTINGS, runs=3)

    assert exit_code == 1
    assert "no cases" in capsys.readouterr().err


def test_eval_known_needs_no_ticker_and_no_sec(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_sec() -> None:
        raise AssertionError("eval-known must not use the SEC client")

    monkeypatch.setattr(cli, "SecClient", no_sec)
    monkeypatch.setattr(cli, "_eval_known", lambda runs: runs)

    assert cli.main(["eval-known", "--runs", "2"]) == 2
