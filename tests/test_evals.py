import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pytest

from finscope.__main__ import _run_triage_eval
from finscope.config import Settings
from finscope.diff import ParagraphChange, SectionDiff
from finscope.evals import Importance, coverage_of, prompt_version, score_run, score_triage
from finscope.labels import ChangeLabel, SectionLabels, fingerprint
from finscope.llm import LLMResult
from finscope.triage import SCHEMA, SYSTEM_PROMPT

IDS = ["+1", "+2", "-1", "~1", "~2"]


def test_perfect_agreement() -> None:
    human: dict[str, Importance] = {"+1": "high", "+2": "medium", "-1": "low"}

    run = score_run(human, human)

    assert run.agreement == 1.0
    assert run.high_recall == 1.0
    assert run.high_precision == 1.0
    assert run.critical_misses == ()


def test_human_high_rated_low_is_a_critical_miss() -> None:
    gold: dict[str, Importance] = {"+1": "high", "+2": "high", "-1": "low"}
    model: dict[str, Importance] = {"+1": "low", "+2": "medium", "-1": "low"}

    run = score_run(gold, model)

    assert run.critical_misses == ("+1",)  # +2 (high -> medium) is a miss, but not critical
    assert run.high_recall == 0.0
    assert run.agreement == pytest.approx(1 / 3)


def test_unsure_and_unlabeled_are_excluded_and_counted() -> None:
    human = {"+1": "high", "+2": "unsure", "-1": "low"}  # ~1 and ~2 unlabeled
    model: dict[str, Importance] = {
        "+1": "high",
        "+2": "low",
        "-1": "low",
        "~1": "high",
        "~2": "low",
    }

    score = score_triage(IDS, human, [model])

    assert score.coverage == coverage_of(IDS, human)
    assert (score.coverage.scored, score.coverage.unsure, score.coverage.unlabeled) == (2, 1, 2)
    assert score.coverage.describe() == "3/5 labeled, 1 unsure (2 scored)"
    run = score.runs[0]
    assert run is not None
    assert run.agreement == 1.0  # +2 and ~1 would be wrong, but they aren't scored
    assert run.high_precision == 1.0  # model-high ~1 is unlabeled, so it doesn't count


def test_no_human_high_means_recall_is_none() -> None:
    gold: dict[str, Importance] = {"+1": "medium", "-1": "low"}

    run = score_run(gold, {"+1": "high", "-1": "low"})

    assert run.high_recall is None
    assert run.high_precision == 0.0


def test_model_rating_nothing_high_means_precision_is_none() -> None:
    gold: dict[str, Importance] = {"+1": "high", "-1": "low"}

    run = score_run(gold, {"+1": "medium", "-1": "low"})

    assert run.high_precision is None
    assert run.high_recall == 0.0


def test_nothing_scored_gives_none_not_a_crash() -> None:
    score = score_triage(IDS, {"+1": "unsure"}, [dict.fromkeys(IDS, "low")])

    assert score.agreement.mean is None and score.high_recall.mean is None


def test_runs_that_disagree_are_listed_as_unstable() -> None:
    human = {"+1": "high", "-1": "low"}
    first: dict[str, Importance] = {
        "+1": "high",
        "+2": "low",
        "-1": "low",
        "~1": "low",
        "~2": "low",
    }
    second = {**first, "-1": "medium"}

    score = score_triage(IDS, human, [first, second])

    [unstable] = score.unstable
    assert (unstable.id, unstable.human, unstable.ratings) == ("-1", "low", ("low", "medium"))
    assert score.agreement.mean == 0.75
    assert (score.agreement.min, score.agreement.max) == (0.5, 1.0)


def test_failed_run_is_kept_as_none_and_skipped_in_the_numbers() -> None:
    human = {"+1": "high"}
    ok: dict[str, Importance] = dict.fromkeys(IDS, "low")

    score = score_triage(IDS, human, [ok, None])

    assert score.runs[1] is None
    assert score.unstable == ()  # one successful run can't disagree with itself
    [miss] = score.critical_misses
    assert (miss.id, miss.ratings) == ("+1", ("low", None))


def test_confusion_matrix_counts() -> None:
    gold: dict[str, Importance] = {"a": "high", "b": "high", "c": "medium", "d": "low", "e": "low"}
    model: dict[str, Importance] = {"a": "high", "b": "low", "c": "high", "d": "low", "e": "medium"}

    run = score_run(gold, model)

    assert run.confusion == {
        "high": {"high": 1, "medium": 0, "low": 1},
        "medium": {"high": 1, "medium": 0, "low": 0},
        "low": {"high": 0, "medium": 1, "low": 1},
    }
    # Summed over runs: the same run twice doubles every cell.
    score = score_triage(list(gold), gold, [model, model])
    assert score.confusion["high"] == {"high": 2, "medium": 0, "low": 2}


def test_prompt_version_changes_with_the_prompt_or_schema() -> None:
    version = prompt_version(SYSTEM_PROMPT, SCHEMA)

    assert len(version) == 12
    assert prompt_version(SYSTEM_PROMPT, SCHEMA) == version  # stable
    assert prompt_version(SYSTEM_PROMPT + " Be strict.", SCHEMA) != version
    assert prompt_version(SYSTEM_PROMPT, {**SCHEMA, "required": []}) != version


def test_scoring_module_does_not_import_openai() -> None:
    code = "import sys, finscope.evals; print('openai' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "False"


# --- CLI end to end, with a fake provider -------------------------------------------------------


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
LABELS = SectionLabels(
    company="TSLA",
    form="10-K",
    part="I",
    item="1A",
    new_accession="0001628280-26-003952",
    old_accession="0001628280-25-003063",
    labels=tuple(
        ChangeLabel(id=change_id, fingerprint=fingerprint(c), label=label)
        for change_id, c, label in [
            ("+1", ROBOTAXI, "high"),
            ("-1", CEO, "low"),
            ("~1", DEMAND, "unsure"),
        ]
    ),
)
SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_model="fake-model",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)


def triage_answer(**importance: str) -> str:
    ids = {"added": "+1", "removed": "-1", "modified": "~1"}
    items = [
        {"id": ids[kind], "importance": level, "reason": f"{kind} looks {level}."}
        for kind, level in importance.items()
    ]
    return json.dumps({"items": items})


class FakeProvider:
    """complete_json returns the next canned answer: one per triage run."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)

    def complete_json(
        self, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> LLMResult:
        return LLMResult(
            text=self.answers.pop(0), model="fake-model", input_tokens=1_000, output_tokens=200
        )


def test_cli_runs_scores_and_saves_the_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = FakeProvider(
        triage_answer(added="high", removed="low", modified="low"),
        triage_answer(added="low", removed="low"),  # ~1 missing: the run fails
        triage_answer(added="medium", removed="low", modified="high"),
    )

    exit_code = _run_triage_eval(
        DIFF,
        LABELS,
        tmp_path / "labels" / LABELS.file_name,
        provider=provider,
        settings=SETTINGS,
        runs=3,
        results_dir=tmp_path / "results",
    )

    assert exit_code == 0
    [path] = (tmp_path / "results").iterdir()  # one file, no .partial left behind
    assert path.name.endswith("Z_TSLA_10-K_0001628280-26-003952_I-1A.json")
    saved = json.loads(path.read_text())
    assert saved["model"] == "fake-model"
    assert saved["prompt_version"] == prompt_version(SYSTEM_PROMPT, SCHEMA)
    assert saved["labels_file"] == LABELS.file_name
    assert [run["error"] is None for run in saved["runs"]] == [True, False, True]
    assert "missing: ~1" in saved["runs"][1]["error"]
    assert saved["runs"][0]["items"][0] == {
        "id": "+1",
        "importance": "high",
        "reason": "added looks high.",
    }
    assert saved["score"]["coverage"] == {"changes": 3, "scored": 2, "unsure": 1, "unlabeled": 0}
    assert [run and run["agreement"] for run in saved["score"]["runs"]] == [1.0, None, 0.5]
    assert saved["score"]["unstable"] == [
        {"id": "+1", "human": "high", "ratings": ["high", None, "medium"]},
        {"id": "~1", "human": "unsure", "ratings": ["low", None, "high"]},
    ]
    assert (saved["input_tokens"], saved["output_tokens"]) == (2_000, 400)  # successful runs
    assert Decimal(saved["cost_usd"]) == Decimal("0.0072")

    out = capsys.readouterr().out
    assert "Running triage 3 time(s): 3 API call(s)." in out
    assert out.index("API call(s)") < out.index("run 1/3")  # announced before calling
    assert "run 2: FAILED" in out
    assert "mean:  agreement 75% (50%-100%)" in out
    assert "[+1] high, failed, medium   (human: high)" in out


def test_cli_refuses_when_nothing_is_scorable(tmp_path: Path) -> None:
    only_unsure = LABELS.model_copy(update={"labels": LABELS.labels[2:]})
    provider = FakeProvider()  # no answers: any call would fail the test

    exit_code = _run_triage_eval(
        DIFF,
        only_unsure,
        tmp_path / LABELS.file_name,
        provider=provider,
        settings=SETTINGS,
        runs=3,
        results_dir=tmp_path / "results",
    )

    assert exit_code == 1
    assert not (tmp_path / "results").exists()
