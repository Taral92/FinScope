from decimal import Decimal

import pytest

from finscope import judge, triage
from finscope.config import Settings
from finscope.versions import (
    agent_prompt_version,
    current_versions,
    judge_prompt_version,
    triage_prompt_version,
)

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_model="fake-model",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)


def test_current_versions_name_the_model_and_every_prompt() -> None:
    versions = current_versions(SETTINGS, judged=True)

    assert versions.model == "fake-model"
    assert versions.triage == triage_prompt_version()
    assert versions.agent == agent_prompt_version()
    assert versions.judge == judge_prompt_version()
    assert all(len(v) == 12 for v in (versions.triage, versions.agent, versions.judge))
    # Three different prompts, three different versions.
    assert len({versions.triage, versions.agent, versions.judge}) == 3


def test_no_judge_version_when_no_judge_runs() -> None:
    assert current_versions(SETTINGS, judged=False).judge is None


def test_editing_a_prompt_changes_only_its_own_version(monkeypatch: pytest.MonkeyPatch) -> None:
    before = current_versions(SETTINGS, judged=True)

    monkeypatch.setattr(triage, "SYSTEM_PROMPT", triage.SYSTEM_PROMPT + "\nBe strict.")
    after = current_versions(SETTINGS, judged=True)

    assert after.triage != before.triage
    assert (after.agent, after.judge, after.model) == (before.agent, before.judge, before.model)


def test_editing_the_judge_schema_changes_its_version(monkeypatch: pytest.MonkeyPatch) -> None:
    before = judge_prompt_version()

    monkeypatch.setattr(judge, "SCHEMA", {**judge.SCHEMA, "description": "changed"})

    assert judge_prompt_version() != before


def test_the_judge_model_is_recorded_only_when_it_differs() -> None:
    own = SETTINGS.model_copy(update={"judge_model": "judge-x"})
    same = SETTINGS.model_copy(update={"judge_model": SETTINGS.llm_model})

    assert current_versions(own, judged=True).judge_model == "judge-x"
    assert current_versions(own, judged=False).judge_model is None  # no judge ran
    assert current_versions(same, judged=True).judge_model is None
    assert current_versions(SETTINGS, judged=True).judge_model is None


def test_the_label_names_a_separate_judge_model() -> None:
    own = current_versions(SETTINGS.model_copy(update={"judge_model": "judge-x"}), judged=True)
    shared = current_versions(SETTINGS, judged=True)

    assert own.label.endswith(f"judge {judge_prompt_version()} on judge-x")
    assert shared.label.endswith(f"judge {judge_prompt_version()}")
    assert own != shared  # so eval-alerts keeps reports from the two judges apart
