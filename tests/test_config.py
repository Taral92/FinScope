from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from finscope.__main__ import main
from finscope.config import Settings, SettingsError, get_settings, settings_problem


def test_valid_user_agent_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINSCOPE_SEC_USER_AGENT", "FinScope Jane Doe jane@example.com")

    # _env_file=None keeps a developer's local .env from leaking into the test.
    settings = Settings(_env_file=None)

    assert settings.sec_user_agent == "FinScope Jane Doe jane@example.com"
    assert settings.log_level == "INFO"


def test_user_agent_without_email_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINSCOPE_SEC_USER_AGENT", "FinScope Jane Doe")

    with pytest.raises(ValidationError, match="contact email"):
        Settings(_env_file=None)


@pytest.mark.parametrize("rate", ["0", "-1", "10.5"])
def test_request_rate_outside_sec_limit_is_rejected(
    monkeypatch: pytest.MonkeyPatch, rate: str
) -> None:
    monkeypatch.setenv("FINSCOPE_SEC_USER_AGENT", "FinScope Jane Doe jane@example.com")
    monkeypatch.setenv("FINSCOPE_SEC_MAX_REQUESTS_PER_SECOND", rate)

    with pytest.raises(ValidationError, match="sec_max_requests_per_second"):
        Settings(_env_file=None)


# --- a plain message when the settings can't load ---------------------------------------------


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    """get_settings caches its result: clear it so each test loads from scratch."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_no_env_file_in_the_folder_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_settings: None
) -> None:
    monkeypatch.chdir(tmp_path)  # a folder with no .env, like a terminal opened elsewhere
    monkeypatch.delenv("FINSCOPE_SEC_USER_AGENT", raising=False)

    with pytest.raises(SettingsError) as raised:
        get_settings()

    message = str(raised.value)
    assert message.startswith(f"no .env in {tmp_path}.")
    assert "start it from the project folder" in message
    assert "ValidationError" not in message and "Field required" not in message


def test_an_invalid_env_file_names_the_file_and_the_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh_settings: None
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FINSCOPE_SEC_USER_AGENT", raising=False)
    (tmp_path / ".env").write_text('FINSCOPE_SEC_USER_AGENT="FinScope no email"\n', "utf-8")

    with pytest.raises(SettingsError) as raised:
        get_settings()

    message = str(raised.value)
    assert message.startswith(f"the settings in {tmp_path / '.env'} are not valid:")
    assert "contact email" in message  # pydantic's own reason, kept: it says what's wrong


def test_settings_problem_prefers_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as raised:
        Settings(_env_file=None, sec_user_agent="no email")

    assert settings_problem(raised.value, tmp_path).startswith("no .env in")
    (tmp_path / ".env").touch()
    assert settings_problem(raised.value, tmp_path).startswith("the settings in")


def test_the_cli_prints_one_error_line_instead_of_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fresh_settings: None,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FINSCOPE_SEC_USER_AGENT", raising=False)

    assert main(["feedback"]) == 1

    err = capsys.readouterr().err
    assert err.startswith(f"error: no .env in {tmp_path}.")
    assert "Traceback" not in err
