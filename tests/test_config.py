import pytest
from pydantic import ValidationError

from finscope.config import Settings


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
