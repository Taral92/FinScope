"""Application settings, loaded from environment variables and an optional `.env` file."""

import re
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Deliberately loose: we only need to confirm a contact email is present, not validate it fully.
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


class Settings(BaseSettings):
    """FinScope configuration. Every field maps to a `FINSCOPE_`-prefixed env var."""

    model_config = SettingsConfigDict(
        env_prefix="FINSCOPE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SEC EDGAR rejects requests whose User-Agent lacks contact info,
    # e.g. "FinScope Jane Doe jane@example.com".
    sec_user_agent: str
    log_level: str = "INFO"

    @field_validator("sec_user_agent")
    @classmethod
    def _require_email(cls, value: str) -> str:
        if not _EMAIL_PATTERN.search(value):
            raise ValueError("must contain a contact email address, as required by SEC EDGAR")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first call."""
    return Settings()
