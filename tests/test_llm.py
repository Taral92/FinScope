from typing import Any, cast

import pytest
from openai import OpenAI
from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from finscope.config import Settings
from finscope.llm import LLMError, OpenAIProvider

SETTINGS = Settings(_env_file=None, sec_user_agent="FinScope Test test@example.com")


def make_response(
    content: list[Any],
    *,
    status: str = "completed",
    incomplete_reason: str | None = None,
) -> Response:
    """A real SDK Response object, built offline without validation."""
    message = ResponseOutputMessage.model_construct(
        id="msg_1", type="message", role="assistant", status="completed", content=content
    )
    usage = ResponseUsage.model_construct(
        input_tokens=1_200,
        output_tokens=150,  # includes the 40 reasoning tokens below
        total_tokens=1_350,
        input_tokens_details=InputTokensDetails.model_construct(cached_tokens=0),
        output_tokens_details=OutputTokensDetails.model_construct(reasoning_tokens=40),
    )
    response = Response.model_construct(
        id="resp_1",
        model="gpt-6.1-terra",
        status=status,
        output=[message],
        incomplete_details=(
            IncompleteDetails.model_construct(reason=incomplete_reason)
            if incomplete_reason
            else None
        ),
        error=None,
        usage=usage,
    )
    response._request_id = "req_123"
    return response


def text(value: str) -> ResponseOutputText:
    return ResponseOutputText.model_construct(type="output_text", text=value, annotations=[])


class FakeResponses:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Response:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    """Stands in for openai.OpenAI: only `client.responses.create` is used."""

    def __init__(self, response: Response) -> None:
        self.responses = FakeResponses(response)


def provider_returning(response: Response) -> tuple[OpenAIProvider, FakeClient]:
    client = FakeClient(response)
    return OpenAIProvider(SETTINGS, client=cast(OpenAI, client)), client


def test_text_usage_and_request_id_are_extracted() -> None:
    provider, client = provider_returning(make_response([text("  WHAT CHANGED: ...  ")]))

    result = provider.complete("system rules", "user question")

    assert result.text == "WHAT CHANGED: ..."
    assert result.model == "gpt-6.1-terra"
    assert (result.input_tokens, result.output_tokens) == (1_200, 150)
    assert result.request_id == "req_123"
    [call] = client.responses.calls
    assert (call["instructions"], call["input"]) == ("system rules", "user question")
    assert call["model"] == SETTINGS.llm_model


def test_refusal_raises() -> None:
    refusal = ResponseOutputRefusal.model_construct(type="refusal", refusal="I can't help.")
    provider, _ = provider_returning(make_response([refusal]))

    with pytest.raises(LLMError, match="refused: I can't help"):
        provider.complete("system", "user")


def test_incomplete_response_raises_with_reason() -> None:
    response = make_response(
        [text("WHAT CHANGED: The para")], status="incomplete", incomplete_reason="max_output_tokens"
    )
    provider, _ = provider_returning(response)

    with pytest.raises(LLMError, match="incomplete: max_output_tokens"):
        provider.complete("system", "user")


def test_empty_answer_raises() -> None:
    provider, _ = provider_returning(make_response([text("   ")]))

    with pytest.raises(LLMError, match="empty answer"):
        provider.complete("system", "user")


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None, sec_user_agent="FinScope Test test@example.com")

    with pytest.raises(LLMError, match="OPENAI_API_KEY is not set"):
        OpenAIProvider(settings)
