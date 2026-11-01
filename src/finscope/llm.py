"""A minimal LLM provider layer: one interface, one OpenAI implementation, cost in exact Decimal."""

from decimal import Decimal
from typing import Protocol

import openai
from openai import OpenAI
from openai.types.responses import Response
from openai.types.shared_params import ReasoningEffort
from pydantic import BaseModel, ConfigDict

from finscope.config import Settings

_TOKENS_PER_PRICE_UNIT = Decimal(1_000_000)  # prices are quoted per 1M tokens


class LLMError(Exception):
    """The model call failed or produced no usable answer."""


class LLMResult(BaseModel):
    """The model's answer plus what it cost in tokens."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    input_tokens: int
    # Includes any hidden reasoning tokens: reasoning models "think" before answering, and
    # those tokens are reported in usage.output_tokens_details.reasoning_tokens and billed
    # as output tokens, even though they never appear in the text.
    output_tokens: int
    request_id: str | None = None  # quote this when asking the provider's support about a call


class LLMProvider(Protocol):
    """Anything that can answer a system + user prompt. Lets callers (and tests) swap models."""

    def complete(self, system: str, user: str) -> LLMResult: ...


class OpenAIProvider:
    """LLMProvider backed by OpenAI's Responses API."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: OpenAI | None = None,
        max_output_tokens: int = 4_000,  # caps cost; includes reasoning tokens
        reasoning_effort: ReasoningEffort = "low",  # short explanations need little reasoning
    ) -> None:
        if client is None:
            if settings.openai_api_key is None:
                raise LLMError("OPENAI_API_KEY is not set (add it to .env)")
            client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._client = client
        self._model = settings.llm_model
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort

    def complete(self, system: str, user: str) -> LLMResult:
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=system,  # developer-level instructions, ranked above user input
                input=user,
                reasoning={"effort": self._reasoning_effort},
                max_output_tokens=self._max_output_tokens,
            )
        except openai.AuthenticationError:
            # Not chained or echoed: the error text can include part of the key.
            raise LLMError("OpenAI rejected the API key (check OPENAI_API_KEY)") from None
        except openai.APIError as error:
            raise LLMError(f"OpenAI request failed: {error}") from error
        return _to_result(response)


def _to_result(response: Response) -> LLMResult:
    """Turn a response into an LLMResult, or raise LLMError if there is no usable answer."""
    if response.status == "incomplete":
        details = response.incomplete_details
        reason = details.reason if details and details.reason else "unknown reason"
        raise LLMError(f"response incomplete: {reason}")
    if response.status not in (None, "completed"):
        message = response.error.message if response.error else "no details"
        raise LLMError(f"response {response.status}: {message}")

    # A refusal is its own content type; output_text would silently skip it and return "".
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "refusal":
                    raise LLMError(f"model refused: {content.refusal}")

    text = response.output_text.strip()
    if not text:
        raise LLMError("model returned an empty answer")
    if response.usage is None:
        raise LLMError("response has no token usage")
    return LLMResult(
        text=text,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        request_id=response._request_id,
    )


def cost_usd(result: LLMResult, settings: Settings) -> Decimal | None:
    """Exact cost from the configured prices, or None if they aren't configured (never guessed).

    Cached input tokens are billed at a lower rate; counting them at the full input price
    makes this an upper bound when the provider's prompt cache hits.
    """
    input_price = settings.llm_input_price_per_mtok
    output_price = settings.llm_output_price_per_mtok
    if input_price is None or output_price is None:
        return None
    return (
        result.input_tokens * input_price + result.output_tokens * output_price
    ) / _TOKENS_PER_PRICE_UNIT
