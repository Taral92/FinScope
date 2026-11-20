from decimal import Decimal
from typing import Any, cast

import httpx2
import openai
import pytest
from openai import OpenAI, omit
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from finscope.config import Settings
from finscope.llm import (
    LLMError,
    OpenAIProvider,
    ToolCall,
    ToolResult,
    ToolSpec,
    cost_for_tokens,
    judge_cost_for_tokens,
    judge_model,
    judge_provider,
    split_cost,
)

SETTINGS = Settings(_env_file=None, sec_user_agent="FinScope Test test@example.com")


def make_response(
    content: list[Any],
    *,
    status: str = "completed",
    incomplete_reason: str | None = None,
    output: list[Any] | None = None,
) -> Response:
    """A real SDK Response object, built offline without validation.

    `content` fills one assistant message; pass `output` instead for other output items.
    """
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
        output=output if output is not None else [message],
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
    """Returns the given responses in order and records every create() call."""

    def __init__(self, responses: list[Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Response:
        # Snapshot the input list: the provider builds a new one each turn, but be safe.
        self.calls.append({**kwargs, "input": _copy(kwargs.get("input"))})
        return self.responses.pop(0)


def _copy(value: Any) -> Any:
    return list(value) if isinstance(value, list) else value


class FakeClient:
    """Stands in for openai.OpenAI: only `client.responses.create` is used."""

    def __init__(self, *responses: Response) -> None:
        self.responses = FakeResponses(list(responses))


def provider_returning(*responses: Response) -> tuple[OpenAIProvider, FakeClient]:
    client = FakeClient(*responses)
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
    assert call["tools"] is omit  # complete() sends no tools, exactly as before
    assert call["text"] is omit  # ...and no output format: plain text


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


# --- Structured output --------------------------------------------------------------------------

LEVEL_SCHEMA = {
    "type": "object",
    "properties": {"level": {"type": "string", "enum": ["low", "high"]}},
    "required": ["level"],
    "additionalProperties": False,
}


def test_complete_json_sends_the_schema_in_the_documented_shape() -> None:
    provider, client = provider_returning(make_response([text('{"level": "high"}')]))

    result = provider.complete_json("system rules", "user question", "rating", LEVEL_SCHEMA)

    assert result.text == '{"level": "high"}'  # the raw JSON; callers parse it
    assert (result.input_tokens, result.output_tokens) == (1_200, 150)
    [call] = client.responses.calls
    assert (call["instructions"], call["input"]) == ("system rules", "user question")
    # Responses API Structured Outputs: text.format with type json_schema and strict mode.
    assert call["text"] == {
        "format": {"type": "json_schema", "name": "rating", "schema": LEVEL_SCHEMA, "strict": True}
    }
    assert call["tools"] is omit


def test_complete_json_refusal_raises() -> None:
    # A refusal doesn't follow the schema: it is a separate content type.
    refusal = ResponseOutputRefusal.model_construct(type="refusal", refusal="I can't help.")
    provider, _ = provider_returning(make_response([refusal]))

    with pytest.raises(LLMError, match="refused: I can't help"):
        provider.complete_json("system", "user", "rating", LEVEL_SCHEMA)


def test_complete_json_truncated_answer_raises() -> None:
    # Cut off by max_output_tokens: the JSON would be incomplete, so it is never returned.
    response = make_response(
        [text('{"level": ')], status="incomplete", incomplete_reason="max_output_tokens"
    )
    provider, _ = provider_returning(response)

    with pytest.raises(LLMError, match="incomplete: max_output_tokens"):
        provider.complete_json("system", "user", "rating", LEVEL_SCHEMA)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None, sec_user_agent="FinScope Test test@example.com")

    with pytest.raises(LLMError, match="OPENAI_API_KEY is not set"):
        OpenAIProvider(settings)


# --- Retries and timeouts (the SDK does the retrying; we configure and report it) -------------


def test_retries_and_timeout_reach_the_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []

    def fake_openai(**kwargs: Any) -> FakeClient:
        created.append(kwargs)
        return FakeClient()

    monkeypatch.setattr("finscope.llm.OpenAI", fake_openai)
    settings = Settings(
        _env_file=None,
        sec_user_agent="FinScope Test test@example.com",
        OPENAI_API_KEY="sk-test",
        llm_max_retries=5,
        llm_timeout_seconds=30,
    )

    OpenAIProvider(settings)

    [kwargs] = created
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["max_retries"] == 5
    assert kwargs["timeout"] == 30


class FailingClient:
    """Stands in for openai.OpenAI after the SDK has given up: create() raises."""

    def __init__(self, error: Exception) -> None:
        self.responses = self
        self._error = error

    def create(self, **kwargs: Any) -> Response:
        raise self._error


def _api_error(error_type: type[openai.APIStatusError], status: int, message: str) -> Exception:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx2.Response(status, request=request, json={"error": {"message": message}})
    return error_type(message, response=response, body=None)


def test_rate_limit_failure_says_it_was_retried() -> None:
    client = FailingClient(_api_error(openai.RateLimitError, 429, "Rate limit reached"))
    provider = OpenAIProvider(SETTINGS, client=cast(OpenAI, client))

    with pytest.raises(LLMError) as info:
        provider.complete("system", "user")

    # SETTINGS has the default 3 retries: 4 attempts in all.
    expected = "OpenAI request failed after 4 attempt(s): HTTP 429 Rate limit reached"
    assert str(info.value) == expected
    assert isinstance(info.value.__cause__, openai.RateLimitError)


def test_timeout_failure_says_it_was_retried() -> None:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    client = FailingClient(openai.APITimeoutError(request=request))
    provider = OpenAIProvider(SETTINGS, client=cast(OpenAI, client))

    with pytest.raises(LLMError, match=r"after 4 attempt\(s\): Request timed out"):
        provider.complete("system", "user")


def test_non_retryable_failure_does_not_claim_retries() -> None:
    client = FailingClient(_api_error(openai.BadRequestError, 400, "Unsupported parameter"))
    provider = OpenAIProvider(SETTINGS, client=cast(OpenAI, client))

    with pytest.raises(LLMError, match=r"^OpenAI request failed: ") as info:
        provider.complete("system", "user")

    assert "attempt" not in str(info.value)


# --- Tool calling ---------------------------------------------------------------------------

METRICS_TOOL = ToolSpec(
    name="get_key_metrics",
    description="Key XBRL metrics.",
    parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
)
REASONING = ResponseReasoningItem.model_construct(id="rs_1", type="reasoning", summary=[])


def function_call(arguments: str = "{}") -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall.model_construct(
        type="function_call",
        id="fc_1",
        call_id="call_1",
        name="get_key_metrics",
        arguments=arguments,
        status="completed",
    )


def test_function_call_is_parsed_into_a_tool_call() -> None:
    provider, client = provider_returning(make_response([], output=[REASONING, function_call()]))

    step, _ = provider.start("system rules", "user question", [METRICS_TOOL])

    assert step.text is None
    assert step.tool_calls == (
        ToolCall(id="call_1", name="get_key_metrics", arguments={}, raw_arguments="{}"),
    )
    [call] = client.responses.calls
    assert call["tools"] == [
        {
            "type": "function",
            "name": "get_key_metrics",
            "description": "Key XBRL metrics.",
            "parameters": METRICS_TOOL.parameters,
            "strict": True,
        }
    ]


def test_tool_results_go_back_in_the_documented_shape() -> None:
    call_turn = make_response([], output=[REASONING, function_call()])
    provider, client = provider_returning(call_turn, make_response([text("ANSWER")]))

    _, state = provider.start("system rules", "user question", [METRICS_TOOL])
    step, _ = provider.resume(state, [ToolResult(call_id="call_1", output="Revenue | $28,046M")])

    assert step.text == "ANSWER"
    second = client.responses.calls[1]
    # Instructions and tools are re-sent: they only apply to a single request.
    assert second["instructions"] == "system rules"
    assert second["tools"] == client.responses.calls[0]["tools"]
    # The history: our message, everything the model returned (reasoning item included),
    # then our function_call_output answering the call_id.
    assert second["input"] == [
        {"role": "user", "content": "user question"},
        REASONING,
        function_call(),
        {"type": "function_call_output", "call_id": "call_1", "output": "Revenue | $28,046M"},
    ]


def test_error_results_are_marked_in_the_output_text() -> None:
    call_turn = make_response([], output=[function_call()])
    provider, client = provider_returning(call_turn, make_response([text("ANSWER")]))

    _, state = provider.start("system", "user", [METRICS_TOOL])
    provider.resume(state, [ToolResult(call_id="call_1", output="unknown tool", is_error=True)])

    assert client.responses.calls[1]["input"][-1]["output"] == "ERROR: unknown tool"


def test_invalid_json_arguments_are_flagged_not_raised() -> None:
    provider, _ = provider_returning(make_response([], output=[function_call('{"count": ')]))

    step, _ = provider.start("system", "user", [METRICS_TOOL])

    [tool_call] = step.tool_calls
    assert tool_call.invalid_json
    assert (tool_call.arguments, tool_call.raw_arguments) == ({}, '{"count": ')


def test_turn_with_neither_text_nor_tool_calls_raises() -> None:
    provider, _ = provider_returning(make_response([], output=[REASONING]))

    with pytest.raises(LLMError, match="neither an answer nor tool calls"):
        provider.start("system", "user", [METRICS_TOOL])


# --- A judge on its own model (FINSCOPE_JUDGE_MODEL) ---------------------------------------------

PRICED = {"llm_input_price_per_mtok": Decimal("1"), "llm_output_price_per_mtok": Decimal("5")}


def judge_settings(**fields: Any) -> Settings:
    return Settings(_env_file=None, sec_user_agent="FinScope Test test@example.com", **fields)


def test_unset_judge_model_means_the_judge_uses_the_main_model() -> None:
    settings = judge_settings(**PRICED)

    assert settings.judge_model is None
    assert judge_model(settings) == settings.llm_model
    # Priced like everything else, so costs are exactly what they were before the setting.
    assert judge_cost_for_tokens(1_000, 100, settings) == cost_for_tokens(1_000, 100, settings)


def test_the_judge_model_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINSCOPE_JUDGE_MODEL", "judge-model-x")
    monkeypatch.setenv("FINSCOPE_JUDGE_INPUT_PRICE_PER_MTOK", "10")

    settings = judge_settings()

    assert judge_model(settings) == "judge-model-x"
    assert settings.judge_input_price_per_mtok == Decimal("10")


def test_a_provider_requests_the_model_it_was_given() -> None:
    client = FakeClient(make_response([text("ok")]))
    provider = OpenAIProvider(SETTINGS, client=cast(OpenAI, client), model="judge-model-x")

    provider.complete("system", "user")

    [call] = client.responses.calls
    assert call["model"] == "judge-model-x"


def test_judge_provider_runs_on_the_judge_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(make_response([text("ok")]), make_response([text("ok")]))
    monkeypatch.setattr("finscope.llm.OpenAI", lambda **kwargs: client)

    judge_provider(judge_settings(OPENAI_API_KEY="sk-test", judge_model="judge-x")).complete(
        "system", "user"
    )
    judge_provider(judge_settings(OPENAI_API_KEY="sk-test")).complete("system", "user")

    assert [call["model"] for call in client.responses.calls] == ["judge-x", SETTINGS.llm_model]


def test_judge_tokens_are_priced_at_the_judges_prices() -> None:
    settings = judge_settings(
        **PRICED,
        judge_model="judge-x",
        judge_input_price_per_mtok=Decimal("10"),
        judge_output_price_per_mtok=Decimal("50"),
    )

    # 1M input + 100k output in total, of which 200k input + 10k output went to the judge.
    cost = split_cost(
        1_000_000,
        100_000,
        judge_input_tokens=200_000,
        judge_output_tokens=10_000,
        settings=settings,
    )

    # Rest: 800k x $1 + 90k x $5 per 1M = 0.80 + 0.45; judge: 200k x $10 + 10k x $50 = 2 + 0.5
    assert cost == Decimal("0.80") + Decimal("0.45") + Decimal("2") + Decimal("0.5")


def test_a_judge_model_without_prices_makes_its_cost_unknown_but_only_when_used() -> None:
    settings = judge_settings(**PRICED, judge_model="judge-x")

    assert judge_cost_for_tokens(1_000, 100, settings) is None
    assert (
        split_cost(1_000, 100, judge_input_tokens=200, judge_output_tokens=10, settings=settings)
        is None
    )
    # No judge call, nothing to price at the missing rate: the rest is still known.
    assert split_cost(
        1_000, 100, judge_input_tokens=0, judge_output_tokens=0, settings=settings
    ) == cost_for_tokens(1_000, 100, settings)
