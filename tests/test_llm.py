from typing import Any, cast

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
from finscope.llm import LLMError, OpenAIProvider, ToolCall, ToolResult, ToolSpec

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
