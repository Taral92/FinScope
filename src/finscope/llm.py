"""A minimal LLM provider layer: one interface, one OpenAI implementation, cost in exact Decimal."""

import json
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Protocol, cast

import openai
from openai import OpenAI, omit
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
)
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


# --- Tool calling, in provider-neutral terms -------------------------------------------------


class ToolSpec(BaseModel):
    """A tool the model may call: its name, what it does, and a JSON schema for its arguments."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """The model asking to run a tool."""

    model_config = ConfigDict(frozen=True)

    id: str  # pass back in ToolResult.call_id so the model knows which call a result answers
    name: str
    arguments: dict[str, Any]  # parsed JSON; {} when invalid_json
    raw_arguments: str  # exactly what the model sent
    invalid_json: bool = False


class ToolResult(BaseModel):
    """Our answer to one ToolCall."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    output: str
    is_error: bool = False


class AgentStep(BaseModel):
    """One model turn: an answer (text), more tool calls, or occasionally both."""

    model_config = ConfigDict(frozen=True)

    text: str | None
    tool_calls: tuple[ToolCall, ...]
    input_tokens: int
    output_tokens: int  # includes reasoning tokens, billed as output (see LLMResult)
    request_id: str | None = None


class ConversationState:
    """Opaque to callers: whatever a provider needs to continue a conversation. Pass it back."""


class LLMProvider(Protocol):
    """Anything that can answer prompts and run a tool-calling conversation."""

    def complete(self, system: str, user: str) -> LLMResult: ...

    def start(
        self, system: str, user: str, tools: list[ToolSpec]
    ) -> tuple[AgentStep, ConversationState]: ...

    def resume(
        self, state: ConversationState, tool_results: list[ToolResult]
    ) -> tuple[AgentStep, ConversationState]: ...


# --- OpenAI implementation ---------------------------------------------------------------------


@dataclass(frozen=True)
class _OpenAIState(ConversationState):
    # `instructions` and `tools` only apply to one request, so every turn re-sends them.
    system: str
    tools: tuple[FunctionToolParam, ...]
    # The whole conversation as input items: our messages, every item the model returned
    # (including reasoning items, which the docs require to be passed back), and tool outputs.
    items: tuple[Any, ...]


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
        return _to_result(self._create(system, user))

    def start(
        self, system: str, user: str, tools: list[ToolSpec]
    ) -> tuple[AgentStep, ConversationState]:
        state = _OpenAIState(
            system=system,
            tools=tuple(_function_tool(tool) for tool in tools),
            items=({"role": "user", "content": user},),
        )
        return self._step(state)

    def resume(
        self, state: ConversationState, tool_results: list[ToolResult]
    ) -> tuple[AgentStep, ConversationState]:
        if not isinstance(state, _OpenAIState):
            raise TypeError("conversation state was not created by OpenAIProvider")
        outputs = tuple(
            {
                "type": "function_call_output",
                "call_id": result.call_id,
                # The API has no error flag for tool outputs, so say it in the text.
                "output": f"ERROR: {result.output}" if result.is_error else result.output,
            }
            for result in tool_results
        )
        return self._step(replace(state, items=state.items + outputs))

    def _step(self, state: _OpenAIState) -> tuple[AgentStep, ConversationState]:
        response = self._create(
            state.system, cast(ResponseInputParam, list(state.items)), list(state.tools)
        )
        step = _to_step(response)
        return step, replace(state, items=state.items + tuple(response.output))

    def _create(
        self,
        system: str,
        input: str | ResponseInputParam,
        tools: list[FunctionToolParam] | None = None,
    ) -> Response:
        try:
            return self._client.responses.create(
                model=self._model,
                instructions=system,  # developer-level instructions, ranked above user input
                input=input,
                tools=tools if tools else omit,
                reasoning={"effort": self._reasoning_effort},
                max_output_tokens=self._max_output_tokens,
            )
        except openai.AuthenticationError:
            # Not chained or echoed: the error text can include part of the key.
            raise LLMError("OpenAI rejected the API key (check OPENAI_API_KEY)") from None
        except openai.APIError as error:
            raise LLMError(f"OpenAI request failed: {error}") from error


def _function_tool(tool: ToolSpec) -> FunctionToolParam:
    # Strict mode: the model's arguments are guaranteed to match the schema, which must then
    # list every property as required and set additionalProperties to false.
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": True,
    }


def _to_result(response: Response) -> LLMResult:
    """Turn a plain (no tools) response into an LLMResult, or raise LLMError."""
    _raise_if_unusable(response)
    text = response.output_text.strip()
    if not text:
        raise LLMError("model returned an empty answer")
    input_tokens, output_tokens = _usage(response)
    return LLMResult(
        text=text,
        model=response.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        request_id=response._request_id,
    )


def _to_step(response: Response) -> AgentStep:
    """Turn a tool-calling turn into an AgentStep, or raise LLMError."""
    _raise_if_unusable(response)
    calls = tuple(_tool_call(item) for item in response.output if item.type == "function_call")
    text = response.output_text.strip() or None
    if text is None and not calls:
        raise LLMError("model returned neither an answer nor tool calls")
    input_tokens, output_tokens = _usage(response)
    return AgentStep(
        text=text,
        tool_calls=calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        request_id=response._request_id,
    )


def _raise_if_unusable(response: Response) -> None:
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


def _tool_call(item: ResponseFunctionToolCall) -> ToolCall:
    try:
        arguments = json.loads(item.arguments)
    except json.JSONDecodeError:
        arguments = None
    if not isinstance(arguments, dict):
        return ToolCall(
            id=item.call_id,
            name=item.name,
            arguments={},
            raw_arguments=item.arguments,
            invalid_json=True,
        )
    return ToolCall(
        id=item.call_id, name=item.name, arguments=arguments, raw_arguments=item.arguments
    )


def _usage(response: Response) -> tuple[int, int]:
    if response.usage is None:
        raise LLMError("response has no token usage")
    return response.usage.input_tokens, response.usage.output_tokens


# --- Cost ---------------------------------------------------------------------------------------


def cost_usd(result: LLMResult, settings: Settings) -> Decimal | None:
    """Exact cost of one call from the configured prices, or None if they aren't configured."""
    return cost_for_tokens(result.input_tokens, result.output_tokens, settings)


def cost_for_tokens(input_tokens: int, output_tokens: int, settings: Settings) -> Decimal | None:
    """Exact cost from the configured prices, or None if they aren't configured (never guessed).

    Cached input tokens are billed at a lower rate; counting them at the full input price
    makes this an upper bound when the provider's prompt cache hits.
    """
    input_price = settings.llm_input_price_per_mtok
    output_price = settings.llm_output_price_per_mtok
    if input_price is None or output_price is None:
        return None
    return (input_tokens * input_price + output_tokens * output_price) / _TOKENS_PER_PRICE_UNIT
