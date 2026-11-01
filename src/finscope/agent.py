"""A small tool-calling agent loop, written by hand. Knows the provider interface only."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from finscope.llm import AgentStep, LLMProvider, ToolCall, ToolResult, ToolSpec


@dataclass(frozen=True)
class Tool:
    """A tool the agent may use: what the model sees, the Python behind it, its argument model."""

    spec: ToolSpec
    function: Callable[[Any], str]  # receives a validated instance of `arguments`
    arguments: type[BaseModel]


class TraceCall(BaseModel):
    """One tool call as it happened, for the trace."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: str  # the raw JSON the model sent
    result_chars: int
    is_error: bool
    error: str | None = None


class TraceStep(BaseModel):
    """One model call: the tools it asked for (if any) and its token usage."""

    model_config = ConfigDict(frozen=True)

    number: int
    tool_calls: tuple[TraceCall, ...]
    answered: bool  # True when this step produced the final text
    input_tokens: int
    output_tokens: int
    request_id: str | None


class AgentRun(BaseModel):
    """The final answer plus everything that led to it."""

    model_config = ConfigDict(frozen=True)

    text: str
    trace: tuple[TraceStep, ...]

    @property
    def input_tokens(self) -> int:
        return sum(step.input_tokens for step in self.trace)

    @property
    def output_tokens(self) -> int:
        return sum(step.output_tokens for step in self.trace)


class AgentError(Exception):
    """The agent stopped without an answer. Carries the trace so far for debugging."""

    def __init__(self, message: str, trace: list[TraceStep]) -> None:
        super().__init__(message)
        self.trace = tuple(trace)


def run_agent(
    system: str,
    user: str,
    tools: dict[str, Tool],
    provider: LLMProvider,
    *,
    max_steps: int = 6,
    max_tool_output_chars: int = 4_000,
) -> AgentRun:
    """Let the model call tools until it answers with text, within max_steps model calls."""
    trace: list[TraceStep] = []
    step, state = provider.start(system, user, [tool.spec for tool in tools.values()])
    while True:
        results = [_run_tool(call, tools, max_tool_output_chars) for call in step.tool_calls]
        trace.append(_trace_step(len(trace) + 1, step, results))
        if not step.tool_calls:
            if step.text is None:
                raise AgentError("model returned no answer", trace)
            return AgentRun(text=step.text, trace=tuple(trace))
        if len(trace) >= max_steps:
            # Without a cap, a model that keeps calling tools would loop (and bill) forever.
            raise AgentError("step limit reached", trace)
        step, state = provider.resume(state, [result for result, _ in results])


def _run_tool(
    call: ToolCall, tools: dict[str, Tool], max_chars: int
) -> tuple[ToolResult, TraceCall]:
    """Run one call. Every failure becomes an error result the model can read and fix."""
    tool = tools.get(call.name)
    if tool is None:
        return _error(call, f"unknown tool {call.name!r}; available: {', '.join(tools)}")
    if call.invalid_json:
        return _error(call, "arguments are not a valid JSON object")
    try:
        arguments = tool.arguments.model_validate(call.arguments)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc']) or 'arguments'}: {issue['msg']}"
            for issue in error.errors()
        )
        return _error(call, f"invalid arguments: {problems}")
    try:
        output = tool.function(arguments)
    except Exception as error:  # any tool bug: report it to the model, keep the loop alive
        return _error(call, f"tool failed: {type(error).__name__}: {error}")

    if len(output) > max_chars:
        output = f"{output[:max_chars]}\n[truncated {len(output) - max_chars} chars]"
    result = ToolResult(call_id=call.id, output=output)
    return result, TraceCall(
        name=call.name, arguments=call.raw_arguments, result_chars=len(output), is_error=False
    )


def _error(call: ToolCall, message: str) -> tuple[ToolResult, TraceCall]:
    result = ToolResult(call_id=call.id, output=message, is_error=True)
    trace_call = TraceCall(
        name=call.name,
        arguments=call.raw_arguments,
        result_chars=len(message),
        is_error=True,
        error=message,
    )
    return result, trace_call


def _trace_step(
    number: int, step: AgentStep, results: list[tuple[ToolResult, TraceCall]]
) -> TraceStep:
    return TraceStep(
        number=number,
        tool_calls=tuple(trace_call for _, trace_call in results),
        answered=not step.tool_calls,
        input_tokens=step.input_tokens,
        output_tokens=step.output_tokens,
        request_id=step.request_id,
    )
