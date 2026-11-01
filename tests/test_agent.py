import json
from collections.abc import Callable

import pytest
from pydantic import BaseModel, ConfigDict, Field

from finscope.agent import AgentError, Tool, run_agent
from finscope.llm import AgentStep, ConversationState, LLMResult, ToolCall, ToolResult, ToolSpec


class ScriptedProvider:
    """Implements LLMProvider by replaying pre-planned steps; records the tool results sent."""

    def __init__(self, steps: list[AgentStep]) -> None:
        self.steps = list(steps)
        self.tools_offered: list[ToolSpec] = []
        self.resumed_with: list[list[ToolResult]] = []

    def complete(self, system: str, user: str) -> LLMResult:
        raise NotImplementedError

    def start(
        self, system: str, user: str, tools: list[ToolSpec]
    ) -> tuple[AgentStep, ConversationState]:
        self.tools_offered = tools
        return self.steps.pop(0), ConversationState()

    def resume(
        self, state: ConversationState, tool_results: list[ToolResult]
    ) -> tuple[AgentStep, ConversationState]:
        self.resumed_with.append(tool_results)
        return self.steps.pop(0), state


def call(name: str, raw_arguments: str, call_id: str = "call_1") -> ToolCall:
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return ToolCall(
            id=call_id, name=name, arguments={}, raw_arguments=raw_arguments, invalid_json=True
        )
    return ToolCall(id=call_id, name=name, arguments=arguments, raw_arguments=raw_arguments)


def step(
    *calls: ToolCall, text: str | None = None, tokens: tuple[int, int] = (100, 10)
) -> AgentStep:
    return AgentStep(text=text, tool_calls=calls, input_tokens=tokens[0], output_tokens=tokens[1])


class CountArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=1, le=3)


def lookup_tool(function: Callable[[CountArgs], str]) -> dict[str, Tool]:
    spec = ToolSpec(
        name="lookup",
        description="Look something up.",
        parameters={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": False,
        },
    )
    return {"lookup": Tool(spec=spec, function=function, arguments=CountArgs)}


TOOLS = lookup_tool(lambda args: f"found {args.count} items")


def test_tool_result_is_sent_back_and_loop_stops_on_answer() -> None:
    provider = ScriptedProvider([step(call("lookup", '{"count": 2}')), step(text="ANSWER")])

    run = run_agent("system", "user", TOOLS, provider)

    assert run.text == "ANSWER"
    assert [spec.name for spec in provider.tools_offered] == ["lookup"]
    assert provider.resumed_with == [[ToolResult(call_id="call_1", output="found 2 items")]]
    assert [(s.number, s.answered) for s in run.trace] == [(1, False), (2, True)]
    assert run.trace[0].tool_calls[0].result_chars == len("found 2 items")


def first_result(steps: list[AgentStep], tools: dict[str, Tool] = TOOLS) -> ToolResult:
    """Run the script and return the result sent back for the first tool call."""
    provider = ScriptedProvider(steps)
    run_agent("system", "user", tools, provider)
    return provider.resumed_with[0][0]


def test_unknown_tool_is_an_error_result_and_the_loop_continues() -> None:
    result = first_result([step(call("delete_everything", "{}")), step(text="ANSWER")])

    assert result.is_error
    assert result.output == "unknown tool 'delete_everything'; available: lookup"


def test_invalid_arguments_are_an_error_result() -> None:
    result = first_result([step(call("lookup", '{"count": 50}')), step(text="ANSWER")])

    assert result.is_error
    assert result.output == "invalid arguments: count: Input should be less than or equal to 3"


def test_bad_json_is_an_error_result() -> None:
    result = first_result([step(call("lookup", '{"count": ')), step(text="ANSWER")])

    assert result.is_error
    assert result.output == "arguments are not a valid JSON object"


def test_a_failing_tool_is_an_error_result_without_a_traceback() -> None:
    def broken(args: CountArgs) -> str:
        raise RuntimeError("database offline")

    result = first_result(
        [step(call("lookup", '{"count": 1}')), step(text="ANSWER")], lookup_tool(broken)
    )

    assert result.is_error
    assert result.output == "tool failed: RuntimeError: database offline"


def test_long_tool_output_is_truncated_with_a_marker() -> None:
    provider = ScriptedProvider([step(call("lookup", '{"count": 1}')), step(text="ANSWER")])

    run_agent(
        "system",
        "user",
        lookup_tool(lambda args: "x" * 5_000),
        provider,
        max_tool_output_chars=4_000,
    )

    output = provider.resumed_with[0][0].output
    assert output == "x" * 4_000 + "\n[truncated 1000 chars]"


def test_step_limit_raises_with_the_trace_so_far() -> None:
    looping = [step(call("lookup", '{"count": 1}', call_id=f"call_{i}")) for i in range(5)]
    provider = ScriptedProvider(looping)

    with pytest.raises(AgentError, match="step limit reached") as error:
        run_agent("system", "user", TOOLS, provider, max_steps=2)

    assert [s.number for s in error.value.trace] == [1, 2]
    assert len(provider.resumed_with) == 1  # exactly max_steps model calls were made


def test_tokens_are_summed_across_steps() -> None:
    provider = ScriptedProvider(
        [
            step(call("lookup", '{"count": 1}'), tokens=(1_000, 50)),
            step(text="ANSWER", tokens=(1_200, 80)),
        ]
    )

    run = run_agent("system", "user", TOOLS, provider)

    assert (run.input_tokens, run.output_tokens) == (2_200, 130)
