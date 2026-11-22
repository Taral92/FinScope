"""An MCP client for the explain agent: tools discovered from an MCP server, called over MCP.

The agent's own tools are Python functions in this process. Here the agent instead gets every
tool FinScope's MCP server (mcp_server.py) offers, discovered at connect time, and each call
goes to the server over the protocol, as Claude Code's calls do. The server runs as a separate
process over stdin/stdout; tests connect to it in-process.

The MCP SDK is async and the agent loop is plain synchronous code, so the client session runs
in a background thread (anyio's blocking portal) and each call waits for its answer there.
"""

import sys
from contextlib import ExitStack
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ConfigDict

from finscope.agent import Tool
from finscope.llm import ToolSpec

Target = MCPServer | StdioServerParameters

# Added to the agent's system prompt for an MCP run: its tools are the server's, which need
# the change's identifiers (the in-process tools are pre-loaded with them).
MCP_PROMPT_NOTE = """

In this run your tools come from FinScope's MCP server, not the ones named above:
- get_change_context gives the change's full text and the paragraphs around it;
- get_key_metrics gives the filing's financial numbers from SEC XBRL data.
They need the change's identifiers: ticker {ticker}, form {form}, item {item}{part},
change_id {change_id}."""


class McpToolError(Exception):
    """The server answered a tool call with an error; the message is the server's."""


class McpArguments(BaseModel):
    """Whatever the model sends: the server validates arguments against its own schema."""

    model_config = ConfigDict(extra="allow")


def server_command(project_dir: Path) -> StdioServerParameters:
    """FinScope's own server, started by this Python in the project folder (for .env, data/)."""
    return StdioServerParameters(
        command=sys.executable, args=["-m", "finscope", "mcp"], cwd=str(project_dir)
    )


def prompt_note(ticker: str, form: str, item: str, part: str | None, change_id: str) -> str:
    return MCP_PROMPT_NOTE.format(
        ticker=ticker,
        form=form,
        item=item,
        part=f", part {part}" if part else "",
        change_id=change_id,
    )


class McpToolbox:
    """A connection to one MCP server, usable from synchronous code. Use it as a context
    manager: the server (a subprocess for stdio) is stopped on exit, even after an error."""

    def __init__(self, target: Target, *, timeout_seconds: float = 60.0) -> None:
        self._target = target
        self._timeout = timeout_seconds
        self._stack = ExitStack()
        self._portal: BlockingPortal | None = None
        self._client: Client | None = None
        self.specs: tuple[ToolSpec, ...] = ()

    def __enter__(self) -> Self:
        try:
            portal = self._stack.enter_context(start_blocking_portal())
            client = Client(self._target, read_timeout_seconds=self._timeout)
            self._client = self._stack.enter_context(portal.wrap_async_context_manager(client))
            self._portal = portal
            listed = portal.call(self._client.list_tools)
        except BaseException:
            self._stack.close()
            raise
        self.specs = tuple(_spec(tool) for tool in listed.tools)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stack.close()

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """The server's text answer, or McpToolError with the server's own error message."""
        if self._portal is None or self._client is None:
            raise RuntimeError("McpToolbox is not connected: use it in a with block")
        result = self._portal.call(self._client.call_tool, name, arguments)
        text = "\n".join(part.text for part in result.content if part.type == "text")
        if result.is_error:
            raise McpToolError(text)
        return text

    def tools(self) -> dict[str, Tool]:
        """The agent's tools: one per discovered spec, each forwarding its call to the server.

        A server error raises McpToolError, which the agent loop hands back to the model as a
        tool error it can read and fix, like any other tool's failure.
        """

        def forward(name: str) -> Any:
            return lambda arguments: self.call(name, arguments.model_dump())

        return {
            spec.name: Tool(spec=spec, function=forward(spec.name), arguments=McpArguments)
            for spec in self.specs
        }


def _spec(tool: Any) -> ToolSpec:
    """An MCP tool as the model sees it. Non-strict: its schema has optional arguments."""
    return ToolSpec(
        name=tool.name,
        description=tool.description or "",
        parameters=tool.input_schema,
        strict=False,
    )
