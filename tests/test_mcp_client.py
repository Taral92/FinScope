"""The MCP client, against the real server code: in-process by default, a subprocess when
`-m integration` is given (that one calls the real SEC EDGAR API)."""

import logging
from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

import finscope.notify  # noqa: F401  (holds the httpx logger at WARNING, as in every run)
from finscope.__main__ import main
from finscope.agent_tools import ExplainContext, filing_tool_outputs
from finscope.config import Settings
from finscope.diff import ParagraphChange
from finscope.edgar import SecClient
from finscope.explain import AGENT_SYSTEM_PROMPT, explain_change_with_agent
from finscope.llm import (
    AgentStep,
    ConversationState,
    ToolCall,
    ToolResult,
    ToolSpec,
    _function_tool,
)
from finscope.mcp_client import (
    McpArguments,
    McpToolbox,
    McpToolError,
    prompt_note,
    server_command,
)
from finscope.mcp_server import build_server
from finscope.sections import Section
from finscope.verify import parse_explanation, sources_for, verify_explanation

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
)
CIK = "0001318605"
BASE = (
    "Our vehicles may be subject to recalls, which could harm our reputation.",
    "Changes in trade policy and tariffs could increase our costs significantly.",
)
NEW_RISK = "Our Optimus humanoid robot program may never reach volume production."


def filing_html(*paragraphs: str) -> str:
    body = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    return (
        f"<html><body><p>PART I</p><p>Item 1A. Risk Factors</p>{body}"
        "<p>Item 1B. Unresolved Staff Comments</p><p>None.</p></body></html>"
    )


OLD = ("0001628280-25-000002", date(2025, 1, 30), filing_html(*BASE))
NEW = ("0001628280-26-000003", date(2026, 1, 29), filing_html(*BASE, NEW_RISK))


def handle(request: httpx.Request) -> httpx.Response:
    """Stand-in for SEC's servers: one company, two 10-Ks, no XBRL facts."""
    url = str(request.url)
    if url == "https://www.sec.gov/files/company_tickers.json":
        return httpx.Response(200, json={"0": {"cik_str": int(CIK), "ticker": "TSLA"}})
    if url == f"https://data.sec.gov/submissions/CIK{CIK}.json":
        recent = {
            "accessionNumber": [NEW[0], OLD[0]],
            "form": ["10-K", "10-K"],
            "filingDate": [NEW[1].isoformat(), OLD[1].isoformat()],
            "reportDate": ["", ""],
            "primaryDocument": [f"{NEW[0]}.htm", f"{OLD[0]}.htm"],
        }
        return httpx.Response(200, json={"filings": {"recent": recent}})
    if url.endswith("/companyfacts/CIK" + CIK + ".json"):
        return httpx.Response(200, json={"facts": {}})
    for accession, _, html in (NEW, OLD):
        if url.endswith(f"/{accession}.htm"):
            return httpx.Response(200, text=html)
    return httpx.Response(404)


@pytest.fixture
def toolbox(tmp_path: Path) -> Iterator[McpToolbox]:
    settings = SETTINGS.model_copy(update={"data_dir": tmp_path})
    with (
        SecClient(settings=settings, transport=httpx.MockTransport(handle)) as sec,
        McpToolbox(build_server(sec)) as box,
    ):
        yield box


def test_the_client_discovers_the_servers_five_tools_as_non_strict_specs(
    toolbox: McpToolbox,
) -> None:
    by_name = {spec.name: spec for spec in toolbox.specs}

    assert set(by_name) == {
        "list_filings",
        "get_section",
        "diff_section",
        "get_change_context",
        "get_key_metrics",
    }
    context = by_name["get_change_context"]
    assert set(context.parameters["required"]) == {"ticker", "change_id"}
    assert "paragraphs around it" in context.description
    assert all(not spec.strict for spec in toolbox.specs)
    assert _function_tool(context)["strict"] is False  # sent to the model as non-strict


def test_a_call_returns_the_servers_text(toolbox: McpToolbox) -> None:
    text = toolbox.call("get_change_context", {"ticker": "TSLA", "change_id": "+1"})

    assert text.startswith("[+1] added")
    assert NEW_RISK in text
    assert BASE[1] in text  # the paragraph just before the change


def test_a_server_error_comes_back_as_its_own_message(toolbox: McpToolbox) -> None:
    with pytest.raises(McpToolError, match=r"no change '\+9'.*ids come from diff_section"):
        toolbox.call("get_change_context", {"ticker": "TSLA", "change_id": "+9"})


def test_each_tool_forwards_to_the_server(toolbox: McpToolbox) -> None:
    tools = toolbox.tools()

    text = tools["get_section"].function(McpArguments(ticker="TSLA", form="10-K", item="1A"))

    assert BASE[0] in text and NEW_RISK in text


def test_a_toolbox_must_be_connected_first() -> None:
    with pytest.raises(RuntimeError, match="not connected"):
        McpToolbox(server_command(Path.cwd())).call("list_filings", {"ticker": "TSLA"})


def test_building_a_server_doesnt_turn_on_url_logging() -> None:
    # The MCP SDK sets the root logger to INFO; httpx would then log every URL, and Telegram's
    # carry the bot token. notify.py holds the httpx logger at WARNING.
    with SecClient(SETTINGS, transport=httpx.MockTransport(handle)) as sec:
        build_server(sec)

    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)


def test_the_prompt_note_gives_the_tools_their_identifiers() -> None:
    note = prompt_note("TSLA", "10-K", "1A", "I", "+4")

    assert "ticker TSLA, form 10-K, item 1A, part I,\nchange_id +4." in note
    assert "part" not in prompt_note("TSLA", "10-K", "1A", None, "+4").split("item 1A")[1][:6]


# --- the agent with MCP tools, end to end ---------------------------------------------------------


class CallsChangeContext:
    """Implements LLMProvider's start/resume: calls get_change_context, then answers."""

    def __init__(self, quote: str) -> None:
        self.quote = quote
        self.tools: list[ToolSpec] = []
        self.results: list[ToolResult] = []

    def start(
        self, system: str, user: str, tools: list[ToolSpec]
    ) -> tuple[AgentStep, ConversationState]:
        self.tools = tools
        self.system = system
        arguments = {"ticker": "TSLA", "change_id": "+1"}
        call = ToolCall(
            id="c1", name="get_change_context", arguments=arguments, raw_arguments=str(arguments)
        )
        return (
            AgentStep(text=None, tool_calls=(call,), input_tokens=100, output_tokens=10),
            ConversationState(),
        )

    def resume(
        self, state: ConversationState, tool_results: list[ToolResult]
    ) -> tuple[AgentStep, ConversationState]:
        self.results += tool_results
        text = (
            "WHAT CHANGED: Tesla added a risk about its Optimus robot program.\n"
            "WHY IT MATTERS: It sits next to the tariff risk.\n"
            "IMPORTANCE: medium - potential: new program\n"
            f'EVIDENCE: "{self.quote}"'
        )
        return AgentStep(text=text, tool_calls=(), input_tokens=300, output_tokens=50), state


def change_context() -> ExplainContext:
    section = Section(part="I", item="1A", title="Risk Factors", start=0, end=0, text=NEW_RISK)
    change = ParagraphChange(
        kind="added",
        old_text=None,
        new_text=NEW_RISK,
        old_start=None,
        old_end=None,
        new_start=0,
        new_end=len(NEW_RISK),
        similarity=None,
    )
    return ExplainContext(change=change, old_section=section, new_section=section, metrics=())


def test_the_agent_uses_mcp_tools_and_their_output_counts_for_the_fact_check(
    toolbox: McpToolbox,
) -> None:
    # The quote is only in the MCP tool's answer (the paragraph before the change).
    provider = CallsChangeContext("tariffs could increase our costs")
    context = change_context()

    explanation = explain_change_with_agent(
        context,
        company="TSLA",
        form="10-K",
        section_title="Part I Item 1A. Risk Factors",
        provider=provider,  # type: ignore[arg-type]
        settings=SETTINGS,
        tools=toolbox.tools(),
        system_prompt=AGENT_SYSTEM_PROMPT + prompt_note("TSLA", "10-K", "1A", "I", "+1"),
    )

    assert {tool.name for tool in provider.tools} == {spec.name for spec in toolbox.specs}
    assert "change_id +1" in provider.system
    [result] = provider.results
    assert not result.is_error and BASE[1] in result.output
    run = explanation.run
    parsed = parse_explanation(run.text, expect_evidence=True)
    check = verify_explanation(parsed, sources_for(context.change, filing_tool_outputs(run)))
    assert check.passed


def test_mcp_without_agent_is_refused_before_any_network_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["explain", "TSLA", "--item", "1A", "--kind", "added", "--n", "1", "--mcp"])

    assert code == 1
    assert "--mcp needs --agent" in capsys.readouterr().err


@pytest.mark.integration
def test_a_real_server_subprocess_is_discovered_and_called() -> None:
    """Starts `python -m finscope mcp` for real: needs .env and calls SEC EDGAR."""
    root = Path(__file__).parents[1]
    with McpToolbox(server_command(root)) as box:
        assert len(box.specs) == 5
        text: Any = box.call("list_filings", {"ticker": "TSLA", "form": "10-K", "limit": 1})
    assert "10-K" in text
