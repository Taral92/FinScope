"""Read-only tools for the explain agent. All data is loaded before the agent starts."""

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from finscope.agent import AgentRun, Tool
from finscope.diff import ParagraphChange, section_paragraphs
from finscope.llm import ToolSpec
from finscope.memory import PastFinding, format_recall, recall
from finscope.prompting import filing_text_block
from finscope.sections import Section
from finscope.xbrl import MetricValue, format_amount, format_period


@dataclass(frozen=True)
class ExplainContext:
    """Everything the tools may look at, fetched up front: no tool touches the network."""

    change: ParagraphChange
    old_section: Section
    new_section: Section
    metrics: tuple[MetricValue, ...]  # xbrl.key_metrics for the new filing
    # Agent memory (memory.py): past findings, never from this filing; empty without memory.
    memory: tuple[PastFinding, ...] = field(default=())
    company: str = ""  # ticker, for recalling this company's history
    section: str = ""  # e.g. "I-1A"


class NearbyParagraphsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=1, le=3)


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Schemas use only what OpenAI's function-calling guide shows for strict mode: every property
# required, no extra properties. Value ranges are stated in descriptions and enforced by the
# argument models above, so an out-of-range value comes back to the model as an error.
_NEARBY_SPEC = ToolSpec(
    name="get_nearby_paragraphs",
    description=(
        "Returns the paragraphs just before and just after the changed paragraph, in the same "
        "section (the new filing for added or modified paragraphs, the old filing for removed "
        "ones). Each paragraph is labeled with its character offset in the filing. Use it to "
        "see what the change is about: its heading, the risk or topic it belongs to, and how "
        "it connects to the text around it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Paragraphs to return on each side of the change, from 1 to 3.",
            }
        },
        "required": ["count"],
        "additionalProperties": False,
    },
)
_METRICS_SPEC = ToolSpec(
    name="get_key_metrics",
    description=(
        "Returns this filing's key financial numbers from SEC XBRL data, one line each: "
        "metric, period, value, prior-year value and percent change (revenue, gross profit, "
        "operating income, net income, diluted EPS, cash). These are the only reliable "
        "numbers: use them whenever your answer mentions a financial figure."
    ),
    parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
)
_MEMORY_SPEC = ToolSpec(
    name="recall_past_findings",
    description=(
        "Returns FinScope's own earlier findings that are similar to this change: from this "
        "company's earlier filings (what was found and how it was rated), and from any "
        "company where the user rated the alert useful, not_useful or wrong. Use it to say "
        "when a risk first appeared or how it grew, and to see how the user rated similar "
        "alerts. Its text is FinScope's past wording, not the filing: never quote it as "
        "evidence."
    ),
    parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
)
# What the model is shown about the tools: part of the agent's prompt version.
TOOL_SPECS = (_NEARBY_SPEC, _METRICS_SPEC, _MEMORY_SPEC)


def build_tools(context: ExplainContext) -> dict[str, Tool]:
    def nearby(args: NearbyParagraphsArgs) -> str:
        return get_nearby_paragraphs(context, args.count)

    def metrics(_: NoArgs) -> str:
        return get_key_metrics(context)

    def memory(_: NoArgs) -> str:
        return recall_past_findings(context)

    return {
        _NEARBY_SPEC.name: Tool(spec=_NEARBY_SPEC, function=nearby, arguments=NearbyParagraphsArgs),
        _METRICS_SPEC.name: Tool(spec=_METRICS_SPEC, function=metrics, arguments=NoArgs),
        _MEMORY_SPEC.name: Tool(spec=_MEMORY_SPEC, function=memory, arguments=NoArgs),
    }


def filing_tool_outputs(run: AgentRun) -> list[str]:
    """The tool results that come from the filing: what the fact-check and the judge check an
    answer against. Memory is FinScope's own past wording, so it is never a source: a quote
    found only there must fail the fact-check."""
    return [
        call.output
        for step in run.trace
        for call in step.tool_calls
        if not call.is_error and call.name != _MEMORY_SPEC.name
    ]


def recall_past_findings(context: ExplainContext) -> str:
    change = context.change
    history, rated = recall(
        context.memory,
        change.new_text or change.old_text or "",
        company=context.company,
        section=context.section,
    )
    return format_recall(history, rated, company=context.company, section=context.section)


def get_nearby_paragraphs(context: ExplainContext, count: int) -> str:
    change = context.change
    if change.kind == "removed":
        section, start, filing = context.old_section, change.old_start, "previous filing"
    else:
        section, start, filing = context.new_section, change.new_start, "this filing"

    paragraphs = section_paragraphs(section)
    index = next((i for i, (offset, _) in enumerate(paragraphs) if offset == start), None)
    if index is None:
        raise ValueError("the changed paragraph is not in its section")

    def blocks(chosen: list[tuple[int, str]]) -> list[str]:
        return [
            filing_text_block(text, source=filing, offset=f"{offset:,}") for offset, text in chosen
        ] or ["(none)"]

    before = paragraphs[max(0, index - count) : index]
    after = paragraphs[index + 1 : index + 1 + count]
    parts = [
        f"Before the changed paragraph (offset {start:,}):",
        *blocks(before),
        "After the changed paragraph:",
        *blocks(after),
    ]
    return "\n\n".join(parts)


def get_key_metrics(context: ExplainContext) -> str:
    return format_metrics(context.metrics)


def format_metrics(metrics: tuple[MetricValue, ...]) -> str:
    """One line per metric: name | period | value | prior | change. Also used by the MCP server."""
    if not metrics:
        return "No XBRL metrics are available for this filing."
    lines = []
    for metric in metrics:
        prior = format_amount(metric.prior.value, metric.unit) if metric.prior else "n/a"
        change = f"{metric.change_pct:+.1f}%" if metric.change_pct is not None else "n/a"
        lines.append(
            f"{metric.name} | {format_period(metric)} | "
            f"{format_amount(metric.current.value, metric.unit)} | prior {prior} | {change}"
        )
    return "\n".join(lines)
