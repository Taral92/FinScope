"""Which model and prompts produced a report, so feedback can later be compared by version.

Each prompt version is evals.prompt_version over that step's system prompt and its schema
(the agent's tool specs stand in for a schema). It covers the system prompt only: a change to
how a user message is built doesn't change the version.
"""

from pydantic import BaseModel, ConfigDict

from finscope import judge, triage
from finscope.agent_tools import TOOL_SPECS
from finscope.config import Settings
from finscope.evals import prompt_version
from finscope.explain import AGENT_SYSTEM_PROMPT


class PromptVersions(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str  # every step uses settings.llm_model for now
    triage: str
    agent: str
    judge: str | None  # None when no judge ran


def triage_prompt_version() -> str:
    return prompt_version(triage.SYSTEM_PROMPT, triage.SCHEMA)


def agent_prompt_version() -> str:
    """Like the triage prompt version, over the agent's system prompt and its tool specs."""
    tools = {"tools": [spec.model_dump(mode="json") for spec in TOOL_SPECS]}
    return prompt_version(AGENT_SYSTEM_PROMPT, tools)


def judge_prompt_version() -> str:
    return prompt_version(judge.SYSTEM_PROMPT, judge.SCHEMA)


def current_versions(settings: Settings, *, judged: bool) -> PromptVersions:
    """The versions a report built now would use; `judged` says whether a judge runs."""
    return PromptVersions(
        model=settings.llm_model,
        triage=triage_prompt_version(),
        agent=agent_prompt_version(),
        judge=judge_prompt_version() if judged else None,
    )
