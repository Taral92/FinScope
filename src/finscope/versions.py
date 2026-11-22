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
    judge_model: str | None = None  # None: the judge ran on `model` (or didn't run)

    @property
    def label(self) -> str:
        """e.g. 'model gpt-6.1-terra, triage 3d30ad40231e, agent 983848dcbf2b, judge -'."""
        return (
            f"model {self.model}, triage {self.triage}, agent {self.agent}, "
            f"judge {self.judge or '-'}" + (f" on {self.judge_model}" if self.judge_model else "")
        )


def triage_prompt_version() -> str:
    return prompt_version(triage.SYSTEM_PROMPT, triage.SCHEMA)


def agent_prompt_version(system_prompt: str = AGENT_SYSTEM_PROMPT) -> str:
    """Like the triage prompt version, over the agent's system prompt and its tool specs.

    `system_prompt` gives the version of another agent prompt, e.g. a pairwise candidate.
    """
    # `strict` is how a spec is sent, not what the model reads, and it is the same for all of
    # these: leaving it out keeps the versions from before the field existed.
    tools = {"tools": [spec.model_dump(mode="json", exclude={"strict"}) for spec in TOOL_SPECS]}
    return prompt_version(system_prompt, tools)


def judge_prompt_version() -> str:
    return prompt_version(judge.SYSTEM_PROMPT, judge.SCHEMA)


def current_versions(settings: Settings, *, judged: bool) -> PromptVersions:
    """The versions a report built now would use; `judged` says whether a judge runs."""
    return PromptVersions(
        model=settings.llm_model,
        triage=triage_prompt_version(),
        agent=agent_prompt_version(),
        judge=judge_prompt_version() if judged else None,
        # Recorded only when it differs, so reports from before this field compare as equal.
        judge_model=(
            settings.judge_model
            if judged and settings.judge_model not in (None, settings.llm_model)
            else None
        ),
    )
