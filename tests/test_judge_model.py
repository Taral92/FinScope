"""The judge on its own model (FINSCOPE_JUDGE_MODEL) in the evals that run it."""

from datetime import UTC, datetime
from decimal import Decimal

from finscope.config import Settings
from finscope.known_cases import JUDGE_CASES_PATH
from finscope.known_judge import JudgeCaseResult, JudgeRun, build_judge_result
from finscope.pairwise import PairResult, PairwiseCall, summarize_pairs
from finscope.report import HighItem

SETTINGS = Settings(
    _env_file=None,
    sec_user_agent="FinScope Test test@example.com",
    llm_model="main-model",
    llm_input_price_per_mtok=Decimal("2"),
    llm_output_price_per_mtok=Decimal("8"),
    judge_model="judge-x",
    judge_input_price_per_mtok=Decimal("10"),
    judge_output_price_per_mtok=Decimal("50"),
)


def test_the_known_judge_eval_names_and_prices_the_judge_model() -> None:
    run = JudgeRun(verdict="supported", reason="ok", error=None, input_tokens=700, output_tokens=40)
    case = JudgeCaseResult(name="a", expected="supported", runs=(run, run))

    result = build_judge_result(
        [case],
        cases_file=JUDGE_CASES_PATH,
        runs=2,
        settings=SETTINGS,
        created_at=datetime(2026, 11, 20, tzinfo=UTC),
    )

    assert result.model == "judge-x"
    assert result.cost_usd == (Decimal(1_400) * 10 + Decimal(80) * 50) / 1_000_000


def item(change_id: str) -> HighItem:
    return HighItem(
        id=change_id,
        triage_reason="reason",
        explanation=None,
        error=None,
        parsed=None,
        verification=None,
        input_tokens=300,
        output_tokens=50,
    )


def test_pairwise_prices_the_judge_calls_at_the_judge_prices() -> None:
    call = PairwiseCall(
        first="a", pick="tie", reason="same", model="judge-x", input_tokens=700, output_tokens=40
    )
    result = PairResult(
        id="+1",
        a=item("+1"),
        b=item("+1"),
        outcome="tie",
        decided_by="judge",
        calls=(call, call),
    )

    summary = summarize_pairs([result], SETTINGS, extra_input_tokens=1_000, extra_output_tokens=200)

    # Triage 1,000 / 200 and both agent runs 600 / 100 at $2 / $8; two judge calls at $10 / $50.
    assert (
        summary.cost_usd
        == (Decimal(1_600) * 2 + Decimal(300) * 8 + Decimal(1_400) * 10 + Decimal(80) * 50)
        / 1_000_000
    )
