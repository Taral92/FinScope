"""Exact financial numbers from SEC's XBRL "company facts" data. No AI, no estimates."""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# How many days a "current period" lasts, per form. A 10-Q also reports year-to-date
# totals (6 or 9 months), which fall outside the quarter range and are ignored.
_PERIOD_DAYS = {"10-Q": (80, 100), "10-K": (350, 380)}
# The comparison period ends about a year before the current one; the tolerance absorbs
# 52/53-week fiscal years (e.g. Apple's quarters end on the last Saturday of a month).
_ONE_YEAR = timedelta(days=365)
_TOLERANCE_DAYS = 10


class Fact(BaseModel):
    """One reported number, e.g. Revenues = 28,046,000,000 USD for Jul 1 - Sep 30, 2026."""

    model_config = ConfigDict(frozen=True)

    concept: str  # us-gaap concept name, e.g. "Revenues"
    unit: str  # "USD", "USD/shares", ...
    value: Decimal
    start: date | None  # None for instant facts (a balance on a single day)
    end: date
    # fy/fp describe the filing that reported the fact, not the fact's own period: last
    # year's comparison numbers in a FY2026 10-Q are tagged fiscal_year=2026 too.
    fiscal_year: int | None
    fiscal_period: str | None
    form: str
    accession_number: str
    filed: date

    @property
    def days(self) -> int | None:
        """Length of the period in days, or None for an instant."""
        return None if self.start is None else (self.end - self.start).days


@dataclass(frozen=True)
class Metric:
    name: str
    concepts: tuple[str, ...]  # candidate us-gaap concepts, most preferred first
    kind: Literal["duration", "instant"]
    unit: str


METRICS = (
    Metric(
        "Revenue",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ),
        "duration",
        "USD",
    ),
    Metric("Gross profit", ("GrossProfit",), "duration", "USD"),
    Metric("Operating income", ("OperatingIncomeLoss",), "duration", "USD"),
    Metric("Net income", ("NetIncomeLoss",), "duration", "USD"),
    Metric("Diluted EPS", ("EarningsPerShareDiluted",), "duration", "USD/shares"),
    Metric("Cash", ("CashAndCashEquivalentsAtCarryingValue",), "instant", "USD"),
)


class MetricValue(BaseModel):
    """A metric as one filing reported it: this period, and the same period a year earlier."""

    model_config = ConfigDict(frozen=True)

    name: str
    current: Fact
    prior: Fact | None
    change_pct: Decimal | None  # None when there's no prior value or it is zero

    @property
    def concept(self) -> str:
        return self.current.concept

    @property
    def unit(self) -> str:
        return self.current.unit


def parse_facts(company_facts: dict[str, Any], concepts: set[str]) -> list[Fact]:
    """Pull the requested us-gaap concepts out of a companyfacts JSON document."""
    us_gaap = company_facts["facts"].get("us-gaap", {})
    facts = []
    for concept in sorted(concepts):
        for unit, entries in us_gaap.get(concept, {}).get("units", {}).items():
            for entry in entries:
                facts.append(
                    Fact(
                        concept=concept,
                        unit=unit,
                        # The JSON parser hands us floats for decimals; str() gives back the
                        # exact digits SEC sent (0.1 stays "0.1"), and Decimal keeps them.
                        value=Decimal(str(entry["val"])),
                        start=entry.get("start"),
                        end=entry["end"],
                        fiscal_year=entry.get("fy"),
                        fiscal_period=entry.get("fp"),
                        form=entry["form"],
                        accession_number=entry["accn"],
                        filed=entry["filed"],
                    )
                )
    return facts


def key_metrics(facts: list[Fact], accession_number: str, form: str) -> list[MetricValue]:
    """The METRICS as reported by one filing. Metrics it doesn't report are left out."""
    try:
        period_days = _PERIOD_DAYS[form.removesuffix("/A")]
    except KeyError:
        raise ValueError(f"no standard reporting period for form {form!r}") from None

    # Every filing repeats earlier periods for comparison, and companyfacts lists each
    # repetition as its own fact. Keeping only this filing's facts means we report exactly
    # what this filing said, including any restated comparison numbers.
    in_filing = [fact for fact in facts if fact.accession_number == accession_number]

    results = []
    for metric in METRICS:
        for concept in metric.concepts:
            candidates = [
                fact
                for fact in in_filing
                if fact.concept == concept
                and fact.unit == metric.unit
                and _is_reporting_period(fact, metric.kind, period_days)
            ]
            if candidates:
                current = max(candidates, key=lambda fact: fact.end)
                prior = _year_earlier(current, candidates)
                if prior is None and metric.kind == "instant":
                    # Balance sheets compare with the last fiscal year-end, not necessarily
                    # the same day a year earlier, so use the filing's own comparison date.
                    prior = _latest_before(current, candidates)
                results.append(
                    MetricValue(
                        name=metric.name,
                        current=current,
                        prior=prior,
                        change_pct=_change_pct(current, prior),
                    )
                )
                break  # first concept that the filing reports wins
    return results


def _is_reporting_period(fact: Fact, kind: str, period_days: tuple[int, int]) -> bool:
    if kind == "instant":
        return fact.days is None
    shortest, longest = period_days
    return fact.days is not None and shortest <= fact.days <= longest


def _year_earlier(current: Fact, candidates: list[Fact]) -> Fact | None:
    target = current.end - _ONE_YEAR
    for fact in candidates:
        if abs((fact.end - target).days) <= _TOLERANCE_DAYS:
            return fact
    return None


def _latest_before(current: Fact, candidates: list[Fact]) -> Fact | None:
    earlier = [fact for fact in candidates if fact.end < current.end]
    return max(earlier, key=lambda fact: fact.end, default=None)


def _change_pct(current: Fact, prior: Fact | None) -> Decimal | None:
    if prior is None or prior.value == 0:
        return None
    # abs() keeps the sign meaningful when the prior value is negative: a loss of 500
    # turning into a profit of 1,000 is +300%, not -300%.
    return (current.value - prior.value) / abs(prior.value) * 100


def format_period(value: MetricValue) -> str:
    """'2026-07-01 to 2026-09-30' for durations; 'end vs prior end' for instants."""
    current, prior = value.current, value.prior
    if current.start is not None:
        return f"{current.start.isoformat()} to {current.end.isoformat()}"
    # Instants: show the comparison date, since it isn't always exactly a year earlier.
    if prior is not None:
        return f"{current.end.isoformat()} vs {prior.end.isoformat()}"
    return f"at {current.end.isoformat()}"


def format_amount(value: Decimal, unit: str) -> str:
    """USD as readable millions ('$28,046M'), EPS with 2 decimals ('$0.22')."""
    sign = "-" if value < 0 else ""
    if unit == "USD":
        return f"{sign}${abs(value) / 1_000_000:,.0f}M"
    if unit == "USD/shares":
        return f"{sign}${abs(value):,.2f}"
    return f"{value:,} {unit}"
