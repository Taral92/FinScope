from datetime import date
from decimal import Decimal
from typing import Any

from finscope.xbrl import Fact, MetricValue, key_metrics, parse_facts

THIS_10Q = "0001628280-26-066775"  # the filing under test (Q3 2026)
OLDER_10Q = "0001628280-25-045968"  # last year's Q3 10-Q


def entry(
    start: str | None, end: str, val: float, accn: str = THIS_10Q, filed: str = "2026-10-22"
) -> dict[str, Any]:
    """One fact as it appears in companyfacts JSON (instants have no "start")."""
    fact: dict[str, Any] = {
        "end": end,
        "val": val,
        "accn": accn,
        "fy": 2026,
        "fp": "Q3",
        "form": "10-Q",
        "filed": filed,
    }
    if start is not None:
        fact["start"] = start
    return fact


def company_facts(us_gaap: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    concepts = {concept: {"units": units} for concept, units in us_gaap.items()}
    return {"cik": 1318605, "entityName": "Tesla, Inc.", "facts": {"us-gaap": concepts}}


FAKE_TESLA = company_facts(
    {
        "Revenues": {
            "USD": [
                # This filing: Q3 and 9-month year-to-date, for this year and last year.
                entry("2026-07-01", "2026-09-30", 28_046_000_000),
                entry("2026-01-01", "2026-09-30", 78_669_000_000),
                entry("2025-07-01", "2025-09-30", 28_095_000_000),
                entry("2025-01-01", "2025-09-30", 69_926_000_000),
                # Last year's 10-Q reported Q3 2025 before a (pretend) restatement.
                entry("2025-07-01", "2025-09-30", 28_000_000_000, OLDER_10Q, "2025-10-23"),
            ]
        },
        "OperatingIncomeLoss": {
            "USD": [
                entry("2026-07-01", "2026-09-30", 1_000_000),
                entry("2025-07-01", "2025-09-30", 0),
            ]
        },
        "NetIncomeLoss": {
            "USD": [
                entry("2026-07-01", "2026-09-30", 1_000),
                entry("2025-07-01", "2025-09-30", -500),
            ]
        },
        "EarningsPerShareDiluted": {
            "USD/shares": [
                entry("2026-07-01", "2026-09-30", 0.1),
                entry("2025-07-01", "2025-09-30", 0.4),
            ]
        },
        "CashAndCashEquivalentsAtCarryingValue": {
            "USD": [
                entry(None, "2026-09-30", 17_853_000_000),
                entry(None, "2025-12-31", 16_513_000_000),
                entry(None, "2025-09-30", 18_289_000_000),
            ]
        },
    }
)

# Apple-style: an old filing used "Revenues", the current one uses the newer concept.
FAKE_APPLE = company_facts(
    {
        "Revenues": {"USD": [entry("2017-07-02", "2017-09-30", 52_579_000_000, OLDER_10Q)]},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "USD": [
                entry("2026-03-29", "2026-06-27", 94_036_000_000),
                entry("2025-03-30", "2025-06-28", 85_777_000_000),
            ]
        },
        # Balance-sheet style: this quarter-end vs. the last fiscal year-end only.
        "CashAndCashEquivalentsAtCarryingValue": {
            "USD": [
                entry(None, "2026-06-27", 39_544_000_000),
                entry(None, "2025-09-27", 35_934_000_000),
            ]
        },
    }
)

ALL_CONCEPTS = {
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    "CashAndCashEquivalentsAtCarryingValue",
}


def metrics(data: dict[str, Any], accession_number: str = THIS_10Q) -> dict[str, MetricValue]:
    values = key_metrics(parse_facts(data, ALL_CONCEPTS), accession_number, "10-Q")
    return {value.name: value for value in values}


def test_parse_facts_reads_values_periods_and_filing_details() -> None:
    facts = parse_facts(FAKE_TESLA, {"CashAndCashEquivalentsAtCarryingValue"})

    assert facts[0] == Fact(
        concept="CashAndCashEquivalentsAtCarryingValue",
        unit="USD",
        value=Decimal("17853000000"),
        start=None,
        end=date(2026, 9, 30),
        fiscal_year=2026,
        fiscal_period="Q3",
        form="10-Q",
        accession_number=THIS_10Q,
        filed=date(2026, 10, 22),
    )


def test_quarter_is_chosen_over_year_to_date() -> None:
    revenue = metrics(FAKE_TESLA)["Revenue"]

    assert revenue.current.value == Decimal("28046000000")
    assert (revenue.current.start, revenue.current.end) == (date(2026, 7, 1), date(2026, 9, 30))


def test_prior_is_same_quarter_last_year_as_reported_in_this_filing() -> None:
    prior = metrics(FAKE_TESLA)["Revenue"].prior

    assert prior is not None
    assert (prior.start, prior.end) == (date(2025, 7, 1), date(2025, 9, 30))
    assert prior.value == Decimal("28095000000")  # not the older filing's 28,000,000,000
    assert prior.accession_number == THIS_10Q


def test_facts_from_other_filings_are_ignored() -> None:
    older = metrics(FAKE_TESLA, accession_number=OLDER_10Q)

    # The older filing only reported revenue, and only its own numbers.
    assert list(older) == ["Revenue"]
    assert older["Revenue"].current.value == Decimal("28000000000")
    assert older["Revenue"].prior is None


def test_falls_back_to_next_revenue_concept() -> None:
    revenue = metrics(FAKE_APPLE)["Revenue"]

    assert revenue.concept == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert revenue.current.value == Decimal("94036000000")


def test_change_pct() -> None:
    found = metrics(FAKE_TESLA)

    assert found["Revenue"].change_pct == Decimal(28_046 - 28_095) / 28_095 * 100
    assert found["Net income"].change_pct == Decimal(300)  # -500 -> 1,000
    assert found["Operating income"].change_pct is None  # prior was zero


def test_cash_prefers_the_same_date_a_year_earlier() -> None:
    cash = metrics(FAKE_TESLA)["Cash"]

    assert cash.current.end == date(2026, 9, 30)
    assert cash.prior is not None
    assert cash.prior.end == date(2025, 9, 30)  # not the nearer 2025-12-31


def test_cash_falls_back_to_the_filings_other_date() -> None:
    cash = metrics(FAKE_APPLE)["Cash"]

    assert cash.current.end == date(2026, 6, 27)
    assert cash.prior is not None
    assert cash.prior.end == date(2025, 9, 27)  # the last fiscal year-end


def test_values_stay_exact_decimals() -> None:
    eps = metrics(FAKE_TESLA)["Diluted EPS"]

    assert eps.current.value == Decimal("0.1")
    # (0.1 - 0.4) / 0.4 * 100 is exactly -75; with floats Python gives -75.00000000000001.
    assert eps.change_pct == Decimal(-75)


def test_missing_metrics_are_left_out() -> None:
    assert "Gross profit" not in metrics(FAKE_TESLA)
    assert list(metrics(FAKE_APPLE)) == ["Revenue", "Cash"]
