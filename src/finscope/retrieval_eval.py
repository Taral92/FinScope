"""Retrieval eval for the agent's memory: does recall bring back the right past findings?

Plain code, no LLM, no cost. Known-answer cases (evals/known/retrieval_cases.json): made-up past
findings, and queries (changed paragraphs) that each name the findings relevant to them, or
none. memory.rank, the exact code the agent's recall uses, ranks the findings for each query;
the top k above the threshold are what recall would show. Standard retrieval metrics:
- recall@k: relevant findings retrieved / relevant findings (all queries pooled);
- precision@k: relevant findings retrieved / findings retrieved (all queries pooled);
- MRR: mean over queries with a relevant finding of 1 / rank of the first one retrieved
  (0 when none is);
- false recall: share of queries with nothing relevant that still retrieve something.
A sweep repeats them for each scoring method and threshold, to see whether the ones memory uses
(overlap at MIN_SIMILARITY) are the right choice.
"""

from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from finscope.evals import write_json_atomic
from finscope.memory import (
    RATED_LIMIT,
    PastFinding,
    Similarity,
    jaccard_similarity,
    overlap_similarity,
    rank,
)

RETRIEVAL_CASES_PATH = Path("evals") / "known" / "retrieval_cases.json"
K = RATED_LIMIT  # recall shows up to this many findings, so that is what is scored
METHODS: dict[str, Similarity] = {"overlap": overlap_similarity, "jaccard": jaccard_similarity}
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
QueryKind = Literal["easy", "paraphrase", "hard-negative", "nothing-relevant"]


class RetrievalCasesError(Exception):
    """The case file can't be used; the message says why."""


class CaseFinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    company: str
    section: str
    what_changed: str = Field(min_length=1)
    triage_reason: str

    def as_past_finding(self) -> PastFinding:
        return PastFinding(
            company=self.company,
            form="10-K",
            section=self.section,
            filing_date=date(2025, 1, 1),
            accession=f"known-{self.company}",
            change_id=self.id,  # the id is what the metrics compare
            what_changed=self.what_changed,
            triage_reason=self.triage_reason,
            importance="high",
            tier="alert",
            verdict=None,
            note=None,
        )


class Query(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    company: str
    section: str
    text: str = Field(min_length=1)
    relevant: tuple[str, ...]  # finding ids; empty: nothing should come back
    kind: QueryKind
    why: str = Field(min_length=1)

    @model_validator(mode="after")
    def _kind_fits(self) -> "Query":
        if (self.kind == "nothing-relevant") != (not self.relevant):
            raise ValueError(f"query {self.name!r}: only nothing-relevant queries have no relevant")
        return self


class RetrievalCases(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    findings: tuple[CaseFinding, ...]
    queries: tuple[Query, ...]


class QueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: QueryKind
    relevant: tuple[str, ...]
    retrieved: tuple[str, ...]  # the top K above the threshold, most similar first

    @property
    def hits(self) -> int:
        return sum(1 for finding in self.retrieved if finding in self.relevant)

    @property
    def reciprocal_rank(self) -> float:
        first = next((i for i, f in enumerate(self.retrieved, 1) if f in self.relevant), None)
        return 1 / first if first else 0.0


class Scores(BaseModel):
    """The metrics for one scoring method at one threshold."""

    model_config = ConfigDict(frozen=True)

    method: str
    threshold: float
    recall_at_k: float | None  # None when no query has a relevant finding
    precision_at_k: float | None  # None when nothing was retrieved at all
    mrr: float | None
    false_recall: float | None  # None when no query is nothing-relevant
    queries: tuple[QueryResult, ...]


class RetrievalEvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at: datetime
    cases_file: str
    k: int
    current: Scores  # what memory uses now: overlap at MIN_SIMILARITY
    sweep: tuple[Scores, ...]  # every method at every threshold

    @property
    def file_name(self) -> str:
        return f"{self.created_at:%Y%m%dT%H%M%SZ}_retrieval_known.json"


def load_retrieval_cases(path: Path) -> RetrievalCases:
    """The cases, or a RetrievalCasesError: bad JSON, an unknown or duplicated id."""
    try:
        cases = RetrievalCases.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RetrievalCasesError(f"can't read {path}: {error}") from error
    except ValidationError as error:
        raise RetrievalCasesError(f"{path} is not a valid case file:\n{error}") from error
    problems = []
    ids = Counter(finding.id for finding in cases.findings)
    if duplicates := sorted(i for i, n in ids.items() if n > 1):
        problems.append(f"duplicated finding ids: {', '.join(duplicates)}")
    names = Counter(query.name for query in cases.queries)
    if duplicates := sorted(n for n, count in names.items() if count > 1):
        problems.append(f"duplicated query names: {', '.join(duplicates)}")
    for query in cases.queries:
        if unknown := sorted(set(query.relevant) - set(ids)):
            problems.append(f"query {query.name!r} names unknown findings: {', '.join(unknown)}")
    if problems:
        raise RetrievalCasesError(f"{path}: " + "; ".join(problems))
    return cases


def score(cases: RetrievalCases, *, method: str, threshold: float, k: int = K) -> Scores:
    memory = tuple(finding.as_past_finding() for finding in cases.findings)
    results = tuple(
        QueryResult(
            name=query.name,
            kind=query.kind,
            relevant=query.relevant,
            retrieved=tuple(
                finding.change_id
                for _, finding in rank(
                    memory, query.text, min_similarity=threshold, similarity=METHODS[method]
                )[:k]
            ),
        )
        for query in cases.queries
    )
    with_relevant = [r for r in results if r.relevant]
    without = [r for r in results if not r.relevant]
    retrieved = sum(len(r.retrieved) for r in results)
    relevant = sum(len(r.relevant) for r in with_relevant)
    hits = sum(r.hits for r in results)
    return Scores(
        method=method,
        threshold=threshold,
        recall_at_k=hits / relevant if relevant else None,
        precision_at_k=hits / retrieved if retrieved else None,
        mrr=sum(r.reciprocal_rank for r in with_relevant) / len(with_relevant)
        if with_relevant
        else None,
        false_recall=sum(1 for r in without if r.retrieved) / len(without) if without else None,
        queries=results,
    )


def run_retrieval_eval(
    cases: RetrievalCases,
    *,
    cases_file: Path,
    current_threshold: float,
    created_at: datetime,
    thresholds: Sequence[float] = THRESHOLDS,
) -> RetrievalEvalResult:
    return RetrievalEvalResult(
        created_at=created_at,
        cases_file=str(cases_file),
        k=K,
        current=score(cases, method="overlap", threshold=current_threshold),
        sweep=tuple(
            score(cases, method=method, threshold=threshold)
            for method in METHODS
            for threshold in thresholds
        ),
    )


def save_retrieval_eval(result: RetrievalEvalResult, directory: Path) -> Path:
    path = directory / result.file_name
    write_json_atomic(path, result)
    return path
