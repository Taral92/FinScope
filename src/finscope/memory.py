"""Agent memory: what FinScope found before, and how you rated it. Plain code, no LLM.

Two kinds, recalled by one tool (agent_tools.recall_past_findings):
- company history: findings from this company's earlier filings, same section;
- your preferences: past findings from any company that you rated, with your verdict.
Recall is word overlap between the changed paragraph and each past finding's WHAT CHANGED and
triage reason: shared words over the smaller set's words (the overlap coefficient), since a
paragraph is much longer than a finding and plain Jaccard would punish that. No embeddings.

The findings come from files FinScope already writes (memory_store.load_memory). This module
imports none of them: the explain agent's tools import it, and the reports import those tools.
"""

import re
from collections.abc import Callable
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

from finscope.triage import Importance

# decide.Tier and feedback.Verdict, repeated: both modules import this one through the
# agent's tools, so importing them here would be a cycle. A test keeps the copies equal.
Tier = Literal["alert", "digest", "review"]
Verdict = Literal["useful", "not_useful", "wrong"]

HISTORY_LIMIT = 2  # earlier findings for this company shown per recall
RATED_LIMIT = 3  # rated findings from any company shown per recall
# Set from real Starbucks paragraphs: +5 (reporting rules) matched its removed old version at
# 0.50 and +1 (strategy) Disney's strategy alert at 0.15; unrelated findings scored lower.
MIN_SIMILARITY = 0.15
MAX_TEXT_CHARS = 300  # of one past WHAT CHANGED in the tool's answer

_WORD = re.compile(r"[a-z][a-z'-]+")
# Words too common in risk factors to say anything about what a change is about.
_STOPWORDS = frozenset(
    {
        # grammar
        "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been",
        "being", "but", "by", "can", "could", "do", "does", "for", "from", "had", "has",
        "have", "if", "in", "into", "is", "it", "its", "may", "might", "more", "most", "must",
        "no", "not", "of", "on", "or", "other", "our", "such", "than", "that", "the", "their",
        "them", "there", "these", "they", "this", "those", "to", "under", "us", "was", "we",
        "were", "what", "when", "which", "while", "who", "will", "with", "would",
        "over", "across", "among", "against", "between", "before", "during", "including",
        "through", "within", "without", "some", "certain", "each", "both", "only",
        # words nearly every risk factor uses
        "company", "company's", "companies", "business", "results", "adversely", "affect",
        "affected", "financial", "operations", "risk", "risks", "added", "disclosure",
        # how a change is described, not what it is about: "costs" and "new" alone matched an
        # insurance-cost query to the tariff findings (retrieval eval, false recall)
        "new", "cost", "costs", "expense", "expenses", "raise", "raises", "rise", "rising",
        "increase", "increases", "increased", "reduce", "reduced", "higher", "lower",
    }
)  # fmt: skip


class PastFinding(BaseModel):
    """One explained change from an earlier report, and your verdict if you rated it."""

    model_config = ConfigDict(frozen=True)

    company: str  # ticker, e.g. TSLA
    form: str
    section: str  # e.g. "I-1A", as in report file names
    filing_date: date
    accession: str  # the filing the finding is about
    change_id: str
    what_changed: str  # the agent's WHAT CHANGED; the answer passed the fact-check
    triage_reason: str
    importance: Importance | None
    tier: Tier
    verdict: Verdict | None  # None: not rated
    note: str | None


def recall(
    memory: tuple[PastFinding, ...], change_text: str, *, company: str, section: str
) -> tuple[list[PastFinding], list[PastFinding]]:
    """(this company's earlier findings in this section, rated findings from any company),
    each most similar first and above MIN_SIMILARITY. A finding is shown in one list only."""
    ranked = [finding for _, finding in rank(memory, change_text)]
    history = [f for f in ranked if f.company == company and f.section == section][:HISTORY_LIMIT]
    rated = [f for f in ranked if f.verdict is not None and f not in history][:RATED_LIMIT]
    return history, rated


Similarity = Callable[[frozenset[str], frozenset[str]], float]


def rank(
    memory: tuple[PastFinding, ...],
    change_text: str,
    *,
    min_similarity: float = MIN_SIMILARITY,
    similarity: Similarity | None = None,
) -> list[tuple[float, PastFinding]]:
    """Every finding scoring at least `min_similarity`, most similar first: what recall picks
    from. `similarity` defaults to overlap_similarity; the retrieval eval tries others."""
    score_of = similarity or overlap_similarity
    query = words(change_text)
    scored = [
        (score, finding)
        for finding in memory
        if (score := score_of(query, words(f"{finding.what_changed} {finding.triage_reason}")))
        >= min_similarity
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1].filing_date, pair[1].change_id))
    return scored


def format_recall(
    history: list[PastFinding], rated: list[PastFinding], *, company: str, section: str
) -> str:
    """The tool's answer: both lists, each finding on one line."""
    lines = [
        "FinScope's own earlier findings (its past wording, not filing text: never quote it "
        "as evidence).",
        f"Earlier filings of {company}, section {section}:",
    ]
    lines += [_line(f, with_company=False) for f in history] or ["(none similar)"]
    lines.append("Similar findings the user rated, any company:")
    lines += [_line(f, with_company=True) for f in rated] or ["(none similar)"]
    return "\n".join(lines)


def _line(finding: PastFinding, *, with_company: bool) -> str:
    where = f"{finding.company} " if with_company else ""
    rated = f", user said {finding.verdict}" if finding.verdict else ", not rated"
    if finding.note:
        rated += f' ("{finding.note}")'
    text = finding.what_changed
    if len(text) > MAX_TEXT_CHARS:
        text = text[: MAX_TEXT_CHARS - 1] + "…"
    return (
        f"- {where}{finding.form} filed {finding.filing_date.isoformat()} [{finding.change_id}]: "
        f"rated {finding.importance or 'none'}, {finding.tier}{rated}. {text}"
    )


def words(text: str) -> frozenset[str]:
    """The text's words that say what it is about: lowercased, stopwords left out."""
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS)


def overlap_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Shared words over the smaller set's words: 1 when one's words all appear in the other."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Shared words over all words: punishes a long paragraph compared with a short finding."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
