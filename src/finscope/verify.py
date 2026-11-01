"""Fact-check an explanation against the text the model was given. Plain code, no LLM.

Known limits (by design: a false alarm is safe, a missed fabrication is not):
- Numbers are compared by value only: "$28,046M" matches "28,046" in a source, but a
  rewritten form such as "$28.0 billion" is flagged even when it is a correct rounding.
- A number counts as supported if it appears anywhere in the sources, even in an unrelated
  place (a "2025" claim is satisfied by any 2025 in the text). Numbers written as words
  ("three") are not checked.
- A quote only has to appear somewhere in one source; whether it supports the claim it is
  attached to is not checked. Neither is the reasoning in WHY IT MATTERS or IMPORTANCE.
"""

import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from finscope.diff import ParagraphChange
from finscope.text import unify_typography

_LABELS = ("WHAT CHANGED", "WHY IT MATTERS", "IMPORTANCE", "EVIDENCE")
_LABELED_LINE = re.compile(rf"\s*({'|'.join(_LABELS)})\s*:\s*(.*)", re.IGNORECASE)
_LEVEL_AND_REASON = re.compile(r"(\w+)\s*(?:[-:]\s*(.*))?", re.DOTALL)
_QUOTED = re.compile(r'"([^"]+)"')  # applied after curly quotes become straight ones
_ELLIPSIS = re.compile(r"\.\.\.|…")
_WHITESPACE = re.compile(r"\s+")
# A number as a claim: "28,046", "$28,046M", "12.5%", "2025". The lookarounds skip digits
# inside identifiers such as "Q3", "1A" or "10-K".
_NUMBER = r"\d(?:[\d,]*\d)?(?:\.\d+)?"  # ends in a digit: "2025," is 2025
_CLAIMED_NUMBER = re.compile(rf"(?<![\w.])\$?({_NUMBER})(?:%|[MBK](?![a-zA-Z]))?(?![\w-])")
# Any digits in a source count: being generous here can only prevent false alarms.
_SOURCE_NUMBER = re.compile(_NUMBER)


class Issue(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["format", "quote", "number"]
    message: str


class ParsedExplanation(BaseModel):
    """The labeled parts of an answer. Missing parts are None; parse problems are issues."""

    model_config = ConfigDict(frozen=True)

    what_changed: str | None
    why_it_matters: str | None
    importance: Literal["low", "medium", "high"] | None
    importance_reason: str | None
    evidence: tuple[str, ...]
    issues: tuple[Issue, ...]


class Verification(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    issues: tuple[Issue, ...]
    quotes_checked: int
    numbers_checked: int


def parse_explanation(text: str, *, expect_evidence: bool = True) -> ParsedExplanation:
    """Split an answer into its labeled lines. Lines without a label continue the previous one.

    `expect_evidence=False` is for the single-call explainer, whose format has no EVIDENCE line.
    """
    parts: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        if match := _LABELED_LINE.fullmatch(line):
            current = match[1].upper()
            parts[current] = match[2].strip()
        elif current and line.strip():
            parts[current] += " " + line.strip()

    issues = [
        Issue(kind="format", message=f"missing {label} line")
        for label in _LABELS
        if label not in parts and (label != "EVIDENCE" or expect_evidence)
    ]

    importance: str | None = None
    reason: str | None = None
    if "IMPORTANCE" in parts:
        match = _LEVEL_AND_REASON.fullmatch(unify_typography(parts["IMPORTANCE"]))
        if match and match[1].lower() in ("low", "medium", "high"):
            importance = match[1].lower()  # "Medium" and "MEDIUM" both become "medium"
            reason = (match[2] or "").strip() or None
        else:
            message = f"unknown IMPORTANCE level in {parts['IMPORTANCE']!r}"
            issues.append(Issue(kind="format", message=message))

    evidence: tuple[str, ...] = ()
    if "EVIDENCE" in parts:
        evidence = tuple(q.strip() for q in _QUOTED.findall(unify_typography(parts["EVIDENCE"])))
        if not evidence:
            issues.append(Issue(kind="format", message="EVIDENCE line has no quoted text"))

    return ParsedExplanation(
        what_changed=parts.get("WHAT CHANGED"),
        why_it_matters=parts.get("WHY IT MATTERS"),
        importance=importance,
        importance_reason=reason,
        evidence=evidence,
        issues=tuple(issues),
    )


def sources_for(change: ParagraphChange, tool_outputs: list[str] | None = None) -> list[str]:
    """Exactly what the model saw: the changed paragraph(s), plus any tool results."""
    paragraphs = [text for text in (change.old_text, change.new_text) if text is not None]
    return paragraphs + (tool_outputs or [])


def verify_explanation(parsed: ParsedExplanation, sources: list[str]) -> Verification:
    """Check quotes and numbers against the sources: exactly the texts the model saw."""
    issues = list(parsed.issues)
    normalized_sources = [_normalize(source) for source in sources]

    for quote in parsed.evidence:
        if not _quote_found(quote, normalized_sources):
            issues.append(
                Issue(kind="quote", message=f'"{_shorten(quote)}" not found in the sources')
            )

    source_numbers = {
        _number_value(match)
        for source in normalized_sources
        for match in _SOURCE_NUMBER.findall(source)
    }
    claims = " ".join(part for part in (parsed.what_changed, parsed.why_it_matters) if part)
    numbers_checked = 0
    for match in _CLAIMED_NUMBER.finditer(unify_typography(claims)):
        numbers_checked += 1
        if _number_value(match[1]) not in source_numbers:
            issues.append(Issue(kind="number", message=f"{match[0]} not found in the sources"))

    return Verification(
        passed=not issues,
        issues=tuple(issues),
        quotes_checked=len(parsed.evidence),
        numbers_checked=numbers_checked,
    )


def _quote_found(quote: str, normalized_sources: list[str]) -> bool:
    """True if the quote is in one source; a quote with "..." needs every piece in that source."""
    pieces = [_normalize(piece) for piece in _ELLIPSIS.split(quote)]
    pieces = [piece for piece in pieces if piece]
    return bool(pieces) and any(
        all(piece in source for piece in pieces) for source in normalized_sources
    )


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", unify_typography(text)).strip()


def _number_value(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def _shorten(text: str, width: int = 80) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"
