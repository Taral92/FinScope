"""Send a check's decision to Telegram. Plain code, no LLM.

The message follows decide.py's rule: only an alert carries the agent's text. A review entry
shows its fixed reason, and digest entries are only counted.

The bot token is part of every Telegram URL, and httpx puts the URL in its error messages.
So no httpx error is ever passed on: every failure becomes a NotifyError whose message we
write ourselves. httpx also logs every request URL at INFO level, and the MCP SDK turns
logging on when an MCP server is built in-process (found 2026-11-22), so the "httpx" logger is
held at WARNING here: request URLs, and with them the token, are never logged.
"""

import logging
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Self

import httpx

from finscope.config import Settings
from finscope.detect import CheckResult
from finscope.synthesis import FilingSummary

# Set on import, before any client exists. A level on this logger wins over whatever level the
# root logger is given later (MCPServer sets INFO), so URLs stay out of every log.
logging.getLogger("httpx").setLevel(logging.WARNING)

TELEGRAM_API = "https://api.telegram.org"
# Telegram's limit for one message, counted in UTF-16 code units.
MAX_MESSAGE_CHARS = 4096
# Room for the "(12/12) " part number put in front of each part of a split message.
_PART_PREFIX_RESERVE = 16
# One failed entry's error in a failure notice: enough to see what broke, not a whole dump.
MAX_ERROR_CHARS = 300


class NotifyError(Exception):
    """Sending failed. The message is ours and never contains the bot token."""


def format_failure_message(failures: Sequence[tuple[str, str]], checked: int) -> str:
    """The notice for a check where entries failed: each entry's label and error.

    A scheduled check runs unattended, so a failure that only reached the log would go unseen.
    """
    lines = [f"FinScope check: {len(failures)} of {checked} failed"]
    for label, error in failures:
        short = error if len(error) <= MAX_ERROR_CHARS else error[: MAX_ERROR_CHARS - 1] + "…"
        lines.append(f"{label}: {short}")
    lines.append("The other entries ran; a failed one is retried on the next check.")
    return "\n".join(lines)


def telegram_configured(settings: Settings) -> bool:
    return settings.telegram_bot_token is not None and settings.telegram_chat_id is not None


def format_alert_message(result: CheckResult) -> str | None:
    """The plain-text message for one check result, or None when there is nothing to send."""
    decision = result.decision
    alerts, reviews = decision.tier("alert"), decision.tier("review")
    if not alerts and not reviews:
        return None

    entry, filing = result.entry, result.new_filing
    blocks = [
        f"FinScope: {entry.ticker} {entry.form}, {_counts(len(alerts), len(reviews))}\n"
        f"{result.report.section_title}\n"
        f"Filed {filing.filing_date.isoformat()}, accession {filing.accession_number}"
    ]
    blocks += _entry_blocks(result)
    blocks.append(f"Also: {len(decision.tier('digest'))} digest")
    return "\n\n".join(blocks)


def format_filing_message(
    results: Sequence[CheckResult], *, summary: FilingSummary | None = None
) -> str | None:
    """One message for the sections of one filing compared in one check; None if there is
    nothing to send. A single section with something to send reads exactly as
    format_alert_message, so a watchlist with one section per company sees no change. The
    summary's kept points, if any, come first; dropped points are never shown."""
    sending = [r for r in results if r.decision.tier("alert") or r.decision.tier("review")]
    if len(sending) <= 1:
        return format_alert_message(sending[0]) if sending else None

    first = sending[0]
    alerts = sum(len(r.decision.tier("alert")) for r in sending)
    reviews = sum(len(r.decision.tier("review")) for r in sending)
    digest = sum(len(r.decision.tier("digest")) for r in results)
    filing = first.new_filing
    blocks = [
        f"FinScope: {first.entry.ticker} {first.entry.form}, {_counts(alerts, reviews)} "
        f"in {len(sending)} sections\n"
        f"Filed {filing.filing_date.isoformat()}, accession {filing.accession_number}"
    ]
    if summary is not None and summary.kept:
        lines = [
            f"{number}. {point.text} [{'; '.join(point.refs)}]"
            for number, point in enumerate(summary.kept, start=1)
        ]
        blocks.append("SUMMARY\n" + "\n".join(lines))
    for result in sending:
        blocks.append(f"== {result.report.section_title} ==")
        blocks += _entry_blocks(result)
    blocks.append(f"Also: {digest} digest")
    return "\n\n".join(blocks)


def _counts(alerts: int, reviews: int) -> str:
    counts = [f"{alerts} alert{'' if alerts == 1 else 's'}"] if alerts else []
    counts += [f"{reviews} to review"] if reviews else []
    return ", ".join(counts)


def _entry_blocks(result: CheckResult) -> list[str]:
    """A section's alerts (with the agent's text) then its reviews (fixed reasons only)."""
    blocks = [
        f"ALERT [{alert.id}]\n"
        f"WHAT CHANGED: {alert.what_changed}\n"
        f"WHY IT MATTERS: {alert.why_it_matters}"
        for alert in result.decision.tier("alert")
    ]
    blocks += [
        f"REVIEW [{review.id}]: {review.reason}" for review in result.decision.tier("review")
    ]
    return blocks


def split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split at blank lines (entry boundaries) into parts Telegram accepts, numbered "(1/2)".

    A single entry longer than a whole message is cut, the only case that splits an entry.
    """
    if _length(text) <= limit:
        return [text]
    budget = limit - _PART_PREFIX_RESERVE
    parts: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        for piece in _cut(block, budget):
            joined = f"{current}\n\n{piece}" if current else piece
            if current and _length(joined) > budget:
                parts.append(current)
                current = piece
            else:
                current = joined
    parts.append(current)
    return [f"({number}/{len(parts)}) {part}" for number, part in enumerate(parts, 1)]


def _length(text: str) -> int:
    """Length as Telegram counts it: UTF-16 code units, so an emoji counts as 2."""
    return len(text.encode("utf-16-le")) // 2


def _cut(block: str, budget: int) -> list[str]:
    """`block` in pieces of at most `budget` UTF-16 units (just [block] when it fits)."""
    pieces = []
    start = used = 0
    for index, char in enumerate(block):
        size = 2 if ord(char) > 0xFFFF else 1
        if used + size > budget:
            pieces.append(block[start:index])
            start, used = index, 0
        used += size
    pieces.append(block[start:])
    return pieces


class TelegramNotifier:
    """A Telegram bot that sends plain-text messages. Needs FINSCOPE_TELEGRAM_BOT_TOKEN."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        if settings.telegram_bot_token is None:
            raise NotifyError("Telegram not configured: set FINSCOPE_TELEGRAM_BOT_TOKEN")
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._client = httpx.Client(timeout=30.0, transport=transport)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def send(self, text: str) -> None:
        """Send `text` to the configured chat, split into several messages if it's too long."""
        if self._chat_id is None:
            raise NotifyError("Telegram chat not configured: set FINSCOPE_TELEGRAM_CHAT_ID")
        parts = split_message(text)
        for number, part in enumerate(parts, 1):
            where = f" (message {number} of {len(parts)})" if len(parts) > 1 else ""
            # Plain text, no parse_mode: nothing in a filing can break Markdown or HTML parsing.
            self._call("sendMessage", {"chat_id": self._chat_id, "text": part}, where)

    def chats(self) -> list[tuple[str, str]]:
        """(chat ID, name) of every chat in the bot's recent updates, each chat once."""
        updates = self._call("getUpdates", {}, "")
        found: dict[str, str] = {}
        for update in updates if isinstance(updates, list) else []:
            # The chat sits in whichever kind of update it is: message, channel_post, ...
            for value in update.values() if isinstance(update, dict) else []:
                chat = value.get("chat") if isinstance(value, dict) else None
                if isinstance(chat, dict) and "id" in chat:
                    found.setdefault(str(chat["id"]), _chat_name(chat))
        return list(found.items())

    def _call(self, method: str, payload: Mapping[str, Any], where: str) -> Any:
        """POST one Bot API method and return its "result". Raises NotifyError on any failure.

        The NotifyError is raised outside the `except` block on purpose, so it has no
        __context__ either: a traceback can't show the httpx error and its URL.
        """
        token = self._token.get_secret_value()
        error_name = None
        try:
            response = self._client.post(f"{TELEGRAM_API}/bot{token}/{method}", json=payload)
        except Exception as error:  # any httpx error; its text may hold the URL
            error_name = type(error).__name__
        if error_name is not None:
            raise NotifyError(f"Telegram {method} failed{where}: {error_name}")

        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or body.get("ok") is not True:
            description = body.get("description") if isinstance(body, dict) else None
            # Telegram's own words, minus the token in case it ever echoes it back.
            safe = str(description or "no description").replace(token, "<token>")
            raise NotifyError(
                f"Telegram {method} failed{where}: HTTP {response.status_code}: {safe}"
            )
        return body.get("result")


def _chat_name(chat: Mapping[str, Any]) -> str:
    if title := chat.get("title"):
        return f"{title} ({chat.get('type', 'chat')})"
    name = " ".join(str(chat[key]) for key in ("first_name", "last_name") if chat.get(key))
    if username := chat.get("username"):
        name = f"{name} @{username}".strip()
    return f"{name or 'no name'} ({chat.get('type', 'chat')})"
