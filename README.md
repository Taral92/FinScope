# FinScope

[![CI](https://github.com/Taral92/FinScope/actions/workflows/ci.yml/badge.svg)](https://github.com/Taral92/FinScope/actions/workflows/ci.yml)

An autonomous monitor for SEC filings. FinScope watches companies' filings on
[SEC EDGAR](https://www.sec.gov/edgar). When a new filing appears, AI agents compare it to the
previous one, find what changed and why it matters, fact-check their claims, and alert you only
when it's important.

## How it works

1. **Detect** a new filing on EDGAR.
2. **Prepare** it: download, split into sections, extract XBRL numbers.
3. **Analyze** the changes with AI agents.
4. **Fact-check** the agents' claims.
5. **Decide & alert** only when the change is important.
6. **Learn**: user feedback feeds evals.

## Setup

```sh
uv sync
cp .env.example .env   # then set your name and contact email
uv run pytest
```

## Use the data from Claude Code

FinScope is also an [MCP](https://modelcontextprotocol.io) server. `.mcp.json` registers it, so
opening this folder in Claude Code gives Claude five read-only tools: `list_filings`,
`get_section`, `diff_section`, `get_change_context` and `get_key_metrics`. Ask, for example,
"what did Apple add to its 10-K risk factors this year?" and Claude answers from the real diff
and SEC XBRL numbers. No model runs inside the server. Start it by hand with
`uv run python -m finscope mcp`.

**Status:** early development.
