# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

This repo currently holds two unrelated, independently-runnable projects, plus a couple of stray scripts:

- **Root (`app.py`, `requirements.txt`, `foundations/`)** — the deployed Hugging Face Space: a Gradio chat app where an LLM answers questions in character as the repo owner, using their LinkedIn profile/summary as context (`foundations/app.py`, class `Me`). This is intentionally minimal — a previous restructuring ("HF deploy") stripped the rest of the original course repo (the numbered `1_foundations`...`5_autogen` lab folders and their `community_contributions`) down to just this, to keep the deployed Space small.
- **`6_mcp/`** — a separate, self-contained project: an autonomous multi-agent paper-trading floor. Deliberately kept out of the HF-deploy trim (unlike the other numbered folders) because it's the active project. Everything below is about this directory.
- `LEANscriptOptionsbacktest.py`, `pdf_to_text.py` — standalone one-off utility scripts at root, unrelated to either app above.
- `6_mcp/community_contributions/` — a large grab-bag of unrelated student/community submissions bundled with the original course material. Treat as reference/noise, not part of the core architecture, unless a task specifically points there.

There is no root README, no lint config, and no test suite for either core project (test files that do exist live inside individual `6_mcp/community_contributions/*` submissions and are unrelated to the core code).

## Running things

**HF chat app** (root): `python app.py` — needs `OPENAI_API_KEY`; reads `foundations/me/ProfileLinkedin.txt` and `foundations/me/summaryRJ.txt`.

**Trading floor** (`6_mcp/`), each run from inside `6_mcp/`:
- `uv run reset.py` — resets all four trader accounts back to their base strategy/starting balance (wipes prior state).
- `uv run trading_floor.py` — starts the scheduler: runs every trader once, then sleeps `RUN_EVERY_N_MINUTES` (default 60) and repeats. Skips runs while the market is closed unless `RUN_EVEN_WHEN_MARKET_IS_CLOSED=true`.
- `uv run app.py` — Gradio dashboard visualizing account state/logs across traders.
- Individual MCP servers (`accounts_server.py`, `push_server.py`, `market_server.py`, `regime_server.py`) are not run directly — they're launched as subprocesses over stdio by the trading floor per `mcp_params.py`.

⚠️ There is currently no dependency manifest (`pyproject.toml`/`uv.lock`) for `6_mcp/` — it was removed along with the rest of the original course scaffolding in the HF-deploy restructuring. The required packages (`openai-agents`, `mcp`, `polygon-api-client`, `numpy`, `gradio`, `plotly`, `pandas`, `python-dotenv`) happen to be present in this environment's venv, but nothing in-repo currently declares/pins them for this subproject — don't assume `uv run` resolves dependencies from scratch in a clean environment.

## Architecture: the trading floor (`6_mcp/`)

Four trader personas, each an independent LLM agent, run in parallel every cycle (`trading_floor.py` → `traders.py`):

| Trader | Persona | Strategy focus |
|---|---|---|
| Warren | Warren Buffett | value investing, long holds |
| George | George Soros | aggressive macro/contrarian |
| Ray | Ray Dalio | systematic, risk-parity |
| Cathie | Cathie Wood | aggressive crypto ETF bets |

Persona strategy text lives in `reset.py`; it's injected verbatim into the agent's prompt (`templates.py`) alongside live account state.

- **Model selection**: all four traders use `gpt-4o-mini` unless `USE_MANY_MODELS=true`, which spreads them across GPT-4.1 Mini / DeepSeek / Gemini / Grok (`trading_floor.py`).
- **Two tool surfaces per trader** (`traders.py: Trader.create_agent`): its own "trader" MCP servers (accounts, push notifications, market data, and — Cathie only — the regime signal tool), plus a `Researcher` sub-agent exposed as a callable `Tool`, which has its own MCP servers (web fetch, Brave search, and a per-trader libsql memory store) for qualitative news research (`mcp_params.py`).
- **Trade vs. rebalance**: `Trader.run()` alternates each cycle between `trade_message` (look for new opportunities) and `rebalance_message` (review existing positions), via the `self.do_trade` flip (`templates.py`).
- **Persistence**: account state and logs live in a local SQLite DB (`database.py`, `accounts.py`).
- **Market data tier**: `market.py` picks between free EOD-only, paid 15-min-delayed, or realtime data based on `POLYGON_PLAN`; the "market" MCP server swaps between a local `market_server.py` (free tier) and the full `mcp_polygon` package (paid/realtime), per `mcp_params.py`.

**Cathie's regime-signal tool** (`regime_signal.py` / `regime_server.py`, wired in via `mcp_params.py: TRADERS_WITH_REGIME_TOOL`) is the one quantitative addition on top of the otherwise fully LLM-reasoning-driven traders:
- Labels each day Bull/Sideways/Bear from a *rolling* 20-day cumulative return (needs a rolling window to have a live "today" label).
- Fits the transition matrix from *non-overlapping* 20-day strides rather than daily — adjacent rolling windows share 19/20 days of underlying data, so counting daily transitions as independent samples overstates confidence (pseudo-replication).
- Forecasts N-days-ahead regime probabilities via true matrix exponentiation (`numpy.linalg.matrix_power`), not by squaring a single scalar probability — squaring a scalar silently assumes zero autocorrelation, which contradicts the tool's own "stickiness"/persistence concept.
- Reports an explicit `confidence` label tied to transition-matrix sample size, and refuses to return a signal at all below a minimum history threshold rather than guessing.
- It's framed to Cathie (in her `reset.py` strategy text) as a *lagging* trend-context signal to weigh alongside her news research — not a standalone buy/sell instruction, and not a profitability guarantee.

## Environment variables

| Var | Used by | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `foundations/app.py`, traders | LLM calls |
| `POLYGON_API_KEY`, `POLYGON_PLAN` | `market.py`, `regime_signal.py` | market data access/tier (`free`/`paid`/`realtime`) |
| `BRAVE_API_KEY` | Researcher MCP server | web search |
| `PUSHOVER_TOKEN`, `PUSHOVER_USER` | `push_server.py`, `foundations/app.py` | push notifications |
| `RUN_EVERY_N_MINUTES`, `RUN_EVEN_WHEN_MARKET_IS_CLOSED`, `USE_MANY_MODELS` | `trading_floor.py` | scheduler behavior |
