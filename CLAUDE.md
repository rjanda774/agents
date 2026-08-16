# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

This repo currently holds two unrelated, independently-runnable projects, plus a couple of stray scripts:

- **Root (`app.py`, `requirements.txt`, `foundations/`)** — the deployed Hugging Face Space: a Gradio chat app where an LLM answers questions in character as the repo owner, using their LinkedIn profile/summary as context (`foundations/app.py`, class `Me`). This is intentionally minimal — a previous restructuring ("HF deploy") stripped the rest of the original course repo (the numbered `1_foundations`...`5_autogen` lab folders and their `community_contributions`) down to just this, to keep the deployed Space small.
- **`6_mcp/`** — a separate, self-contained project: an autonomous agent-driven paper-trading floor (currently a single trader, Cathie, who trades options credit spreads). Deliberately kept out of the HF-deploy trim (unlike the other numbered folders) because it's the active project. Everything below is about this directory.
- `LEANscriptOptionsbacktest.py`, `pdf_to_text.py` — standalone one-off utility scripts at root, unrelated to either app above.

`6_mcp/community_contributions/` (a grab-bag of unrelated student/community submissions bundled with the original course material) has been removed. There is no root README, no lint config, and no test suite.

## Running things

**HF chat app** (root): `python app.py` — needs `OPENAI_API_KEY`; reads `foundations/me/ProfileLinkedin.txt` and `foundations/me/summaryRJ.txt`.

**Trading floor** (`6_mcp/`), each run from inside `6_mcp/`:
- `uv run reset.py` — resets Cathie's stock account (cash/strategy text; she never actually trades stocks, see below) back to its starting balance (wipes prior state).
- `uv run reset_cathie_options.py` — separately resets Cathie's *options* account (`cathie_options` in `accounts.db`): clears open/closed spread positions and resets cash to $10,000. Run this too when you want a clean slate, since options positions live in a separate pseudo-account from the stock one `reset.py` touches.
- `uv run trading_floor.py` — starts the scheduler: runs every trader once, then sleeps `RUN_EVERY_N_MINUTES` (default 60) and repeats. Skips runs while the market is closed unless `RUN_EVEN_WHEN_MARKET_IS_CLOSED=true`.
- `uv run app.py` — Gradio dashboard visualizing account state/logs across traders.
- `python install_options_deps.py` — installs `yfinance`/`optionlab`/`scipy` (the options system's dependencies; not needed by anything else in `6_mcp/`).
- Individual MCP servers (`accounts_server.py`, `push_server.py`, `market_server.py`, `regime_server.py`, `options_trading_wrapper.py`) are not run directly — they're launched as subprocesses over stdio by the trading floor per `mcp_params.py`.

⚠️ There is currently no dependency manifest (`pyproject.toml`/`uv.lock`) for `6_mcp/` — it was removed along with the rest of the original course scaffolding in the HF-deploy restructuring. The required packages (`openai-agents`, `mcp`, `polygon-api-client`, `numpy`, `gradio`, `plotly`, `pandas`, `python-dotenv`) happen to be present in this environment's venv, but nothing in-repo currently declares/pins them for this subproject — don't assume `uv run` resolves dependencies from scratch in a clean environment. **`yfinance`, `optionlab`, and `scipy`** (needed only by the options system) are *not* present in this environment's venv and need `python install_options_deps.py` before Cathie's options tools will actually run.

## Architecture: the trading floor (`6_mcp/`)

A single trader persona, Cathie (named for Cathie Wood; aggressive crypto ETF bets), runs each cycle (`trading_floor.py` → `traders.py`). The floor originally ran four personas (Warren/Buffett-value, George/Soros-macro, Ray/Dalio-risk-parity, plus Cathie); the other three were removed to simplify things, but `traders.py`/`trading_floor.py` still support an arbitrary list of traders via `names`/`lastnames`/`model_names` — add entries there and a matching strategy in `reset.py` to bring more traders back.

Persona strategy text lives in `reset.py`; it's injected verbatim into the agent's prompt (`templates.py`) alongside live account state.

- **Model selection**: Cathie uses `gpt-4o-mini` by default, overridable via `MODEL_NAME` (`trading_floor.py`).
- **Two tool surfaces per trader** (`traders.py: Trader.create_agent`): its own "trader" MCP servers (accounts, push notifications, market data, and — Cathie only — the regime signal tool), plus a `Researcher` sub-agent exposed as a callable `Tool`, which has its own MCP servers (web fetch, Brave search, and a per-trader libsql memory store) for qualitative news research (`mcp_params.py`).
- **Trade vs. rebalance**: `Trader.run()` alternates each cycle between `trade_message` (look for new opportunities) and `rebalance_message` (review existing positions), via the `self.do_trade` flip (`templates.py`).
- **Persistence**: account state and logs live in a local SQLite DB (`database.py`, `accounts.py`).
- **Market data tier**: `market.py` picks between free EOD-only, paid 15-min-delayed, or realtime data based on `POLYGON_PLAN`; the "market" MCP server swaps between a local `market_server.py` (free tier) and the full `mcp_polygon` package (paid/realtime), per `mcp_params.py`.

**Cathie's regime-signal tool** (`regime_signal.py` / `regime_server.py`, wired in via `mcp_params.py: TRADERS_WITH_REGIME_TOOL`) is a quantitative addition on top of the otherwise fully LLM-reasoning-driven traders:
- Labels each day Bull/Sideways/Bear from a *rolling* 20-day cumulative return (needs a rolling window to have a live "today" label).
- Fits the transition matrix from *non-overlapping* 20-day strides rather than daily — adjacent rolling windows share 19/20 days of underlying data, so counting daily transitions as independent samples overstates confidence (pseudo-replication).
- Forecasts N-days-ahead regime probabilities via true matrix exponentiation (`numpy.linalg.matrix_power`), not by squaring a single scalar probability — squaring a scalar silently assumes zero autocorrelation, which contradicts the tool's own "stickiness"/persistence concept.
- Reports an explicit `confidence` label tied to transition-matrix sample size, and refuses to return a signal at all below a minimum history threshold rather than guessing.
- It's framed to Cathie (in her `reset.py` strategy text and `templates.py` instructions) as a *lagging* trend-context signal to weigh alongside her news research when picking directional bias — not a standalone buy/sell instruction, and not a profitability guarantee.

**Cathie's options credit-spread system** is her actual trading mechanism — she does not buy/sell stocks at all (`templates.py: trader_instructions` forbids it explicitly; enforced by instruction, not by removing her `accounts_server` access, since she still needs `get_balance`/`change_strategy`). It's a separate, more substantial subsystem:
- **`options_models.py`** — `CreditSpread`/`OptionLeg`/`OptionsAccount` pydantic models. Positions are tracked in a pseudo-account named `cathie_options` in the same `accounts.db` (via `database.py: read_account`/`write_account`), completely separate from her real `cathie` stock `Account` (which stays untouched at its starting balance, since she never trades stocks).
- **`options_trading_server.py`** (run via `options_trading_wrapper.py`, an MCP-tool wrapper) — five tools: `get_options_chain` and `analyze_credit_spread` (read-only: real chains/Greeks from `yfinance`, P/L and probability-of-profit via `optionlab`'s Black-Scholes), `sell_credit_spread` and `close_credit_spread` (mutate `cathie_options`), `get_options_positions` (read it back). Only trades **bull put** and **bear call** credit spreads — defined-risk, premium-selling strategies — never naked options.
- **Server-enforced rules** (rejected outright by `sell_credit_spread`, not just requested by prompt): 25–45 day expirations only, max loss capped at 3% of the options account's current cash (`OptionsAccount.cash` in `options_models.py`, credited/debited on open/close), and a $100 minimum net premium per trade. Everything else -- short-leg delta 0.20–0.30, minimum probability-of-profit, mandatory position-management checks (take profit at 75% captured, cut losses if the short strike is breached, force-close inside 7 days-to-expiration to avoid assignment/pin risk) -- is prompt-only guidance in `templates.py`'s Cathie-specific `trader_instructions`/`trade_message`/`rebalance_message`, which the LLM has been observed to not always follow exactly (e.g. opening spreads below the stated premium floor before it was moved server-side); treat those as strong steering, not guarantees, unless/until they're moved server-side too.
- **Dashboard** (`app.py`): Cathie gets an options-specific panel (cash/premium-collected/open-risk breakdown, a positions log table, live unrealized P/L fetched from `yfinance` in `get_options_summary`) instead of the stock holdings/transactions tables other traders would get.
- yfinance and OptionLab need no API key (free); see the dependency note above for installing them.

## Environment variables

| Var | Used by | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `foundations/app.py`, traders | LLM calls |
| `POLYGON_API_KEY`, `POLYGON_PLAN` | `market.py`, `regime_signal.py` | market data access/tier (`free`/`paid`/`realtime`) |
| `BRAVE_API_KEY` | Researcher MCP server | web search |
| `PUSHOVER_TOKEN`, `PUSHOVER_USER` | `push_server.py`, `foundations/app.py` | push notifications |
| `RUN_EVERY_N_MINUTES`, `RUN_EVEN_WHEN_MARKET_IS_CLOSED`, `MODEL_NAME` | `trading_floor.py` | scheduler behavior / model override |
