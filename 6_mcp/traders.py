from contextlib import AsyncExitStack
from accounts_client import read_accounts_resource, read_strategy_resource
from accounts import Account
from database import write_log
from tracers import make_trace_id
from agents import Agent, Tool, Runner, OpenAIChatCompletionsModel, trace
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import os
import json
from datetime import datetime
from agents.mcp import MCPServerStdio
from templates import (
    researcher_instructions,
    trader_instructions,
    trade_message,
    rebalance_message,
    new_trades_message,
    research_tool,
)
from mcp_params import trader_mcp_server_params, researcher_mcp_server_params, TRADERS_WITH_OPTIONS_TOOL

load_dotenv(override=True)

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GROK_BASE_URL = "https://api.x.ai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_TURNS = 30

# Cathie's new-trade pass gets its own, larger turn budget (see run_options_cycle): per-
# candidate evaluation needs ~4-6 tool calls each, plus the mandatory dark-pool and regime
# calls. It used to be pass 2 of two; the paper position-review pass 1 was removed on
# 2026-10-14 with paper trading (real positions are closed by code, live_trading.check_exits).
TRADE_PASS_MAX_TURNS = 90

openrouter_client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key)
deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
grok_client = AsyncOpenAI(base_url=GROK_BASE_URL, api_key=grok_api_key)
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)


def get_model(model_name: str):
    if "/" in model_name:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=openrouter_client)
    elif "deepseek" in model_name:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=deepseek_client)
    elif "grok" in model_name:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=grok_client)
    elif "gemini" in model_name:
        return OpenAIChatCompletionsModel(model=model_name, openai_client=gemini_client)
    else:
        return model_name


async def get_researcher(mcp_servers, model_name) -> Agent:
    researcher = Agent(
        name="Researcher",
        instructions=researcher_instructions(),
        model=get_model(model_name),
        mcp_servers=mcp_servers,
    )
    return researcher


async def get_researcher_tool(mcp_servers, model_name) -> Tool:
    researcher = await get_researcher(mcp_servers, model_name)
    return researcher.as_tool(tool_name="Researcher", tool_description=research_tool())


class Trader:
    def __init__(self, name: str, lastname="Trader", model_name="gpt-4o-mini"):
        self.name = name
        self.lastname = lastname
        self.agent = None
        self.model_name = model_name
        self.do_trade = True

    async def create_agent(self, trader_mcp_servers, researcher_mcp_servers) -> Agent:
        tool = await get_researcher_tool(researcher_mcp_servers, self.model_name)
        self.agent = Agent(
            name=self.name,
            instructions=trader_instructions(self.name),
            model=get_model(self.model_name),
            tools=[tool],
            mcp_servers=trader_mcp_servers,
        )
        return self.agent

    async def get_account_report(self, accounts_server=None) -> str:
        if self.name in TRADERS_WITH_OPTIONS_TOOL:
            # Real money only (paper trading removed 2026-10-14): her account is the live
            # Schwab balances plus her real trades.
            import live_trading
            try:
                data = {"real_schwab_account": await asyncio.to_thread(live_trading.real_account_snapshot)}
            except Exception as e:
                data = {"real_schwab_account": {"error": f"couldn't read the real account: {e}"}}
            return json.dumps(data)
        account = await read_accounts_resource(self.name, accounts_server)
        account_json = json.loads(account)
        account_json.pop("portfolio_value_time_series", None)
        return json.dumps(account_json)

    async def run_agent(self, trader_mcp_servers, researcher_mcp_servers):
        self.agent = await self.create_agent(trader_mcp_servers, researcher_mcp_servers)
        # accounts_server is always trader_mcp_servers' first entry (see
        # mcp_params.trader_mcp_server_params), given an explicit name in
        # run_with_mcp_servers so it can be found by name rather than by position.
        # Reused below instead of accounts_client.py opening its own ad-hoc connection
        # per call -- see the comment atop accounts_client.py for why that mattered.
        accounts_server = next((s for s in trader_mcp_servers if s.name == "accounts_server.py"), None)
        if self.name in TRADERS_WITH_OPTIONS_TOOL:
            await self.run_options_cycle(accounts_server)
            return
        account = await self.get_account_report(accounts_server)
        strategy = await read_strategy_resource(self.name, accounts_server)
        message = (
            trade_message(self.name, strategy, account)
            if self.do_trade
            else rebalance_message(self.name, strategy, account)
        )
        await Runner.run(self.agent, message, max_turns=MAX_TURNS)

    async def run_options_cycle(self, accounts_server=None):
        """Options traders (currently just Cathie). Code first: update the real-trade
        ledger from Schwab and close any real position that meets an exit rule. Then, only
        if a new real trade is possible right now (approve mode, Schwab readable, limits
        not used up), the spread scan and one new-trade Runner.run pass.

        Real money only since 2026-10-14: the paper position-review pass and paper trades
        were removed at the user's request. Without Schwab she doesn't trade at all.
        """
        import live_trading
        code_notes = []
        if live_trading.execution_mode() != "simulated":
            # Record fills at their real price, cancel orders working 30+ minutes, expire
            # unapproved stagings. Best effort: a Schwab hiccup must never cost the cycle.
            try:
                for msg in await asyncio.to_thread(live_trading.reconcile):
                    print(f"{self.name} live: {msg}")
                    code_notes.append(msg)
            except Exception as e:
                print(f"{self.name}: live-trade reconcile failed: {e}")
            # Exits on real positions (step 5): code checks the rules and sends the close
            # orders itself -- Cathie can't close real positions. Best effort, like reconcile.
            try:
                for msg in await asyncio.to_thread(live_trading.check_exits):
                    print(f"{self.name} exits: {msg}")
                    code_notes.append(msg)
            except Exception as e:
                print(f"{self.name}: real-position exit check failed: {e}")

        # No room for a new real trade (trading off, no Schwab, daily limit used, too many
        # staged, or STOP_TRADING): skip the scan and the new-trade pass -- every trade she
        # picked would be refused at staging anyway.
        try:
            blocked = live_trading.new_trades_blocked()
        except Exception as e:
            blocked = f"couldn't check whether real trading is possible ({e})"
        if blocked:
            line = f"Skipped the new-trade search: {blocked}."
            print(f"{self.name}: {line}")
            write_log(self.name, "live", line)
            return

        strategy = await read_strategy_resource(self.name, accounts_server)

        # Code-driven spread scan (spread_scanner.py): checks every candidate source's
        # option chains against the hard rules so Cathie picks from spreads that actually
        # qualify, instead of hand-checking 3-5 of ~300 tickers (she was observed skipping
        # every screener name). Best effort: a failed scan falls back to the old routine.
        scan_report = None
        try:
            import spread_scanner
            scan_report = await asyncio.to_thread(spread_scanner.scan)
            line = spread_scanner.summary_line(scan_report)
            print(f"{self.name}: {line}")
            write_log(self.name, "scan", line)
        except Exception as e:
            print(f"{self.name}: spread scan failed: {e}")

        account = await self.get_account_report(accounts_server)
        code_summary = "\n".join(code_notes) or "No fills, closes or cancellations this cycle."
        trade_message_text = new_trades_message(strategy, account, code_summary, scan_report)
        await Runner.run(self.agent, trade_message_text, max_turns=TRADE_PASS_MAX_TURNS)

    async def run_with_mcp_servers(self):
        async with AsyncExitStack() as stack:
            trader_mcp_servers = [
                await stack.enter_async_context(
                    # Named after the launched script (e.g. "accounts_server.py") rather
                    # than left to MCPServerStdio's default -- every local server here is
                    # spawned with the same PYTHON interpreter, so the default name
                    # ("stdio: <interpreter path>") would be identical across all of them
                    # and useless for picking accounts_server back out of the list by name
                    # (see run_agent above).
                    MCPServerStdio(params, name=params["args"][0], client_session_timeout_seconds=120)
                )
                for params in trader_mcp_server_params(self.name)
            ]
            async with AsyncExitStack() as stack:
                researcher_mcp_servers = [
                    await stack.enter_async_context(
                        MCPServerStdio(params, client_session_timeout_seconds=120)
                    )
                    for params in researcher_mcp_server_params(self.name)
                ]
                await self.run_agent(trader_mcp_servers, researcher_mcp_servers)

    async def run_with_trace(self):
        # Options traders run the same cycle every time (see run_options_cycle), so the
        # do_trade-based trading/rebalancing label other traders use would be misleading.
        if self.name in TRADERS_WITH_OPTIONS_TOOL:
            trace_name = f"{self.name}-cycle"
        else:
            trace_name = f"{self.name}-trading" if self.do_trade else f"{self.name}-rebalancing"
        trace_id = make_trace_id(f"{self.name.lower()}")
        with trace(trace_name, trace_id=trace_id):
            await self.run_with_mcp_servers()

    async def run(self):
        try:
            await self.run_with_trace()
        except Exception as e:
            import traceback
            print(f"Error running trader {self.name}: {e}")
            traceback.print_exc()
        self.do_trade = not self.do_trade

        # Record a portfolio-value data point after every run, not just on trades,
        # so the dashboard chart reflects price movement between trades too. In its own
        # try/except, separate from run_with_trace's above -- this whole block does raw
        # SQLite I/O (Account.get/read_account/account.save, all via database.py) with no
        # guard of its own, unlike everything inside run_with_trace which is already
        # covered by the try/except above. It runs unconditionally after every cycle, for
        # every trader, so an uncaught exception here -- e.g. "database is locked" from
        # app.py's dashboard reading the same accounts.db concurrently, or any other
        # transient disk/DB hiccup -- would propagate out of run() uncaught, through
        # asyncio.gather in trading_floor.py's scheduler loop, and kill the entire process
        # permanently, the same failure mode market.py's is_market_open() had. This is a
        # best-effort chart data point, not core trading logic, so it's safe to skip on
        # failure rather than let it take the whole floor down.
        try:
            account = Account.get(self.name)
            if self.name in TRADERS_WITH_OPTIONS_TOOL:
                # Cathie's chart is the real Schwab account's Day Net Liq, recorded by
                # live_trading.record_account_value whenever balances are read (this
                # cycle's account report, the dashboard). Nothing to add here.
                return
            portfolio_value = account.calculate_portfolio_value()
            account.portfolio_value_time_series.append(
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), portfolio_value)
            )
            account.save()
        except Exception as e:
            print(f"{self.name}: failed to record portfolio-value data point: {e}")
