from contextlib import AsyncExitStack
from accounts_client import read_accounts_resource, read_strategy_resource
from accounts import Account
from database import read_account
from tracers import make_trace_id
from agents import Agent, Tool, Runner, OpenAIChatCompletionsModel, trace
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import json
from datetime import date, datetime
from agents.mcp import MCPServerStdio
from templates import (
    researcher_instructions,
    trader_instructions,
    trade_message,
    rebalance_message,
    close_positions_message,
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

# Cathie's cycle runs as two separate Runner.run passes instead of one (see
# run_agent_two_pass) -- a single combined "close positions, then find new trades" call was
# routinely hitting a shared MAX_TURNS budget partway through (her workflow got substantially
# heavier than the original single-symbol template this constant was tuned for: per-candidate
# evaluation now needs ~4-6 tool calls each, across 3-5 candidates, on top of position review).
# Splitting means each half gets its own budget, and one running long doesn't cost the other
# its progress. Position review is the lighter half (bounded by how many open positions
# exist); new-trade search is the heavier half (bounded by how many candidates get evaluated).
CLOSE_PASS_MAX_TURNS = 20
TRADE_PASS_MAX_TURNS = 45

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
            return self._get_options_account_summary()
        account = await read_accounts_resource(self.name, accounts_server)
        account_json = json.loads(account)
        account_json.pop("portfolio_value_time_series", None)
        return json.dumps(account_json)

    def _get_options_account_summary(self) -> str:
        """Compact options-account summary (open positions, no closed-position/rationale bloat)."""
        data = read_account(f"{self.name.lower()}_options")
        if not data:
            return json.dumps(
                {"cash": 10_000, "open_positions": [], "closed_positions": [], "total_premium_collected": 0}
            )

        today = date.today().isoformat()
        open_positions = [p for p in data.get("open_positions", []) if p.get("expiration_date", "9999") >= today]
        closed_positions = data.get("closed_positions", [])

        compact_open = [
            {
                "id": p.get("position_id", "")[:8],
                "sym": p.get("symbol"),
                "type": p.get("spread_type"),
                "short": p.get("short_leg", {}).get("strike"),
                "long": p.get("long_leg", {}).get("strike"),
                "exp": p.get("expiration_date"),
                "contracts": p.get("short_leg", {}).get("contracts"),
                "premium": round(p.get("net_premium_collected", 0), 2),
                "max_loss": round(p.get("max_loss", 0), 2),
                "dte": (date.fromisoformat(p["expiration_date"]) - date.today()).days
                if p.get("expiration_date")
                else None,
            }
            for p in open_positions
        ]
        total_realized = sum(p.get("realized_pnl", 0) or 0 for p in closed_positions)
        total_collected = sum(
            p.get("net_premium_collected", 0) or 0 for p in data.get("open_positions", []) + closed_positions
        )

        return json.dumps(
            {
                "cash": round(data.get("cash", 10_000), 2),
                "total_premium_collected": round(total_collected, 2),
                "total_realized_pnl": round(total_realized, 2),
                "open_positions_count": len(compact_open),
                "closed_positions_count": len(closed_positions),
                "open_positions": compact_open,
            }
        )

    async def run_agent(self, trader_mcp_servers, researcher_mcp_servers):
        self.agent = await self.create_agent(trader_mcp_servers, researcher_mcp_servers)
        # accounts_server is always trader_mcp_servers' first entry (see
        # mcp_params.trader_mcp_server_params), given an explicit name in
        # run_with_mcp_servers so it can be found by name rather than by position.
        # Reused below instead of accounts_client.py opening its own ad-hoc connection
        # per call -- see the comment atop accounts_client.py for why that mattered.
        accounts_server = next((s for s in trader_mcp_servers if s.name == "accounts_server.py"), None)
        if self.name in TRADERS_WITH_OPTIONS_TOOL:
            await self.run_agent_two_pass(accounts_server)
            return
        account = await self.get_account_report(accounts_server)
        strategy = await read_strategy_resource(self.name, accounts_server)
        message = (
            trade_message(self.name, strategy, account)
            if self.do_trade
            else rebalance_message(self.name, strategy, account)
        )
        await Runner.run(self.agent, message, max_turns=MAX_TURNS)

    async def run_agent_two_pass(self, accounts_server=None):
        """Options traders (currently just Cathie) run their cycle as two independent
        Runner.run passes on the same agent: close_positions_message first, then
        new_trades_message. See CLOSE_PASS_MAX_TURNS/TRADE_PASS_MAX_TURNS above for why.

        Only pass 1 gets its own try/except -- if it errors or runs out of turns, pass 2
        still runs (with a placeholder note in place of pass 1's summary), so a slow/stuck
        position review doesn't also cost the new-trade search. Pass 2's own exceptions are
        left to propagate to run()'s existing catch-all, same as the single-pass path.
        """
        strategy = await read_strategy_resource(self.name, accounts_server)

        account_before_close = await self.get_account_report(accounts_server)
        close_message = close_positions_message(strategy, account_before_close)
        close_summary = "(Position review did not complete -- it errored or ran out of turns; treat existing positions as unreviewed this cycle.)"
        try:
            close_result = await Runner.run(self.agent, close_message, max_turns=CLOSE_PASS_MAX_TURNS)
            close_summary = close_result.final_output
        except Exception as e:
            print(f"{self.name}: position-review pass failed: {e}")

        # Re-read live state so the new-trade pass sees any closes pass 1 just made.
        account_after_close = await self.get_account_report(accounts_server)
        trade_message_text = new_trades_message(strategy, account_after_close, close_summary)
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
        # Options traders always run both the close and new-trade passes every cycle now
        # (see run_agent_two_pass), so the do_trade-based trading/rebalancing label -- which
        # only reflected the single-pass alternation other traders still use -- would be
        # misleading for them.
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
        # so the dashboard chart reflects price movement between trades too.
        account = Account.get(self.name)
        if self.name in TRADERS_WITH_OPTIONS_TOOL:
            # Options traders' real capital lives entirely in their separate cathie_options
            # pseudo-account -- the stock Account above is a second, untouched $10,000 pool
            # they never actually trade (see CLAUDE.md), so calculate_portfolio_value() on it
            # is always flat, and blending its starting balance in on top of options cash
            # would double-count capital and not match the "Cash: $X" figure the dashboard
            # already shows. Track live options cash directly instead; still stored on the
            # stock Account's own time series field so app.py's chart plumbing is untouched.
            options_data = read_account(f"{self.name.lower()}_options")
            portfolio_value = options_data.get("cash", 10_000.0) if options_data else 10_000.0
        else:
            portfolio_value = account.calculate_portfolio_value()
        account.portfolio_value_time_series.append(
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), portfolio_value)
        )
        account.save()
