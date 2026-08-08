from contextlib import AsyncExitStack
from accounts_client import read_accounts_resource, read_strategy_resource
from tracers import make_trace_id
from agents import Agent, Tool, Runner, OpenAIChatCompletionsModel, trace  # type: ignore
from openai import AsyncOpenAI  # type: ignore
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from agents.mcp import MCPServerStdio  # type: ignore
from templates import (
    researcher_instructions,
    trader_instructions,
    trade_message,
    rebalance_message,
    research_tool,
)
from mcp_params import trader_mcp_server_params, researcher_mcp_server_params
from accounts import Account

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

    async def get_account_report(self) -> str:
        if self.name == "Cathie":
            return self._get_cathie_options_summary()
        account = await read_accounts_resource(self.name)
        account_json = json.loads(account)
        account_json.pop("portfolio_value_time_series", None)
        account_json["transactions"] = account_json.get("transactions", [])[-10:]
        return json.dumps(account_json)

    def _get_cathie_options_summary(self) -> str:
        """Return a compact options account summary for Cathie (no stock account bloat)."""
        import sqlite3
        from datetime import date
        try:
            conn = sqlite3.connect("accounts.db")
            cur = conn.cursor()
            cur.execute("SELECT account FROM accounts WHERE name = ?", ("cathie_options",))
            row = cur.fetchone()
            conn.close()
            if not row or not row[0]:
                return json.dumps({"cash": 10000, "open_positions": [], "closed_positions": [], "total_premium_collected": 0})
            data = json.loads(row[0])
        except Exception as e:
            return json.dumps({"error": str(e)})

        today = date.today().isoformat()
        open_pos = [p for p in data.get("open_positions", []) if p.get("expiration_date", "9999") >= today]
        closed_pos = data.get("closed_positions", [])

        # Compact open positions — strip bulky fields, keep what matters
        compact_open = []
        for p in open_pos:
            compact_open.append({
                "id": p.get("position_id", "")[:8],
                "sym": p.get("symbol"),
                "type": p.get("spread_type"),
                "short": p.get("short_strike"),
                "long": p.get("long_strike"),
                "exp": p.get("expiration_date"),
                "contracts": p.get("num_contracts"),
                "premium": round(p.get("premium_collected", 0), 2),
                "max_loss": round(p.get("max_loss", 0), 2),
                "dte": (date.fromisoformat(p["expiration_date"]) - date.today()).days if p.get("expiration_date") else None,
            })

        # Closed: just totals, not the full list
        total_realized = sum(p.get("realized_pnl", 0) or 0 for p in closed_pos)
        total_collected = sum(p.get("premium_collected", 0) or 0 for p in data.get("open_positions", []) + closed_pos)

        return json.dumps({
            "cash": round(data.get("cash", 0), 2),
            "total_premium_collected": round(total_collected, 2),
            "total_realized_pnl": round(total_realized, 2),
            "open_positions_count": len(compact_open),
            "closed_positions_count": len(closed_pos),
            "open_positions": compact_open,  # compact, no rationale blobs
        })

    async def run_agent(self, trader_mcp_servers, researcher_mcp_servers):
        self.agent = await self.create_agent(trader_mcp_servers, researcher_mcp_servers)
        account = await self.get_account_report()
        strategy = await read_strategy_resource(self.name)
        message = (
            trade_message(self.name, strategy, account)
            if self.do_trade
            else rebalance_message(self.name, strategy, account)
        )
        await Runner.run(self.agent, message, max_turns=MAX_TURNS)

    async def run_with_mcp_servers(self):
        async with AsyncExitStack() as stack:
            # Cathie gets special options spreads server, others get standard config
            from mcp_params import cathie_mcp_server_params
            mcp_params = cathie_mcp_server_params() if self.name == "Cathie" else trader_mcp_server_params
            
            # Debug logging
            if self.name == "Cathie":
                print(f"\n{'='*60}")
                print(f"CATHIE LOADING MCP SERVERS:")
                for param in mcp_params:
                    print(f"  - {param['command']} {' '.join(param['args'])}")
                print(f"{'='*60}\n")
            
            trader_mcp_servers = [
                await stack.enter_async_context(
                    MCPServerStdio(params, client_session_timeout_seconds=120)
                )
                for params in mcp_params
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
        trace_name = f"{self.name}-trading" if self.do_trade else f"{self.name}-rebalancing"
        trace_id = make_trace_id(f"{self.name.lower()}")
        with trace(trace_name, trace_id=trace_id):
            await self.run_with_mcp_servers()

    async def run(self):
        try:
            await self.run_with_trace()
        except Exception as e:
            print(f"Error running trader {self.name}: {e}")
        self.do_trade = not self.do_trade
        
        # Update portfolio value time series after each run
        account = Account.get(self.name)
        portfolio_value = account.calculate_portfolio_value()
        account.portfolio_value_time_series.append(
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), portfolio_value)
        )
        account.save()
