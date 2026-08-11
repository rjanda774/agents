import os
from dotenv import load_dotenv
from market import is_paid_polygon, is_realtime_polygon

load_dotenv(override=True)

brave_env = {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
polygon_api_key = os.getenv("POLYGON_API_KEY")

# The MCP server for the Trader to read Market Data

if is_paid_polygon or is_realtime_polygon:
    market_mcp = {
        "command": "uvx",
        "args": ["--from", "git+https://github.com/polygon-io/mcp_polygon@v0.1.0", "mcp_polygon"],
        "env": {"POLYGON_API_KEY": polygon_api_key},
    }
else:
    market_mcp = {"command": "uv", "args": ["run", "market_server.py"]}


# The full set of MCP servers for the trader: Accounts, Push Notification and the Market

regime_mcp = {"command": "uv", "args": ["run", "regime_server.py"]}

# Traders who additionally get the Markov regime-signal tool (see regime_signal.py).
# Scoped narrowly for now -- it's a lagging trend-context signal, not proven useful
# for every strategy style, and it needs real price history to say anything trustworthy.
TRADERS_WITH_REGIME_TOOL = {"Cathie"}


def trader_mcp_server_params(name: str):
    params = [
        {"command": "uv", "args": ["run", "accounts_server.py"]},
        {"command": "uv", "args": ["run", "push_server.py"]},
        market_mcp,
    ]
    if name in TRADERS_WITH_REGIME_TOOL:
        params.append(regime_mcp)
    return params

# The full set of MCP servers for the researcher: Fetch, Brave Search and Memory


def researcher_mcp_server_params(name: str):
    return [
        {"command": "uvx", "args": ["mcp-server-fetch"]},
        {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": brave_env,
        },
        {
            "command": "npx",
            "args": ["-y", "mcp-memory-libsql"],
            "env": {"LIBSQL_URL": f"file:./memory/{name}.db"},
        },
    ]
