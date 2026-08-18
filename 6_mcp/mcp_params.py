import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from dotenv import load_dotenv
from market import is_paid_polygon, is_realtime_polygon

load_dotenv(override=True)

polygon_api_key = os.getenv("POLYGON_API_KEY")

# uvx resolves mcp-server-fetch's dependencies in its own isolated environment, which can
# land on a different `mcp` package version than the one actually installed here -- and
# mcp-server-fetch has lagged behind at least one breaking rename in `mcp` (McpError ->
# MCPError), causing an ImportError inside the child process that surfaces to us as an
# opaque "Connection closed" on session.initialize(). Pin uvx to the same `mcp` version
# already proven to work in this environment, read dynamically so this doesn't silently
# break again after a future `pip install --upgrade mcp`.
try:
    _installed_mcp_version = _pkg_version("mcp")
except PackageNotFoundError:
    _installed_mcp_version = None

# The mcp SDK does NOT pass the parent process's full environment to spawned stdio
# servers by default -- only a small curated allowlist. On Windows that allowlist can
# exclude APPDATA, which is where per-user packages live (e.g.
# C:\Users\<you>\AppData\Roaming\Python\Python314\site-packages). Without it, the child
# process can't import anything non-stdlib and dies immediately -- which the parent sees
# as an opaque "Connection closed" McpError on session.initialize(), not an import error.
# Pass the full environment through explicitly to every spawned server to avoid this.
FULL_ENV = dict(os.environ)

# Local MCP servers are launched with this exact interpreter (sys.executable), not `uv run` --
# guarantees the exact same interpreter/env as the parent process regardless of how
# trading_floor.py itself was launched (uv run or a hand-managed venv's plain python; see
# CLAUDE.md's "Running things"). This also happens to sidestep a real problem `uv run
# <script>.py` used to hit here when 6_mcp/ had no pyproject.toml/uv.lock: it treated every
# invocation as an ad-hoc script and could print resolution/status text to stdout, corrupting
# the MCP stdio JSON-RPC stream. A manifest/lock exists now, but sys.executable stays the
# right choice regardless -- it's about matching the parent's interpreter, not the lock file.
PYTHON = sys.executable

# The MCP server for the Trader to read Market Data

if is_paid_polygon or is_realtime_polygon:
    market_mcp = {
        "command": "uvx",
        "args": ["--from", "git+https://github.com/polygon-io/mcp_polygon@v0.1.0", "mcp_polygon"],
        "env": {**FULL_ENV, "POLYGON_API_KEY": polygon_api_key},
    }
else:
    market_mcp = {"command": PYTHON, "args": ["market_server.py"], "env": FULL_ENV}


# The full set of MCP servers for the trader: Accounts, Push Notification and the Market

regime_mcp = {"command": PYTHON, "args": ["regime_server.py"], "env": FULL_ENV}
options_mcp = {"command": PYTHON, "args": ["options_trading_wrapper.py"], "env": FULL_ENV}

# Traders who additionally get the Markov regime-signal tool (see regime_signal.py).
# Scoped narrowly for now -- it's a lagging trend-context signal, not proven useful
# for every strategy style, and it needs real price history to say anything trustworthy.
TRADERS_WITH_REGIME_TOOL = {"Cathie"}

# Traders who trade options credit spreads instead of stocks (see options_trading_server.py).
# They keep accounts_server for get_balance/change_strategy, but their strategy/prompt
# forbids buy_shares/sell_shares -- market_mcp stays too, purely for reference pricing.
TRADERS_WITH_OPTIONS_TOOL = {"Cathie"}


def trader_mcp_server_params(name: str):
    params = [
        {"command": PYTHON, "args": ["accounts_server.py"], "env": FULL_ENV},
        {"command": PYTHON, "args": ["push_server.py"], "env": FULL_ENV},
        market_mcp,
    ]
    if name in TRADERS_WITH_REGIME_TOOL:
        params.append(regime_mcp)
    if name in TRADERS_WITH_OPTIONS_TOOL:
        params.append(options_mcp)
    return params

# The full set of MCP servers for the researcher: Fetch, Brave Search and Memory


def researcher_mcp_server_params(name: str):
    fetch_args = (
        ["--with", f"mcp=={_installed_mcp_version}"] if _installed_mcp_version else []
    ) + ["mcp-server-fetch"]
    return [
        {"command": "uvx", "args": fetch_args, "env": FULL_ENV},
        {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {**FULL_ENV, "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")},
        },
        {
            "command": "npx",
            "args": ["-y", "mcp-memory-libsql"],
            "env": {**FULL_ENV, "LIBSQL_URL": f"file:./memory/{name}.db"},
        },
    ]
