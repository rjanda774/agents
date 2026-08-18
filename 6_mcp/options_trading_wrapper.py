#!/usr/bin/env python
"""
Wrapper for options_trading_server.py - with full error logging
"""
import sys
import traceback
from datetime import datetime
from pathlib import Path

log_file = Path(__file__).parent / "options_trading_debug.log"

# Append, not overwrite -- this wrapper is spawned as a fresh subprocess every trading
# cycle, so opening in "w" mode was wiping out history from every prior cycle on each
# restart. Each session gets its own timestamped header instead.
with open(log_file, "a", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write(f"OPTIONS TRADING WRAPPER STARTING -- {datetime.now().isoformat()}\n")
    f.write("=" * 60 + "\n")
    f.write(f"Python: {sys.executable}\n")
    f.write(f"Version: {sys.version}\n")
    f.write(f"CWD: {Path.cwd()}\n")
    f.write("=" * 60 + "\n")
    f.flush()

try:
    sys.path.insert(0, str(Path(__file__).parent))

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("Testing imports...\n"); f.flush()

    import mcp
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"mcp OK: {mcp.__file__}\n"); f.flush()

    import yfinance
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"yfinance OK: {yfinance.__version__}\n"); f.flush()

    import optionlab
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"optionlab OK: {optionlab.__version__}\n"); f.flush()

    from options_trading_server import mcp as trading_mcp
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("options_trading_server imported OK\n")
        f.write("Starting MCP server...\n"); f.flush()

    trading_mcp.run(transport='stdio')

except Exception as e:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"FATAL ERROR: {e}\n")
        f.write(traceback.format_exc())
        f.write("=" * 60 + "\n")
    sys.exit(1)
