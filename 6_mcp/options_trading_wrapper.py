#!/usr/bin/env python
"""
Wrapper for options_trading_server.py
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the server
from options_trading_server import mcp

if __name__ == "__main__":
    mcp.run(transport='stdio')
