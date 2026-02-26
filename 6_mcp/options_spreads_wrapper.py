#!/usr/bin/env python
#!/usr/bin/env python
"""
Wrapper for options_spreads_server.py that logs all errors
"""
import sys
import traceback
from pathlib import Path

# Write to a log file since stdio is used by MCP
log_file = Path(__file__).parent / "options_server_debug.log"

try:
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("OPTIONS SPREADS SERVER WRAPPER STARTING\n")
        f.write("="*60 + "\n")
        f.write(f"Python: {sys.executable}\n")
        f.write(f"Path: {sys.path}\n")
        f.write(f"CWD: {Path.cwd()}\n")
        f.write("="*60 + "\n")
        f.flush()
        
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import the server module
        f.write("Importing options_spreads_server...\n")
        f.flush()
        
        from mcp.server.fastmcp import FastMCP
        from options_spreads_server import mcp
        
        f.write("SUCCESS: Import successful\n")
        f.write("Starting MCP server...\n")
        f.flush()
        
        # Run the server
        mcp.run(transport='stdio')
        
except Exception as e:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "="*60 + "\n")
        f.write("FATAL ERROR:\n")
        f.write(str(e) + "\n")
        f.write(traceback.format_exc())
        f.write("="*60 + "\n")
    sys.exit(1)