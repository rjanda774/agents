"""Compatibility shim for the mcp SDK's v1/v2 FastMCP rename.

mcp>=2.0 renamed `mcp.server.fastmcp.FastMCP` to `mcp.server.mcpserver.MCPServer` and moved
the module -- a hard import-time break (ModuleNotFoundError), not a missing-method
AttributeError like the MCPServer.read_resource gap accounts_client.py already guards
against. It's a harder failure than that one too: this kills the *_server.py subprocess
before any of its own code runs, so a per-call hasattr check can't help -- it has to be
resolved at import time. The rest of the class's API used by this codebase (constructor's
`name` param, `.tool()`, `.resource()`, `.run(transport="stdio")`) is otherwise identical
between the two, confirmed by diffing both versions' wheels.

mcp<2.0 is what's actually installed locally right now (see CLAUDE.md's "Known constraint"
on openai-agents/mcp version drift between uv.lock and the user's hand-managed env), and
only has the old `mcp.server.fastmcp` path. Import FastMCP from here in every *_server.py
file instead of directly from `mcp.server.fastmcp`, so this codebase runs unmodified
against either mcp major version.
"""
try:
    from mcp.server.fastmcp import FastMCP  # mcp < 2.0
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp >= 2.0
