"""
Test if FastMCP can be imported
"""
print("Testing FastMCP import...")

try:
    from mcp.server.fastmcp import FastMCP
    print("✓ FastMCP imported successfully!")
    
    # Try to create a server
    mcp = FastMCP("test_server")
    print("✓ FastMCP server created successfully!")
    
    print("\n" + "="*60)
    print("SUCCESS: FastMCP is working!")
    print("="*60)
    
except ImportError as e:
    print(f"✗ ERROR: Cannot import FastMCP")
    print(f"  {e}")
    print("\nSOLUTION:")
    print("  Run: pip install fastmcp --break-system-packages")
    print("="*60)
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    print("="*60)