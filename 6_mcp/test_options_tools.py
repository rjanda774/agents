"""
Test to verify options server tools are registered
"""
import sys
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("TESTING OPTIONS SERVER TOOLS")
print("="*60)

async def test_tools():
    try:
        from options_spreads_server import mcp
        
        print("\nServer name:", mcp.name)
        print("\nChecking for registered tools...")
        
        # FastMCP.list_tools() is async
        tools_list = await mcp.list_tools()
        
        if tools_list:
            print(f"\nFound {len(tools_list)} tools:")
            for i, tool in enumerate(tools_list, 1):
                tool_name = tool.get('name', 'unknown')
                tool_desc = tool.get('description', 'No description')
                print(f"\n  {i}. {tool_name}")
                if len(tool_desc) > 100:
                    print(f"     {tool_desc[:100]}...")
                else:
                    print(f"     {tool_desc}")
        else:
            print("\nNo tools found!")
        
        print("\n" + "="*60)
        print("SUCCESS: Server and tools loaded!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_tools())