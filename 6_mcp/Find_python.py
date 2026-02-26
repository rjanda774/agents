"""
Find which Python executable has FastMCP installed
"""
import subprocess
import sys

print("Checking Python executables...")
print("="*60)

candidates = [
    "python",
    "python3",
    "C:\\Python314\\python.exe",
    "C:\\Python313\\python.exe",
    "C:\\Python312\\python.exe",
    sys.executable,  # The one running this script
]

working_python = None

for python_path in candidates:
    try:
        result = subprocess.run(
            [python_path, "-c", "from mcp.server.fastmcp import FastMCP; print('OK')"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and "OK" in result.stdout:
            print(f"✓ {python_path} - FastMCP available")
            if working_python is None:
                working_python = python_path
        else:
            print(f"✗ {python_path} - FastMCP not available")
            
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"✗ {python_path} - Not found or timeout")
    except Exception as e:
        print(f"✗ {python_path} - Error: {e}")

print("="*60)

if working_python:
    print(f"\n✓ FOUND: Use this Python: {working_python}")
    print(f"\nIn mcp_params.py, change:")
    print(f'  {{"command": "python", "args": ["options_spreads_server.py"]}}')
    print(f"To:")
    print(f'  {{"command": "{working_python}", "args": ["options_spreads_server.py"]}}')
else:
    print("\n✗ NO PYTHON WITH FASTMCP FOUND")
    print("  Install FastMCP: pip install fastmcp")