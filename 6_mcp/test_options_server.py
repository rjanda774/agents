"""
Test if options_spreads_server.py can start properly
"""
import subprocess
import sys
import time

print("="*60)
print("TESTING OPTIONS SPREADS SERVER")
print("="*60)

print("\nStarting options_spreads_server.py...")
print("(Will run for 3 seconds then stop)\n")

try:
    # Start the server
    process = subprocess.Popen(
        ["uv", "run", "options_spreads_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait 3 seconds
    time.sleep(3)
    
    # Check if it's still running
    if process.poll() is None:
        print("✓ OPTIONS SERVER STARTED SUCCESSFULLY!")
        print("✓ Server is running (will terminate now)")
        process.terminate()
        process.wait(timeout=5)
    else:
        print("✗ SERVER EXITED IMMEDIATELY")
        stdout, stderr = process.communicate()
        if stdout:
            print(f"\nSTDOUT:\n{stdout}")
        if stderr:
            print(f"\nSTDERR:\n{stderr}")
        sys.exit(1)
        
except FileNotFoundError:
    print("✗ ERROR: 'uv' command not found")
    print("  Make sure uv is installed and in your PATH")
    sys.exit(1)
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
