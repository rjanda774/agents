#!/usr/bin/env python
"""
Install options trading dependencies for Cathie
"""
import subprocess
import sys

print("="*60)
print("INSTALLING OPTIONS TRADING DEPENDENCIES")
print("="*60)

dependencies = [
    "numpy",         # Required by optionlab
    "scipy",         # Required by optionlab for Black-Scholes
    "pandas",        # Already installed but needed by optionlab
    "yfinance",      # Free options data from Yahoo Finance
    "optionlab",     # Credit spread analysis and P/L calculations
]

for package in dependencies:
    print(f"\nInstalling {package}...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            package, "--break-system-packages"
        ])
        print(f"✓ {package} installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        # Continue anyway, some might already be installed

print("\n" + "="*60)
print("VERIFYING INSTALLATION")
print("="*60)

# Test imports
try:
    import yfinance
    print("✓ yfinance OK")
except ImportError:
    print("✗ yfinance FAILED")

try:
    import optionlab
    print("✓ optionlab OK")
except ImportError as e:
    print(f"✗ optionlab FAILED: {e}")

print("\n" + "="*60)
print("INSTALLATION COMPLETE")
print("="*60)
