"""
Test if optionlab and yfinance are properly installed and importable
"""
import sys

print("="*60)
print("TESTING OPTIONS DEPENDENCIES")
print("="*60)

print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Test yfinance
print("\n1. Testing yfinance...")
try:
    import yfinance as yf
    print("   ✓ yfinance installed successfully")
    
    # Try to get a ticker
    ticker = yf.Ticker("SPY")
    info = ticker.info
    print(f"   ✓ Can fetch data: SPY = ${info.get('regularMarketPrice', 'N/A')}")
except ImportError as e:
    print(f"   ✗ yfinance NOT installed: {e}")
except Exception as e:
    print(f"   ✗ yfinance error: {e}")

# Test optionlab
print("\n2. Testing optionlab...")
try:
    import optionlab
    print("   ✓ optionlab installed successfully")
    print(f"   ✓ Version: {optionlab.__version__ if hasattr(optionlab, '__version__') else 'unknown'}")
    
    # Try to import key classes
    from optionlab import BSM, Spread
    from optionlab.models import OptionType, OrderType
    print("   ✓ Can import BSM, Spread, OptionType, OrderType")
    
except ImportError as e:
    print(f"   ✗ optionlab NOT installed: {e}")
    print(f"   → Solution: pip install optionlab --break-system-packages")
except Exception as e:
    print(f"   ✗ optionlab error: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
