# Real Options Trading System for Cathie

## Overview

Cathie now has a **completely separate, real options trading system** that:
- ✅ Uses **yfinance** for FREE real-time options data
- ✅ Uses **OptionLab** for accurate credit spread analysis
- ✅ Properly tracks credit spread positions (premium collected, max risk, expiration)
- ✅ Is **COMPLETELY SEPARATE** from Warren/George/Ray's stock trading (which uses Polygon)

---

## Architecture

### Stock Trading System (Warren, George, Ray)
- **Data**: Polygon API (free tier, end-of-day stock prices)
- **Tools**: `buy_shares`, `sell_shares`, `get_price`
- **Tracking**: Stock holdings in `holdings` dict, transactions in `transactions` list

### Options Trading System (Cathie ONLY)
- **Data**: yfinance (FREE real options chains with Greeks, IV, volume)
- **Analysis**: OptionLab (professional credit spread P/L calculations)
- **Tools**: `get_options_chain`, `analyze_credit_spread`, `sell_credit_spread`, `get_options_positions`
- **Tracking**: Options positions in separate `{name}_options` account with `CreditSpread` objects

---

## Installation

### Step 1: Install Dependencies
```bash
cd C:\udemy\agents\rjanda774\agents\6_mcp
python install_options_deps.py
```

This installs:
- `yfinance` - Free options data from Yahoo Finance
- `optionlab` - Professional credit spread analysis

### Step 2: Copy New Files
Replace these files in your `6_mcp` directory:
1. `options_models.py` (NEW - position tracking)
2. `options_trading_server.py` (NEW - real options tools)
3. `options_trading_wrapper.py` (NEW - server wrapper)
4. `mcp_params.py` (UPDATED - Cathie uses new server)
5. `templates.py` (UPDATED - Cathie's new instructions)

### Step 3: Restart System
```bash
# Terminal 1
python trading_floor.py

# Terminal 2  
python app.py
```

---

## How It Works

### 1. Cathie Gets Real Options Data

**Tool**: `get_options_chain(symbol, expiration_date)`

```python
# Cathie calls this to see available options
result = get_options_chain("SPY", "2026-03-21")

# Returns REAL market data:
{
  "symbol": "SPY",
  "current_price": 551.23,
  "expiration_date": "2026-03-21",
  "puts": [
    {"strike": 545, "bid": 3.20, "ask": 3.35, "volume": 1523, "impliedVolatility": 0.142},
    {"strike": 540, "bid": 2.15, "ask": 2.25, "volume": 2341, "impliedVolatility": 0.138},
    ...
  ],
  "calls": [...]
}
```

### 2. Cathie Analyzes Spreads

**Tool**: `analyze_credit_spread(symbol, spread_type, short_strike, long_strike, ...)`

```python
# Cathie analyzes a potential bull put spread
analysis = analyze_credit_spread(
    symbol="SPY",
    spread_type="bull_put",
    short_strike=540,  # Sell this put
    long_strike=535,   # Buy this put (protection)
    expiration_date="2026-03-21",
    contracts=5
)

# OptionLab calculates:
{
  "premium": {
    "net_collected": "$375.00"  # Cash collected upfront!
  },
  "profit_loss": {
    "max_profit": "$375.00",    # If expires worthless
    "max_loss": "$2,125.00",    # If stock drops below 535
    "breakeven": "$539.25",
    "return_on_risk": "17.6%"
  },
  "probability_of_profit": "68.5%",
  "days_to_expiration": 24
}
```

### 3. Cathie Sells the Spread

**Tool**: `sell_credit_spread(...)`

```python
# Cathie executes the trade
result = sell_credit_spread(
    name="cathie",
    symbol="SPY",
    spread_type="bull_put",
    short_strike=540,
    long_strike=535,
    expiration_date="2026-03-21",
    contracts=5,
    rationale="High PoP bull put spread on strong SPY support"
)

# Position is ACTUALLY RECORDED:
{
  "status": "POSITION OPENED",
  "position_id": "a3f9c2d1",
  "financials": {
    "premium_collected": "$375.00",   # REAL tracking!
    "max_risk": "$2,125.00",
    "max_profit": "$375.00"
  },
  "account_summary": {
    "open_positions": 1,
    "total_premium_collected": 375.00,
    "total_unrealized_pnl": 375.00,
    "total_open_risk": 2125.00
  }
}
```

### 4. Cathie Tracks Her Positions

**Tool**: `get_options_positions(name)`

```python
# View all positions
positions = get_options_positions("cathie")

# Returns:
{
  "account_summary": {
    "open_positions": 3,
    "closed_positions": 0,
    "total_premium_collected": 1225.00,
    "total_realized_pnl": 0.00,
    "total_unrealized_pnl": 1225.00,
    "total_open_risk": 6500.00
  },
  "open_positions": [
    {
      "position_id": "a3f9c2d1",
      "symbol": "SPY",
      "spread_type": "bull_put",
      "short_leg": {...},
      "long_leg": {...},
      "net_premium_collected": 375.00,
      "days_to_expiration": 24,
      ...
    },
    ...
  ]
}
```

---

## Credit Spread Mechanics (Properly Modeled!)

### Bull Put Spread (Bullish/Neutral)
```
Current Price: $550
Sell 540 Put  → Collect $2.15 premium
Buy  535 Put  → Pay    $1.40 premium
                ─────────────────────
Net Premium:     $0.75 × 100 × 5 contracts = $375

Max Profit: $375 (if SPY stays above $540 at expiration)
Max Loss: $2,125 (if SPY drops below $535)
Breakeven: $539.25
```

### Bear Call Spread (Bearish/Neutral)
```
Current Price: $550
Sell 560 Call → Collect $1.80 premium
Buy  565 Call → Pay    $1.15 premium
                ─────────────────────
Net Premium:     $0.65 × 100 × 5 contracts = $325

Max Profit: $325 (if SPY stays below $560 at expiration)
Max Loss: $2,175 (if SPY rises above $565)
Breakeven: $560.65
```

---

## What Cathie Will Do

1. **Call `get_options_chain("SPY")`** to see available options
2. **Call `analyze_credit_spread(...)`** to evaluate 2-3 potential spreads
3. **Call `sell_credit_spread(...)`** to OPEN the best position
4. **System tracks**:
   - Premium collected (cash in)
   - Max risk (potential loss)
   - Days to expiration
   - P/L over time

---

## Database Storage

### Options Account (Separate from Stock Account)
```
accounts.db:
  - cathie_options:
      name: "cathie"
      open_positions: [CreditSpread, CreditSpread, ...]
      closed_positions: [...]
      total_premium_collected: 1225.00
      total_realized_pnl: 0.00
```

### Stock Account (Unchanged)
```
accounts.db:
  - cathie:
      name: "cathie"
      balance: 10000.00
      holdings: {}   # No stock holdings for Cathie
      transactions: []
```

**Completely separate!** Cathie's options and stock accounts don't interfere.

---

## Advantages of This System

1. ✅ **FREE real options data** - yfinance provides actual market prices
2. ✅ **Accurate P/L** - OptionLab uses Black-Scholes for Greeks and P/L
3. ✅ **Proper mechanics** - Tracks premium collected, not share purchases
4. ✅ **Separate systems** - Options don't interfere with stock trading
5. ✅ **Scalable** - Can add more options strategies later (iron condors, calendars, etc.)

---

## Future Enhancements

- Auto-close spreads at 50% max profit
- Track theta decay day-by-day
- Portfolio-level risk management
- Integration with Interactive Brokers for REAL execution
- Support for iron condors, butterflies, calendars

---

## Testing

After installation, check:
```bash
# See if Cathie calls the new tools
type options_tool_calls.log

# Or check OpenAI logs for:
get_options_chain()
analyze_credit_spread()
sell_credit_spread()
```

Cathie should now properly trade credit spreads! 🎯
