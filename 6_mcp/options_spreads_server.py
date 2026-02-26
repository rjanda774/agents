"""
Options Spreads MCP Server
Provides tools for analyzing and trading option spreads, focusing on credit spreads for monthly income.

This server integrates with options pricing libraries to help Cathie:
- Find optimal credit spreads (Bull Put, Bear Call)
- Calculate probability of profit
- Analyze risk/reward ratios
- Execute spread trades
"""

from mcp.server.fastmcp import FastMCP  # type: ignore
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import sys
import os

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mcp = FastMCP("options_spreads_server")


@mcp.tool()
async def find_credit_spread(
    name: str,
    symbol: str,
    spread_type: str = "bull_put",
    expiration_days: int = 30,
    min_premium: float = 0.50,
    max_risk: float = 500.0
) -> str:
    """Find optimal credit spread opportunities for monthly income.
    
    This tool searches for credit spreads that match income generation criteria.
    Credit spreads collect premium upfront and profit if the stock stays above (bull put)
    or below (bear call) certain price levels.
    
    Args:
        name: Account holder name
        symbol: Stock ticker (e.g., "SPY", "QQQ", "AAPL")
        spread_type: Type of spread - "bull_put" (bullish) or "bear_call" (bearish)
        expiration_days: Target days to expiration (typically 30-45 for monthly income)
        min_premium: Minimum premium to collect per spread (default $0.50 = $50 per contract)
        max_risk: Maximum risk per spread in dollars (default $500)
    
    Returns:
        JSON string with spread recommendations including strikes, premium, and probability of profit
    """
    
    # Debug logging
    from pathlib import Path
    debug_log = Path(__file__).parent / "options_tool_calls.log"
    with open(debug_log, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"TOOL CALLED: find_credit_spread\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Args: name={name}, symbol={symbol}, spread_type={spread_type}\n")
        f.write(f"{'='*60}\n")
    
    # TODO: Integrate with real options data (yfinance, Polygon, etc.)
    # For now, return simulated recommendations
    
    current_price = _get_mock_stock_price(symbol)
    expiration_date = (datetime.now() + timedelta(days=expiration_days)).strftime("%Y-%m-%d")
    
    if spread_type == "bull_put":
        # Bull Put Spread: Sell higher strike put, buy lower strike put
        # Profit if stock stays above short strike
        spread_width = 5.0  # $5 spread width
        short_strike = round(current_price * 0.95, 0)  # 5% below current price
        long_strike = short_strike - spread_width
        premium = 0.75  # $75 per spread
        pop = 68.0  # 68% probability of profit
        
        spread_info = {
            "spread_type": "Bull Put Spread",
            "symbol": symbol,
            "current_price": current_price,
            "expiration_date": expiration_date,
            "days_to_expiration": expiration_days,
            "short_put_strike": short_strike,
            "long_put_strike": long_strike,
            "spread_width": spread_width,
            "premium_collected": premium,
            "max_profit": premium * 100,  # Per contract
            "max_loss": (spread_width - premium) * 100,  # Per contract
            "breakeven": short_strike - premium,
            "probability_of_profit": pop,
            "return_on_risk": round((premium / (spread_width - premium)) * 100, 1),
            "strategy": f"Sell {int(short_strike)} Put, Buy {int(long_strike)} Put",
            "market_outlook": "Moderately bullish - profit if {0} stays above ${1:.2f}".format(symbol, short_strike - premium)
        }
    
    else:  # bear_call
        # Bear Call Spread: Sell lower strike call, buy higher strike call
        # Profit if stock stays below short strike
        spread_width = 5.0
        short_strike = round(current_price * 1.05, 0)  # 5% above current price
        long_strike = short_strike + spread_width
        premium = 0.65  # $65 per spread
        pop = 65.0
        
        spread_info = {
            "spread_type": "Bear Call Spread",
            "symbol": symbol,
            "current_price": current_price,
            "expiration_date": expiration_date,
            "days_to_expiration": expiration_days,
            "short_call_strike": short_strike,
            "long_call_strike": long_strike,
            "spread_width": spread_width,
            "premium_collected": premium,
            "max_profit": premium * 100,
            "max_loss": (spread_width - premium) * 100,
            "breakeven": short_strike + premium,
            "probability_of_profit": pop,
            "return_on_risk": round((premium / (spread_width - premium)) * 100, 1),
            "strategy": f"Sell {int(short_strike)} Call, Buy {int(long_strike)} Call",
            "market_outlook": f"Moderately bearish - profit if {symbol} stays below ${short_strike + premium:.2f}"
        }
    
    return json.dumps(spread_info, indent=2)


@mcp.tool()
async def analyze_spread_greeks(
    symbol: str,
    short_strike: float,
    long_strike: float,
    expiration_days: int,
    option_type: str = "put"
) -> str:
    """Calculate Greeks for an option spread to understand risk characteristics.
    
    Greeks measure how the spread value changes with market conditions:
    - Delta: How much spread value changes per $1 stock move
    - Theta: Daily time decay (usually positive for credit spreads)
    - Vega: Sensitivity to volatility changes
    - Gamma: Rate of delta change
    
    Args:
        symbol: Stock ticker
        short_strike: Strike price of sold option (higher for puts, lower for calls)
        long_strike: Strike price of bought option (lower for puts, higher for calls)
        expiration_days: Days until expiration
        option_type: "put" or "call"
    
    Returns:
        JSON string with Greeks analysis
    """
    
    current_price = _get_mock_stock_price(symbol)
    
    # Mock Greeks calculation
    # TODO: Use Black-Scholes or real market data for accurate Greeks
    
    if option_type == "put":
        # Short put has positive delta, long put has negative delta
        net_delta = 0.25  # Slightly bullish position
        net_theta = 2.50  # Earn $2.50/day from time decay
        net_vega = -15.0  # Lose value if volatility increases
        net_gamma = -0.03  # Delta becomes less positive as stock moves up
    else:  # call
        net_delta = -0.22  # Slightly bearish position
        net_theta = 2.30
        net_vega = -14.0
        net_gamma = -0.02
    
    greeks_info = {
        "symbol": symbol,
        "current_price": current_price,
        "spread_type": f"{option_type.capitalize()} Credit Spread",
        "short_strike": short_strike,
        "long_strike": long_strike,
        "days_to_expiration": expiration_days,
        "greeks": {
            "delta": net_delta,
            "theta": net_theta,
            "vega": net_vega,
            "gamma": net_gamma
        },
        "interpretation": {
            "delta": f"Position gains ${abs(net_delta) * 100:.0f} per $1 stock move {'up' if net_delta > 0 else 'down'}",
            "theta": f"Earning ${net_theta:.2f} per day from time decay" if net_theta > 0 else f"Losing ${abs(net_theta):.2f} per day",
            "vega": f"Position loses ${abs(net_vega):.0f} if implied volatility increases 1%",
            "gamma": f"Delta changes by {abs(net_gamma):.3f} per $1 stock move"
        },
        "risk_summary": "Credit spreads benefit from time decay (positive theta) but are hurt by volatility increases (negative vega)"
    }
    
    return json.dumps(greeks_info, indent=2)


@mcp.tool()
async def calculate_spread_profit_loss(
    premium_collected: float,
    spread_width: float,
    num_contracts: int = 1
) -> str:
    """Calculate profit/loss scenarios for a credit spread.
    
    Args:
        premium_collected: Premium received per spread (e.g., 0.75 = $75)
        spread_width: Distance between strikes (e.g., 5.0 = $5)
        num_contracts: Number of spreads to trade (default 1)
    
    Returns:
        JSON string with profit/loss calculations
    """
    
    max_profit = premium_collected * 100 * num_contracts
    max_loss = (spread_width - premium_collected) * 100 * num_contracts
    net_credit = premium_collected * 100 * num_contracts
    
    return_on_risk = (premium_collected / (spread_width - premium_collected)) * 100
    
    pl_info = {
        "num_contracts": num_contracts,
        "premium_per_spread": f"${premium_collected * 100:.2f}",
        "spread_width": f"${spread_width:.2f}",
        "net_credit_received": f"${net_credit:.2f}",
        "max_profit": f"${max_profit:.2f}",
        "max_loss": f"${max_loss:.2f}",
        "return_on_risk": f"{return_on_risk:.1f}%",
        "risk_reward_ratio": f"1:{spread_width/premium_collected:.2f}",
        "margin_required": f"${max_loss:.2f}",
        "scenarios": {
            "best_case": f"Stock stays favorable, options expire worthless, keep entire ${max_profit:.2f} premium",
            "breakeven": f"Stock at short strike ± ${premium_collected:.2f}",
            "worst_case": f"Stock moves against position, lose ${max_loss:.2f} (capped at spread width)"
        },
        "monthly_income_potential": f"${max_profit:.2f} per month if trading {num_contracts} spread(s) monthly"
    }
    
    return json.dumps(pl_info, indent=2)


@mcp.tool()
async def screen_credit_spread_candidates(
    name: str,
    target_monthly_income: float = 500.0,
    risk_per_trade: float = 1000.0,
    min_pop: float = 65.0
) -> str:
    """Screen multiple stocks for optimal credit spread opportunities.
    
    Finds the best candidates for generating consistent monthly income through credit spreads.
    Focuses on high-probability trades with good risk/reward ratios.
    
    Args:
        name: Account holder name
        target_monthly_income: Target monthly income in dollars
        risk_per_trade: Maximum risk willing to take per spread
        min_pop: Minimum probability of profit (default 65%)
    
    Returns:
        JSON string with top spread candidates ranked by attractiveness
    """
    
    # Debug logging
    from pathlib import Path
    debug_log = Path(__file__).parent / "options_tool_calls.log"
    with open(debug_log, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"TOOL CALLED: screen_credit_spread_candidates\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Args: name={name}, target_monthly_income={target_monthly_income}\n")
        f.write(f"{'='*60}\n")
    
    # Mock screening results
    # TODO: Integrate with real options scanner (TastyTrade, TD Ameritrade, etc.)
    
    candidates = [
        {
            "rank": 1,
            "symbol": "SPY",
            "name": "SPDR S&P 500 ETF",
            "current_price": 550.00,
            "spread_type": "Bull Put Spread",
            "short_strike": 540,
            "long_strike": 535,
            "premium": 0.85,
            "pop": 72.0,
            "max_profit": 85,
            "max_loss": 415,
            "return_on_risk": 20.5,
            "iv_rank": 45,  # Implied Volatility Rank
            "score": 95,
            "rationale": "High liquidity, moderate IV, strong support at 540"
        },
        {
            "rank": 2,
            "symbol": "QQQ",
            "name": "Invesco QQQ Trust",
            "current_price": 485.00,
            "spread_type": "Bull Put Spread",
            "short_strike": 475,
            "long_strike": 470,
            "premium": 0.75,
            "pop": 68.0,
            "max_profit": 75,
            "max_loss": 425,
            "return_on_risk": 17.6,
            "iv_rank": 52,
            "score": 88,
            "rationale": "Tech sector strength, elevated IV favorable for selling premium"
        },
        {
            "rank": 3,
            "symbol": "IWM",
            "name": "iShares Russell 2000 ETF",
            "current_price": 215.00,
            "spread_type": "Bear Call Spread",
            "short_strike": 220,
            "long_strike": 225,
            "premium": 0.70,
            "pop": 66.0,
            "max_profit": 70,
            "max_loss": 430,
            "return_on_risk": 16.3,
            "iv_rank": 58,
            "score": 82,
            "rationale": "Small-cap weakness, high IV, resistance at 220"
        }
    ]
    
    contracts_needed = int(target_monthly_income / candidates[0]["max_profit"]) + 1
    
    result = {
        "screening_criteria": {
            "target_monthly_income": f"${target_monthly_income:.2f}",
            "max_risk_per_trade": f"${risk_per_trade:.2f}",
            "min_probability_of_profit": f"{min_pop}%"
        },
        "recommendation": {
            "top_pick": candidates[0]["symbol"],
            "contracts_needed": contracts_needed,
            "estimated_monthly_income": f"${candidates[0]['max_profit'] * contracts_needed:.2f}",
            "total_risk": f"${candidates[0]['max_loss'] * contracts_needed:.2f}"
        },
        "candidates": candidates,
        "strategy_notes": [
            "Focus on 30-45 DTE (days to expiration) for optimal time decay",
            "Target 65%+ probability of profit for consistent income",
            "Close at 50% max profit to lock in gains early",
            "Never risk more than 2-3% of portfolio on single trade",
            "Diversify across multiple uncorrelated underlyings"
        ]
    }
    
    return json.dumps(result, indent=2)


@mcp.tool()
async def execute_credit_spread(
    name: str,
    symbol: str,
    short_strike: float,
    long_strike: float,
    expiration_date: str,
    option_type: str,
    num_contracts: int,
    rationale: str
) -> str:
    """Execute a credit spread trade (simulation - does not execute real trades).
    
    This simulates placing a multi-leg option order. In a real implementation,
    this would connect to a brokerage API to execute the trade.
    
    Args:
        name: Account holder name
        symbol: Stock ticker
        short_strike: Strike of option being sold
        long_strike: Strike of option being bought
        expiration_date: Expiration date (YYYY-MM-DD)
        option_type: "put" or "call"
        num_contracts: Number of spreads
        rationale: Trading rationale and strategy fit
    
    Returns:
        Confirmation message with trade details
    """
    
    # This is a simulation - in production, would call actual brokerage API
    
    try:
        from accounts import Account
        account = Account.get(name)
        has_accounts = True
    except ImportError:
        # If accounts module not available, return simulated response
        has_accounts = False
        account = None
    
    # Mock premium calculation
    premium_per_spread = 0.75 if option_type == "put" else 0.65
    total_premium = premium_per_spread * 100 * num_contracts
    spread_width = abs(short_strike - long_strike)
    max_risk = (spread_width - premium_per_spread) * 100 * num_contracts
    
    # For simulation purposes, we'll treat this as buying a synthetic position
    # In reality, option spreads have different accounting
    
    trade_summary = {
        "status": "SIMULATED (Not a real trade)",
        "trade_type": f"{option_type.capitalize()} Credit Spread",
        "symbol": symbol,
        "expiration": expiration_date,
        "short_leg": f"SELL {num_contracts} {symbol} {short_strike} {option_type.upper()}",
        "long_leg": f"BUY {num_contracts} {symbol} {long_strike} {option_type.upper()}",
        "premium_collected": f"${total_premium:.2f}",
        "max_risk": f"${max_risk:.2f}",
        "max_profit": f"${total_premium:.2f}",
        "account": name,
        "current_balance": f"${account.balance:.2f}" if has_accounts and account else "N/A",
        "rationale": rationale,
        "note": "This is a SIMULATION. Real option trading requires a brokerage account with options approval and would use a proper options API."
    }
    
    return json.dumps(trade_summary, indent=2)


# Helper function for mock data
def _get_mock_stock_price(symbol: str) -> float:
    """Get mock stock price for simulation."""
    mock_prices = {
        "SPY": 550.00,
        "QQQ": 485.00,
        "IWM": 215.00,
        "AAPL": 185.00,
        "MSFT": 425.00,
        "TSLA": 195.00,
        "NVDA": 875.00,
        "AMZN": 178.00
    }
    return mock_prices.get(symbol, 100.00)


if __name__ == "__main__":
    mcp.run(transport='stdio')
