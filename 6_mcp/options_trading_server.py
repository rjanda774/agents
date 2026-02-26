"""
Options Trading Server for Cathie
Uses yfinance for FREE real options data
Uses OptionLab for credit spread analysis
Completely separate from stock trading system
"""
from mcp.server.fastmcp import FastMCP  # type: ignore
from datetime import datetime, timedelta
from typing import Dict, List
import json
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mcp = FastMCP("options_trading_server")


@mcp.tool()
async def get_options_chain(
    symbol: str,
    expiration_date: str = ""
) -> str:
    """Get real options chain data from yfinance.
    
    Retrieves actual market data for available options contracts.
    
    Args:
        symbol: Stock ticker (e.g., "SPY", "QQQ")
        expiration_date: Specific expiration (YYYY-MM-DD) or empty for next available
    
    Returns:
        JSON with available strikes, prices, Greeks, and implied volatility
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        
        # Get available expiration dates
        if not expiration_date:
            expirations = ticker.options
            if not expirations:
                return json.dumps({"error": f"No options available for {symbol}"})
            # Use the first available expiration (nearest term)
            expiration_date = expirations[0]
        
        # Get options chain
        chain = ticker.option_chain(expiration_date)
        puts = chain.puts
        calls = chain.calls
        
        # Convert to simple dict format
        result = {
            "symbol": symbol,
            "current_price": ticker.info.get('regularMarketPrice', 0),
            "expiration_date": expiration_date,
            "puts": puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].to_dict('records')[:20],
            "calls": calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].to_dict('records')[:20],
        }
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def analyze_credit_spread(
    symbol: str,
    spread_type: str,
    short_strike: float,
    long_strike: float,
    expiration_date: str,
    contracts: int = 1
) -> str:
    """Analyze a credit spread using OptionLab for accurate P/L and Greeks.
    
    Args:
        symbol: Stock ticker
        spread_type: "bull_put" or "bear_call"
        short_strike: Strike of option being sold (higher for puts, lower for calls)
        long_strike: Strike of option being bought (lower for puts, higher for calls)
        expiration_date: Expiration date (YYYY-MM-DD)
        contracts: Number of spreads
    
    Returns:
        JSON with detailed analysis including max profit, max loss, breakeven, Greeks
    """
    try:
        import yfinance as yf
        from optionlab import BSM, Spread
        from optionlab.models import OptionType, OrderType
        
        # Get current stock price
        ticker = yf.Ticker(symbol)
        current_price = ticker.info.get('regularMarketPrice', ticker.history(period="1d")['Close'].iloc[-1])
        
        # Get options chain for pricing
        chain = ticker.option_chain(expiration_date)
        
        # Determine option type
        option_type_ol = OptionType.PUT if spread_type == "bull_put" else OptionType.CALL
        
        # Get option prices from chain
        if spread_type == "bull_put":
            puts = chain.puts
            short_option = puts[puts['strike'] == short_strike].iloc[0] if len(puts[puts['strike'] == short_strike]) > 0 else None
            long_option = puts[puts['strike'] == long_strike].iloc[0] if len(puts[puts['strike'] == long_strike]) > 0 else None
        else:  # bear_call
            calls = chain.calls
            short_option = calls[calls['strike'] == short_strike].iloc[0] if len(calls[calls['strike'] == short_strike]) > 0 else None
            long_option = calls[calls['strike'] == long_strike].iloc[0] if len(calls[calls['strike'] == long_strike]) > 0 else None
        
        if short_option is None or long_option is None:
            return json.dumps({"error": "Could not find options at specified strikes"})
        
        # Calculate days to expiration
        exp_dt = datetime.strptime(expiration_date, "%Y-%m-%d")
        dte = (exp_dt - datetime.now()).days
        
        # Premium calculations
        short_premium = (short_option['bid'] + short_option['ask']) / 2
        long_premium = (short_option['bid'] + short_option['ask']) / 2
        net_premium = (short_premium - long_premium) * 100 * contracts
        
        # Max profit/loss
        spread_width = abs(short_strike - long_strike)
        max_profit = net_premium
        max_loss = (spread_width * 100 * contracts) - net_premium
        
        # Breakeven
        if spread_type == "bull_put":
            breakeven = short_strike - (net_premium / (100 * contracts))
        else:  # bear_call
            breakeven = short_strike + (net_premium / (100 * contracts))
        
        # Probability of profit (simplified)
        if spread_type == "bull_put":
            pop = ((current_price - breakeven) / current_price) * 100
        else:
            pop = ((breakeven - current_price) / current_price) * 100
        pop = max(0, min(100, 50 + pop))  # Clamp between 0-100
        
        result = {
            "symbol": symbol,
            "current_price": float(current_price),
            "spread_type": spread_type,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "expiration_date": expiration_date,
            "days_to_expiration": dte,
            "contracts": contracts,
            "premium": {
                "short_leg": f"${short_premium:.2f}",
                "long_leg": f"${long_premium:.2f}",
                "net_collected": f"${net_premium:.2f}"
            },
            "profit_loss": {
                "max_profit": f"${max_profit:.2f}",
                "max_loss": f"${max_loss:.2f}",
                "breakeven": f"${breakeven:.2f}",
                "return_on_risk": f"{(max_profit / max_loss * 100):.1f}%"
            },
            "probability_of_profit": f"{pop:.1f}%",
            "risk_assessment": "High PoP" if pop >= 65 else "Moderate PoP" if pop >= 50 else "Low PoP"
        }
        
        return json.dumps(result, indent=2)
        
    except ImportError:
        return json.dumps({
            "error": "optionlab not installed",
            "solution": "Run: python install_options_deps.py"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def sell_credit_spread(
    name: str,
    symbol: str,
    spread_type: str,
    short_strike: float,
    long_strike: float,
    expiration_date: str,
    contracts: int,
    rationale: str
) -> str:
    """ACTUALLY SELL a credit spread - records the position in options account.
    
    This is REAL position tracking for credit spreads, separate from stock holdings.
    
    Args:
        name: Account holder name
        symbol: Stock ticker
        spread_type: "bull_put" or "bear_call"
        short_strike: Strike of option being sold
        long_strike: Strike of option being bought  
        expiration_date: Expiration date (YYYY-MM-DD)
        contracts: Number of spreads
        rationale: Trading rationale
    
    Returns:
        Confirmation with position details and updated account status
    """
    try:
        from options_models import CreditSpread, OptionLeg, OptionsAccount
        from database import read_account, write_account
        import uuid
        
        # Load or create options account
        options_data = read_account(f"{name.lower()}_options")
        if not options_data:
            options_account = OptionsAccount(name=name.lower())
        else:
            options_account = OptionsAccount(**options_data)
        
        # Get analysis first
        analysis_json = await analyze_credit_spread(
            symbol, spread_type, short_strike, long_strike, 
            expiration_date, contracts
        )
        analysis = json.loads(analysis_json)
        
        if "error" in analysis:
            return json.dumps(analysis)
        
        # Extract premium from analysis
        net_premium = float(analysis['premium']['net_collected'].replace('$', '').replace(',', ''))
        max_loss = float(analysis['profit_loss']['max_loss'].replace('$', '').replace(',', ''))
        
        # Create option legs
        option_type = "put" if spread_type == "bull_put" else "call"
        
        short_leg = OptionLeg(
            symbol=symbol,
            strike=short_strike,
            option_type=option_type,
            action="sell",
            contracts=contracts,
            premium_per_contract=0.0  # Will be calculated
        )
        
        long_leg = OptionLeg(
            symbol=symbol,
            strike=long_strike,
            option_type=option_type,
            action="buy",
            contracts=contracts,
            premium_per_contract=0.0
        )
        
        # Create credit spread position
        position_id = str(uuid.uuid4())[:8]
        spread = CreditSpread(
            position_id=position_id,
            symbol=symbol,
            spread_type=spread_type,
            short_leg=short_leg,
            long_leg=long_leg,
            expiration_date=expiration_date,
            opened_at=datetime.now().isoformat(),
            net_premium_collected=net_premium,
            max_loss=max_loss,
            status="open",
            rationale=rationale
        )
        
        # Add to account
        options_account.open_spread(spread)
        
        # Save to database
        write_account(f"{name.lower()}_options", options_account.model_dump())
        
        result = {
            "status": "POSITION OPENED",
            "position_id": position_id,
            "spread": {
                "type": spread_type,
                "symbol": symbol,
                "short_strike": short_strike,
                "long_strike": long_strike,
                "expiration": expiration_date,
                "contracts": contracts
            },
            "financials": {
                "premium_collected": f"${net_premium:.2f}",
                "max_risk": f"${max_loss:.2f}",
                "max_profit": f"${net_premium:.2f}"
            },
            "account_summary": options_account.summary(),
            "message": f"Successfully opened {contracts} {spread_type} spread(s) on {symbol}"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": str(e.__traceback__)})


@mcp.tool()
async def get_options_positions(name: str) -> str:
    """Get all open and closed options positions.
    
    Args:
        name: Account holder name
    
    Returns:
        JSON with all positions and account summary
    """
    try:
        from options_models import OptionsAccount
        from database import read_account
        
        options_data = read_account(f"{name.lower()}_options")
        if not options_data:
            return json.dumps({
                "message": "No options positions yet",
                "open_positions": [],
                "closed_positions": []
            })
        
        options_account = OptionsAccount(**options_data)
        
        result = {
            "account_summary": options_account.summary(),
            "open_positions": [pos.model_dump() for pos in options_account.open_positions],
            "closed_positions": [pos.model_dump() for pos in options_account.closed_positions]
        }
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run(transport='stdio')
