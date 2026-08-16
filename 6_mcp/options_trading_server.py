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
import concurrent.futures
from pathlib import Path


def fetch_with_timeout(fn, timeout_seconds=30):
    """Run fn() in a thread and raise TimeoutError if it takes too long.
    Prevents yfinance from hanging the MCP server indefinitely."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"yfinance request timed out after {timeout_seconds}s")


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
    Always call this without an expiration_date first to see the available dates,
    then call again with the chosen expiration_date to get the full chain.
    
    Args:
        symbol: Stock ticker (e.g., "SPY", "QQQ")
        expiration_date: Specific expiration (YYYY-MM-DD) or empty to list available dates
    
    Returns:
        JSON with available strikes, prices, Greeks, implied volatility, and delta
    """
    try:
        import yfinance as yf
        import datetime as dt_mod

        def _fetch():
            t = yf.Ticker(symbol)
            exps = t.options  # all available expiration dates as strings
            if not exps:
                raise ValueError(f"No options available for {symbol}")
            price = t.fast_info.get("lastPrice", 0)
            return t, exps, price

        t, all_exps, current_price = fetch_with_timeout(_fetch, timeout_seconds=30)

        today = dt_mod.date.today()
        min_date = today + dt_mod.timedelta(days=25)
        max_date = today + dt_mod.timedelta(days=45)

        # Filter to expirations in the valid 25-45 day window
        valid_exps = []
        for e in all_exps:
            try:
                d = dt_mod.date.fromisoformat(e)
                days = (d - today).days
                if 25 <= days <= 45:
                    valid_exps.append({"date": e, "days_to_expiration": days})
            except ValueError:
                pass

        # If no expiration_date specified, return the list of valid dates only
        if not expiration_date:
            return json.dumps({
                "symbol": symbol,
                "current_price": current_price,
                "today": str(today),
                "valid_expiration_window": f"{min_date} to {max_date} (25-45 days out)",
                "valid_expirations": valid_exps,
                "instruction": (
                    "Choose one date from valid_expirations above and call get_options_chain again "
                    "with that expiration_date to see strikes, premiums, and Greeks. "
                    "Do NOT use any expiration not listed here."
                ) if valid_exps else "No expirations fall in the 25-45 day window for this symbol. Try a different underlying."
            }, indent=2)

        # Validate the requested expiration is in the valid window
        try:
            exp_date = dt_mod.date.fromisoformat(expiration_date)
            days_to_exp = (exp_date - today).days
        except ValueError:
            return json.dumps({"error": f"Invalid expiration_date format: '{expiration_date}'. Use YYYY-MM-DD."})

        if days_to_exp < 25:
            return json.dumps({
                "error": f"Expiration {expiration_date} is only {days_to_exp} days away — too soon. "
                         f"Valid window is {min_date} to {max_date}. "
                         f"Valid expirations for {symbol}: {[e['date'] for e in valid_exps] or 'none in window'}"
            })
        if days_to_exp > 45:
            return json.dumps({
                "error": f"Expiration {expiration_date} is {days_to_exp} days away — too far out. "
                         f"Valid window is {min_date} to {max_date}. "
                         f"Valid expirations for {symbol}: {[e['date'] for e in valid_exps] or 'none in window'}"
            })

        def _fetch_chain():
            return t.option_chain(expiration_date)

        chain = fetch_with_timeout(_fetch_chain, timeout_seconds=30)
        puts  = chain.puts
        calls = chain.calls

        def extract(df):
            cols = ["strike", "lastPrice", "bid", "ask", "volume", "impliedVolatility"]
            if "delta" in df.columns:
                cols.append("delta")
            return df[cols].to_dict("records")[:25]

        result = {
            "symbol": symbol,
            "current_price": current_price,
            "today": str(today),
            "expiration_date": expiration_date,
            "days_to_expiration": days_to_exp,
            "puts":  extract(puts),
            "calls": extract(calls),
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

    IMPORTANT: Call get_options_chain first to see available strikes and expiration dates.
    This tool will automatically snap to the nearest available strike if your requested
    strike doesn't exist exactly - so approximate strikes are fine.

    For a bull_put spread (bullish): short_strike > long_strike (e.g. short=540, long=535)
    For a bear_call spread (bearish): short_strike < long_strike (e.g. short=550, long=555)

    Args:
        symbol: Stock ticker (e.g. "SPY", "QQQ", "IWM")
        spread_type: "bull_put" (neutral/bullish) or "bear_call" (neutral/bearish)
        short_strike: Strike to SELL - use a strike ~5% OTM from current price
        long_strike: Strike to BUY - use a strike $5 further OTM than short_strike
        expiration_date: Expiration date (YYYY-MM-DD) - use a date from get_options_chain
        contracts: Number of spreads (start with 1-3)

    Returns:
        JSON with max profit, max loss, breakeven, probability of profit, Greeks
    """
    try:
        import yfinance as yf
        import datetime as dt
        from optionlab.models import Inputs
        from optionlab.black_scholes import get_bs_info
        from optionlab import run_strategy

        # Fetch price + options chain with timeout to prevent MCP server hang
        def _fetch_market_data():
            t = yf.Ticker(symbol)
            price = t.fast_info.get("lastPrice") or t.history(period="1d")["Close"].iloc[-1]
            chain = t.option_chain(expiration_date)
            return float(price), chain

        current_price, chain = fetch_with_timeout(_fetch_market_data, timeout_seconds=30)

        if spread_type == "bull_put":
            option_type = "put"
            df = chain.puts
        else:  # bear_call
            option_type = "call"
            df = chain.calls

        if df.empty:
            return json.dumps({"error": f"No {option_type} options available for {symbol} on {expiration_date}"})

        # Snap to nearest available strikes rather than failing on exact match
        available_strikes = df['strike'].values
        actual_short = available_strikes[abs(available_strikes - short_strike).argmin()]
        actual_long  = available_strikes[abs(available_strikes - long_strike).argmin()]

        short_row = df[df['strike'] == actual_short]
        long_row  = df[df['strike'] == actual_long]

        if short_row.empty or long_row.empty:
            return json.dumps({"error": "Could not find options at specified strikes"})

        # Warn if we snapped to different strikes
        snapped = {}
        if actual_short != short_strike:
            snapped["short_strike_adjusted"] = f"{short_strike} -> {actual_short}"
        if actual_long != long_strike:
            snapped["long_strike_adjusted"] = f"{long_strike} -> {actual_long}"
        short_strike = actual_short
        long_strike  = actual_long

        # Use midpoint of bid/ask for fair value
        short_premium = (short_row.iloc[0]['bid'] + short_row.iloc[0]['ask']) / 2
        long_premium  = (long_row.iloc[0]['bid']  + long_row.iloc[0]['ask'])  / 2
        iv = float(short_row.iloc[0]['impliedVolatility'])

        # Guard: zero or near-zero premium means the option is illiquid or too far OTM
        if short_premium <= 0.01:
            return json.dumps({
                "error": f"Short strike {short_strike} has zero or negligible premium (${short_premium:.4f}). "
                         f"This strike is too far out-of-the-money or illiquid. "
                         f"Move the short strike closer to the current price (${current_price:.2f}) until premium > $0.05. "
                         f"Try a strike with delta closer to 0.25."
            })
        if long_premium <= 0.0:
            # Long leg with zero premium is fine to use $0.01 as floor — it's just cheap protection
            long_premium = 0.01

        net_premium_per_spread = short_premium - long_premium
        if net_premium_per_spread <= 0.01:
            return json.dumps({
                "error": f"Net premium is ${net_premium_per_spread:.4f} — not worth trading. "
                         f"Short leg: ${short_premium:.4f}, Long leg: ${long_premium:.4f}. "
                         f"Move the short strike closer to ATM to collect more premium, "
                         f"or try a different underlying with higher implied volatility."
            })

        # Guard: zero IV means yfinance returned bad data for this expiration
        if iv <= 0.001:
            return json.dumps({
                "error": f"Implied volatility is zero for strike {short_strike} on {expiration_date}. "
                         f"This usually means the option has no market data (illiquid or bad data). "
                         f"Try a different strike or a different expiration date."
            })

        # Days to expiration
        exp_date = dt.date.fromisoformat(expiration_date)
        today    = dt.date.today()
        dte      = (exp_date - today).days

        if dte <= 0:
            return json.dumps({"error": "Expiration date must be in the future"})

        years_to_exp = dte / 365.0

        # --- OptionLab: P/L profile + probability of profit ---
        price_range = abs(short_strike - long_strike) * 4
        inputs = Inputs(
            stock_price=float(current_price),
            start_date=today,
            target_date=exp_date,
            volatility=iv,
            interest_rate=0.05,
            min_stock=round(float(current_price) - price_range, 2),
            max_stock=round(float(current_price) + price_range, 2),
            strategy=[
                {"type": option_type, "strike": float(short_strike), "premium": float(short_premium), "n": contracts, "action": "sell"},
                {"type": option_type, "strike": float(long_strike),  "premium": float(long_premium),  "n": contracts, "action": "buy"},
            ],
        )
        result_ol = run_strategy(inputs)

        pop          = round(result_ol.probability_of_profit * 100, 1)
        profit_ranges = [
            f"${r[0]:.2f} to {'∞' if r[1] == float('inf') else f'${r[1]:.2f}'}"
            for r in result_ol.profit_ranges
        ]

        # --- Black-Scholes Greeks ---
        short_bs = get_bs_info(s=float(current_price), x=short_strike, r=0.05, vol=iv, years_to_maturity=years_to_exp)
        long_bs  = get_bs_info(s=float(current_price), x=long_strike,  r=0.05, vol=iv, years_to_maturity=years_to_exp)

        if option_type == "put":
            net_delta = round(long_bs.put_delta  - short_bs.put_delta,  4)
            net_theta = round(short_bs.put_theta - long_bs.put_theta,   4)
        else:
            net_delta = round(long_bs.call_delta  - short_bs.call_delta, 4)
            net_theta = round(short_bs.call_theta - long_bs.call_theta,  4)
        net_vega  = round(long_bs.vega  - short_bs.vega,  4)
        net_gamma = round(long_bs.gamma - short_bs.gamma, 4)

        # P/L summary (net_premium_per_spread already computed above)
        net_premium_per_spread = round(short_premium - long_premium, 4)
        spread_width = abs(short_strike - long_strike)
        max_profit_dollars = round(net_premium_per_spread * 100 * contracts, 2)
        max_loss_dollars   = round((spread_width - net_premium_per_spread) * 100 * contracts, 2)

        if spread_type == "bull_put":
            breakeven = round(short_strike - net_premium_per_spread, 2)
        else:
            breakeven = round(short_strike + net_premium_per_spread, 2)

        result = {
            "symbol": symbol,
            "current_price": round(float(current_price), 2),
            "spread_type": spread_type,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "strikes_adjusted": snapped if snapped else None,
            "expiration_date": expiration_date,
            "days_to_expiration": dte,
            "contracts": contracts,
            "implied_volatility": f"{iv*100:.1f}%",
            "premium": {
                "short_leg": round(short_premium, 2),
                "long_leg":  round(long_premium, 2),
                "net_per_spread": net_premium_per_spread,
            },
            "profit_loss": {
                "max_profit": f"${max_profit_dollars:.2f}",
                "max_loss":   f"${max_loss_dollars:.2f}",
                "breakeven":  f"${breakeven:.2f}",
                "return_on_risk": f"{(max_profit_dollars / max_loss_dollars * 100):.1f}%" if max_loss_dollars else "N/A",
                "profit_ranges": profit_ranges,
            },
            "probability_of_profit": f"{pop:.1f}%",
            "risk_assessment": "High PoP" if pop >= 65 else "Moderate PoP" if pop >= 50 else "Low PoP",
            "greeks": {
                "net_delta": net_delta,
                "net_theta": net_theta,
                "net_vega":  net_vega,
                "net_gamma": net_gamma,
            },
        }

        return json.dumps(result, indent=2)

    except ImportError as e:
        return json.dumps({
            "error": f"Missing dependency: {e}",
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
        from database import read_account, write_account, write_log
        import uuid
        import datetime as dt_mod

        # HARD ENFORCEMENT: expiration must be 25-45 calendar days from today
        today = dt_mod.date.today()
        try:
            exp_date = dt_mod.date.fromisoformat(expiration_date)
            days_to_exp = (exp_date - today).days
        except ValueError:
            return json.dumps({"error": f"Invalid expiration_date format: '{expiration_date}'. Use YYYY-MM-DD."})

        if days_to_exp < 25:
            return json.dumps({
                "error": f"TRADE REJECTED: expiration {expiration_date} is only {days_to_exp} days away. "
                         f"Minimum is 25 days. Today is {today}. Choose an expiration between "
                         f"{today + dt_mod.timedelta(days=25)} and {today + dt_mod.timedelta(days=45)}."
            })
        if days_to_exp > 45:
            return json.dumps({
                "error": f"TRADE REJECTED: expiration {expiration_date} is {days_to_exp} days away. "
                         f"Maximum is 45 days. Today is {today}. Choose an expiration between "
                         f"{today + dt_mod.timedelta(days=25)} and {today + dt_mod.timedelta(days=45)}."
            })

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
        # Fields match the output of analyze_credit_spread
        net_premium = float(analysis['premium']['net_per_spread']) * 100 * contracts
        max_loss = float(analysis['profit_loss']['max_loss'].replace('$', '').replace(',', ''))
        short_premium_per = float(analysis['premium']['short_leg'])
        long_premium_per  = float(analysis['premium']['long_leg'])

        # HARD ENFORCEMENT: minimum net premium per trade. The strategy text/prompt has
        # asked for this all along, but it's been observed slipping through in practice
        # (e.g. a live $69.50 and $18.50 trade both under the stated $100 floor) -- prompt
        # instructions aren't reliable enough on their own, so enforce it here too.
        MIN_NET_PREMIUM = 100.0
        if net_premium < MIN_NET_PREMIUM:
            return json.dumps({
                "error": (
                    f"TRADE REJECTED: net premium ${net_premium:.2f} is below the ${MIN_NET_PREMIUM:.2f} "
                    f"minimum. Not worth the capital/risk for this little income. Increase `contracts`, "
                    f"move the short strike closer to the money (more premium, still within the "
                    f"0.20-0.30 delta band), or pick a different underlying with richer premium."
                )
            })

        # HARD ENFORCEMENT: never risk more than 3% of available cash on a single trade.
        # Checked against current cash, not cash-after-this-trade's-premium -- what you'd
        # actually have to cover the max loss from is what you have going in.
        MAX_RISK_PCT = 0.03
        max_allowed_risk = options_account.cash * MAX_RISK_PCT
        if max_loss > max_allowed_risk:
            return json.dumps({
                "error": (
                    f"TRADE REJECTED: max loss ${max_loss:.2f} would risk "
                    f"{(max_loss / options_account.cash * 100) if options_account.cash else float('inf'):.1f}% "
                    f"of available cash (${options_account.cash:.2f}), exceeding the "
                    f"{MAX_RISK_PCT * 100:.0f}% per-trade cap. Max allowed risk right now is "
                    f"${max_allowed_risk:.2f}. Reduce `contracts`, pick a narrower spread width, "
                    f"or choose strikes closer together to lower max loss."
                )
            })

        # Create option legs
        option_type = "put" if spread_type == "bull_put" else "call"
        
        short_leg = OptionLeg(
            symbol=symbol,
            strike=short_strike,
            option_type=option_type,
            action="sell",
            contracts=contracts,
            premium_per_contract=short_premium_per
        )

        long_leg = OptionLeg(
            symbol=symbol,
            strike=long_strike,
            option_type=option_type,
            action="buy",
            contracts=contracts,
            premium_per_contract=long_premium_per
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
        write_log(name, "account", f"Opened {spread_type} on {symbol} | strikes {short_strike}/{long_strike} | exp {expiration_date} | {contracts} contracts | premium ${net_premium:.2f} | max loss ${max_loss:.2f}")

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
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})



@mcp.tool()
async def close_credit_spread(
    name: str,
    position_id: str,
    reason: str
) -> str:
    """Close (buy back) an existing credit spread position.

    Use this tool to exit a position by buying back the spread. This is done when:
    - The position has captured 75%+ of max profit (closing premium <= 25% of original)
    - The underlying is moving against the position and you want to cut losses
    - Expiration is within 7 days (avoid pin risk / assignment risk)

    To close: buy back the short leg and sell back the long leg.
    The closing cost is fetched live from the market.

    Args:
        name: Account holder name
        position_id: The position_id from get_options_positions
        reason: Why you are closing (profit target hit / loss management / near expiry)

    Returns:
        Confirmation with P&L realized
    """
    try:
        import yfinance as yf
        import datetime as dt_mod
        from options_models import OptionsAccount
        from database import read_account, write_account, write_log

        options_data = read_account(f"{name.lower()}_options")
        if not options_data:
            return json.dumps({"error": "No options account found"})

        options_account = OptionsAccount(**options_data)

        # Find the position
        position = next((p for p in options_account.open_positions if p.position_id == position_id), None)
        if not position:
            return json.dumps({"error": f"Position {position_id} not found in open positions"})

        # Guard: cannot close a position that has already expired
        today = dt_mod.date.today()
        try:
            exp_date = dt_mod.date.fromisoformat(position.expiration_date)
            if exp_date < today:
                return json.dumps({
                    "error": (
                        f"Cannot close position {position_id} — it expired on {position.expiration_date}. "
                        f"In the real world, expired options cannot be traded. "
                        f"If it expired out-of-the-money, the premium is already yours and no action is needed. "
                        f"If it expired in-the-money, assignment would have already occurred. "
                        f"Do not attempt to close expired positions."
                    )
                })
        except ValueError:
            pass  # If date parse fails, let it proceed

        # Fetch current market prices for closing cost
        try:
            def _fetch():
                t = yf.Ticker(position.symbol)
                chain = t.option_chain(position.expiration_date)
                return chain

            chain = fetch_with_timeout(_fetch, timeout_seconds=30)
            df = chain.puts if position.spread_type == "bull_put" else chain.calls

            available = df['strike'].values
            short_strike = available[abs(available - position.short_leg.strike).argmin()]
            long_strike  = available[abs(available - position.long_leg.strike).argmin()]

            short_row = df[df['strike'] == short_strike]
            long_row  = df[df['strike'] == long_strike]

            if short_row.empty or long_row.empty:
                closing_cost = position.net_premium_collected * 0.25  # fallback estimate
                price_source = "estimated (market data unavailable)"
            else:
                # Cost to close = buy back short + sell back long
                short_ask = (short_row.iloc[0]['bid'] + short_row.iloc[0]['ask']) / 2
                long_bid  = (long_row.iloc[0]['bid']  + long_row.iloc[0]['ask'])  / 2
                closing_cost = (short_ask - long_bid) * 100 * position.short_leg.contracts
                price_source = "live market"
        except Exception as e:
            closing_cost = position.net_premium_collected * 0.25
            price_source = f"estimated (fetch error: {e})"

        # Close the position
        closed = options_account.close_spread(position_id, closing_cost)
        if not closed:
            return json.dumps({"error": f"Failed to close position {position_id}"})

        write_account(f"{name.lower()}_options", options_account.model_dump())

        pnl = closed.profit_loss()
        pct_captured = (pnl / position.net_premium_collected * 100) if position.net_premium_collected else 0
        write_log(name, "account", f"Closed {position.spread_type} on {position.symbol} | P&L ${pnl:.2f} ({pct_captured:.1f}% captured) | reason: {reason}")

        return json.dumps({
            "status": "POSITION CLOSED",
            "position_id": position_id,
            "symbol": position.symbol,
            "spread_type": position.spread_type,
            "original_premium": f"${position.net_premium_collected:.2f}",
            "closing_cost": f"${closing_cost:.2f}",
            "realized_pnl": f"${pnl:.2f}",
            "pct_premium_captured": f"{pct_captured:.1f}%",
            "price_source": price_source,
            "reason": reason,
            "account_summary": options_account.summary()
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


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
