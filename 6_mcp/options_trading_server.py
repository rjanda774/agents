"""
Options Trading Server for Cathie
Real market data (option chains, quotes) from Schwab's Market Data API when
configured (schwab_client.py), falling back automatically to yfinance
otherwise or on any Schwab failure -- every response that touches live chain
data carries a "data_source" field ("schwab" / "yfinance" / "yfinance (schwab
fallback)") so it's always visible which one actually supplied the numbers.
Uses OptionLab for credit spread P/L analysis and probability-of-profit.
With CATHIE_EXECUTION_MODE unset or "simulated" (the default), trades are fully
simulated: sell_credit_spread/close_credit_spread only write to the local
`cathie_options` pseudo-account (accounts.db/options_models.py). With "approve",
sell_credit_spread stages a REAL trade for the user's approval instead
(live_trading.py); nothing here ever sends an order itself -- only the user's
approve_orders.py does, via schwab_execution.py. See CLAUDE.md, "Live trading
on Schwab".
Completely separate from stock trading system
"""
from mcp.server.fastmcp import FastMCP  # type: ignore
from datetime import datetime, timedelta
from typing import Dict, List
import json
import math
import sys
import os
import concurrent.futures
import datetime as dt_mod
from pathlib import Path

import schwab_client


def fetch_with_timeout(fn, timeout_seconds=30):
    """Run fn() in a thread and raise TimeoutError if it takes too long.
    Prevents yfinance from hanging the MCP server indefinitely."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"yfinance request timed out after {timeout_seconds}s")


def _get_next_earnings_date(symbol: str):
    """Best-effort lookup of `symbol`'s next upcoming earnings date via yfinance.

    Returns a `datetime.date`, or None if it can't be determined (no data, or the
    lookup itself failed/timed out) -- callers should treat None as "unknown", not
    "no earnings", and decide how to fail accordingly.
    """
    import yfinance as yf
    import datetime as dt_mod

    def _fetch():
        t = yf.Ticker(symbol)
        # yfinance's Ticker.calendar shape has varied across versions (dict vs.
        # DataFrame-like), so extract defensively rather than assume one structure.
        cal = t.calendar
        raw_dates = None
        if isinstance(cal, dict):
            raw_dates = cal.get("Earnings Date")
        if not raw_dates:
            return None
        if not isinstance(raw_dates, (list, tuple)):
            raw_dates = [raw_dates]
        parsed = []
        for d in raw_dates:
            if isinstance(d, dt_mod.datetime):
                parsed.append(d.date())
            elif isinstance(d, dt_mod.date):
                parsed.append(d)
            else:
                try:
                    parsed.append(dt_mod.date.fromisoformat(str(d)))
                except ValueError:
                    continue
        return min(parsed) if parsed else None

    try:
        return fetch_with_timeout(_fetch, timeout_seconds=15)
    except Exception:
        return None


# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mcp = FastMCP("options_trading_server")


def _finite_or(v, default=None):
    """`v` as a float if it's a real finite number, else `default`.

    yfinance hands back pandas NaN for missing quote fields, and NaN is truthy, so the
    old `float(x or 0.0)` idiom let it straight through. NaN is poison for every
    server-side gate here: all comparisons against it are False, so a NaN premium or
    max loss silently skips the premium floor and the risk cap instead of tripping them.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _regular_session_open(now: datetime | None = None) -> tuple[bool, str]:
    """(is the US stock market in its regular session right now, current time in ET).

    Regular session = Monday-Friday, 9:30-16:00 America/New_York. Clock-based only, so
    it doesn't know about exchange holidays or early closes -- a holiday weekday counts
    as open here."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    now_et = (now or datetime.now(et)).astimezone(et)
    is_open = now_et.weekday() < 5 and dt_mod.time(9, 30) <= now_et.time() < dt_mod.time(16, 0)
    return is_open, now_et.strftime("%a %Y-%m-%d %H:%M ET")


def _yfinance_data_source_label() -> str:
    """Label to attach to a yfinance-sourced response -- distinguishes "Schwab was
    never configured" from "Schwab was configured but this particular call fell
    back", so a spot-check of the logs can tell the two apart."""
    return "yfinance" if not schwab_client.is_configured() else "yfinance (schwab fallback)"


def _fetch_valid_expirations(symbol: str, min_date: dt_mod.date, max_date: dt_mod.date):
    """List expirations for `symbol` in [min_date, max_date] plus current price.

    Tries Schwab first (real broker data) when configured, falling back to
    yfinance on any failure or when Schwab isn't configured at all.

    Returns (current_price, valid_expirations, data_source, source_note) where
    valid_expirations is a list of {"date": "YYYY-MM-DD", "days_to_expiration": int},
    and source_note is a human-readable explanation when a Schwab fallback happened
    (None otherwise).
    """
    source_note = None
    if schwab_client.is_configured():
        try:
            def _call():
                return schwab_client.get_option_chain(symbol, min_date, max_date)
            data = fetch_with_timeout(_call, timeout_seconds=30)
            valid_exps = [
                {"date": d, "days_to_expiration": info["days_to_expiration"]}
                for d, info in sorted(data["expirations"].items())
            ]
            return data["current_price"], valid_exps, "schwab", None
        except Exception as e:
            source_note = f"Schwab market data unavailable ({e}); fell back to yfinance."

    import yfinance as yf

    def _fetch():
        t = yf.Ticker(symbol)
        exps = t.options
        if not exps:
            raise ValueError(f"No options available for {symbol}")
        price = t.fast_info.get("lastPrice", 0)
        return exps, price

    all_exps, price = fetch_with_timeout(_fetch, timeout_seconds=30)
    today = dt_mod.date.today()
    valid_exps = []
    for e in all_exps:
        try:
            d = dt_mod.date.fromisoformat(e)
            days = (d - today).days
            if 25 <= days <= 45:
                valid_exps.append({"date": e, "days_to_expiration": days})
        except ValueError:
            pass
    return float(price), valid_exps, _yfinance_data_source_label(), source_note


def _fetch_chain(symbol: str, expiration_date: str) -> dict:
    """Fetch a normalized option chain for one specific expiration.

    Tries Schwab first (real bid/ask/open interest/IV and real broker-computed
    Greeks) when configured, falling back to yfinance (which has no Greeks at
    all -- delta is computed here via Black-Scholes in that path only) on any
    Schwab failure or when it isn't configured.

    Returns:
        {
            "current_price": float,
            "puts":  [ {strike, bid, ask, lastPrice, volume, openInterest,
                        impliedVolatility, delta, gamma, theta, vega}, ... ],
            "calls": [ ... same shape ... ],   # gamma/theta/vega are None on
                                                # the yfinance path -- get_options_chain
                                                # only ever needed delta there before
            "data_source": "schwab" | "yfinance" | "yfinance (schwab fallback)",
            "source_note": str | None,
        }
    Both lists are sorted by strike. Raises if neither source can produce data.
    """
    if schwab_client.is_configured():
        try:
            def _call():
                d = dt_mod.date.fromisoformat(expiration_date)
                return schwab_client.get_option_chain(symbol, d, d)
            data = fetch_with_timeout(_call, timeout_seconds=30)
            exp_info = data["expirations"].get(expiration_date)
            if not exp_info:
                raise ValueError(
                    f"Schwab returned no contracts for {symbol} expiring {expiration_date}"
                )
            return {
                "current_price": data["current_price"],
                "puts": exp_info["puts"],
                "calls": exp_info["calls"],
                "data_source": "schwab",
                "source_note": None,
            }
        except Exception as e:
            source_note = f"Schwab market data unavailable ({e}); fell back to yfinance for this chain."
    else:
        source_note = None

    import yfinance as yf
    from optionlab.black_scholes import get_bs_info

    def _fetch():
        t = yf.Ticker(symbol)
        price = t.fast_info.get("lastPrice") or t.history(period="1d")["Close"].iloc[-1]
        chain = t.option_chain(expiration_date)
        return float(price), chain

    price, chain = fetch_with_timeout(_fetch, timeout_seconds=30)

    today = dt_mod.date.today()
    exp_date = dt_mod.date.fromisoformat(expiration_date)
    years_to_exp = max((exp_date - today).days, 0) / 365.0

    def compute_delta(strike: float, iv, option_type: str):
        try:
            if not iv or iv <= 0 or years_to_exp <= 0:
                return None
            bs = get_bs_info(s=price, x=float(strike), r=0.05, vol=float(iv), years_to_maturity=years_to_exp)
            return round(bs.put_delta if option_type == "put" else bs.call_delta, 4)
        except Exception:
            return None

    def _rows(df, option_type: str):
        rows = []
        for _, row in df.iterrows():
            iv = _finite_or(row.get("impliedVolatility")) or None  # NaN or 0 -> None
            volume = _finite_or(row.get("volume"))
            open_interest = _finite_or(row.get("openInterest"))
            rows.append(
                {
                    "strike": float(row["strike"]),
                    "bid": _finite_or(row.get("bid"), 0.0),
                    "ask": _finite_or(row.get("ask"), 0.0),
                    "lastPrice": _finite_or(row.get("lastPrice"), 0.0),
                    "volume": int(volume) if volume is not None else None,
                    "openInterest": int(open_interest) if open_interest is not None else None,
                    "impliedVolatility": iv,
                    "delta": compute_delta(row["strike"], iv, option_type),
                    "gamma": None,
                    "theta": None,
                    "vega": None,
                }
            )
        return sorted(rows, key=lambda r: r["strike"])

    return {
        "current_price": price,
        "puts": _rows(chain.puts, "put"),
        "calls": _rows(chain.calls, "call"),
        "data_source": _yfinance_data_source_label(),
        "source_note": source_note,
    }

# Screens chosen for likely options liquidity -- Yahoo's screener has ~19 predefined
# queries total (see yfinance.PREDEFINED_SCREENER_QUERIES), most of which are mutual
# fund/ETF category screens irrelevant here. This is the subset of equity screens
# whose filters (market cap, day volume, etc.) tend to surface names that actually have
# tight, liquid options markets -- small_cap_gainers/most_shorted_stocks and the fund/
# ETF screens are deliberately left out since thin or non-existent options chains are
# common there. Still just a starting list, same as CATHIE_ETF_UNIVERSE -- results are
# unverified until checked with get_options_chain.
SCREENER_QUERIES = {
    "most_actives",
    "day_gainers",
    "day_losers",
    "growth_technology_stocks",
    "undervalued_large_caps",
    "aggressive_small_caps",
}


# Yahoo's screener API hard-caps a single request's result count at 250 (yfinance raises
# ValueError above this) -- there is no further pagination support in yfinance.screen(),
# so 250 is also the most "all the tickers on this screener" can mean here in one call.
SCREENER_MAX_COUNT = 250

# FIXED: count previously defaulted to SCREENER_MAX_COUNT (250) -- with all six screens
# now mandatory every new-trade pass (see templates.py), that meant up to 6 x 250 = 1500
# candidate records injected into the conversation in one cycle, pretty-printed. On a
# 128k-context model (gpt-4o-mini) this reliably blew the context window outright
# (observed: "Your input exceeds the context window of this model"), not just bloated it.
# 40 is still a large jump from the original hardcoded 15 -- comfortably enough to widen
# real candidate discovery -- while keeping six calls' worth of results well within
# budget alongside everything else a trade pass accumulates. count can still be raised
# up to SCREENER_MAX_COUNT for a single deliberately exhaustive call; just don't do that
# on all six screens in the same pass.
SCREENER_DEFAULT_COUNT = 40

# HARD FILTER: discard candidates priced under $50 before they're even returned. Observed
# in practice: aggressive_small_caps has no price floor at all in Yahoo's own filter
# definition (day_gainers/day_losers only require >= $5), and Cathie was seen opening
# credit spreads on ~$2 stocks sourced from the screener. That's a bad fit for this
# strategy regardless of trade frequency -- $5-wide strikes (see long_strike = short_strike
# +/- $5 in the trading rules) don't make sense against a $2 underlying, and sub-$50 names
# tend to have wide bid-ask spreads and thin open interest on top of that. The existing
# ">$100" guidance for individual-stock candidates has always been prompt-only and
# evidently wasn't enough on its own, so this is enforced here instead, before results
# ever reach the model.
MIN_SCREENER_PRICE = 50.0


@mcp.tool()
async def get_stock_screener(query: str = "most_actives", count: int = SCREENER_DEFAULT_COUNT) -> str:
    """Screen for liquid, actively-traded stocks as extra credit-spread candidates,
    beyond the named ETF universe.

    Pulls live results from Yahoo Finance's screener (via yfinance.screen). Use this
    to widen candidate discovery beyond the named ETF universe and whatever the
    research tool happened to surface -- especially useful for finding genuinely
    different names cycle over cycle instead of converging on the same handful.
    Defaults to 40 results per call -- comfortably more than you need to shortlist
    3-5 candidates from, and safe to call across all six screens in one pass without
    risking the model's context window. Raise `count` (up to 250, Yahoo's own cap)
    only for a single screen you specifically want exhaustive coverage of, not on
    every screen in the same pass -- six calls at 250 each was observed to blow the
    context window outright on smaller models.

    This is a discovery tool only: results are NOT pre-verified as optionable. Always
    follow up with get_options_chain(symbol) for any candidate you're seriously
    considering, to confirm it actually has listed options in the 25-45 day window
    with real open interest -- plenty of liquid stocks still have thin or no options
    market. Candidates priced under $50 are already discarded server-side (see
    MIN_SCREENER_PRICE) -- some screens (e.g. aggressive_small_caps) have no price
    floor of their own and can otherwise return very low-priced stocks that don't
    suit this strategy's $5-wide strikes.

    Args:
        query: which screen to run. One of:
            "most_actives" (default) -- highest day-volume large/mid caps, the safest
                bet for a liquid options market.
            "day_gainers" / "day_losers" -- today's biggest movers (%>3 up / %>2.5 down),
                useful for picking directional bias off of momentum.
            "growth_technology_stocks" -- high revenue/EPS growth tech names.
            "undervalued_large_caps" -- low P/E, low PEG large caps.
            "aggressive_small_caps" -- higher-volume small caps; options liquidity is
                less reliable here than the others, double-check open interest.
        count: how many results to return (1-250, default 40). See above -- avoid
            raising this on every screen in the same pass; it's meant for one
            deliberately exhaustive call, not the default across all six.

    Returns:
        JSON list of candidates (symbol, name, price, day % change, volume), or an
        "error" key if the screener request itself failed -- fall back to the named
        ETF universe and your own research in that case. Includes `total_matches` (how
        many stocks match this screen at Yahoo, which can exceed 250) alongside
        `candidate_count` (how many were actually returned, post price-filter) and
        `filtered_low_price_count` (how many results were discarded for being under
        $50) so you can tell whether this call already covers the full screen.
    """
    if query not in SCREENER_QUERIES:
        return json.dumps(
            {"error": f"Unknown query '{query}'. Choose one of: {sorted(SCREENER_QUERIES)}"}
        )
    count = max(1, min(count, SCREENER_MAX_COUNT))

    try:
        import yfinance as yf

        def _fetch():
            return yf.screen(query, count=count)

        result = fetch_with_timeout(_fetch, timeout_seconds=30)
    except Exception as e:
        return json.dumps(
            {
                "error": f"Screener request failed: {e}. Fall back to your named ETF "
                "universe and your own research for candidates this cycle instead."
            }
        )

    quotes = result.get("quotes", []) if isinstance(result, dict) else []
    total_matches = result.get("total") if isinstance(result, dict) else None
    candidates = []
    filtered_low_price_count = 0
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol:
            continue
        price = q.get("regularMarketPrice")
        # Discard anything under MIN_SCREENER_PRICE server-side -- including missing/null
        # price, since we can't confirm it clears the bar. Not just a prompt suggestion:
        # these never reach the model at all.
        if price is None or price < MIN_SCREENER_PRICE:
            filtered_low_price_count += 1
            continue
        candidates.append(
            {
                "symbol": symbol,
                "name": q.get("shortName") or q.get("longName"),
                "price": price,
                "day_change_pct": q.get("regularMarketChangePercent"),
                "volume": q.get("regularMarketVolume"),
                "avg_volume_3m": q.get("averageDailyVolume3Month"),
                "market_cap": q.get("marketCap"),
            }
        )

    return json.dumps(
        {
            "query": query,
            "total_matches": total_matches,
            "candidate_count": len(candidates),
            "filtered_low_price_count": filtered_low_price_count,
            "candidates": candidates,
            "note": (
                f"Unverified candidates -- call get_options_chain(symbol) before treating "
                f"any of these as a real trade candidate. {filtered_low_price_count} result(s) "
                f"under ${MIN_SCREENER_PRICE:.0f} were already discarded and are not shown."
                + (
                    f" ALSO NOTE: total_matches ({total_matches}) exceeds the {count} results "
                    "returned -- this screen has more candidates than this call returned; "
                    "raise `count` (up to 250) if you want a more exhaustive look at this "
                    "one screen specifically."
                    if isinstance(total_matches, int) and total_matches > len(quotes)
                    else ""
                )
            ),
        },
        # Compact, not pretty-printed -- this tool is called up to six times per pass
        # (see templates.py's mandatory-per-screen instruction), and indent=2 alone adds
        # meaningful token overhead multiplied across every one of those calls.
        default=str,
    )


@mcp.tool()
async def get_custom_watchlist() -> str:
    """Get the user's own hand-picked candidate tickers from watchlist.txt.

    This is a plain text file (one ticker per line, '#' comments allowed) that
    the person running this trading floor edits directly -- add or remove
    tickers any time; changes take effect on your very next cycle since this
    reads the file fresh on every call, no restart needed.

    Like get_stock_screener, this is a discovery tool only: these are
    unverified candidates, not pre-checked for optionability or liquidity.
    Always follow up with get_options_chain(symbol) before treating any of
    them as a real candidate, and every server-enforced rule (25-45 DTE,
    delta, premium floor, risk cap, earnings) still applies regardless of
    where a candidate came from -- a ticker being on this list is a suggestion
    to consider, not an instruction to trade it.

    Returns:
        JSON with the ticker list and a count, or a friendly note (not an
        error) if the file is missing or empty -- an empty watchlist is a
        normal, common state, not a failure.
    """
    try:
        from universe import load_custom_watchlist

        tickers = load_custom_watchlist()
        if not tickers:
            return json.dumps(
                {
                    "tickers": [],
                    "count": 0,
                    "note": (
                        "Your custom watchlist (watchlist.txt) is empty or doesn't exist yet. "
                        "This is normal, not an error -- it's an optional extra candidate source "
                        "on top of your named ETF universe and get_stock_screener. Nothing to do "
                        "here; the person running this trading floor can add tickers to "
                        "watchlist.txt at any time."
                    ),
                }
            )
        return json.dumps(
            {
                "tickers": tickers,
                "count": len(tickers),
                "note": (
                    "Unverified candidates from the user's own hand-picked watchlist -- call "
                    "get_options_chain(symbol) before treating any of these as a real candidate, "
                    "same as get_stock_screener's results."
                ),
            }
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


DARK_POOL_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tradealgo_darkpool.json")
DARK_POOL_STALE_HOURS = 30  # fetched twice a trading day; older than this means fetches stopped


@mcp.tool()
async def get_dark_pool_activity(symbol: str = "") -> str:
    """Get TradeAlgo's dark-pool ("DarkFlow") activity: tickers with unusual off-exchange
    (dark-pool / ATS) buying or selling, plus each one's options flow and TradeAlgo's AI
    sentiment. Fetched from the user's own TradeAlgo account twice a trading day and read
    here from a saved file -- no network call, so this is cheap.

    Two lists:
      - "intraday": today's flagged tickers, each with direction "up" (dark-pool activity
        alongside an up trend) or "down".
      - "historical": TradeAlgo's "Historic ATS Gainers" -- the top 20 tickers flagged in
        an EARLIER session (check from_time) whose price rose from first flagged
        (from_price) to last unflagged (to_price) -- percent_change > 0 -- ranked by
        irregular_vol, how unusual their dark-pool volume was.

    How to use it: a research signal to weigh, like get_market_regime -- not a trade
    instruction. Heavy dark-pool buying plus an up trend can support a bull put spread's
    bullish bias; heavy selling plus a down trend can support a bear call spread. Tickers
    here are unverified candidates: confirm each with get_options_chain before treating it
    as real, and every server-enforced rule (25-45 DTE, delta, premium floor, risk cap,
    earnings) still applies regardless of where a ticker came from.

    Args:
        symbol: Optional ticker (e.g. "NVDA") to look up just that one. Empty returns all.

    Returns:
        JSON with fetched_at, age_hours, a "stale" flag, and the matching tickers -- or a
        friendly note (not an error) if no TradeAlgo data has been fetched yet.
    """
    try:
        if not os.path.exists(DARK_POOL_CACHE_PATH):
            return json.dumps({
                "note": (
                    "No TradeAlgo dark-pool data has been fetched yet. That's fine -- carry on "
                    "with your other candidate sources."
                )
            })
        with open(DARK_POOL_CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)

        fetched_at = cache.get("fetched_at")
        age_hours = None
        try:
            age_hours = round((datetime.now() - datetime.fromisoformat(fetched_at)).total_seconds() / 3600, 1)
        except (TypeError, ValueError):
            pass
        stale = age_hours is None or age_hours > DARK_POOL_STALE_HOURS

        want = symbol.strip().upper()
        result = {"fetched_at": fetched_at, "age_hours": age_hours, "stale": stale}
        for section in ("intraday", "historical"):
            tickers = (cache.get(section) or {}).get("tickers") or []
            if want:
                tickers = [t for t in tickers if str(t.get("ticker", "")).upper() == want]
            result[section] = tickers
        if stale:
            result["stale_note"] = (
                f"This data is {age_hours} hours old (or its age is unknown) -- the scheduled "
                "fetch may have stopped. Treat it as out of date."
            )
        if want and not (result["intraday"] or result["historical"]):
            result["note"] = f"{want} isn't on TradeAlgo's dark-pool lists in the latest fetch."
        if cache.get("errors"):
            result["fetch_errors"] = cache["errors"]
        return json.dumps(result, separators=(",", ":"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_options_chain(
    symbol: str,
    expiration_date: str = ""
) -> str:
    """Get a real options chain: real broker data from Schwab's Market Data API
    when configured, falling back automatically to yfinance otherwise (or on any
    Schwab failure). Every response carries a "data_source" field so it's always
    visible which one actually supplied the numbers -- "schwab", "yfinance", or
    "yfinance (schwab fallback)".

    Retrieves actual market data for available options contracts.
    Always call this without an expiration_date first to see the available dates,
    then call again with the chosen expiration_date to get the full chain.

    Args:
        symbol: Stock ticker (e.g., "SPY", "QQQ")
        expiration_date: Specific expiration (YYYY-MM-DD) or empty to list available dates

    Returns:
        JSON with available strikes, prices, open interest, implied volatility, and
        delta for each strike (delta is Schwab's real broker-reported value when
        data_source is "schwab", else computed via Black-Scholes). Deliberately does
        NOT include gamma/theta/vega here -- none of your rules ever need them at the
        per-strike level, only delta (the 0.20 band) and open interest/premium; the
        fuller Greeks for your actual chosen spread are in analyze_credit_spread's
        output instead, where it's one spread, not up to 50 strikes.
    """
    try:
        today = dt_mod.date.today()
        min_date = today + dt_mod.timedelta(days=25)
        max_date = today + dt_mod.timedelta(days=45)

        # If no expiration_date specified, return the list of valid dates only
        if not expiration_date:
            current_price, valid_exps, data_source, source_note = _fetch_valid_expirations(
                symbol, min_date, max_date
            )
            result = {
                "symbol": symbol,
                "current_price": current_price,
                "today": str(today),
                "data_source": data_source,
                "valid_expiration_window": f"{min_date} to {max_date} (25-45 days out)",
                "valid_expirations": valid_exps,
                "instruction": (
                    "Choose one date from valid_expirations above and call get_options_chain again "
                    "with that expiration_date to see strikes, premiums, and Greeks. "
                    "Do NOT use any expiration not listed here."
                ) if valid_exps else "No expirations fall in the 25-45 day window for this symbol. Try a different underlying."
            }
            if source_note:
                result["source_note"] = source_note
            return json.dumps(result, indent=2)

        # Validate the requested expiration is in the valid window
        try:
            exp_date = dt_mod.date.fromisoformat(expiration_date)
            days_to_exp = (exp_date - today).days
        except ValueError:
            return json.dumps({"error": f"Invalid expiration_date format: '{expiration_date}'. Use YYYY-MM-DD."})

        if days_to_exp < 25 or days_to_exp > 45:
            try:
                _, valid_exps, _, _ = _fetch_valid_expirations(symbol, min_date, max_date)
                valid_list = [e["date"] for e in valid_exps] or "none in window"
            except Exception:
                valid_list = "(unable to fetch)"
            too = "only" if days_to_exp < 25 else ""
            direction = "too soon" if days_to_exp < 25 else "too far out"
            return json.dumps({
                "error": f"Expiration {expiration_date} is {too} {days_to_exp} days away — {direction}. "
                         f"Valid window is {min_date} to {max_date}. "
                         f"Valid expirations for {symbol}: {valid_list}"
            })

        chain = _fetch_chain(symbol, expiration_date)

        def _round(v, ndigits):
            return round(v, ndigits) if isinstance(v, (int, float)) else v

        def _public(rows):
            # FIXED: this used to include gamma/theta/vega whenever a record had them
            # (previously that meant "never, on yfinance" -- now that Schwab is
            # actually configured and working, it means "always, with real values").
            # None of Cathie's rules ever check gamma/theta/vega at the per-strike
            # listing level -- only delta (the 0.20 band) and open interest/premium
            # matter here; the fuller Greeks she does use for her chosen spread are
            # already in analyze_credit_spread's output, which is one spread, not up
            # to 50 strikes. Reproduced live: switching from yfinance to a working
            # Schwab connection alone was enough to blow gpt-4o-mini's context window
            # again in the new-trade pass (3-5 candidates x up to 2 calls each x up to
            # 50 real, non-null Greek values per call), the same class of bug as the
            # get_stock_screener context overflow in CLAUDE.md's history -- so drop
            # gamma/theta/vega here unconditionally, not just when null, and round the
            # remaining floats to keep each record as compact as the old yfinance-only
            # payload regardless of which source supplied it.
            out = []
            for r in rows[:25]:
                out.append(
                    {
                        "strike": r["strike"],
                        "lastPrice": _round(r["lastPrice"], 2),
                        "bid": _round(r["bid"], 2),
                        "ask": _round(r["ask"], 2),
                        "volume": r["volume"],
                        "openInterest": r["openInterest"],
                        "impliedVolatility": _round(r["impliedVolatility"], 4),
                        "delta": _round(r["delta"], 4),
                    }
                )
            return out

        result = {
            "symbol": symbol,
            "current_price": chain["current_price"],
            "today": str(today),
            "expiration_date": expiration_date,
            "days_to_expiration": days_to_exp,
            "data_source": chain["data_source"],
            "puts": _public(chain["puts"]),
            "calls": _public(chain["calls"]),
        }
        if chain["source_note"]:
            result["source_note"] = chain["source_note"]

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
    """Analyze a credit spread using OptionLab for accurate P/L and probability of
    profit, priced off real market data -- Schwab's Market Data API when configured,
    falling back automatically to yfinance otherwise (or on any Schwab failure; see
    "data_source" in the response).

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
        JSON with max profit, max loss, breakeven, probability of profit, Greeks.
        Greeks are Schwab's own real, broker-reported values when data_source is
        "schwab"; otherwise (a yfinance path) they're computed via Black-Scholes,
        since yfinance's raw chain has no Greeks at all.
    """
    try:
        import datetime as dt
        from optionlab.models import Inputs
        from optionlab.black_scholes import get_bs_info
        from optionlab import run_strategy

        chain = _fetch_chain(symbol, expiration_date)
        current_price = chain["current_price"]
        option_type = "put" if spread_type == "bull_put" else "call"
        rows = chain["puts"] if option_type == "put" else chain["calls"]

        if not rows:
            return json.dumps({"error": f"No {option_type} options available for {symbol} on {expiration_date}"})

        # Snap to nearest available strikes rather than failing on exact match
        short_row = min(rows, key=lambda r: abs(r["strike"] - short_strike))
        long_row  = min(rows, key=lambda r: abs(r["strike"] - long_strike))

        # Warn if we snapped to different strikes
        snapped = {}
        if short_row["strike"] != short_strike:
            snapped["short_strike_adjusted"] = f"{short_strike} -> {short_row['strike']}"
        if long_row["strike"] != long_strike:
            snapped["long_strike_adjusted"] = f"{long_strike} -> {long_row['strike']}"
        short_strike = short_row["strike"]
        long_strike  = long_row["strike"]

        # Use midpoint of bid/ask for fair value
        short_premium = (short_row["bid"] + short_row["ask"]) / 2
        long_premium  = (long_row["bid"]  + long_row["ask"])  / 2
        iv = float(short_row["impliedVolatility"] or 0.0)

        # Guard: a non-finite quote (NaN/inf from any data source) would slip past every
        # "<=" check below, since comparisons against NaN are always False, and end up as
        # a NaN premium/max loss that also slips past sell_credit_spread's gates.
        if not all(math.isfinite(v) for v in (short_premium, long_premium, iv, float(current_price))):
            return json.dumps({
                "error": f"Market data for {symbol} {expiration_date} is incomplete (non-numeric "
                         f"bid/ask, implied volatility, or underlying price). Can't price this "
                         f"spread reliably -- try a different strike, expiration, or underlying."
            })

        # Guard: zero or near-zero premium means the option is illiquid or too far OTM
        if short_premium <= 0.01:
            return json.dumps({
                "error": f"Short strike {short_strike} has zero or negligible premium (${short_premium:.4f}). "
                         f"This strike is too far out-of-the-money or illiquid. "
                         f"Move the short strike closer to the current price (${current_price:.2f}) until premium > $0.05. "
                         f"Try a strike with delta closer to 0.15."
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

        # Guard: zero IV means the data source returned bad data for this expiration
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

        # --- OptionLab: P/L profile + probability of profit (source-independent --
        # just needs strikes/premiums/IV/dates, same math regardless of who supplied them) ---
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

        # --- Greeks: Schwab's own real, broker-reported values when available (it
        # returns delta/gamma/theta/vega directly per contract); otherwise fall back
        # to Black-Scholes, since yfinance's raw chain has no Greeks at all. ---
        if (
            chain["data_source"] == "schwab"
            and short_row["delta"] is not None
            and long_row["delta"] is not None
        ):
            short_leg_delta = round(short_row["delta"], 4)
            net_delta = round(long_row["delta"] - short_row["delta"], 4)
            net_theta = round((short_row["theta"] or 0.0) - (long_row["theta"] or 0.0), 4)
            net_vega  = round((long_row["vega"]  or 0.0) - (short_row["vega"]  or 0.0), 4)
            net_gamma = round((long_row["gamma"] or 0.0) - (short_row["gamma"] or 0.0), 4)
            greeks_source = "schwab (real, broker-reported)"
        else:
            short_bs = get_bs_info(s=float(current_price), x=short_strike, r=0.05, vol=iv, years_to_maturity=years_to_exp)
            long_bs  = get_bs_info(s=float(current_price), x=long_strike,  r=0.05, vol=iv, years_to_maturity=years_to_exp)

            if option_type == "put":
                short_leg_delta = round(short_bs.put_delta, 4)
                net_delta = round(long_bs.put_delta  - short_bs.put_delta,  4)
                net_theta = round(short_bs.put_theta - long_bs.put_theta,   4)
            else:
                short_leg_delta = round(short_bs.call_delta, 4)
                net_delta = round(long_bs.call_delta  - short_bs.call_delta, 4)
                net_theta = round(short_bs.call_theta - long_bs.call_theta,  4)
            net_vega  = round(long_bs.vega  - short_bs.vega,  4)
            net_gamma = round(long_bs.gamma - short_bs.gamma, 4)
            greeks_source = "Black-Scholes (yfinance has no Greeks)"

        # P/L summary (net_premium_per_spread already computed above)
        net_premium_per_spread = round(short_premium - long_premium, 4)
        spread_width = abs(short_strike - long_strike)
        max_profit_dollars = round(net_premium_per_spread * 100 * contracts, 2)
        max_loss_dollars   = round((spread_width - net_premium_per_spread) * 100 * contracts, 2)
        if not (math.isfinite(max_profit_dollars) and math.isfinite(max_loss_dollars)):
            return json.dumps({"error": f"Computed P/L for {symbol} is not a finite number -- market data is unusable right now."})

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
            "data_source": chain["data_source"],
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
                "source": greeks_source,
                "short_leg_delta": short_leg_delta,
                "net_delta": net_delta,
                "net_theta": net_theta,
                "net_vega":  net_vega,
                "net_gamma": net_gamma,
            },
        }
        if chain["source_note"]:
            result["source_note"] = chain["source_note"]

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

        # HARD ENFORCEMENT: no earnings from today through EARNINGS_BUFFER_DAYS_AFTER_EXPIRATION
        # days past this trade's own expiration. "Skip if earnings falls in the window" has been
        # prompt-only guidance in templates.py all along ("EARNINGS AVOIDANCE... If it does, skip
        # it entirely") -- and, like every other prompt-only rule in this file before it got moved
        # server-side, that wasn't enough on its own: a live NVDA bull put spread was opened one
        # day before NVDA's own earnings release. An earnings move can blow through both strikes
        # regardless of how far OTM they are, so this is now checked here rather than trusted to
        # have been checked already. The buffer extends past expiration, not just up to it --
        # implied vol (and the position's own remaining vega/gamma risk near the tail of its life)
        # starts pricing in an upcoming earnings date well before the event itself, so a spread
        # expiring just a few days ahead of earnings is still exposed to that run-up.
        # Fails OPEN (allows the trade, with a warning) if the earnings date can't be determined --
        # yfinance's calendar data isn't always populated -- rather than blocking every trade
        # whenever that data is flaky; the warning is included in the success response so a gap
        # here stays visible instead of silent.
        EARNINGS_BUFFER_DAYS_AFTER_EXPIRATION = 20
        # ETFs have no earnings, so their date is always "unknown" -- which would make real
        # trades (approve mode fails closed on unknown) impossible on Cathie's named ETFs.
        # Those are exempt from both earnings checks.
        import trade_rules
        from universe import CATHIE_ETF_UNIVERSE
        is_named_etf = symbol.upper() in CATHIE_ETF_UNIVERSE
        next_earnings = None if is_named_etf else _get_next_earnings_date(symbol)
        earnings_warning = None
        if is_named_etf:
            pass
        elif next_earnings is not None:
            danger_end = exp_date + dt_mod.timedelta(days=EARNINGS_BUFFER_DAYS_AFTER_EXPIRATION)
            if today <= next_earnings <= danger_end:
                return json.dumps({
                    "error": (
                        f"TRADE REJECTED: {symbol} has earnings on {next_earnings.isoformat()}, "
                        f"which falls within the danger window (today through {danger_end.isoformat()}"
                        f" -- {EARNINGS_BUFFER_DAYS_AFTER_EXPIRATION} days past this trade's "
                        f"{expiration_date} expiration). Earnings moves can blow through both "
                        "strikes regardless of delta. Pick a different underlying, choose an "
                        "expiration that clears this window, or wait until after the earnings date."
                    )
                })
        else:
            earnings_warning = (
                f"Could not verify {symbol}'s next earnings date (yfinance calendar data "
                "unavailable) -- proceeding without an automated earnings check for this trade. "
                "Double-check earnings timing yourself before relying on this position."
            )

        # HARD ENFORCEMENT (trade_rules.py, added 2026-10-08): no new spread within 5 trading
        # days after an earnings report. The forward check above can't see a report that
        # already happened -- yfinance then shows the NEXT date -- which let PENG through on
        # its own earnings day.
        last_earnings = None
        if not is_named_etf:
            last_earnings = trade_rules.get_last_earnings_date(symbol)
            recent = trade_rules.recent_earnings_problem(symbol, last_earnings)
            if recent:
                return json.dumps({"error": f"TRADE REJECTED: {recent}. Pick a different underlying."})
            if last_earnings is None:
                note = (f"Could not verify when {symbol} last reported earnings -- the 5-trading-day "
                        "post-earnings check wasn't applied.")
                earnings_warning = f"{earnings_warning} {note}" if earnings_warning else note

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

        # HARD ENFORCEMENT (trade_rules.py): individual stocks must trade at $100+; the
        # named ETF universe is exempt. Was prompt-only guidance before 2026-10-08.
        floor = trade_rules.price_floor_problem(symbol, analysis.get("current_price"))
        if floor:
            return json.dumps({"error": f"TRADE REJECTED: {floor}. Pick a different underlying."})
        
        # Extract premium from analysis
        # Fields match the output of analyze_credit_spread
        net_premium = float(analysis['premium']['net_per_spread']) * 100 * contracts
        max_loss = float(analysis['profit_loss']['max_loss'].replace('$', '').replace(',', ''))
        short_premium_per = float(analysis['premium']['short_leg'])
        long_premium_per  = float(analysis['premium']['long_leg'])
        short_leg_delta = analysis.get('greeks', {}).get('short_leg_delta')

        # HARD ENFORCEMENT: every gate below is a comparison, and comparisons against NaN
        # are always False -- so a NaN premium or max loss would pass the premium floor and
        # the risk cap and get written into the ledger. analyze_credit_spread already
        # rejects non-finite quotes; this re-checks at the point of mutation regardless.
        if not all(math.isfinite(v) for v in (net_premium, max_loss, short_premium_per, long_premium_per)):
            return json.dumps({
                "error": "TRADE REJECTED: premium or max loss is not a finite number (bad market "
                         "data). Nothing was opened -- try again later or pick a different candidate."
            })

        # HARD ENFORCEMENT: short leg must be genuinely far OTM (|delta| < 0.20). The prompt
        # has asked for a delta band all along, but get_options_chain never actually returned
        # real delta (yfinance's raw chain has no delta column at all -- that check was
        # silently unenforceable), which let strikes close to or even in-the-money through.
        # Real delta is now computed via Black-Scholes in both get_options_chain and here.
        # Cap raised from 0.15 to 0.20 to qualify more candidate strikes -- accepted
        # tradeoff: strikes this close to the money carry a somewhat higher
        # assignment/breach probability and a lower win rate than under the old cap.
        MAX_SHORT_DELTA = 0.20
        if short_leg_delta is None:
            return json.dumps({
                "error": "TRADE REJECTED: could not compute short leg delta (missing/zero implied "
                         "volatility or bad inputs from analyze_credit_spread). Cannot verify this "
                         "strike is far enough out-of-the-money -- try a different strike or underlying."
            })
        if abs(short_leg_delta) > MAX_SHORT_DELTA:
            return json.dumps({
                "error": (
                    f"TRADE REJECTED: short leg delta {short_leg_delta} (magnitude "
                    f"{abs(short_leg_delta):.3f}) exceeds the {MAX_SHORT_DELTA} cap. This strike is too "
                    f"close to the money. Move the short strike further out-of-the-money (lower strike "
                    f"for a call, higher strike for a put) and retry."
                )
            })

        # HARD ENFORCEMENT: minimum net premium per trade. The strategy text/prompt has
        # asked for this all along, but it's been observed slipping through in practice
        # (e.g. a live $69.50 and $18.50 trade both under the stated $100 floor) -- prompt
        # instructions aren't reliable enough on their own, so enforce it here too.
        # Lowered 100 -> 50 -- the $100 floor combined with the 25-45 DTE/delta<0.20/3%-cash
        # gates was plausibly excluding too many otherwise-valid trades on a $10k account,
        # contributing to very low trade frequency. $50 still rules out trading for pennies.
        MIN_NET_PREMIUM = 50.0
        if net_premium < MIN_NET_PREMIUM:
            return json.dumps({
                "error": (
                    f"TRADE REJECTED: net premium ${net_premium:.2f} is below the ${MIN_NET_PREMIUM:.2f} "
                    f"minimum. Not worth the capital/risk for this little income. Increase `contracts` "
                    f"or pick a different underlying with richer premium (you can't just move the short "
                    f"strike closer to the money for more premium -- that would breach the delta cap)."
                )
            })

        # REAL TRADING (CLAUDE.md, "Live trading on Schwab", step 4). In approve mode the
        # trade is staged in the cathie_live ledger for the user to approve by hand
        # (approve_orders.py) instead of opened on paper. The paper-cash risk cap below
        # doesn't apply to real money: live_trading checks max loss against the real
        # account's Funds Available and Day Net Liq, now and again at approval.
        import live_trading
        mode = live_trading.execution_mode()
        if mode != "simulated":
            if mode != "approve":
                return json.dumps({"error": f"CATHIE_EXECUTION_MODE={mode!r} isn't supported "
                                            "(use 'simulated' or 'approve'). Nothing was traded."})
            if earnings_warning:
                # Paper trades fail open on unknown earnings; real money fails closed.
                missing = " and ".join(
                    part for part, unknown in (("next earnings date", next_earnings is None),
                                               ("last report date", last_earnings is None)) if unknown
                )
                return json.dumps({"error": f"TRADE REJECTED: {symbol}'s {missing} can't be verified "
                                            "right now, and real trades require both. Pick a different "
                                            "underlying."})
            staged = live_trading.stage(
                symbol, spread_type, short_strike, long_strike, expiration_date, contracts,
                net_premium / (100 * contracts), max_loss, rationale,
            )
            if analysis.get("data_source"):
                staged["data_source"] = analysis["data_source"]
            return json.dumps(staged, indent=2)

        # HARD ENFORCEMENT: never risk more than 8% of available cash on a single trade,
        # AND never risk more than 5x the premium actually collected for that trade --
        # both must hold, so the effective cap is whichever of the two is smaller. Raised
        # from a flat 3%-of-cash cap (no premium-ratio check existed before) after low
        # trade frequency suggested 3% was too tight to clear on a $10k account for most
        # otherwise-valid setups; the added 5x-premium check keeps a thin-premium trade
        # from using the full 8% anyway, since a bigger cash cushion alone doesn't make a
        # weak risk/reward trade sound. Checked against current cash, not
        # cash-after-this-trade's-premium -- what you'd actually have to cover the max
        # loss from is what you have going in.
        MAX_RISK_PCT = 0.08
        MAX_RISK_TO_PREMIUM_RATIO = 5.0
        max_allowed_risk_pct = options_account.cash * MAX_RISK_PCT
        max_allowed_risk_premium = net_premium * MAX_RISK_TO_PREMIUM_RATIO
        max_allowed_risk = min(max_allowed_risk_pct, max_allowed_risk_premium)
        if max_loss > max_allowed_risk:
            limiting_rule = (
                f"the {MAX_RISK_PCT * 100:.0f}%-of-cash cap "
                f"({(max_loss / options_account.cash * 100) if options_account.cash else float('inf'):.1f}% "
                f"of ${options_account.cash:.2f} available)"
                if max_allowed_risk_pct <= max_allowed_risk_premium
                else f"the {MAX_RISK_TO_PREMIUM_RATIO:.0f}x-net-premium cap "
                     f"(net premium collected is only ${net_premium:.2f})"
            )
            return json.dumps({
                "error": (
                    f"TRADE REJECTED: max loss ${max_loss:.2f} exceeds {limiting_rule}. "
                    f"Max allowed risk right now is ${max_allowed_risk:.2f}. Reduce `contracts`, "
                    f"pick a narrower spread width, or choose strikes closer together to lower max loss."
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
            "message": f"Successfully opened {contracts} {spread_type} spread(s) on {symbol}",
            "data_source": analysis.get("data_source"),
        }
        if earnings_warning:
            result["earnings_check_warning"] = earnings_warning
        if analysis.get("source_note"):
            result["source_note"] = analysis["source_note"]

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

    HARD ENFORCEMENT: this only succeeds if one of these is actually true, verified here
    server-side against live data -- not just whatever `reason` you pass:
    - The position has captured 75%+ of max profit (closing cost <= 25% of original premium)
    - The underlying has breached the short strike (moving against the position)
    - 7 or fewer days remain to expiration (avoid pin risk / assignment risk)
    If none of those hold, the call is rejected and the position stays open. Don't call this
    just because you feel like reconsidering a position -- only when one of the three
    conditions above is genuinely met.

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
        from options_models import OptionsAccount
        from database import read_account, write_account, write_log

        # Real Schwab trades (approve mode) are closed by code, not here -- live_trading.check_exits.
        import live_trading
        real = live_trading.load_ledger().get(position_id) if live_trading.execution_mode() != "simulated" else None
        if real is not None:
            return json.dumps({"error": (
                f"{position_id} is a REAL Schwab trade ({real.label()}, status {real.status}). "
                "Real positions are closed automatically by code when an exit rule fires (75% "
                "profit, breach, 7 DTE), not by this tool. Nothing was closed. If you think it "
                "should be closed for another reason, say so in your summary."
            )})

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

        # Fetch current market prices AND the underlying's current price together --
        # need both to verify whether a close is actually justified, not just take the
        # LLM's word for it via `reason`. Real Schwab data when configured, falling
        # back to yfinance automatically (or on any Schwab failure) -- same source
        # priority as get_options_chain/analyze_credit_spread.
        market_data_available = False
        current_price = None
        closing_cost = None
        try:
            chain = _fetch_chain(position.symbol, position.expiration_date)
            current_price = chain["current_price"]
            rows = chain["puts"] if position.spread_type == "bull_put" else chain["calls"]

            short_row = min(rows, key=lambda r: abs(r["strike"] - position.short_leg.strike))
            long_row  = min(rows, key=lambda r: abs(r["strike"] - position.long_leg.strike))

            short_bid_q, short_ask_q = short_row["bid"], short_row["ask"]
            long_bid_q,  long_ask_q  = long_row["bid"],  long_row["ask"]

            # Guard: a 0/0 bid-ask on a leg is not a real quote, it's the data source's way
            # of saying no market maker has an active price right now -- routinely true
            # outside regular trading hours, or for a thin/deep-OTM strike, not proof the
            # option is worth nothing. Trusting it as-is silently turned "no live data"
            # into a fabricated $0.00 closing cost, which trivially satisfies the
            # 75%-captured profit rule below and force-closes a position that was never
            # actually verified against a real price. Require at least one nonzero side
            # on each leg before treating this as a usable quote; otherwise fall through
            # to the existing (correct) "no market data" handling below, which assumes
            # breakeven rather than fabricating a gain.
            # Also require every quote (and the underlying price used for the breach check) to
            # be finite: NaN on one side would otherwise pass the "> 0" check via the other
            # side and produce a NaN closing_cost -- and a NaN P&L written to the ledger if
            # the breach or expiry trigger then fired.
            quotes_finite = all(
                math.isfinite(_finite_or(v, float("nan")))
                for v in (short_bid_q, short_ask_q, long_bid_q, long_ask_q, current_price)
            )
            if quotes_finite and (short_bid_q > 0 or short_ask_q > 0) and (long_bid_q > 0 or long_ask_q > 0):
                # Cost to close = buy back short + sell back long
                short_ask = (short_bid_q + short_ask_q) / 2
                long_bid  = (long_bid_q  + long_ask_q)  / 2
                closing_cost = (short_ask - long_bid) * 100 * position.short_leg.contracts
                price_source = f"live market ({chain['data_source']})"
                market_data_available = True
        except Exception:
            pass  # handled below -- no fabricated "estimate", we just don't have data

        # HARD ENFORCEMENT: verify a real closing rule actually applies, server-side.
        # We've seen positions get closed within minutes of opening them for no valid
        # reason -- don't just trust `reason`, check the actual numbers.
        days_to_exp = (dt_mod.date.fromisoformat(position.expiration_date) - today).days
        expiry_trigger = days_to_exp <= 7  # computable without any market data

        breach_trigger = False
        profit_trigger = False
        if market_data_available:
            if position.spread_type == "bull_put":
                breach_trigger = current_price <= position.short_leg.strike
            else:
                breach_trigger = current_price >= position.short_leg.strike
            if position.net_premium_collected:
                profit_trigger = closing_cost <= 0.25 * position.net_premium_collected

        # HARD ENFORCEMENT: breach and profit exits act on prices, so only during the
        # regular session. Overnight/pre/after-market underlying prints are thin, and
        # option quotes don't update at all outside regular hours, so they're stale.
        # Observed live: ORCL dipped to $133 overnight, below a $135 short put, and this
        # tool closed the spread at 00:52 ET for a -$185 loss; ORCL was back at ~$138
        # by the next day. The 7-day rule depends only on the date, so it still applies
        # at any hour.
        session_open, market_time = _regular_session_open()
        held_until_open = []
        if not session_open:
            if breach_trigger:
                held_until_open.append("short strike breached")
            if profit_trigger:
                held_until_open.append("profit target")
            breach_trigger = profit_trigger = False

        if not (expiry_trigger or breach_trigger or profit_trigger):
            detail = (
                f"current price ${current_price:.2f} vs short strike {position.short_leg.strike} "
                f"(not breached); closing cost ${closing_cost:.2f} is "
                f"{(closing_cost / position.net_premium_collected * 100) if position.net_premium_collected else 0:.1f}% "
                f"of original premium ${position.net_premium_collected:.2f} (need <=25% to take profit)."
                if market_data_available else
                "live market data was unavailable to check the breach/profit conditions."
            )
            hours_note = ""
            if not session_open:
                hours_note = (
                    f" It's {market_time}, outside regular market hours (Mon-Fri 9:30-16:00 ET): "
                    "breach and profit exits are only acted on during regular hours, when prices "
                    "and option quotes are live."
                )
                if held_until_open:
                    hours_note += (
                        f" At these off-hours prices, {' and '.join(held_until_open)} would apply "
                        "-- re-check during market hours."
                    )
            return json.dumps({
                "error": (
                    f"CLOSE REJECTED: no exit rule applies to position {position_id} yet. "
                    f"Days to expiration: {days_to_exp} (need <=7 to force-close). {detail}"
                    f"{hours_note} Leave this position open."
                )
            })

        if not market_data_available:
            # Only reachable via expiry_trigger (the only rule verifiable without market
            # data). Don't fabricate a profit estimate -- assume breakeven (pay back the
            # full original premium) rather than silently reporting a fake gain.
            closing_cost = position.net_premium_collected
            price_source = "ESTIMATED (market data unavailable) -- assumed breakeven, forced close at DTE<=7"

        # Which rule(s) actually justified this close, as verified above -- recorded in
        # the ledger, the log and the response, because the model's own `reason` can't be
        # trusted (ORCL 2026-09-29 was logged as "DTE <= 7 days" with 17 days left; it
        # was really a breach close).
        rules = []
        if expiry_trigger:
            rules.append(f"7 or fewer days to expiration ({days_to_exp} left)")
        if breach_trigger:
            op, leg = ("<=", "put") if position.spread_type == "bull_put" else (">=", "call")
            rules.append(
                f"short strike breached ({position.symbol} ${current_price:.2f} {op} "
                f"{position.short_leg.strike} short {leg})"
            )
        if profit_trigger:
            rules.append(
                f"profit target (closing cost ${closing_cost:.2f} <= 25% of "
                f"${position.net_premium_collected:.2f} premium)"
            )
        close_rule = "; ".join(rules)
        underlying_price = round(float(current_price), 2) if market_data_available else None

        # Close the position
        closed = options_account.close_spread(position_id, closing_cost)
        if not closed:
            return json.dumps({"error": f"Failed to close position {position_id}"})
        closed.close_rule = close_rule
        closed.underlying_price_at_close = underlying_price

        write_account(f"{name.lower()}_options", options_account.model_dump())

        pnl = closed.profit_loss()
        pct_captured = (pnl / position.net_premium_collected * 100) if position.net_premium_collected else 0
        price_text = f"${underlying_price:.2f}" if underlying_price is not None else "unavailable"
        write_log(
            name, "account",
            f"Closed {position.spread_type} on {position.symbol} | P&L ${pnl:.2f} "
            f"({pct_captured:.1f}% captured) | rule: {close_rule} | {position.symbol} price "
            f"{price_text} at {market_time} ({price_source}) | model's stated reason: {reason}"
        )

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
            "close_rule": close_rule,
            "underlying_price": underlying_price,
            "checked_at": market_time,
            "reason": reason,
            "note": (
                "close_rule is the exit rule the server actually verified; when you report "
                "this close, give close_rule as the reason, not your own `reason` text."
            ),
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
            result = {
                "message": "No paper options positions yet",
                "open_positions": [],
                "closed_positions": []
            }
        else:
            options_account = OptionsAccount(**options_data)
            result = {
                "account_summary": options_account.summary(),
                "open_positions": [pos.model_dump() for pos in options_account.open_positions],
                "closed_positions": [pos.model_dump() for pos in options_account.closed_positions]
            }
        # Real trades (approve mode) live in their own ledger; show them alongside, clearly
        # labelled, so Cathie doesn't re-propose a staged spread or forget a real one.
        import live_trading
        if live_trading.execution_mode() != "simulated":
            result["real_schwab_trades"] = live_trading.ledger_summary()
        
        return json.dumps(result, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run(transport='stdio')
