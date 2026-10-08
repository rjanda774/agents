"""
Entry rules shared by every place a credit spread can be opened -- the spread
scanner, sell_credit_spread (paper and approve mode) and approval of real trades
(live_trading.check_for_approval) -- so they can't drift apart.

Added 2026-10-08 after the scanner's first live run surfaced PENG (a ~$72 stock)
with a bull put 60/55 on the same day it reported earnings. The existing earnings
rule only looks FORWARD (today through 20 days past expiration), and once a company
reports, yfinance shows its NEXT date, so "just reported" slipped through. The user
chose both rules below.
"""
import concurrent.futures
import datetime as dt

# Individual stocks must trade at $100+. Cathie's named ETF universe is exempt: many of
# those are liquid and steady well under $100 (XLF, EEM, SLV, TLT...).
MIN_STOCK_PRICE = 100.0

# No new spread within this many trading days after an earnings report: the stock can
# keep moving hard for days, and its option prices are distorted meanwhile.
RECENT_EARNINGS_TRADING_DAYS = 5

# Wide-market rule: the natural credit (what you'd get selling at the bid and buying at
# the ask) may be at most this fraction of the spread's WIDTH below the mid. Measured
# against the width, not the credit: on a $1,000 stock each leg's bid-ask is wide in
# dollars while a $10-wide spread's credit is ~$1, so a percent-of-credit rule rejected
# spreads that were fine (MU 940/930 on 2026-10-08: mid 1.88, natural 1.25 = 34% of the
# credit but 6.3% of the width). User chose 10% on 2026-10-08: $0.50 on a $5 spread,
# $1.00 on a $10 spread. Replaced "natural no more than 25% below mid".
MAX_GAP_PCT_OF_WIDTH = 0.10


def price_floor_problem(symbol: str, price: float | None) -> str | None:
    """Why `symbol` at `price` breaks the $100 rule, or None if it's fine."""
    from universe import CATHIE_ETF_UNIVERSE
    if symbol.upper() in CATHIE_ETF_UNIVERSE:
        return None
    if price is None or not price >= MIN_STOCK_PRICE:
        shown = f"${price:,.2f}" if isinstance(price, (int, float)) else "unknown"
        return (f"{symbol} trades at {shown}; individual stocks must be ${MIN_STOCK_PRICE:.0f}+ "
                "(only the named ETF universe is exempt)")
    return None


def wide_market_problem(mid: float, natural: float, width: float) -> str | None:
    """Why a spread's market is too wide to trust its mid, or None if it's fine."""
    limit = MAX_GAP_PCT_OF_WIDTH * width
    gap = mid - natural
    if not gap <= limit + 1e-9:
        return (f"market too wide: natural {natural:.2f} is ${gap:.2f} below mid {mid:.2f} "
                f"({gap / width:.1%} of the ${width:g} width; limit {MAX_GAP_PCT_OF_WIDTH:.0%} = ${limit:.2f})")
    return None


def get_last_earnings_date(symbol: str, timeout: float = 15) -> dt.date | None:
    """Most recent earnings date on or before today, via yfinance's earnings-dates
    table (which lists past and upcoming reports). None if it can't be determined."""
    def _fetch():
        import yfinance as yf
        df = yf.Ticker(symbol).get_earnings_dates(limit=8)
        if df is None or len(df) == 0:
            return None
        today = dt.date.today()
        past = [ts.date() for ts in df.index if ts.date() <= today]
        return max(past) if past else None

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(_fetch).result(timeout=timeout)
    except Exception:
        return None
    finally:
        pool.shutdown(wait=False)


def recent_earnings_problem(symbol: str, last_date: dt.date | None,
                            today: dt.date | None = None) -> str | None:
    """Why `symbol` is still inside its post-earnings cooling-off, or None. Counts
    weekdays (exchange holidays aren't known, so a holiday counts as a trading day).
    Unknown -> None here; callers decide whether unknown is acceptable."""
    import numpy as np
    if last_date is None:
        return None
    today = today or dt.date.today()
    traded_since = int(np.busday_count(last_date, today))
    if traded_since < RECENT_EARNINGS_TRADING_DAYS:
        return (f"{symbol} reported earnings on {last_date.isoformat()}, {traded_since} trading day(s) ago; "
                f"wait {RECENT_EARNINGS_TRADING_DAYS} trading days after a report")
    return None
