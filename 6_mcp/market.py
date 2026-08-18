from polygon import RESTClient
from dotenv import load_dotenv
import os
from datetime import datetime
import random
from database import write_market, read_market
from functools import lru_cache
from datetime import timezone
import time
import threading

# Polygon free tier = 5 requests/minute
# Rate limiter: track timestamps of recent calls and wait if needed
_polygon_lock = threading.Lock()
_polygon_call_times = []
_POLYGON_MAX_CALLS = 5
_POLYGON_WINDOW = 62  # slightly over 60s to be safe

def _polygon_rate_limited_call(fn):
    """Call fn() respecting Polygon free tier limit of 5 requests/minute."""
    with _polygon_lock:
        now = time.monotonic()
        # Drop timestamps older than the window
        while _polygon_call_times and now - _polygon_call_times[0] > _POLYGON_WINDOW:
            _polygon_call_times.pop(0)
        # If at limit, wait until oldest call falls outside window
        if len(_polygon_call_times) >= _POLYGON_MAX_CALLS:
            wait = _POLYGON_WINDOW - (now - _polygon_call_times[0]) + 0.1
            if wait > 0:
                print(f"Polygon rate limit: waiting {wait:.1f}s...")
                time.sleep(wait)
            # Re-prune after sleeping
            now = time.monotonic()
            while _polygon_call_times and now - _polygon_call_times[0] > _POLYGON_WINDOW:
                _polygon_call_times.pop(0)
        _polygon_call_times.append(time.monotonic())
    return fn()

load_dotenv(override=True)

polygon_api_key = os.getenv("POLYGON_API_KEY")
polygon_plan = os.getenv("POLYGON_PLAN")

is_paid_polygon = polygon_plan == "paid"
is_realtime_polygon = polygon_plan == "realtime"


def is_market_open() -> bool:
    def _call():
        client = RESTClient(polygon_api_key)
        market_status = client.get_market_status()
        return market_status.market == "open"
    try:
        return _polygon_rate_limited_call(_call)
    except Exception as e:
        # Unlike get_share_price, there's no safe "just make something up" fallback here -
        # trading blind on a guessed market status is worse than skipping a cycle. Fail
        # safe by reporting closed, so trading_floor.py's scheduler loop skips this cycle
        # and tries again next time instead of crashing the whole process.
        print(f"Was not able to check market status via Polygon due to {e}; treating market as closed for this cycle")
        return False


def get_all_share_prices_polygon_eod() -> dict[str, float]:
    """With much thanks to student Reema R. for fixing the timezone issue with this!"""
    def _call():
        client = RESTClient(polygon_api_key)
        probe = client.get_previous_close_agg("SPY")[0]
        last_close = datetime.fromtimestamp(probe.timestamp / 1000, tz=timezone.utc).date()
        results = client.get_grouped_daily_aggs(last_close, adjusted=True, include_otc=False)
        return {result.ticker: result.close for result in results}
    # Counts as 2 calls (prev close + grouped daily)
    _polygon_rate_limited_call(lambda: None)  # reserve a slot
    return _polygon_rate_limited_call(_call)


@lru_cache(maxsize=2)
def get_market_for_prior_date(today):
    market_data = read_market(today)
    if not market_data:
        market_data = get_all_share_prices_polygon_eod()
        write_market(today, market_data)
    return market_data


def get_share_price_polygon_eod(symbol) -> float:
    today = datetime.now().date().strftime("%Y-%m-%d")
    market_data = get_market_for_prior_date(today)
    return market_data.get(symbol, 0.0)


def get_share_price_polygon_min(symbol) -> float:
    def _call():
        client = RESTClient(polygon_api_key)
        result = client.get_snapshot_ticker("stocks", symbol)
        return result.min.close or result.prev_day.close
    return _polygon_rate_limited_call(_call)


def get_share_price_polygon(symbol) -> float:
    if is_paid_polygon:
        return get_share_price_polygon_min(symbol)
    else:
        return get_share_price_polygon_eod(symbol)


def get_share_price(symbol) -> float:
    if polygon_api_key:
        try:
            return get_share_price_polygon(symbol)
        except Exception as e:
            print(f"Was not able to use the polygon API due to {e}; using a random number")
    return float(random.randint(1, 100))
