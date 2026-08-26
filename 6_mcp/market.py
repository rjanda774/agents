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

# Rate limiter: track timestamps of recent calls and wait if needed.
_polygon_lock = threading.Lock()
_polygon_call_times = []
# Free tier = Polygon's documented 5 requests/minute. Paid/realtime plans allow
# substantially more -- this cap used to be hardcoded to the free-tier number
# regardless of POLYGON_PLAN, needlessly throttling paid/realtime users to 5/min too.
# Still a real cap on paid/realtime rather than "unlimited": exact per-plan limits
# vary and a bounded queue here is safer than risking a real 429 from Polygon itself.
_POLYGON_FREE_MAX_CALLS = 5
_POLYGON_PAID_MAX_CALLS = 100
_POLYGON_WINDOW = 62  # slightly over 60s to be safe
# Cap how long a single caller will wait for a slot rather than blocking indefinitely.
# Callers here (is_market_open, share-price lookups, get_market_regime) are invoked
# from async MCP tool handlers with their own ~120s client-side timeout -- this must
# stay comfortably under that, or a caller stuck waiting past its own timeout budget
# never gets the chance to fail cleanly; it just hangs until the client gives up.
_POLYGON_MAX_WAIT = 90.0


def _polygon_rate_limit() -> tuple[int, float]:
    if is_realtime_polygon or is_paid_polygon:
        return _POLYGON_PAID_MAX_CALLS, _POLYGON_WINDOW
    return _POLYGON_FREE_MAX_CALLS, _POLYGON_WINDOW


def _polygon_rate_limited_call(fn, max_wait: float = _POLYGON_MAX_WAIT):
    """Call fn() respecting the Polygon rate limit, waiting for a free slot if needed.

    Used to hold _polygon_lock across both the wait AND the call to fn() itself,
    fully serializing every Polygon call in the process -- which meant the wait for
    any one call grew unboundedly with how many callers were already queued ahead of
    it (each had to clear its own window first). Under load from get_market_regime
    being called once per candidate, that reliably exceeded the 120s MCP client
    timeout: calls didn't fail, they just hung until the client gave up, and because
    the wait happened synchronously inside the server's single-threaded event loop
    (see regime_server.py), the whole server stopped responding to anything else
    while it slept -- cascading into every other in-flight call timing out too.
    Now only holds the lock long enough to reserve a slot or read the queue depth,
    and raises TimeoutError instead of blocking past max_wait, so a caller with its
    own timeout budget gets a clear, fast failure instead of a silent hang.
    """
    deadline = time.monotonic() + max_wait
    max_calls, window = _polygon_rate_limit()
    while True:
        with _polygon_lock:
            now = time.monotonic()
            while _polygon_call_times and now - _polygon_call_times[0] > window:
                _polygon_call_times.pop(0)
            if len(_polygon_call_times) < max_calls:
                _polygon_call_times.append(now)
                break
            wait = window - (now - _polygon_call_times[0]) + 0.1
        if time.monotonic() + wait > deadline:
            raise TimeoutError(
                f"Polygon rate limit queue is too deep right now -- this call would need "
                f"to wait {wait:.0f}s, exceeding its {max_wait:.0f}s budget. Too many "
                "Polygon requests queued; try again shortly."
            )
        print(f"Polygon rate limit: waiting {wait:.1f}s for a slot...")
        time.sleep(min(wait, max(0.0, deadline - time.monotonic())))
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
