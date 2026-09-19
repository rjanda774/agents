"""
Thin wrapper around `schwab-py` for real Schwab market data: option chains and
underlying quotes only. This module never places orders -- there is no order-
placement code here at all. Cathie's trades stay simulated: sell_credit_spread/
close_credit_spread (options_trading_server.py) only ever write to the local
`cathie_options` pseudo-account in accounts.db, exactly as before this file
existed. This module just supplies real numbers (bid/ask/open interest/Greeks)
for that simulated ledger to reason about, sourced from Schwab's Market Data
API instead of yfinance.

Why not real paper trading at Schwab? Schwab's developer platform has no
paper-trading sandbox -- unlike the old TD Ameritrade API, every order the
Trader API's endpoints accept posts against your real, funded account. The
old thinkorswim "paperMoney" simulator still exists, but only inside the
desktop/web app; there's no API for it. So "real data, simulated fills" (this
module + the existing local ledger) is the closest available approximation
to paper trading against a real broker feed. See CLAUDE.md for the fuller
writeup of this tradeoff.

Setup (one-time, interactive, must be done on a machine with a real browser --
see schwab_auth_setup.py):
  1. Register an app at https://developer.schwab.com ("Trader API - Individual").
     Approval is not instant.
  2. Set SCHWAB_APP_KEY / SCHWAB_APP_SECRET / SCHWAB_CALLBACK_URL in .env.
  3. Run `uv run schwab_auth_setup.py` once to complete the OAuth login and
     write a token file to SCHWAB_TOKEN_PATH.
  4. Schwab refresh tokens expire after 7 days (a hard platform limit, not a
     bug here) -- re-run step 3 whenever calls start failing/falling back.

Every function in this module is written to fail loudly with a clear
exception rather than fabricate data -- callers in options_trading_server.py
catch these and fall back to yfinance, tagging the result with which source
actually supplied the numbers (see `data_source`/`source_note` in that file).
"""
import os
import threading
import time
import datetime as dt
from dotenv import load_dotenv

load_dotenv(override=True)

SCHWAB_APP_KEY = os.getenv("SCHWAB_APP_KEY")
SCHWAB_APP_SECRET = os.getenv("SCHWAB_APP_SECRET")
SCHWAB_CALLBACK_URL = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
SCHWAB_TOKEN_PATH = os.getenv(
    "SCHWAB_TOKEN_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "schwab_token.json"),
)


class SchwabNotConfiguredError(RuntimeError):
    """SCHWAB_APP_KEY/SECRET aren't set, or the one-time login hasn't been done yet."""


class SchwabAuthError(RuntimeError):
    """A token file exists but couldn't be used -- most likely the 7-day refresh
    token has expired or been revoked. Re-running schwab_auth_setup.py fixes this."""


_client = None
_client_lock = threading.Lock()

# Schwab doesn't publish a single official rate limit the way Polygon does, but a
# per-app cap in the same ballpark as the old TDA API (~120 requests/minute) is the
# commonly-observed figure. Mirrors market.py's Polygon limiter in spirit -- reserve a
# slot before calling, wait for one to free up rather than hammering the API, and give
# up with a clear TimeoutError rather than hanging past the caller's own MCP client
# timeout if the queue gets too deep.
_MAX_CALLS = 100
_WINDOW = 62.0
_MAX_WAIT = 60.0
_rate_lock = threading.Lock()
_call_times: list[float] = []


def _rate_limited_call(fn, max_wait: float = _MAX_WAIT):
    deadline = time.monotonic() + max_wait
    while True:
        with _rate_lock:
            now = time.monotonic()
            while _call_times and now - _call_times[0] > _WINDOW:
                _call_times.pop(0)
            if len(_call_times) < _MAX_CALLS:
                _call_times.append(now)
                break
            wait = _WINDOW - (now - _call_times[0]) + 0.1
        if time.monotonic() + wait > deadline:
            raise TimeoutError(
                f"Schwab API rate-limit queue is too deep right now (would need to wait "
                f"{wait:.0f}s, exceeding the {max_wait:.0f}s budget). Try again shortly."
            )
        time.sleep(min(wait, max(0.0, deadline - time.monotonic())))
    return fn()


def is_configured() -> bool:
    """True if Schwab creds are set AND the one-time login has already produced a
    token file. Deliberately does not import `schwab` or touch the network -- safe
    to call cheaply from every options_trading_server.py tool to decide whether to
    even attempt Schwab before falling back to yfinance."""
    return bool(SCHWAB_APP_KEY and SCHWAB_APP_SECRET and os.path.exists(SCHWAB_TOKEN_PATH))


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        if not (SCHWAB_APP_KEY and SCHWAB_APP_SECRET):
            raise SchwabNotConfiguredError(
                "SCHWAB_APP_KEY/SCHWAB_APP_SECRET are not set in .env."
            )
        if not os.path.exists(SCHWAB_TOKEN_PATH):
            raise SchwabNotConfiguredError(
                f"No Schwab token file at {SCHWAB_TOKEN_PATH} -- run "
                "`uv run schwab_auth_setup.py` once to log in."
            )
        try:
            import schwab.auth
            _client = schwab.auth.client_from_token_file(
                SCHWAB_TOKEN_PATH, SCHWAB_APP_KEY, SCHWAB_APP_SECRET
            )
        except ImportError as e:
            raise SchwabNotConfiguredError(
                "the 'schwab-py' package isn't installed -- run `uv sync` in 6_mcp/."
            ) from e
        except Exception as e:
            raise SchwabAuthError(
                f"Schwab token file exists but couldn't be used ({e}). The refresh "
                "token most likely expired (Schwab refresh tokens are only valid for "
                "7 days) or was revoked -- re-run `uv run schwab_auth_setup.py`."
            ) from e
        return _client


def get_quote(symbol: str) -> float:
    """Real last/mark price for `symbol` from Schwab's Market Data API."""
    client = _get_client()

    def _call():
        resp = client.get_quote(symbol)
        resp.raise_for_status()
        return resp.json()

    data = _rate_limited_call(_call)
    entry = data.get(symbol, {}) if isinstance(data, dict) else {}
    quote = entry.get("quote", {})
    price = quote.get("lastPrice") or quote.get("mark") or quote.get("closePrice")
    if price is None:
        raise ValueError(f"Schwab quote for {symbol} had no usable price field: {entry}")
    return float(price)


def get_option_chain(symbol: str, from_date: dt.date, to_date: dt.date) -> dict:
    """Real option chain from Schwab's Market Data API for expirations in
    [from_date, to_date] (inclusive), with real bid/ask/open interest/IV and
    real broker-computed Greeks per contract -- Schwab's chain endpoint returns
    delta/gamma/theta/vega directly, unlike yfinance's raw chain which has none.

    Returns:
        {
            "symbol": ...,
            "current_price": float,
            "expirations": {
                "YYYY-MM-DD": {
                    "days_to_expiration": int,
                    "puts":  [ {strike, bid, ask, lastPrice, volume, openInterest,
                                impliedVolatility, delta, gamma, theta, vega}, ... ],
                    "calls": [ ... same shape ... ],
                }, ...
            }
        }
    Raises on any failure (auth, network, no data) -- callers decide how to fall back.
    """
    client = _get_client()

    def _call():
        resp = client.get_option_chain(
            symbol,
            contract_type=client.Options.ContractType.ALL,
            from_date=from_date,
            to_date=to_date,
            include_underlying_quote=True,
        )
        resp.raise_for_status()
        return resp.json()

    data = _rate_limited_call(_call)

    call_map = data.get("callExpDateMap") or {}
    put_map = data.get("putExpDateMap") or {}
    if not call_map and not put_map:
        raise ValueError(
            f"Schwab option chain for {symbol} returned no contracts "
            f"(status={data.get('status')!r})"
        )

    underlying_price = data.get("underlyingPrice")
    if underlying_price is None:
        underlying = data.get("underlying") or {}
        underlying_price = underlying.get("last") or underlying.get("mark")
    if underlying_price is None:
        raise ValueError(f"Schwab option chain for {symbol} had no underlying price")

    def _extract(date_map: dict) -> dict:
        by_exp: dict[str, list[dict]] = {}
        for exp_key, strikes in date_map.items():
            # Schwab keys each expiration as "YYYY-MM-DD:<days>".
            exp_date_str = exp_key.split(":")[0]
            for contracts in strikes.values():
                for c in contracts:
                    iv_pct = c.get("volatility")
                    by_exp.setdefault(exp_date_str, []).append(
                        {
                            "strike": float(c.get("strikePrice")),
                            "bid": float(c.get("bid") or 0.0),
                            "ask": float(c.get("ask") or 0.0),
                            "lastPrice": float(c.get("last") or 0.0),
                            "volume": c.get("totalVolume"),
                            "openInterest": c.get("openInterest"),
                            # Schwab reports volatility as a percentage number (e.g.
                            # 23.4 meaning 23.4%) -- convert to the plain decimal
                            # fraction (0.234) the rest of this codebase expects.
                            "impliedVolatility": (iv_pct / 100.0) if iv_pct else None,
                            "delta": c.get("delta"),
                            "gamma": c.get("gamma"),
                            "theta": c.get("theta"),
                            "vega": c.get("vega"),
                            "days_to_expiration": c.get("daysToExpiration"),
                        }
                    )
        return by_exp

    puts_by_exp = _extract(put_map)
    calls_by_exp = _extract(call_map)

    expirations = {}
    for exp_date_str in set(puts_by_exp) | set(calls_by_exp):
        puts = sorted(puts_by_exp.get(exp_date_str, []), key=lambda r: r["strike"])
        calls = sorted(calls_by_exp.get(exp_date_str, []), key=lambda r: r["strike"])
        dte = (puts + calls)[0].get("days_to_expiration") if (puts or calls) else None
        expirations[exp_date_str] = {
            "days_to_expiration": dte,
            "puts": puts,
            "calls": calls,
        }

    return {
        "symbol": symbol,
        "current_price": float(underlying_price),
        "expirations": expirations,
    }
