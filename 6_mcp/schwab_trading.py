"""
Read-only access to your real Schwab brokerage account through the Trader API:
balances, option positions, and recent orders. Step 1 of moving Cathie from
simulated fills to real order execution (see CLAUDE.md, "Live trading on Schwab").

THIS MODULE CANNOT PLACE, CHANGE, OR CANCEL ORDERS. It only calls schwab-py's
get_account_numbers / get_account / get_orders_for_account, plus preview_order
(step 2), which asks Schwab to validate an order without sending it. Order
placement is a later, separate step, gated behind its own settings and your
approval of each order.

Needs the "Accounts and Trading Production" API product on your developer.schwab.com
app, in addition to Market Data. Without it, the calls here fail with HTTP 401/403
while market data (schwab_client.py) keeps working. If you added that product to an
app you'd already logged into, you may need to revoke and log in again (Stop Linking
under Linked Apps and Websites, then `uv run schwab_auth_setup.py`) before the token
covers it.

Uses the same token file, client, and rate limiter as schwab_client.py.

Which account: if your login can see more than one Schwab account, set
SCHWAB_ACCOUNT_NUMBER in .env (the full number, or just its last 4+ digits) so
nothing ever reads from (or later trades in) the wrong one. With exactly one
account it's picked automatically.
"""
import datetime as dt
import os
import threading

from dotenv import load_dotenv

from schwab_client import _get_client, _rate_limited_call

load_dotenv(override=True)

SCHWAB_ACCOUNT_NUMBER = (os.getenv("SCHWAB_ACCOUNT_NUMBER") or "").strip()

_account_hash = None
_account_number = None
_account_lock = threading.Lock()


class SchwabAccountError(RuntimeError):
    """The account to use couldn't be determined, or account data couldn't be read."""


def mask_account_number(number: str) -> str:
    """'12345678' -> '****5678'. Account numbers never get printed or logged in full."""
    number = str(number or "")
    return "****" + number[-4:] if len(number) > 4 else "****"


def _get_json(fn):
    """Run one rate-limited Schwab call and return its JSON body, turning HTTP errors
    into a message that says what most likely went wrong."""
    def _call():
        resp = fn()
        status = getattr(resp, "status_code", None)
        if status in (401, 403):
            raise SchwabAccountError(
                f"Schwab refused account access (HTTP {status}). Most likely the "
                "'Accounts and Trading Production' API product isn't approved on your "
                "developer.schwab.com app yet, or your token predates it: revoke the "
                "app (client.schwab.com -> Security Settings -> Linked Apps and "
                "Websites -> Stop Linking) and re-run `uv run schwab_auth_setup.py`."
            )
        resp.raise_for_status()
        return resp.json()

    return _rate_limited_call(_call)


def get_account_hash() -> tuple[str, str]:
    """(account_hash, account_number) for the account Cathie uses. The hash is what
    Schwab's API calls take; it's cached for the life of the process."""
    global _account_hash, _account_number
    if _account_hash is not None:
        return _account_hash, _account_number
    with _account_lock:
        if _account_hash is not None:
            return _account_hash, _account_number
        client = _get_client()
        rows = _get_json(client.get_account_numbers) or []
        accounts = [
            (str(r.get("accountNumber", "")), r.get("hashValue"))
            for r in rows
            if r.get("hashValue")
        ]
        if not accounts:
            raise SchwabAccountError("Schwab returned no accounts for this login.")

        if SCHWAB_ACCOUNT_NUMBER:
            matches = [a for a in accounts if a[0].endswith(SCHWAB_ACCOUNT_NUMBER)]
            if len(matches) != 1:
                seen = ", ".join(mask_account_number(n) for n, _ in accounts)
                raise SchwabAccountError(
                    f"SCHWAB_ACCOUNT_NUMBER (ending {SCHWAB_ACCOUNT_NUMBER[-4:]}) matches "
                    f"{len(matches)} of this login's accounts ({seen}); it must match "
                    "exactly one. Use more digits, or the full account number."
                )
            number, account_hash = matches[0]
        elif len(accounts) == 1:
            number, account_hash = accounts[0]
        else:
            seen = ", ".join(mask_account_number(n) for n, _ in accounts)
            raise SchwabAccountError(
                f"This login can see {len(accounts)} accounts ({seen}). Set "
                "SCHWAB_ACCOUNT_NUMBER in .env (full number or its last 4+ digits) to "
                "pick the one Cathie uses."
            )
        _account_hash, _account_number = account_hash, number
        return _account_hash, _account_number


def get_account(include_positions: bool = True) -> dict:
    """The raw `securitiesAccount` object: type, balances, and (optionally) positions."""
    client = _get_client()
    account_hash, _ = get_account_hash()
    fields = [client.Account.Fields.POSITIONS] if include_positions else None
    data = _get_json(lambda: client.get_account(account_hash, fields=fields))
    account = (data or {}).get("securitiesAccount")
    if not account:
        raise SchwabAccountError(f"Schwab account response had no securitiesAccount: {list((data or {}).keys())}")
    # aggregatedBalance sits next to securitiesAccount in the response, not inside it;
    # carry it along so balance_fields() sees every balance in one place.
    if data.get("aggregatedBalance") and "aggregatedBalance" not in account:
        account = {**account, "aggregatedBalance": data["aggregatedBalance"]}
    return account


def balance_fields(account: dict) -> dict[str, float]:
    """Every numeric balance Schwab reports, flattened to 'group.field' -> value,
    e.g. 'currentBalances.availableFunds'. Groups: initialBalances, currentBalances,
    projectedBalances, plus the top-level aggregatedBalance when present."""
    out = {}
    for group in ("initialBalances", "currentBalances", "projectedBalances", "aggregatedBalance"):
        for key, value in (account.get(group) or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{group}.{key}"] = float(value)
    return out


# Which API balance fields are schwab.com's "Funds Available for Trading" and "Day Net
# Liquidating Value". Confirmed live 2026-10-07 with schwab_account_check.py --match:
# the site showed $10,638.12 / $17,877.07 and these fields read $10,643.38 / $17,880.72
# moments later (option marks move; every other field was far off). Funds Available also
# equalled availableFundsNonMarginableTrade and buyingPowerNonMarginableTrade that day;
# availableFunds is the one named for it.
FUNDS_AVAILABLE_FIELD = "currentBalances.availableFunds"
NET_LIQ_FIELD = "currentBalances.liquidationValue"

# The per-trade risk cap the user chose for real money (CLAUDE.md, "Live trading on
# Schwab"): max loss <= the smaller of 8% of Day Net Liq and 5x the net premium, and
# always within Funds Available for Trading.
MAX_RISK_PCT_OF_NET_LIQ = 0.08
MAX_RISK_TO_PREMIUM_RATIO = 5.0


def risk_balances(account: dict) -> dict:
    """{'funds_available', 'net_liq'} from a get_account() result. Raises if either is
    missing, rather than letting a risk check run against a guess."""
    fields = balance_fields(account)
    missing = [f for f in (FUNDS_AVAILABLE_FIELD, NET_LIQ_FIELD) if f not in fields]
    if missing:
        raise SchwabAccountError(f"Schwab account data is missing {', '.join(missing)}; can't size risk.")
    return {"funds_available": fields[FUNDS_AVAILABLE_FIELD], "net_liq": fields[NET_LIQ_FIELD]}


def check_spread_risk(max_loss: float, net_premium: float, balances: dict) -> list[str]:
    """The real-money risk rules, as a list of failures (empty means it passes).
    Amounts in dollars for the whole order (all contracts)."""
    problems = []
    if max_loss > balances["funds_available"]:
        problems.append(
            f"max loss ${max_loss:,.2f} exceeds Funds Available for Trading "
            f"${balances['funds_available']:,.2f}"
        )
    pct_cap = balances["net_liq"] * MAX_RISK_PCT_OF_NET_LIQ
    if max_loss > pct_cap:
        problems.append(
            f"max loss ${max_loss:,.2f} exceeds {MAX_RISK_PCT_OF_NET_LIQ:.0%} of Day Net Liq "
            f"(${pct_cap:,.2f} of ${balances['net_liq']:,.2f})"
        )
    premium_cap = net_premium * MAX_RISK_TO_PREMIUM_RATIO
    if max_loss > premium_cap:
        problems.append(
            f"max loss ${max_loss:,.2f} exceeds {MAX_RISK_TO_PREMIUM_RATIO:g}x the net premium "
            f"(${premium_cap:,.2f} on ${net_premium:,.2f})"
        )
    return problems


def build_open_order(spread_type: str, short_symbol: str, long_symbol: str,
                     contracts: int, net_credit: float):
    """A schwab-py OrderBuilder for opening a credit spread as one NET_CREDIT limit
    order (DAY, regular session) -- never a market order. Symbols must be Schwab's own
    contract symbols (from its option chain), not hand-built ones."""
    from schwab.orders.options import bear_call_vertical_open, bull_put_vertical_open

    if contracts < 1:
        raise ValueError("contracts must be at least 1")
    if not (net_credit > 0):
        raise ValueError(f"net credit must be positive, got {net_credit}")
    price = f"{net_credit:.2f}"
    if spread_type == "bull_put":
        return bull_put_vertical_open(long_symbol, short_symbol, contracts, price)
    if spread_type == "bear_call":
        return bear_call_vertical_open(short_symbol, long_symbol, contracts, price)
    raise ValueError(f"spread_type must be 'bull_put' or 'bear_call', got {spread_type!r}")


def mask_account_fields(value):
    """Copy of a Schwab JSON response with every account-identifying value masked,
    safe to print or paste into a chat."""
    if isinstance(value, dict):
        return {
            k: (mask_account_number(v) if "account" in k.lower() and isinstance(v, (str, int)) else mask_account_fields(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [mask_account_fields(v) for v in value]
    return value


def preview_order(order) -> tuple[int, dict | str]:
    """Ask Schwab to validate `order` (an OrderBuilder or its dict) WITHOUT sending it:
    Schwab's previewOrder endpoint. Returns (HTTP status, response body with account
    fields masked). Never raises on an HTTP error status -- a rejected preview is a
    normal, informative result here."""
    client = _get_client()
    account_hash, _ = get_account_hash()

    def _call():
        resp = client.preview_order(account_hash, order)
        try:
            body = resp.json()
        except Exception:
            body = getattr(resp, "text", "") or ""
        return resp.status_code, body

    status, body = _rate_limited_call(_call)
    return status, mask_account_fields(body)


def parse_occ_symbol(symbol: str) -> dict | None:
    """Schwab option symbols use the OCC layout: root padded to 6 chars, YYMMDD,
    C/P, then strike x 1000 as 8 digits -- e.g. 'SPY   261106P00540000'.
    Returns {underlying, expiration, option_type, strike}, or None if it doesn't fit."""
    s = (symbol or "").strip()
    if len(s) < 16:
        return None
    tail = s[-15:]
    root = s[:-15].strip()
    date_part, cp, strike_part = tail[:6], tail[6], tail[7:]
    if cp not in ("C", "P") or not date_part.isdigit() or not strike_part.isdigit() or not root:
        return None
    try:
        expiration = dt.datetime.strptime(date_part, "%y%m%d").date()
    except ValueError:
        return None
    return {
        "underlying": root,
        "expiration": expiration.isoformat(),
        "option_type": "call" if cp == "C" else "put",
        "strike": int(strike_part) / 1000.0,
    }


def option_positions(account: dict) -> list[dict]:
    """The account's option positions, one row per contract. `quantity` is signed:
    negative for short (sold) contracts, positive for long (bought) ones."""
    rows = []
    for p in account.get("positions") or []:
        inst = p.get("instrument") or {}
        if inst.get("assetType") != "OPTION":
            continue
        symbol = inst.get("symbol", "")
        parsed = parse_occ_symbol(symbol) or {}
        quantity = float(p.get("longQuantity") or 0) - float(p.get("shortQuantity") or 0)
        rows.append({
            "symbol": symbol,
            "underlying": inst.get("underlyingSymbol") or parsed.get("underlying"),
            "option_type": (inst.get("putCall") or "").lower() or parsed.get("option_type"),
            "strike": parsed.get("strike"),
            "expiration": parsed.get("expiration"),
            "quantity": quantity,
            "average_price": p.get("averagePrice"),
            "market_value": p.get("marketValue"),
        })
    rows.sort(key=lambda r: (r["underlying"] or "", r["expiration"] or "", r["strike"] or 0))
    return rows


def recent_orders(days: int = 7, max_results: int = 50) -> list[dict]:
    """Orders entered in the last `days` days (Schwab allows up to 60), newest first,
    summarized: id, status, when, type, price, and each leg."""
    client = _get_client()
    account_hash, _ = get_account_hash()
    now = dt.datetime.now(dt.timezone.utc)
    data = _get_json(lambda: client.get_orders_for_account(
        account_hash,
        max_results=max_results,
        from_entered_datetime=now - dt.timedelta(days=days),
        to_entered_datetime=now,
    )) or []
    orders = []
    for o in data:
        orders.append({
            "order_id": o.get("orderId"),
            "status": o.get("status"),
            "entered_time": o.get("enteredTime"),
            "order_type": o.get("orderType"),
            "price": o.get("price"),
            "quantity": o.get("quantity"),
            "filled_quantity": o.get("filledQuantity"),
            "legs": [
                {
                    "instruction": leg.get("instruction"),
                    "symbol": (leg.get("instrument") or {}).get("symbol"),
                    "quantity": leg.get("quantity"),
                }
                for leg in o.get("orderLegCollection") or []
            ],
        })
    orders.sort(key=lambda o: o["entered_time"] or "", reverse=True)
    return orders
