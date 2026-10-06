"""
Read-only access to your real Schwab brokerage account through the Trader API:
balances, option positions, and recent orders. Step 1 of moving Cathie from
simulated fills to real order execution (see CLAUDE.md, "Live trading on Schwab").

THIS MODULE CANNOT PLACE, CHANGE, OR CANCEL ORDERS. It only calls schwab-py's
get_account_numbers / get_account / get_orders_for_account. Order placement is a
later, separate step, gated behind its own settings and your approval of each order.

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
