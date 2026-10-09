"""
The ONLY module in this project that sends orders to Schwab: place, read back,
and cancel. Everything else (schwab_client.py, schwab_trading.py) only reads or
previews. Keep it that way, so "what can touch real money" stays one file.

Step 3 of live trading (see CLAUDE.md, "Live trading on Schwab"): used by
schwab_order_smoketest.py to place one deliberately unfillable order and cancel
it. Since step 4, live_trading.py also sends Cathie's opening orders after the
user approves each one, and (step 5) her closing orders automatically when an
exit rule fires.

Every call goes through the same client, token, account selection and rate
limiter as the read-only modules.
"""
import re
import time

from schwab_client import _get_client, _rate_limited_call
from schwab_trading import SchwabAccountError, get_account_hash, mask_account_fields

# Statuses where the order is finished and nothing more will happen to it.
TERMINAL_STATUSES = {"CANCELED", "FILLED", "REJECTED", "EXPIRED", "REPLACED"}

_ORDER_ID_IN_LOCATION = re.compile(r"/orders/(\d+)")


class OrderError(RuntimeError):
    """Schwab refused or failed an order request. The message includes Schwab's reason."""


def _error_text(resp) -> str:
    try:
        return str(mask_account_fields(resp.json()))[:800]
    except Exception:
        return (getattr(resp, "text", "") or "")[:800]


def place_order(order) -> int | None:
    """Send `order` (a schwab-py OrderBuilder or its dict) to Schwab. Returns Schwab's
    order ID, or None if Schwab accepted it but didn't say which ID it got -- the
    caller must then find it (find_recent_order) before it can track or cancel it.
    Raises OrderError if Schwab rejected the request."""
    client = _get_client()
    account_hash, _ = get_account_hash()

    def _call():
        return client.place_order(account_hash, order)

    resp = _rate_limited_call(_call)
    if resp.status_code not in (200, 201):
        raise OrderError(f"Schwab rejected the order (HTTP {resp.status_code}): {_error_text(resp)}")
    m = _ORDER_ID_IN_LOCATION.search(resp.headers.get("Location", "") or "")
    return int(m.group(1)) if m else None


def get_order(order_id: int) -> dict:
    """Schwab's current record of one order (status, filled quantity, legs...),
    account fields masked."""
    client = _get_client()
    account_hash, _ = get_account_hash()

    def _call():
        resp = client.get_order(order_id, account_hash)
        if resp.status_code != 200:
            raise OrderError(f"couldn't read order {order_id} (HTTP {resp.status_code}): {_error_text(resp)}")
        return resp.json()

    return mask_account_fields(_rate_limited_call(_call))


def cancel_order(order_id: int) -> None:
    """Ask Schwab to cancel an order. Cancellation is asynchronous: confirm with
    wait_for_status(order_id, {"CANCELED"}). Raises OrderError if Schwab refuses
    (e.g. the order already filled)."""
    client = _get_client()
    account_hash, _ = get_account_hash()

    def _call():
        return client.cancel_order(order_id, account_hash)

    resp = _rate_limited_call(_call)
    if resp.status_code not in (200, 201, 204):
        raise OrderError(f"Schwab refused to cancel order {order_id} (HTTP {resp.status_code}): {_error_text(resp)}")


def wait_for_status(order_id: int, wanted: set[str], timeout: float = 30.0, poll: float = 2.0) -> dict:
    """Poll an order until its status is in `wanted` or is terminal, or `timeout`
    seconds pass. Returns the last order record seen (check its 'status')."""
    deadline = time.monotonic() + timeout
    while True:
        order = get_order(order_id)
        status = order.get("status")
        if status in wanted or status in TERMINAL_STATUSES or time.monotonic() >= deadline:
            return order
        time.sleep(poll)


def find_recent_order(leg_symbols: set[str], since_minutes: int = 10,
                      closing: bool | None = None, exclude_ids: set | None = None) -> dict | None:
    """The newest order from the last few minutes whose legs are exactly
    `leg_symbols`. Fallback for when place_order() couldn't read the new order's ID,
    so an order we just sent can still be found and cancelled. `closing=True` only
    matches closing orders (every leg ..._TO_CLOSE), False only opening ones: a
    spread's open and close orders have the same legs. `exclude_ids`: orders already
    accounted for (e.g. an earlier close of the same strikes), never returned."""
    import datetime as dt

    client = _get_client()
    account_hash, _ = get_account_hash()
    now = dt.datetime.now(dt.timezone.utc)

    def _call():
        resp = client.get_orders_for_account(
            account_hash,
            from_entered_datetime=now - dt.timedelta(minutes=since_minutes),
            to_entered_datetime=now + dt.timedelta(minutes=1),
        )
        if resp.status_code != 200:
            raise SchwabAccountError(f"couldn't list recent orders (HTTP {resp.status_code})")
        return resp.json() or []

    def _is_close(o):
        return all((leg.get("instruction") or "").endswith("_TO_CLOSE") for leg in o.get("orderLegCollection") or [])

    matches = [
        o for o in _rate_limited_call(_call)
        if {(leg.get("instrument") or {}).get("symbol") for leg in o.get("orderLegCollection") or []} == leg_symbols
        and (closing is None or _is_close(o) == closing)
        and o.get("orderId") not in (exclude_ids or set())
    ]
    matches.sort(key=lambda o: o.get("enteredTime") or "", reverse=True)
    return mask_account_fields(matches[0]) if matches else None
