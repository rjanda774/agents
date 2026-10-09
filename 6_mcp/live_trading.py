"""
Step 4 of live trading (see CLAUDE.md, "Live trading on Schwab"): Cathie proposes
real credit spreads, the user approves each one, code tracks the fills.

CATHIE_EXECUTION_MODE (in .env) decides what sell_credit_spread does:
  simulated  (default)  -- unchanged: opens a paper position in cathie_options.
  approve               -- STAGES the trade in the cathie_live ledger instead. Nothing
                           reaches Schwab until the user runs `uv run approve_orders.py`,
                           which re-checks it live and asks y/N before sending.
  live                  -- not built yet; refused.
Anything else is refused too, so a typo can't silently pick a mode.

Real trades live in their own ledger, the `cathie_live` row in accounts.db, never
mixed with Cathie's simulated history in `cathie_options`.

Hard limits on real orders, all enforced here in code (not by prompt):
  - 1 contract per order; at most 2 orders sent per trading day (ET); at most 3 trades
    waiting for approval at once.
  - Only during the regular session (Mon-Fri 9:30-16:00 ET).
  - Live re-checks at approval: 25-45 DTE, short delta < 0.20, $50+ premium, $100+ stock
    price and no earnings report in the last 5 trading days (named ETFs exempt), quotes on
    both legs, natural no more than 10% of the spread's width below mid, max loss within Funds Available and
    the 8%-of-Day-Net-Liq / 5x-premium caps, and no underlying the user already holds
    options on (their own positions are never touched or merged with Cathie's).
  - A file named STOP_TRADING in this folder blocks all staging and sending.
Unfilled orders are cancelled once they've been working 30+ minutes, at the next
reconcile (each trading cycle, or whenever approve_orders.py runs).

Exits (step 5) are automatic, run by code each cycle (check_exits), never by Cathie:
  - 7 days or less to expiration, or the stock past the short strike: close at the
    natural price (pay the ask on the short leg, take the bid on the long) so it fills.
  - 75%+ of the credit captured (cost to close <= 25% of it): close at the mid.
  Only during the regular session; previewed first; a close left working 30+ minutes is
  cancelled at the next reconcile and re-sent at the then-current price. STOP_TRADING
  stops closes too. The user's own positions are never touched.
"""
import datetime as dt
import json
import math
import os
import sqlite3
import uuid
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

LEDGER_NAME = "cathie_live"
STOP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STOP_TRADING")

MAX_CONTRACTS_PER_ORDER = 1
MAX_ORDERS_PER_DAY = 2
MAX_STAGED = 3
UNFILLED_CANCEL_MINUTES = 30
MIN_NET_PREMIUM = 50.0
MAX_SHORT_DELTA = 0.20
MIN_DTE, MAX_DTE = 25, 45
# Exits (step 5, user's choice 2026-10-09: automatic, same rules as paper trades).
PROFIT_TAKE_FRACTION = 0.25   # close when the cost to close is <= 25% of the credit
CLOSE_AT_DTE = 7              # close at 7 days or less to expiration
MISSING_CONFIRM_MINUTES = 30  # legs gone on two checks this far apart = closed outside Cathie
# Schwab's refresh token lasts 7 days from the login; stop new trades in its last 12h.
TOKEN_LIFETIME_DAYS = 7.0
TOKEN_HALT_OPENS_DAYS = 6.5
TOKEN_WARN_DAYS = 6.0

ACTIVE_STATUSES = {"staged", "pending", "open", "closing"}


def execution_mode() -> str:
    """Current CATHIE_EXECUTION_MODE, re-read from .env on every call so an edit takes
    effect on the next cycle without restarting anything."""
    load_dotenv(override=True)
    return (os.getenv("CATHIE_EXECUTION_MODE") or "simulated").strip().lower()


def _now_et() -> dt.datetime:
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo("America/New_York"))


def _stamp() -> str:
    return _now_et().strftime("%Y-%m-%d %H:%M ET")


def _parse_et(text: str | None) -> dt.datetime | None:
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def regular_session_open() -> bool:
    """Mon-Fri 9:30-16:00 ET by the clock. Doesn't know holidays or early closes; on a
    holiday Schwab simply won't fill, and the order expires or gets cancelled."""
    now = _now_et()
    return now.weekday() < 5 and dt.time(9, 30) <= now.time() < dt.time(16, 0)


def kill_switch_on() -> bool:
    return os.path.exists(STOP_FILE)


class LiveSpread(BaseModel):
    id: str
    symbol: str
    spread_type: Literal["bull_put", "bear_call"]
    short_strike: float
    long_strike: float
    expiration_date: str
    contracts: int
    # staged -> pending (sent to Schwab) -> open (filled)
    # staged -> rejected (by the user) / expired (not approved the same day)
    # pending -> not_filled (cancelled, rejected or expired at Schwab)
    # open -> closing (close order sent) -> closed; closing -> open if the close didn't fill
    status: Literal["staged", "rejected", "expired", "pending", "open", "not_filled", "closing", "closed"] = "staged"
    staged_at: str
    staged_credit: float | None = None  # Cathie's net credit per spread when she proposed it
    rationale: str = ""
    short_symbol: str | None = None
    long_symbol: str | None = None
    order_id: int | None = None
    limit_credit: float | None = None
    submitted_at: str | None = None
    fill_credit: float | None = None  # actual net credit per spread
    filled_at: str | None = None
    net_premium: float | None = None
    max_loss: float | None = None
    # Closing (step 5)
    close_rule: str | None = None          # profit / breach / dte / expired / outside
    close_order_id: int | None = None
    close_limit_debit: float | None = None
    close_submitted_at: str | None = None
    close_fill_debit: float | None = None  # actual net debit per spread paid to close
    closed_at: str | None = None
    realized_pnl: float | None = None
    past_close_order_ids: list[int] = []   # earlier close attempts (cancelled/rejected)
    legs_missing_since: str | None = None  # first check that found its legs gone at Schwab
    events: list[str] = []

    def note(self, text: str):
        self.events.append(f"{_stamp()} {text}")

    @property
    def width(self) -> float:
        return abs(self.short_strike - self.long_strike)

    def label(self) -> str:
        return (f"{self.symbol} {self.spread_type} {self.short_strike:g}/{self.long_strike:g} "
                f"exp {self.expiration_date} x{self.contracts}")


class LiveLedger(BaseModel):
    spreads: list[LiveSpread] = []
    token_warned_on: str | None = None  # ET date of the last "Schwab login expiring" push

    def get(self, spread_id: str) -> LiveSpread | None:
        return next((s for s in self.spreads if s.id == spread_id), None)


def load_ledger() -> LiveLedger:
    from database import read_account
    data = read_account(LEDGER_NAME)
    return LiveLedger(**data) if data else LiveLedger()


def _mutate(fn):
    """Apply fn(ledger) inside one SQLite write transaction and save, so the trading
    floor and approve_orders.py can't overwrite each other's changes. fn must only
    change the ledger -- never make network calls while the database is locked."""
    from database import DB
    with sqlite3.connect(DB, timeout=30) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT account FROM accounts WHERE name = ?", (LEDGER_NAME,)).fetchone()
        ledger = LiveLedger(**json.loads(row[0])) if row else LiveLedger()
        result = fn(ledger)
        conn.execute(
            "INSERT INTO accounts (name, account) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET account=excluded.account",
            (LEDGER_NAME, ledger.model_dump_json()),
        )
    return result


def _log(message: str):
    try:
        from database import write_log
        write_log("cathie", "live", message)
    except Exception:
        pass


def _push(message: str):
    """Best-effort Pushover notification (same credentials as push_server.py)."""
    try:
        import requests
        user, token = os.getenv("PUSHOVER_USER"), os.getenv("PUSHOVER_TOKEN")
        if user and token:
            requests.post("https://api.pushover.net/1/messages.json",
                          data={"user": user, "token": token, "message": message}, timeout=10)
    except Exception:
        pass


def orders_sent_today(ledger: LiveLedger) -> int:
    today = _now_et().date()
    return sum(1 for s in ledger.spreads
               if (t := _parse_et(s.submitted_at)) is not None and t.date() == today)


def new_trades_blocked() -> str | None:
    """Why no new real trade can be staged right now (approve mode only), or None.
    Checked before Cathie's new-trade pass so she doesn't spend a whole pass picking
    trades stage() is certain to refuse -- seen live 2026-10-08: after 2 orders were sent,
    she "staged" two more that the daily limit had rejected, and reported them as staged."""
    if execution_mode() != "approve":
        return None
    if kill_switch_on():
        return "real trading is stopped (STOP_TRADING file present)"
    try:
        import schwab_client
        age = schwab_client.token_age_days()
    except Exception:
        age = None
    if age is not None and age >= TOKEN_HALT_OPENS_DAYS:
        return (f"the Schwab login is {age:.1f} days old and expires at {TOKEN_LIFETIME_DAYS:g}; "
                "run `uv run schwab_auth_setup.py` to renew it")
    ledger = load_ledger()
    sent = orders_sent_today(ledger)
    if sent >= MAX_ORDERS_PER_DAY:
        return f"the daily limit of {MAX_ORDERS_PER_DAY} real orders is used up ({sent} sent today)"
    staged = sum(1 for s in ledger.spreads if s.status == "staged")
    if staged >= MAX_STAGED:
        return f"{staged} trades are already waiting for approval (limit {MAX_STAGED})"
    return None


def users_own_underlyings(positions: list[dict], ledger: LiveLedger) -> set[str]:
    """Underlyings the user holds options on themselves: Schwab option positions whose
    contract symbols aren't legs of Cathie's own live spreads."""
    cathies = {sym for s in ledger.spreads if s.status in ("pending", "open", "closing")
               for sym in (s.short_symbol, s.long_symbol) if sym}
    return {p["underlying"] for p in positions if p.get("symbol") not in cathies and p.get("underlying")}


def real_account_snapshot() -> dict:
    """Live Schwab balances plus Cathie's active real trades, for her account summary in
    approve mode. Added 2026-10-09: her end-of-cycle push reported the PAPER account's cash
    ($13,461.50) as her balance, since that was the only balance she was shown."""
    ledger = load_ledger()
    snap = {
        "real_trades": [f"{s.status}: {s.label()}" for s in ledger.spreads if s.status in ACTIVE_STATUSES],
        "orders_sent_today": orders_sent_today(ledger),
        "daily_limit": MAX_ORDERS_PER_DAY,
    }
    try:
        import schwab_trading
        balances = schwab_trading.risk_balances(schwab_trading.get_account())
        snap["funds_available_for_trading"] = round(balances["funds_available"], 2)
        snap["day_net_liquidating_value"] = round(balances["net_liq"], 2)
    except Exception as e:
        snap["balances_error"] = f"couldn't read Schwab balances right now: {e}"
    return snap


def overlapping_spread(ledger: LiveLedger, symbol: str, spread_type: str, expiration_date: str,
                       short_strike: float, long_strike: float, exclude_id: str | None = None):
    """Cathie's own staged/pending/open spread sharing a contract (same underlying,
    expiration, put/call and strike) with this one, or None. Two such spreads would merge
    at Schwab: on 2026-10-09 she staged MU 910/900 while holding MU 900/890, whose short
    900 put the new spread's long 900 put would cancel out."""
    for s in ledger.spreads:
        if s.id == exclude_id or s.status not in ACTIVE_STATUSES:
            continue
        if (s.symbol, s.spread_type, s.expiration_date) == (symbol, spread_type, expiration_date) and \
                {s.short_strike, s.long_strike} & {short_strike, long_strike}:
            return s
    return None


# ---------------------------------------------------------------- staging (Cathie)

def stage(symbol: str, spread_type: str, short_strike: float, long_strike: float,
          expiration_date: str, contracts: int, staged_credit: float | None,
          max_loss: float | None, rationale: str) -> dict:
    """Called by sell_credit_spread in approve mode, after all its own rules passed.
    Records the trade for the user's approval; sends nothing. Returns the tool response."""
    mode = execution_mode()
    if mode != "approve":
        return {"error": f"Real trading isn't in approve mode (CATHIE_EXECUTION_MODE={mode!r})."}
    if kill_switch_on():
        return {"error": "TRADE REJECTED: real trading is stopped (STOP_TRADING file present). Nothing was staged."}
    if contracts > MAX_CONTRACTS_PER_ORDER:
        return {"error": f"TRADE REJECTED: real orders are limited to {MAX_CONTRACTS_PER_ORDER} contract. "
                         f"Retry with contracts={MAX_CONTRACTS_PER_ORDER}."}
    symbol = symbol.upper()

    # Best-effort live checks now, so Cathie hears about obvious problems this cycle.
    # Everything is checked again, authoritatively, at approval time.
    warnings = []
    try:
        import schwab_client
        import schwab_trading
        if schwab_client.is_configured():
            account = schwab_trading.get_account(include_positions=True)
            own = users_own_underlyings(schwab_trading.option_positions(account), load_ledger())
            if symbol in own:
                return {"error": f"TRADE REJECTED: the account already holds the user's own {symbol} "
                                 "options, and real trades skip those underlyings. Pick a different one."}
            if max_loss is not None and staged_credit is not None:
                problems = schwab_trading.check_spread_risk(
                    max_loss, staged_credit * 100 * contracts, schwab_trading.risk_balances(account))
                if problems:
                    return {"error": "TRADE REJECTED: " + "; ".join(problems)}
    except Exception as e:
        warnings.append(f"couldn't check against the Schwab account right now ({e}); "
                        "it will be checked when the user approves")

    def _add(ledger: LiveLedger):
        active = [s for s in ledger.spreads if s.status in ACTIVE_STATUSES]
        for s in active:
            if (s.symbol, s.spread_type, s.short_strike, s.long_strike, s.expiration_date) == \
                    (symbol, spread_type, short_strike, long_strike, expiration_date):
                return {"error": f"Already {s.status}: {s.label()} (id {s.id}). Not staged again."}
        clash = overlapping_spread(ledger, symbol, spread_type, expiration_date, short_strike, long_strike)
        if clash:
            return {"error": f"TRADE REJECTED: it shares a strike with Cathie's own {clash.status} spread "
                             f"{clash.label()}, and the two would merge into one position at Schwab. "
                             "Pick strikes that don't touch it, or a different underlying."}
        if sum(1 for s in active if s.status == "staged") >= MAX_STAGED:
            return {"error": f"TRADE REJECTED: {MAX_STAGED} trades are already waiting for the user's "
                             "approval. Don't propose more until those are handled."}
        if orders_sent_today(ledger) >= MAX_ORDERS_PER_DAY:
            return {"error": f"TRADE REJECTED: the daily limit of {MAX_ORDERS_PER_DAY} real orders "
                             "has been reached. Try again next trading day."}
        spread = LiveSpread(
            id=uuid.uuid4().hex[:8], symbol=symbol, spread_type=spread_type,
            short_strike=short_strike, long_strike=long_strike, expiration_date=expiration_date,
            contracts=contracts, staged_at=_now_et().isoformat(), staged_credit=staged_credit,
            rationale=rationale[:1000],
        )
        spread.note(f"staged by Cathie at ~{staged_credit:.2f} credit" if staged_credit else "staged by Cathie")
        ledger.spreads.append(spread)
        return {"id": spread.id, "label": spread.label()}

    added = _mutate(_add)
    if "error" in added:
        return added
    _log(f"Staged for approval: {added['label']} (id {added['id']})")
    _push(f"Cathie staged a REAL trade for your approval: {added['label']}. "
          f"Run `uv run approve_orders.py` today to review it.")
    result = {
        "status": "STAGED FOR APPROVAL",
        "staged_id": added["id"],
        "spread": added["label"],
        "message": ("This is NOT an open position and no order was sent. The trade is waiting for the "
                    "user to approve it by hand; it may be rejected, or re-priced at approval. Report it "
                    "as 'staged for approval', never as opened or filled."),
    }
    if warnings:
        result["warnings"] = warnings
    return result


# ---------------------------------------------------------------- approval (user)

def check_for_approval(spread: LiveSpread) -> dict:
    """Every live check for sending `spread` now. Returns a plan:
    {ok, problems[], lines[] (for display), order, credit, short_symbol, long_symbol,
     net_premium, max_loss}. Never sends anything."""
    import schwab_client
    import schwab_trading

    problems, lines = [], []
    plan = {"ok": False, "problems": problems, "lines": lines}

    if kill_switch_on():
        problems.append("STOP_TRADING file is present")
    if not regular_session_open():
        problems.append(f"market is closed ({_stamp()}); approve during 9:30-16:00 ET")
    exp = dt.date.fromisoformat(spread.expiration_date)
    dte = (exp - _now_et().date()).days
    if not MIN_DTE <= dte <= MAX_DTE:
        problems.append(f"expiration is {dte} days out; must be {MIN_DTE}-{MAX_DTE}")
    if spread.contracts > MAX_CONTRACTS_PER_ORDER:
        problems.append(f"{spread.contracts} contracts; real orders are limited to {MAX_CONTRACTS_PER_ORDER}")
    ledger = load_ledger()
    if orders_sent_today(ledger) >= MAX_ORDERS_PER_DAY:
        problems.append(f"daily limit of {MAX_ORDERS_PER_DAY} real orders already reached")

    chain = schwab_client.get_option_chain(spread.symbol, exp, exp)
    rows = (chain["expirations"].get(spread.expiration_date) or {})
    rows = rows.get("puts" if spread.spread_type == "bull_put" else "calls") or []
    find = lambda k: next((r for r in rows if abs(r["strike"] - k) < 1e-6), None)
    short, long_ = find(spread.short_strike), find(spread.long_strike)
    if not (short and long_ and short.get("symbol") and long_.get("symbol")):
        problems.append("one or both strikes aren't in Schwab's chain for that expiration")
        return plan
    # The order goes out under these symbols, so confirm each one really is the contract
    # meant: right underlying, expiration, put/call and strike.
    want_type = "put" if spread.spread_type == "bull_put" else "call"
    for row, strike in ((short, spread.short_strike), (long_, spread.long_strike)):
        p = schwab_trading.parse_occ_symbol(row["symbol"]) or {}
        if (p.get("underlying"), p.get("expiration"), p.get("option_type"), p.get("strike")) != \
                (spread.symbol, spread.expiration_date, want_type, strike):
            problems.append(f"contract symbol {row['symbol']!r} doesn't match the trade; not sending")
            return plan
    lines.append(f"underlying ${chain['current_price']:,.2f}, {dte} DTE")
    lines.append(f"short {short['symbol']}  bid {short['bid']:.2f} ask {short['ask']:.2f}  delta {short.get('delta')}")
    lines.append(f"long  {long_['symbol']}  bid {long_['bid']:.2f} ask {long_['ask']:.2f}  delta {long_.get('delta')}")

    # trade_rules.py: $100+ for individual stocks, and no trade within 5 trading days after
    # an earnings report (named ETFs exempt from both). Real money: unknown report date = no.
    import trade_rules
    from universe import CATHIE_ETF_UNIVERSE
    floor = trade_rules.price_floor_problem(spread.symbol, chain["current_price"])
    if floor:
        problems.append(floor)
    if spread.symbol not in CATHIE_ETF_UNIVERSE:
        last = trade_rules.get_last_earnings_date(spread.symbol)
        recent = trade_rules.recent_earnings_problem(spread.symbol, last)
        if recent:
            problems.append(recent)
        elif last is None:
            problems.append(f"can't verify when {spread.symbol} last reported earnings")

    quotes = [short["bid"], short["ask"], long_["bid"], long_["ask"]]
    if not all(math.isfinite(q) for q in quotes) or min(quotes) <= 0:
        problems.append("missing bid or ask on a leg; can't price it")
        return plan
    mid = (short["bid"] + short["ask"]) / 2 - (long_["bid"] + long_["ask"]) / 2
    natural = short["bid"] - long_["ask"]
    lines.append(f"credit per spread: mid {mid:.2f}, natural {natural:.2f}")
    if mid <= 0:
        problems.append("no credit at the mid")
        return plan
    wide = trade_rules.wide_market_problem(mid, natural, abs(spread.short_strike - spread.long_strike))
    if wide:
        problems.append(wide)
    delta = short.get("delta")
    if delta is None:
        problems.append("Schwab has no delta for the short leg")
    elif abs(delta) > MAX_SHORT_DELTA:
        problems.append(f"short delta {delta:.3f} is beyond {MAX_SHORT_DELTA}")

    credit = round(mid, 2)
    net_premium = round(credit * 100 * spread.contracts, 2)
    max_loss = round((spread.width - credit) * 100 * spread.contracts, 2)
    lines.append(f"limit credit {credit:.2f} -> premium ${net_premium:,.2f}, max loss ${max_loss:,.2f}"
                 + (f" (Cathie proposed ~{spread.staged_credit:.2f})" if spread.staged_credit else ""))
    if net_premium < MIN_NET_PREMIUM:
        problems.append(f"premium ${net_premium:.2f} is under the ${MIN_NET_PREMIUM:.0f} minimum")

    account = schwab_trading.get_account(include_positions=True)
    balances = schwab_trading.risk_balances(account)
    lines.append(f"Funds Available ${balances['funds_available']:,.2f}, Day Net Liq ${balances['net_liq']:,.2f}")
    problems.extend(schwab_trading.check_spread_risk(max_loss, net_premium, balances))
    if spread.symbol in users_own_underlyings(schwab_trading.option_positions(account), ledger):
        problems.append(f"you already hold your own {spread.symbol} options; real trades skip those underlyings")
    clash = overlapping_spread(ledger, spread.symbol, spread.spread_type, spread.expiration_date,
                               spread.short_strike, spread.long_strike, exclude_id=spread.id)
    if clash and clash.status != "staged":
        problems.append(f"shares a strike with Cathie's {clash.status} spread {clash.label()}; "
                        "they'd merge into one position at Schwab")

    order = schwab_trading.build_open_order(
        spread.spread_type, short["symbol"], long_["symbol"], spread.contracts, credit).build()
    if not problems:
        status, body = schwab_trading.preview_order(order)
        rejects = ((body or {}).get("orderValidationResult") or {}).get("rejects") if isinstance(body, dict) else None
        lines.append(f"Schwab preview: HTTP {status}" + (f", rejects: {rejects}" if rejects else ", accepted"))
        if status >= 400 or rejects:
            problems.append("Schwab's preview rejected the order")

    plan.update(ok=not problems, order=order, credit=credit, short_symbol=short["symbol"],
                long_symbol=long_["symbol"], net_premium=net_premium, max_loss=max_loss)
    return plan


def reject(spread_id: str, why: str = "rejected by the user"):
    def _do(ledger: LiveLedger):
        s = ledger.get(spread_id)
        if s and s.status == "staged":
            s.status = "rejected"
            s.note(why)
    _mutate(_do)
    _log(f"Rejected staged trade {spread_id}: {why}")


def submit(spread_id: str, plan: dict) -> int:
    """Send an approved plan to Schwab. The spread is marked pending (and counted
    toward the daily limit) BEFORE the order goes out, so a crash mid-send leaves a
    record that reconcile() can match to the real order. Returns Schwab's order ID."""
    import schwab_execution

    def _mark(ledger: LiveLedger):
        s = ledger.get(spread_id)
        if s is None or s.status != "staged":
            raise RuntimeError(f"staged trade {spread_id} is no longer waiting for approval")
        if orders_sent_today(ledger) >= MAX_ORDERS_PER_DAY:
            raise RuntimeError(f"daily limit of {MAX_ORDERS_PER_DAY} real orders already reached")
        s.status = "pending"
        s.submitted_at = _now_et().isoformat()
        s.limit_credit = plan["credit"]
        s.short_symbol, s.long_symbol = plan["short_symbol"], plan["long_symbol"]
        s.net_premium, s.max_loss = plan["net_premium"], plan["max_loss"]
        s.note(f"approved; sending at {plan['credit']:.2f} credit")
        return s.label()

    label = _mutate(_mark)
    try:
        order_id = schwab_execution.place_order(plan["order"])
    except Exception as e:
        def _fail(ledger: LiveLedger):
            s = ledger.get(spread_id)
            s.status = "not_filled"
            s.note(f"Schwab didn't accept the order: {e}")
        _mutate(_fail)
        _log(f"Order for {label} NOT placed: {e}")
        raise
    if order_id is None:
        found = schwab_execution.find_recent_order({plan["short_symbol"], plan["long_symbol"]})
        order_id = found.get("orderId") if found else None

    def _record(ledger: LiveLedger):
        s = ledger.get(spread_id)
        s.order_id = order_id
        s.note(f"sent; Schwab order {order_id}" if order_id else "sent; Schwab order ID not yet known")
    _mutate(_record)
    _log(f"Real order sent for {label} at {plan['credit']:.2f} credit, Schwab order {order_id}")
    return order_id


# ---------------------------------------------------------------- fills (code)

def fill_net_credit(order: dict) -> float | None:
    """Actual net credit per spread from an order's executions: sell-leg proceeds minus
    buy-leg cost, divided by spreads filled. None if Schwab didn't report executions."""
    instructions = {leg.get("legId"): leg.get("instruction") for leg in order.get("orderLegCollection") or []}
    total, filled = 0.0, 0.0
    for activity in order.get("orderActivityCollection") or []:
        if activity.get("activityType") != "EXECUTION":
            continue
        filled += float(activity.get("quantity") or 0)
        for leg in activity.get("executionLegs") or []:
            amount = float(leg.get("price") or 0) * float(leg.get("quantity") or 0)
            instr = instructions.get(leg.get("legId")) or ""
            total += amount if instr.startswith("SELL") else -amount
    return round(total / filled, 4) if filled else None


def reconcile() -> list[str]:
    """Bring the ledger up to date with Schwab: record fills at their real price, mark
    cancelled/rejected/expired orders, cancel orders working 30+ minutes, expire staged
    trades not approved the same day. Returns what changed, for logging."""
    import schwab_execution

    messages = []
    ledger = load_ledger()
    now = _now_et()

    stale = [s.id for s in ledger.spreads
             if s.status == "staged" and (t := _parse_et(s.staged_at)) and t.date() < now.date()]
    if stale:
        def _expire(lg: LiveLedger):
            for sid in stale:
                s = lg.get(sid)
                if s and s.status == "staged":
                    s.status = "expired"
                    s.note("not approved the day it was staged")
        _mutate(_expire)
        messages.append(f"{len(stale)} staged trade(s) expired unapproved")

    for s in [s for s in ledger.spreads if s.status == "pending"]:
        order_id, record, update = s.order_id, None, {}
        try:
            if order_id is None:
                found = schwab_execution.find_recent_order({s.short_symbol, s.long_symbol}, since_minutes=24 * 60)
                if found:
                    order_id = found.get("orderId")
                    update["order_id"] = order_id
                elif (t := _parse_et(s.submitted_at)) and (now - t).total_seconds() > 600:
                    update.update(status="not_filled", note="no matching order found at Schwab; check thinkorswim")
            if order_id is not None:
                record = schwab_execution.get_order(order_id)
                status = record.get("status")
                age_min = ((now - t).total_seconds() / 60) if (t := _parse_et(s.submitted_at)) else 0
                if (status not in ("FILLED", "CANCELED", "REJECTED", "EXPIRED", "PENDING_CANCEL")
                        and age_min >= UNFILLED_CANCEL_MINUTES):
                    schwab_execution.cancel_order(order_id)
                    record = schwab_execution.wait_for_status(order_id, {"CANCELED"}, timeout=30)
                    status = record.get("status")
                    update["note"] = f"unfilled after {age_min:.0f} min; cancel requested (Schwab order {order_id})"
                if status == "FILLED":
                    credit = fill_net_credit(record) or s.limit_credit
                    update.update(status="open", fill_credit=credit, filled_at=now.isoformat(),
                                  net_premium=round(credit * 100 * s.contracts, 2),
                                  max_loss=round((s.width - credit) * 100 * s.contracts, 2),
                                  note=f"FILLED at {credit:.2f} credit (Schwab order {order_id})")
                elif status in ("CANCELED", "REJECTED", "EXPIRED"):
                    reason = record.get("statusDescription") or ""
                    why = (f"unfilled after {age_min:.0f} min, so cancelled" if "note" in update
                           else status + (f": {reason}" if reason else ""))
                    update.update(status="not_filled", note=f"Schwab order {order_id} {why}")
        except Exception as e:
            messages.append(f"{s.label()}: couldn't check with Schwab ({e})")
            continue
        if not update:
            continue

        def _apply(lg: LiveLedger, sid=s.id, upd=update):
            x = lg.get(sid)
            if x is None or x.status != "pending":
                return
            note = upd.get("note")
            for k, v in upd.items():
                if k != "note":
                    setattr(x, k, v)
            if note:
                x.note(note)
        _mutate(_apply)
        if update.get("status") or update.get("note"):
            msg = f"{s.label()}: {update.get('note') or update.get('status')}"
            messages.append(msg)
            _log(msg)
            if update.get("status") == "open":
                _push(f"Cathie's REAL trade filled: {msg}. Exits are automatic (75% profit, breach, 7 DTE).")

    for s in [s for s in load_ledger().spreads if s.status == "closing"]:
        try:
            msg = _reconcile_closing(s, now)
        except Exception as e:
            msg = f"{s.label()}: couldn't check its close order with Schwab ({e})"
        if msg:
            messages.append(msg)

    warning = _token_warning()
    if warning:
        messages.append(warning)
    return messages


# ---------------------------------------------------------------- exits (code, step 5)

def _cents_up(x: float) -> float:
    return math.ceil(round(x * 100, 6)) / 100


def exit_verdict(spread: LiveSpread, chain: dict, today: dt.date) -> dict:
    """Which exit rule (if any) an open real spread meets, from Schwab's chain, and the
    limit debit to close it at. Pure: no network, no ledger. Keys: rule (None, 'dte',
    'breach' or 'profit'), debit, mid, natural, price, dte, lines, problem."""
    exp = dt.date.fromisoformat(spread.expiration_date)
    v = {"rule": None, "debit": None, "mid": None, "natural": None, "price": None,
         "dte": (exp - today).days, "lines": [], "problem": None}
    rows = (chain.get("expirations") or {}).get(spread.expiration_date) or {}
    rows = rows.get("puts" if spread.spread_type == "bull_put" else "calls") or []
    find = lambda sym, k: (next((r for r in rows if r.get("symbol") == sym), None)
                           or next((r for r in rows if abs(r["strike"] - k) < 1e-6), None))
    short, long_ = find(spread.short_symbol, spread.short_strike), find(spread.long_symbol, spread.long_strike)
    price = chain.get("current_price")
    v["price"] = price
    if not (short and long_):
        v["problem"] = "its strikes aren't in Schwab's chain right now"
        return v
    quotes = [short.get("bid"), short.get("ask"), long_.get("bid"), long_.get("ask")]
    if not all(isinstance(q, (int, float)) and math.isfinite(q) and q >= 0 for q in quotes) or not quotes[1] > 0:
        v["problem"] = "no usable quote on its legs right now"
        return v
    sb, sa, lb, la = quotes
    mid = (sb + sa) / 2 - (lb + la) / 2
    natural = sa - lb
    v["mid"], v["natural"] = round(mid, 4), round(natural, 4)
    credit = spread.fill_credit or spread.limit_credit or 0
    v["lines"].append(f"stock ${price:,.2f}, {v['dte']} DTE, cost to close: mid {mid:.2f}, natural {natural:.2f} "
                      f"(opened for {credit:.2f})" if isinstance(price, (int, float))
                      else f"{v['dte']} DTE, cost to close: mid {mid:.2f}, natural {natural:.2f} (opened for {credit:.2f})")

    breached = isinstance(price, (int, float)) and math.isfinite(price) and (
        price < spread.short_strike if spread.spread_type == "bull_put" else price > spread.short_strike)
    if v["dte"] <= CLOSE_AT_DTE or breached:
        v["rule"] = "dte" if v["dte"] <= CLOSE_AT_DTE else "breach"
        v["debit"] = max(0.01, min(_cents_up(natural), spread.width))
    elif credit > 0 and mid <= PROFIT_TAKE_FRACTION * credit:
        v["rule"] = "profit"
        v["debit"] = max(0.01, round(max(mid, 0.0), 2))
    return v


RULE_TEXT = {
    "dte": f"{CLOSE_AT_DTE} days or less to expiration",
    "breach": "the stock is past the short strike",
    "profit": f"{100 - PROFIT_TAKE_FRACTION * 100:.0f}%+ of the credit captured",
    "expired": "expired",
    "outside": "no longer in the Schwab account (closed outside Cathie?)",
}


def check_exits(send: bool = True) -> list[str]:
    """Check every open real spread against the exit rules and, with send=True, close
    the ones that meet one (regular session only, not with STOP_TRADING). send=False is
    a dry run that only reports (approve_orders.py --exits). Returns what it found/did."""
    if execution_mode() != "approve":
        return []
    ledger = load_ledger()
    open_ = [s for s in ledger.spreads if s.status == "open"]
    if not open_:
        return []
    import schwab_client
    import schwab_trading

    messages = []
    now = _now_et()
    held = {p["symbol"]: p["quantity"] for p in
            schwab_trading.option_positions(schwab_trading.get_account(include_positions=True))}
    for s in open_:
        exp = dt.date.fromisoformat(s.expiration_date)
        short_q, long_q = held.get(s.short_symbol, 0), held.get(s.long_symbol, 0)
        if short_q >= 0 and long_q <= 0:
            # Neither leg is held any more: expired, or closed outside Cathie (by hand).
            # Before expiration that needs care: a Schwab reply missing its positions list
            # would look exactly like this. So never when Schwab shows no option positions
            # at all, and only once the legs have been gone on two checks 30+ min apart.
            rule = "expired" if exp < now.date() else "outside"
            if rule == "outside":
                first = _parse_et(s.legs_missing_since)
                if not held:
                    messages.append(f"{s.label()}: Schwab reported no option positions at all; "
                                    "not changing anything (check thinkorswim if that's wrong)")
                    continue
                if first is None or (now - first).total_seconds() < MISSING_CONFIRM_MINUTES * 60:
                    if send and first is None:
                        _set_missing(s.id, now.isoformat())
                    messages.append(f"{s.label()}: its legs aren't in the Schwab account right now; "
                                    f"if they're still gone in {MISSING_CONFIRM_MINUTES} min it's recorded as "
                                    "closed outside Cathie")
                    continue
            if send:
                _mark_closed_elsewhere(s.id, rule)
            messages.append(f"{s.label()}: {RULE_TEXT[rule]}; recorded as closed" if send
                            else f"{s.label()}: {RULE_TEXT[rule]}")
            continue
        if s.legs_missing_since and send:
            _set_missing(s.id, None)  # legs are back: it was a glitch
        if short_q > -s.contracts or long_q < s.contracts:
            msg = (f"{s.label()}: Schwab holds {short_q:+g} short / {long_q:+g} long contracts, not the "
                   f"spread's {s.contracts}; not closing anything automatically -- check thinkorswim")
            messages.append(msg)
            if send:
                _log(msg)
            continue
        try:
            chain = schwab_client.get_option_chain(s.symbol, exp, exp)
            v = exit_verdict(s, chain, now.date())
        except Exception as e:
            messages.append(f"{s.label()}: couldn't check exits ({e})")
            continue
        detail = "; ".join(v["lines"])
        if v["problem"]:
            messages.append(f"{s.label()}: can't check exits: {v['problem']}")
            continue
        if not v["rule"]:
            messages.append(f"{s.label()}: no exit rule met ({detail})")
            continue
        how = "natural" if v["rule"] in ("dte", "breach") else "mid"
        verdict = f"{s.label()}: CLOSE -- {RULE_TEXT[v['rule']]}; {detail}; limit {v['debit']:.2f} debit ({how})"
        if not send:
            messages.append(verdict + (" [market closed: would wait for the session]"
                                       if not regular_session_open() else ""))
            continue
        if not regular_session_open():
            messages.append(f"{verdict} -- market closed, will close during the next session")
            continue
        if kill_switch_on():
            messages.append(f"{verdict} -- NOT sent: STOP_TRADING file present")
            continue
        messages.append(_send_close(s, v))
    return messages


def _set_missing(spread_id: str, when: str | None):
    def _fn(lg: LiveLedger):
        x = lg.get(spread_id)
        if x is not None and x.status == "open":
            x.legs_missing_since = when
    _mutate(_fn)


def _mark_closed_elsewhere(spread_id: str, rule: str):
    def _fn(lg: LiveLedger):
        x = lg.get(spread_id)
        if x is None or x.status != "open":
            return None
        x.status, x.close_rule, x.closed_at = "closed", rule, _now_et().isoformat()
        if rule == "expired":
            x.realized_pnl = x.net_premium  # assumed expired worthless
            x.note("expired; no longer held at Schwab. Recorded as expired worthless -- check thinkorswim if not")
        else:
            x.note("no longer held at Schwab before expiration (closed by hand?); P&L unknown here")
        return x.label()
    label = _mutate(_fn)
    if label:
        msg = f"{label}: {RULE_TEXT[rule]}; recorded as closed"
        _log(msg)
        _push(f"Cathie's REAL position {msg}.")


def _known_order_ids() -> set:
    """Every Schwab order ID the ledger already accounts for."""
    return {i for x in load_ledger().spreads
            for i in (x.order_id, x.close_order_id, *x.past_close_order_ids) if i is not None}


def _send_close(s: LiveSpread, v: dict) -> str:
    """Preview, mark closing, send. Marked closing BEFORE sending (like submit) so a crash
    mid-send leaves a record that reconcile can match to the real order."""
    import schwab_execution
    import schwab_trading

    debit = v["debit"]
    order = schwab_trading.build_close_order(s.spread_type, s.short_symbol, s.long_symbol,
                                             s.contracts, debit).build()
    try:
        status, body = schwab_trading.preview_order(order)
        rejects = ((body.get("orderValidationResult") or {}).get("rejects") if isinstance(body, dict) else None)
    except Exception as e:
        status, rejects = 0, [str(e)]
    if status >= 400 or rejects or status == 0:
        msg = f"{s.label()}: close NOT sent, Schwab's preview refused it (HTTP {status}): {str(rejects)[:300]}"
        _log(msg)
        _push(f"Cathie couldn't close a REAL position: {msg}. Check it in thinkorswim.")
        return msg

    rule = v["rule"]

    def _mark(lg: LiveLedger):
        x = lg.get(s.id)
        if x is None or x.status != "open":
            return False
        x.status, x.close_rule, x.close_limit_debit = "closing", rule, debit
        x.close_submitted_at, x.close_order_id = _now_et().isoformat(), None
        x.note(f"closing ({RULE_TEXT[rule]}) at {debit:.2f} debit")
        return True
    if not _mutate(_mark):
        return f"{s.label()}: no longer open; close not sent"

    try:
        order_id = schwab_execution.place_order(order)
    except schwab_execution.OrderError as e:
        def _back(lg: LiveLedger):
            x = lg.get(s.id)
            if x and x.status == "closing":
                x.status, x.close_submitted_at = "open", None
                x.note(f"Schwab didn't accept the close order: {e}")
        _mutate(_back)
        msg = f"{s.label()}: close order NOT accepted by Schwab: {e}"
        _log(msg)
        _push(f"Cathie couldn't close a REAL position: {msg}. Check it in thinkorswim.")
        return msg
    except Exception as e:
        # Unknown whether it reached Schwab: stay 'closing'; reconcile looks it up.
        msg = f"{s.label()}: close order result unknown ({e}); checking at the next reconcile"
        _log(msg)
        return msg
    if order_id is None:
        found = schwab_execution.find_recent_order({s.short_symbol, s.long_symbol}, closing=True,
                                                   exclude_ids=_known_order_ids())
        order_id = found.get("orderId") if found else None

    def _record(lg: LiveLedger):
        x = lg.get(s.id)
        if x:
            x.close_order_id = order_id
            x.note(f"close sent; Schwab order {order_id}" if order_id else "close sent; order ID not yet known")
    _mutate(_record)
    msg = (f"{s.label()}: CLOSING -- {RULE_TEXT[rule]}; sent at {debit:.2f} debit, Schwab order {order_id}")
    _log(msg)
    _push(f"Cathie is closing a REAL position: {msg}.")
    return msg


def _reconcile_closing(s: LiveSpread, now: dt.datetime) -> str | None:
    """reconcile() for one 'closing' spread: record the fill, or put it back to 'open'
    if the close was cancelled/rejected/expired, cancelling it first if it's been
    working 30+ minutes. The next check_exits re-sends it at the current price."""
    import schwab_execution

    order_id, update = s.close_order_id, {}
    sent_at = _parse_et(s.close_submitted_at)
    age_min = (now - sent_at).total_seconds() / 60 if sent_at else 0
    if order_id is None:
        # Only orders entered since this close was sent (plus a little slack), so an
        # earlier, cancelled close attempt on the same legs can't be picked up.
        found = schwab_execution.find_recent_order({s.short_symbol, s.long_symbol},
                                                   since_minutes=int(age_min) + 3, closing=True,
                                                   exclude_ids=_known_order_ids())
        if found:
            order_id = found.get("orderId")
            update["close_order_id"] = order_id
        elif age_min > 10:
            update.update(status="open", close_submitted_at=None,
                          note="no close order found at Schwab; will retry")
    if order_id is not None:
        record = schwab_execution.get_order(order_id)
        status = record.get("status")
        if (status not in ("FILLED", "CANCELED", "REJECTED", "EXPIRED", "PENDING_CANCEL")
                and age_min >= UNFILLED_CANCEL_MINUTES):
            schwab_execution.cancel_order(order_id)
            record = schwab_execution.wait_for_status(order_id, {"CANCELED"}, timeout=30)
            status = record.get("status")
            update["note"] = f"close unfilled after {age_min:.0f} min; cancelled (Schwab order {order_id})"
        if status == "FILLED":
            net = fill_net_credit(record)
            debit = round(-net, 4) if net is not None else s.close_limit_debit
            credit = s.fill_credit or s.limit_credit or 0
            pnl = round((credit - debit) * 100 * s.contracts, 2)
            update.update(status="closed", close_fill_debit=debit, closed_at=now.isoformat(),
                          realized_pnl=pnl,
                          note=f"CLOSED at {debit:.2f} debit ({RULE_TEXT.get(s.close_rule, s.close_rule)}); "
                               f"P&L ${pnl:+,.2f} (Schwab order {order_id})")
        elif status in ("CANCELED", "REJECTED", "EXPIRED"):
            why = update.get("note") or f"close order {order_id} {status}" + (
                f": {record.get('statusDescription')}" if record.get("statusDescription") else "")
            update.update(status="open", close_order_id=None, close_submitted_at=None,
                          past_close_order_ids=[*s.past_close_order_ids, order_id],
                          note=f"{why}; will retry at the current price")
    if not update:
        return None

    def _apply(lg: LiveLedger):
        x = lg.get(s.id)
        if x is None or x.status != "closing":
            return
        for k, val in update.items():
            if k != "note":
                setattr(x, k, val)
        if update.get("note"):
            x.note(update["note"])
    _mutate(_apply)
    msg = f"{s.label()}: {update.get('note') or update.get('status')}"
    _log(msg)
    if update.get("status") == "closed":
        _push(f"Cathie's REAL position closed: {msg}.")
    return msg


def _token_warning() -> str | None:
    """Push once a day when the Schwab login is in its last day."""
    try:
        import schwab_client
        age = schwab_client.token_age_days()
    except Exception:
        return None
    if age is None or age < TOKEN_WARN_DAYS:
        return None
    today = _now_et().date().isoformat()
    left_h = max(0.0, (TOKEN_LIFETIME_DAYS - age) * 24)

    def _fn(lg: LiveLedger):
        if lg.token_warned_on == today:
            return False
        lg.token_warned_on = today
        return True
    msg = (f"Schwab login expires in about {left_h:.0f} hours (it lasts {TOKEN_LIFETIME_DAYS:g} days). "
           "Run `uv run schwab_auth_setup.py` to renew it; new trades stop in its last 12 hours, "
           "and automatic closes stop working once it expires.")
    if _mutate(_fn):
        _log(msg)
        _push(msg)
    return msg


def ledger_summary() -> dict:
    """Compact view of the real-trade ledger for Cathie's get_options_positions."""
    ledger = load_ledger()
    rows = [
        {k: v for k, v in {
            "id": s.id, "spread": s.label(), "status": s.status,
            "limit_credit": s.limit_credit, "fill_credit": s.fill_credit,
            "net_premium": s.net_premium, "max_loss": s.max_loss,
        }.items() if v is not None}
        for s in ledger.spreads if s.status in ACTIVE_STATUSES
    ]
    return {
        "mode": execution_mode(),
        "real_trades": rows,
        "orders_sent_today": orders_sent_today(ledger),
        "daily_limit": MAX_ORDERS_PER_DAY,
        "note": ("Real positions (status 'open') are closed automatically by code when an exit "
                 "rule fires (75% profit, breach, 7 DTE); close_credit_spread can't close them. "
                 "'staged' means waiting for the user's approval, 'pending' means sent to Schwab "
                 "but not filled, 'closing' means a close order is working."),
    }
