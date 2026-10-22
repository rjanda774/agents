#!/usr/bin/env python
"""
Breach alerts: a Pushover warning when a stock gets close to, or past, the short
strike of any credit spread in the Schwab account -- the user's own (MANUAL) and
Cathie's. ALERTS ONLY: this never places, changes or cancels an order.

    uv run breach_watch.py            keep watching: checks every 5 minutes during
                                      market hours (Mon-Fri 9:30-16:00 ET), sleeps otherwise
    uv run breach_watch.py --once     check once now and print every spread's cushion

Added 2026-10-11 at the user's request: Cathie's code only closes the user's own
spreads at 75% profit, never on a breach, and the trading floor checks hourly. SNDK
can move $50-100 in a day, with the user's 1560/1550 bull put only $22 above its
short strike, so this checks far more often and runs on its own, separate from the
trading floor (a third PowerShell window, or Task Scheduler).

Two alert levels per spread, each sent at most once per spread per day:
  NEAR      the stock is within NEAR_PCT (1.5%) of the short strike
  BREACHED  the stock is past the short strike (below it for a bull put / short
            put, above it for a bear call / short call)
Only credit spreads and lone short options are watched; debit spreads and long
options gain when the stock moves that way. Which alerts were sent today is kept in
.breach_alerts.json (gitignored).
"""
import argparse
import datetime as dt
import json
import os
import time

CHECK_EVERY_MINUTES = 5
NEAR_PCT = 1.5
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".breach_alerts.json")


def watched_spreads(positions: list[dict], ledger) -> list[dict]:
    """Every credit spread / lone short option at Schwab, as dicts: owner, underlying,
    option_type, short, long (or None), expiration, qty, max_loss (or None), label."""
    from real_dashboard import _pair_legs, filled_pending

    out = []
    cathies = set()
    filled = filled_pending(ledger, {p["symbol"] for p in positions})
    for s in ledger.spreads:
        if s.status not in ("open", "closing") and s.id not in filled:
            continue
        cathies |= {s.short_symbol, s.long_symbol}
        out.append({
            "owner": "CATHIE", "underlying": s.symbol,
            "option_type": "put" if s.spread_type == "bull_put" else "call",
            "short": s.short_strike, "long": s.long_strike, "expiration": s.expiration_date,
            "qty": s.contracts, "max_loss": s.max_loss,
            "label": f"Cathie's {s.spread_type.replace('_', ' ')} {s.short_strike:g}/{s.long_strike:g} exp {s.expiration_date}",
        })

    groups = {}
    for p in positions:
        if p["symbol"] not in cathies and p.get("quantity"):
            groups.setdefault((p["underlying"], p["expiration"], p["option_type"]), []).append(p)
    for (underlying, expiration, option_type), legs in groups.items():
        for sh, lg, qty in _pair_legs(legs):
            if not sh:
                continue  # a lone long option: a falling/rising stock doesn't hurt it
            if lg and not (lg["strike"] < sh["strike"] if option_type == "put" else lg["strike"] > sh["strike"]):
                continue  # debit spread: the long leg is the closer one
            max_loss = None
            if lg:
                credit = (sh.get("average_price") or 0) - (lg.get("average_price") or 0)
                max_loss = round((abs(sh["strike"] - lg["strike"]) - credit) * 100 * qty, 2)
                kind = "bull put" if option_type == "put" else "bear call"
                label = f"your {kind} {sh['strike']:g}/{lg['strike']:g} exp {expiration}"
            else:
                label = f"your short {option_type} {sh['strike']:g} exp {expiration} (no long leg: undefined risk)"
            out.append({
                "owner": "MANUAL", "underlying": underlying, "option_type": option_type,
                "short": sh["strike"], "long": lg["strike"] if lg else None, "expiration": expiration,
                "qty": int(qty), "max_loss": max_loss, "label": label,
            })
    return out


def assess(spread: dict, price: float, near_pct: float = NEAR_PCT) -> tuple[str, float]:
    """('breached' | 'near' | 'ok', cushion %) for one spread at `price`. Cushion is how
    far the stock is from the short strike, as % of the price; negative once breached."""
    if spread["option_type"] == "put":
        cushion = (price - spread["short"]) / price * 100
    else:
        cushion = (spread["short"] - price) / price * 100
    if cushion < 0:
        return "breached", cushion
    if cushion < near_pct:
        return "near", cushion
    return "ok", cushion


def _message(spread: dict, level: str, price: float, cushion: float) -> str:
    side = "below" if spread["option_type"] == "put" else "above"
    risk = f"; max loss about ${spread['max_loss']:,.0f}" if spread.get("max_loss") is not None else ""
    if level == "breached":
        head = f"BREACHED: {spread['underlying']} ${price:,.2f} is {side} the short {spread['short']:g} strike"
    else:
        head = (f"NEAR: {spread['underlying']} ${price:,.2f} is only {cushion:.1f}% "
                f"(${abs(price - spread['short']):,.2f}) from the short {spread['short']:g} strike")
    if spread.get("hands_off"):
        tail = " It's on your hands-off list, so nothing closes it automatically: decide in thinkorswim."
    elif spread["owner"] == "CATHIE":
        tail = " Cathie's code closes it automatically at the natural price."
    else:
        tail = " Nothing is closed automatically for your own spreads: decide in thinkorswim."
    return f"{head} -- {spread['label']}{risk}.{tail}"


def _load_state(today: str) -> dict:
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        return state if state.get("date") == today else {"date": today, "sent": []}
    except Exception:
        return {"date": today, "sent": []}


def _save_state(state: dict):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_FILE)


def check(send: bool, near_pct: float = NEAR_PCT, verbose: bool = False) -> list[str]:
    """One pass: read positions and prices, send any new alert (when `send`), return
    lines describing every watched spread."""
    import live_trading
    import schwab_client
    import schwab_trading

    positions = schwab_trading.option_positions(schwab_trading.get_account(include_positions=True))
    spreads = watched_spreads(positions, live_trading.load_ledger())
    hands_off = live_trading.hands_off_tickers()
    for sp in spreads:
        sp["hands_off"] = sp["underlying"] in hands_off
    today = live_trading._now_et().date().isoformat()
    state = _load_state(today)
    prices, lines = {}, []
    for spread in sorted(spreads, key=lambda x: (x["underlying"], x["short"])):
        u = spread["underlying"]
        if u not in prices:
            try:
                prices[u] = schwab_client.get_quote(u)
            except Exception as e:
                prices[u] = None
                lines.append(f"{u}: couldn't get a price ({e})")
        price = prices[u]
        if price is None:
            continue
        level, cushion = assess(spread, price, near_pct)
        lines.append(f"{level.upper():<8} {u} ${price:,.2f}  {spread['label']}  cushion {cushion:+.1f}%")
        key = f"{spread['owner']}|{u}|{spread['option_type']}|{spread['short']:g}|{spread['expiration']}|{level}"
        if level != "ok" and send and key not in state["sent"]:
            msg = _message(spread, level, price, cushion)
            live_trading._push(msg)
            live_trading._log(msg)
            state["sent"].append(key)
            lines.append(f"  -> alert sent: {msg}")
    if send:
        _save_state(state)
    if verbose:
        for line in lines:
            print(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Pushover alerts when a stock nears or passes a short strike.")
    parser.add_argument("--once", action="store_true", help="check once now and print every spread")
    parser.add_argument("--near-pct", type=float, default=NEAR_PCT,
                        help=f"warn when the stock is within this %% of the short strike (default {NEAR_PCT})")
    args = parser.parse_args()

    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    import live_trading
    import schwab_client
    if not schwab_client.is_configured():
        raise SystemExit("ERROR: Schwab isn't set up -- run `uv run schwab_auth_setup.py` first.")

    if args.once:
        open_now = live_trading.regular_session_open()
        if not open_now:
            print("Market is closed: showing prices, sending no alerts.")
        check(send=open_now, near_pct=args.near_pct, verbose=True)
        return

    from single_instance import acquire_single_instance_lock
    acquire_single_instance_lock("breach_watch.py")
    print(f"Watching for breaches every {CHECK_EVERY_MINUTES} min during market hours "
          f"(near = within {args.near_pct}%). Alerts only; Ctrl+C to stop.")
    while True:
        if live_trading.regular_session_open():
            try:
                lines = check(send=True, near_pct=args.near_pct)
                flagged = [x for x in lines if not x.startswith("OK")]
                stamp = live_trading._now_et().strftime("%H:%M")
                print(f"[{stamp} ET] checked {sum(1 for x in lines if x[:8].strip() in ('OK', 'NEAR', 'BREACHED'))} "
                      f"spread(s)" + ("" if not flagged else ":\n  " + "\n  ".join(flagged)))
            except Exception as e:
                print(f"check failed: {e}")
        time.sleep(CHECK_EVERY_MINUTES * 60)


if __name__ == "__main__":
    main()
