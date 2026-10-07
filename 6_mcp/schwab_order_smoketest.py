#!/usr/bin/env python
"""
Step 3 check for live Schwab trading: SEND ONE REAL ORDER that can't fill, see it
in thinkorswim, then cancel it. Proves place / read-back / cancel work end to end.

    uv run schwab_order_smoketest.py SYMBOL SPREAD_TYPE SHORT_STRIKE LONG_STRIKE EXPIRATION           (shows the plan only)
    uv run schwab_order_smoketest.py SYMBOL SPREAD_TYPE SHORT_STRIKE LONG_STRIKE EXPIRATION --send    (really sends it)

    uv run schwab_order_smoketest.py CIEN bull_put 385 375 2026-11-06 --send

Why it can't fill: it's 1 contract, priced at a credit of the spread's full width
minus $0.05 (e.g. $9.95 on a $10-wide spread). That's only payable if the spread
is almost certainly finishing in the money, and the script refuses to run unless
the short strike is out of the money and the live mid is under half the width.
Even if it somehow did fill, the most it could lose is $5 plus about $1.34 in
commission and fees, since the max loss of a credit spread is width minus credit.

Sequence:
  1. checks the strikes against Schwab's chain and runs the safety checks;
  2. previews the order with Schwab -- stops here if Schwab would reject it;
  3. asks you to type SEND, then places the order;
  4. reads it back from Schwab and shows its status (QUEUED/WORKING/...);
  5. waits while you find it in thinkorswim's Working orders;
  6. cancels it when you press Enter (or on Ctrl+C or any error) and confirms
     Schwab shows it CANCELED.
If the cancel can't be confirmed, it says so loudly with the order ID: cancel it
in thinkorswim yourself then.
"""
import argparse
import datetime as dt
import json
import sys

LIVE_STATUSES = {"WORKING", "QUEUED", "PENDING_ACTIVATION", "ACCEPTED", "AWAITING_MANUAL_REVIEW",
                 "NEW", "AWAITING_RELEASE_TIME", "PENDING_ACKNOWLEDGEMENT"}


def main():
    parser = argparse.ArgumentParser(description="Place one unfillable credit-spread order, then cancel it.")
    parser.add_argument("symbol")
    parser.add_argument("spread_type", choices=["bull_put", "bear_call"])
    parser.add_argument("short_strike", type=float)
    parser.add_argument("long_strike", type=float)
    parser.add_argument("expiration", help="YYYY-MM-DD")
    parser.add_argument("--send", action="store_true", help="really send the order (otherwise only show the plan)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    if args.spread_type == "bull_put" and not args.short_strike > args.long_strike:
        sys.exit("ERROR: a bull put's short strike must be ABOVE its long strike.")
    if args.spread_type == "bear_call" and not args.short_strike < args.long_strike:
        sys.exit("ERROR: a bear call's short strike must be BELOW its long strike.")
    try:
        expiration = dt.date.fromisoformat(args.expiration)
    except ValueError:
        sys.exit(f"ERROR: expiration {args.expiration!r} isn't YYYY-MM-DD.")

    import schwab_client
    import schwab_execution
    import schwab_trading
    from schwab_order_dryrun import _find, _messages

    if not schwab_client.is_configured():
        sys.exit("ERROR: Schwab isn't set up -- run `uv run schwab_auth_setup.py` first.")

    # 1. Contracts from Schwab's chain, and the can't-fill safety checks.
    try:
        chain = schwab_client.get_option_chain(symbol, expiration, expiration)
    except Exception as e:
        sys.exit(f"ERROR: couldn't get {symbol}'s option chain from Schwab: {e}")
    exp_data = chain["expirations"].get(expiration.isoformat())
    if not exp_data:
        sys.exit(f"ERROR: {symbol} has no options expiring {expiration}.")
    rows = exp_data["puts"] if args.spread_type == "bull_put" else exp_data["calls"]
    short, long_ = _find(rows, args.short_strike), _find(rows, args.long_strike)
    if not (short and long_ and short.get("symbol") and long_.get("symbol")):
        sys.exit(f"ERROR: {args.short_strike:g} and/or {args.long_strike:g} isn't a listed strike for {expiration}.")

    price = chain["current_price"]
    otm = price > args.short_strike if args.spread_type == "bull_put" else price < args.short_strike
    if not otm:
        sys.exit(f"REFUSED: the short strike {args.short_strike:g} isn't out of the money "
                 f"(underlying ${price:,.2f}). Pick strikes further out so the order can't fill.")
    width = abs(args.short_strike - args.long_strike)
    have_quotes = all(r["bid"] > 0 or r["ask"] > 0 for r in (short, long_))
    mid = (short["bid"] + short["ask"]) / 2 - (long_["bid"] + long_["ask"]) / 2
    if have_quotes and mid >= width / 2:
        sys.exit(f"REFUSED: the spread's mid ({mid:.2f}) is at least half its width ({width:g}); "
                 "it's too close to the money for a can't-fill test. Pick strikes further out.")
    credit = round(width - 0.05, 2)
    if credit <= 0:
        sys.exit("REFUSED: spread is too narrow to price an unfillable credit.")

    print(f"{symbol} {args.spread_type} {args.short_strike:g}/{args.long_strike:g} exp {expiration} x1 "
          f"(underlying ${price:,.2f})")
    print(f"  short {short['symbol']}  bid {short['bid']:.2f} ask {short['ask']:.2f}")
    print(f"  long  {long_['symbol']}  bid {long_['bid']:.2f} ask {long_['ask']:.2f}")
    print(f"  live mid credit {mid:.2f}" if have_quotes else "  (no live quotes right now)")
    print(f"  TEST LIMIT CREDIT {credit:.2f} on a {width:g}-wide spread -- can't fill at today's prices.")
    print(f"  If it somehow did fill: max loss ${(width - credit) * 100:,.2f} plus commission.")
    print()

    order = schwab_trading.build_open_order(args.spread_type, short["symbol"], long_["symbol"], 1, credit).build()

    # 2. Preview first; stop if Schwab would reject it.
    status, body = schwab_trading.preview_order(order)
    validation = (body.get("orderValidationResult") or {}) if isinstance(body, dict) else {}
    lines = _messages(validation)
    print(f"Schwab preview: HTTP {status}")
    print("\n".join(lines) if lines else "  no rejects, warnings or alerts")
    if status >= 400 or validation.get("rejects"):
        sys.exit("STOPPED: Schwab's preview rejected this order, so nothing was sent.")
    print()

    if not args.send:
        print("Plan only -- nothing sent. Add --send to place this order for real.")
        return

    # 3. Explicit confirmation, then send.
    try:
        answer = input("This sends a REAL order to your Schwab account. Type SEND to place it: ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip() != "SEND":
        print("Not sent.")
        return

    leg_symbols = {short["symbol"], long_["symbol"]}
    try:
        order_id = schwab_execution.place_order(order)
    except schwab_execution.OrderError as e:
        sys.exit(f"NOT PLACED: {e}")
    if order_id is None:
        print("Schwab accepted the order but didn't return its ID; looking it up...")
        found = schwab_execution.find_recent_order(leg_symbols)
        order_id = found.get("orderId") if found else None
        if order_id is None:
            sys.exit("!!! The order was sent but couldn't be found. CANCEL IT IN THINKORSWIM NOW "
                     f"({symbol} {args.short_strike:g}/{args.long_strike:g} {expiration}, credit {credit:.2f}).")
    print(f"PLACED: Schwab order ID {order_id}")

    # 4-6. Read back, wait for the user, always cancel.
    final = None
    try:
        record = schwab_execution.wait_for_status(order_id, LIVE_STATUSES, timeout=15)
        st = record.get("status")
        print(f"Schwab shows status {st}"
              + (f" ({record['statusDescription']})" if record.get("statusDescription") else ""))
        if st == "FILLED":
            print("!!! It FILLED. That shouldn't be possible at this price -- check the position in thinkorswim.")
            final = record
            return
        if st in schwab_execution.TERMINAL_STATUSES:
            print(f"The order is already {st}; nothing to cancel.")
            final = record
            return
        try:
            input("\nFind it in thinkorswim (Monitor > Working orders), then press Enter to cancel it... ")
        except (EOFError, KeyboardInterrupt):
            print("\n(interrupted -- cancelling now)")
    finally:
        if final is None:
            try:
                schwab_execution.cancel_order(order_id)
                final = schwab_execution.wait_for_status(order_id, {"CANCELED"}, timeout=30)
            except Exception as e:
                final = {"status": f"unknown ({e})"}
            st = final.get("status")
            if st == "CANCELED":
                print(f"CANCELED: Schwab confirms order {order_id} is cancelled. Check it shows as cancelled in thinkorswim.")
            else:
                print(f"!!! Couldn't confirm the cancel (status: {st}). CANCEL ORDER {order_id} IN THINKORSWIM NOW.")
                print(json.dumps({k: final.get(k) for k in ("status", "statusDescription", "filledQuantity")}))


if __name__ == "__main__":
    main()
