#!/usr/bin/env python
"""
Review and approve Cathie's staged REAL trades (CATHIE_EXECUTION_MODE=approve).

    uv run approve_orders.py           review each staged trade, approve or reject it
    uv run approve_orders.py --list    just show the real-trade ledger, change nothing

First it brings the ledger up to date with Schwab (fills, cancellations, orders
left working 30+ minutes get cancelled). Then, for each trade Cathie staged, it
re-checks everything live -- market hours, current prices, delta, premium, your
Funds Available and Day Net Liq, the daily limit, your own positions, and
Schwab's preview -- and shows you the result:

  - if any check fails, it says why and leaves the trade staged (it may pass later
    today, e.g. once the market opens); type r to reject it for good.
  - if every check passes, type y to SEND the order to Schwab, r to reject, or
    press Enter to skip for now.

Prices are taken fresh at approval (the spread's current mid), not from when
Cathie proposed it. Staged trades not approved by the end of their day expire.
Exits aren't automated yet: close real positions yourself in thinkorswim.
"""
import argparse
import sys


def _show_ledger(live_trading):
    ledger = live_trading.load_ledger()
    active = [s for s in ledger.spreads if s.status in live_trading.ACTIVE_STATUSES]
    print(f"Mode: {live_trading.execution_mode()}   orders sent today: "
          f"{live_trading.orders_sent_today(ledger)}/{live_trading.MAX_ORDERS_PER_DAY}"
          + ("   STOP_TRADING is ON" if live_trading.kill_switch_on() else ""))
    if not active:
        print("No staged, pending or open real trades.")
    for s in active:
        extra = ""
        if s.status == "open":
            extra = f"  filled {s.fill_credit:.2f}, premium ${s.net_premium:,.2f}, max loss ${s.max_loss:,.2f}"
        elif s.status == "pending":
            extra = f"  Schwab order {s.order_id}, limit {s.limit_credit:.2f}"
        print(f"  [{s.id}] {s.status.upper():<7} {s.label()}{extra}")
    # Today's trades that are already finished (not filled, rejected, expired), so the
    # outcome of an order sent earlier is visible here too.
    today = live_trading._now_et().date().isoformat()
    done = [s for s in ledger.spreads if s.status not in live_trading.ACTIVE_STATUSES
            and (s.submitted_at or s.staged_at or "")[:10] == today]
    if done:
        print("Earlier today:")
        for s in done:
            last = s.events[-1] if s.events else ""
            print(f"  [{s.id}] {s.status.upper():<10} {s.label()}" + (f"  ({last})" if last else ""))
    print()


def main():
    parser = argparse.ArgumentParser(description="Approve or reject Cathie's staged real trades.")
    parser.add_argument("--list", action="store_true", help="show the ledger only")
    args = parser.parse_args()

    import live_trading
    import schwab_client

    if not schwab_client.is_configured():
        sys.exit("ERROR: Schwab isn't set up -- run `uv run schwab_auth_setup.py` first.")

    try:
        for msg in live_trading.reconcile():
            print(f"update: {msg}")
    except Exception as e:
        print(f"WARNING: couldn't update from Schwab: {e}")
    _show_ledger(live_trading)
    if args.list:
        return

    staged = [s for s in live_trading.load_ledger().spreads if s.status == "staged"]
    if not staged:
        print("Nothing waiting for approval.")
        return

    for spread in staged:
        print("=" * 72)
        print(f"[{spread.id}] {spread.label()}   staged {spread.staged_at[:16].replace('T', ' ')}")
        if spread.rationale:
            print(f"Cathie's reasoning: {spread.rationale[:500]}")
        try:
            plan = live_trading.check_for_approval(spread)
        except Exception as e:
            print(f"  couldn't check it right now: {e}")
            continue
        for line in plan["lines"]:
            print(f"  {line}")

        try:
            if not plan["ok"]:
                print("  NOT READY:")
                for p in plan["problems"]:
                    print(f"    - {p}")
                answer = input("  r = reject for good, Enter = leave it staged: ").strip().lower()
                if answer == "r":
                    live_trading.reject(spread.id)
                    print("  rejected.")
                continue
            answer = input(f"  SEND this order at {plan['credit']:.2f} credit? y = send, r = reject, "
                           "Enter = skip: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped. Nothing more was sent.")
            return

        if answer == "r":
            live_trading.reject(spread.id)
            print("  rejected.")
        elif answer == "y":
            try:
                order_id = live_trading.submit(spread.id, plan)
                print(f"  SENT: Schwab order {order_id}. It shows in thinkorswim's Working orders until it "
                      f"fills; if it hasn't filled in {live_trading.UNFILLED_CANCEL_MINUTES} minutes it gets "
                      "cancelled at the next check.")
            except Exception as e:
                print(f"  NOT SENT: {e}")
        else:
            print("  skipped; still staged.")
    print()
    _show_ledger(live_trading)


if __name__ == "__main__":
    main()
