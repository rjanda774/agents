#!/usr/bin/env python
"""
Step 2 check for live Schwab trading: build a real credit-spread order and ask
Schwab to validate it, WITHOUT SENDING IT. Nothing here can place an order --
it only calls Schwab's previewOrder endpoint (see schwab_trading.py).

    uv run schwab_order_dryrun.py SYMBOL SPREAD_TYPE SHORT_STRIKE LONG_STRIKE EXPIRATION [CONTRACTS] [--credit X]

    uv run schwab_order_dryrun.py SPY bull_put 540 535 2026-11-20
    uv run schwab_order_dryrun.py AAPL bear_call 260 265 2026-11-20 1 --credit 0.85

SPREAD_TYPE is bull_put (short strike above long) or bear_call (short strike
below long). CONTRACTS defaults to 1. Without --credit, the limit price is the
spread's mid from Schwab's live quotes.

Prints:
  1. the two contracts, straight from Schwab's option chain (so the order can't
     name one that doesn't exist), with bid/ask and the mid/natural credit;
  2. the order exactly as it would be sent (NET_CREDIT limit, DAY);
  3. the real-money risk checks Cathie will apply in step 4, against your live
     Funds Available for Trading and Day Net Liq -- shown here, not enforced;
  4. Schwab's preview verdict: any rejects/warnings, and what it would cost.

Output masks your account number; fine to paste into a chat. The full preview
response is saved (also masked) to .schwab_preview_last.json for debugging.
"""
import argparse
import datetime as dt
import json
import os
import sys

PREVIEW_DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".schwab_preview_last.json")


def _money(v) -> str:
    return f"${v:,.2f}"


def _find(rows: list[dict], strike: float) -> dict | None:
    return next((r for r in rows if abs(r["strike"] - strike) < 1e-6), None)


def _messages(validation: dict) -> list[str]:
    """Flatten Schwab's orderValidationResult into readable lines. Its item shape
    isn't well documented, so take whichever message field is present."""
    lines = []
    for kind in ("rejects", "reviews", "warns", "alerts", "accepts"):
        for item in validation.get(kind) or []:
            if isinstance(item, dict):
                text = item.get("message") or item.get("activityMessage") or item.get("validationRuleName") or json.dumps(item)
            else:
                text = str(item)
            lines.append(f"  {kind[:-1].upper():<7} {text}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Build and preview (never send) a Schwab credit-spread order.")
    parser.add_argument("symbol")
    parser.add_argument("spread_type", choices=["bull_put", "bear_call"])
    parser.add_argument("short_strike", type=float)
    parser.add_argument("long_strike", type=float)
    parser.add_argument("expiration", help="YYYY-MM-DD")
    parser.add_argument("contracts", type=int, nargs="?", default=1)
    parser.add_argument("--credit", type=float, help="limit price (net credit per spread); default: live mid")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    if args.spread_type == "bull_put" and not args.short_strike > args.long_strike:
        sys.exit("ERROR: a bull put's short strike must be ABOVE its long strike.")
    if args.spread_type == "bear_call" and not args.short_strike < args.long_strike:
        sys.exit("ERROR: a bear call's short strike must be BELOW its long strike.")
    if args.contracts < 1:
        sys.exit("ERROR: contracts must be at least 1.")
    try:
        expiration = dt.date.fromisoformat(args.expiration)
    except ValueError:
        sys.exit(f"ERROR: expiration {args.expiration!r} isn't YYYY-MM-DD.")

    import schwab_client
    import schwab_trading

    if not schwab_client.is_configured():
        sys.exit("ERROR: Schwab isn't set up -- run `uv run schwab_auth_setup.py` first.")

    # 1. The contracts, from Schwab's own chain.
    try:
        chain = schwab_client.get_option_chain(symbol, expiration, expiration)
    except Exception as e:
        sys.exit(f"ERROR: couldn't get {symbol}'s option chain from Schwab: {e}")
    exp_data = chain["expirations"].get(expiration.isoformat())
    if not exp_data:
        listed = ", ".join(sorted(chain["expirations"])) or "none"
        sys.exit(f"ERROR: {symbol} has no options expiring {expiration} (Schwab returned: {listed}).")
    rows = exp_data["puts"] if args.spread_type == "bull_put" else exp_data["calls"]
    short, long_ = _find(rows, args.short_strike), _find(rows, args.long_strike)
    for label, strike, row in (("short", args.short_strike, short), ("long", args.long_strike, long_)):
        if row is None or not row.get("symbol"):
            near = sorted(r["strike"] for r in sorted(rows, key=lambda r: abs(r["strike"] - strike))[:6])
            sys.exit(f"ERROR: no {label} {strike:g} contract for {expiration}. "
                     f"Nearest strikes: {', '.join(format(k, 'g') for k in near)}")

    mid = (short["bid"] + short["ask"]) / 2 - (long_["bid"] + long_["ask"]) / 2
    natural = short["bid"] - long_["ask"]
    print(f"{symbol} {args.spread_type} {args.short_strike:g}/{args.long_strike:g} exp {expiration} "
          f"x{args.contracts}   (underlying {_money(chain['current_price'])})")
    for label, row in (("short", short), ("long ", long_)):
        delta = f"{row['delta']:.3f}" if row.get("delta") is not None else "n/a"
        print(f"  {label} {row['symbol']:<22} bid {row['bid']:.2f}  ask {row['ask']:.2f}  delta {delta}")
    print(f"  credit per spread: mid {mid:.2f}, natural {natural:.2f}")
    if args.credit is not None:
        credit = args.credit
    else:
        credit = round(mid, 2)
        print(f"  using mid {credit:.2f} as the limit (pass --credit to choose)")
    if credit <= 0:
        sys.exit(f"ERROR: credit {credit:.2f} isn't positive -- no premium to collect at these strikes.")
    print()

    # 2. The order, exactly as it would be sent.
    order = schwab_trading.build_open_order(
        args.spread_type, short["symbol"], long_["symbol"], args.contracts, credit
    ).build()
    print("Order (NOT sent):")
    print(json.dumps(order, indent=2))
    print()

    # 3. Step-4 risk checks against live balances (shown, not enforced, in a dry run).
    width = abs(args.short_strike - args.long_strike)
    net_premium = credit * 100 * args.contracts
    max_loss = (width - credit) * 100 * args.contracts
    print(f"Risk: net premium {_money(net_premium)}, max loss {_money(max_loss)}")
    try:
        balances = schwab_trading.risk_balances(schwab_trading.get_account(include_positions=False))
        print(f"  Funds Available for Trading {_money(balances['funds_available'])}, "
              f"Day Net Liq {_money(balances['net_liq'])}")
        problems = schwab_trading.check_spread_risk(max_loss, net_premium, balances)
        if problems:
            print("  Cathie would REJECT this in step 4:")
            for p in problems:
                print(f"    - {p}")
        else:
            print("  passes the real-money risk checks")
    except Exception as e:
        print(f"  couldn't read balances: {e}")
    print()

    # 4. Schwab's preview verdict.
    try:
        status, body = schwab_trading.preview_order(order)
    except Exception as e:
        sys.exit(f"ERROR: preview request failed: {e}")
    try:
        with open(PREVIEW_DUMP, "w", encoding="utf-8") as f:
            json.dump({"http_status": status, "response": body}, f, indent=2)
    except OSError:
        pass
    print(f"Schwab preview: HTTP {status}")
    if not isinstance(body, dict):
        print(f"  {str(body)[:1000]}")
        return
    validation = body.get("orderValidationResult") or {}
    lines = _messages(validation)
    print("\n".join(lines) if lines else "  no rejects, warnings or alerts")
    strategy = body.get("orderStrategy") or {}
    for key in ("orderValue", "orderBalance", "status"):
        if key in strategy:
            print(f"  {key}: {json.dumps(strategy[key])}")
    if body.get("commissionAndFee"):
        print(f"  commissionAndFee: {json.dumps(body['commissionAndFee'])[:600]}")
    if status >= 400 or not body:
        print(f"  response: {json.dumps(body)[:1000]}")
    print(f"\nFull preview response (account masked): {os.path.basename(PREVIEW_DUMP)}")


if __name__ == "__main__":
    main()
