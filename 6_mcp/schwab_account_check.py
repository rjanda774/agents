#!/usr/bin/env python
"""
Step 1 check for live Schwab trading: read your real account through the Trader
API and print what Cathie would see. READ-ONLY -- this cannot place, change, or
cancel anything (see schwab_trading.py).

    uv run schwab_account_check.py
    uv run schwab_account_check.py --match 10205.34 17935.99
    uv run schwab_account_check.py --days 30

Prints: the account (number masked), its type, every balance field Schwab's API
reports, your current option positions, and orders from the last few days.

--match AMOUNT [AMOUNT ...] marks every balance field within $1 of each amount you
give. Use it with the numbers schwab.com shows at the same moment for "Funds
Available for Trading" and "Day Net Liquidating Value": the API names its fields
differently from the website, and this is how we pin down which field is which
before Cathie relies on them. Balances move intraday, so read both at the same time.

Output contains balances and positions but never your full account number, the
account hash, or any token -- fine to paste back into a chat.
"""
import argparse
import sys


def _money(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    parser = argparse.ArgumentParser(description="Read-only check of your Schwab account via the Trader API.")
    parser.add_argument("--match", type=float, nargs="+", default=[],
                        help="dollar amounts from schwab.com to find among the API's balance fields")
    parser.add_argument("--days", type=int, default=7, help="how many days of orders to list (max 60)")
    args = parser.parse_args()

    import schwab_client
    import schwab_trading

    if not schwab_client.is_configured():
        print("ERROR: Schwab isn't set up -- SCHWAB_APP_KEY/SCHWAB_APP_SECRET missing from .env,")
        print("or no token file yet. Run `uv run schwab_auth_setup.py` first.")
        sys.exit(1)

    try:
        _, number = schwab_trading.get_account_hash()
        account = schwab_trading.get_account(include_positions=True)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Account:   {schwab_trading.mask_account_number(number)}")
    print(f"Type:      {account.get('type')}")
    for key in ("roundTrips", "isDayTrader", "isClosingOnlyRestricted", "pfcbFlag"):
        if key in account:
            print(f"{key + ':':<10} {account[key]}")
    print()

    balances = schwab_trading.balance_fields(account)
    print(f"Balance fields reported by the API ({len(balances)}):")
    width = max((len(k) for k in balances), default=0)
    for key, value in balances.items():
        marks = [f"<-- matches {_money(m)}" for m in args.match if abs(value - m) <= 1.0]
        print(f"  {key:<{width}}  {_money(value):>16}  {' '.join(marks)}")
    for m in args.match:
        if not any(abs(v - m) <= 1.0 for v in balances.values()):
            print(f"  (no field within $1 of {_money(m)} -- balances may have moved; re-read both at once)")
    print()

    positions = schwab_trading.option_positions(account)
    print(f"Option positions ({len(positions)}):")
    if not positions:
        print("  none")
    for p in positions:
        side = "short" if p["quantity"] < 0 else "long"
        print(
            f"  {p['underlying'] or '?':<6} {p['expiration'] or '?':<10} "
            f"{(p['option_type'] or '?'):<4} {p['strike'] if p['strike'] is not None else '?':>8}  "
            f"{side} {abs(p['quantity']):g}  avg {_money(p['average_price'])}  "
            f"mkt value {_money(p['market_value'])}"
        )
    print()

    try:
        orders = schwab_trading.recent_orders(days=args.days)
    except Exception as e:
        print(f"Orders: couldn't read them ({e})")
        return
    print(f"Orders in the last {args.days} days ({len(orders)}):")
    if not orders:
        print("  none")
    for o in orders:
        legs = "; ".join(
            f"{leg['instruction']} {leg['quantity']:g} {leg['symbol']}"
            if isinstance(leg["quantity"], (int, float)) else f"{leg['instruction']} {leg['symbol']}"
            for leg in o["legs"]
        )
        print(
            f"  {o['entered_time']}  {o['status']:<10} {o['order_type'] or '':<10} "
            f"price {o['price']}  filled {o['filled_quantity']}/{o['quantity']}  | {legs}"
        )


if __name__ == "__main__":
    main()
