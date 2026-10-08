#!/usr/bin/env python
"""
Code-driven credit-spread scanner: finds spreads that actually pass Cathie's hard
rules across every candidate source, so she picks from real trades instead of
hand-checking 3-5 of ~300 tickers.

Why: given ~240 screener tickers plus the watchlist and dark-pool lists, Cathie
(gpt-4o-mini) only ever ran get_options_chain on a few -- in practice the watchlist
and dark-pool names, whose prompt wording was strongest -- and missed viable
screener names (AMD and MU on 2026-10, per the user's own thinkorswim check).
Screening hundreds of chains is mechanical work, so code does it now.

What it does, each cycle before Cathie's new-trade pass (traders.py) or by hand
(`uv run spread_scanner.py`):
  1. Candidates: the user's watchlist, TradeAlgo dark-pool tickers, the named ETF
     universe, and all six Yahoo screens (40 each, $50+), de-duplicated, in that
     priority order (the scan stops at a time budget, so the first ones matter most).
  2. One Schwab option-chain call per ticker for the 25-45 DTE window.
  3. Every bull put and bear call, $5 and $10 wide, that passes: short |delta| <= 0.20,
     bid and ask on both legs, open interest >= 100 on both legs, natural no more than
     10% of the spread's width below mid, $50+ premium at the mid, max loss <= 5x premium and <= 8% of Day Net
     Liq (when the account can be read). Best one per ticker and direction, ranked by
     premium / max loss.
  4. Drops individual stocks under $100, anything with earnings from today through 20
     days past expiration or reported in the last 5 trading days (trade_rules.py; the
     named ETFs are exempt from all three), and in approve mode also unknown earnings and
     underlyings the user holds options on themselves (staging would reject those anyway).
Prices are mids for 1 contract. Everything is re-checked when Cathie stages it and
again at approval -- this list only says where to look.

Schwab only: without Schwab market data the scan is skipped (yfinance would take far
too long for ~300 tickers), and Cathie is told to fall back to the old routine.
"""
import concurrent.futures
import datetime as dt
import warnings

import trade_rules

# Importing options_trading_server (for the screener list and earnings lookup) loads the
# MCP library, whose FastMCP settings model warns about an unresolved 'lifespan' field.
# Harmless -- that setting is never used here -- so hide exactly that one warning.
warnings.filterwarnings("ignore", message=r"Field 'lifespan' has an incomplete definition")
import json
import math
import os
import time

MIN_DTE, MAX_DTE = 25, 45
MAX_SHORT_DELTA = 0.20
WIDTHS = (5.0, 10.0)
MIN_OPEN_INTEREST = 100
MIN_NET_PREMIUM = 50.0
MAX_RISK_TO_PREMIUM = 5.0
MAX_RISK_PCT_OF_NET_LIQ = 0.08
EARNINGS_BUFFER_DAYS = 20
STRIKE_COUNT = 40          # strikes each side of the money requested per chain
SCREENER_COUNT = 40
TIME_BUDGET_SECONDS = 360  # stop scanning new tickers after this; report it as partial
TOP_N = 10
WORKERS = 4


def _candidates() -> tuple[dict[str, list[str]], list[str]]:
    """{ticker: [sources]} in priority order, plus notes about sources that failed."""
    from universe import CATHIE_ETF_UNIVERSE, load_custom_watchlist

    found: dict[str, list[str]] = {}
    notes = []

    def add(symbol, source):
        symbol = str(symbol or "").strip().upper()
        if symbol and symbol.replace(".", "").replace("-", "").isalnum():
            found.setdefault(symbol, [])
            if source not in found[symbol]:
                found[symbol].append(source)

    try:
        for t in load_custom_watchlist():
            add(t, "watchlist")
    except Exception as e:
        notes.append(f"watchlist unreadable: {e}")

    darkpool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tradealgo_darkpool.json")
    try:
        if os.path.exists(darkpool):
            with open(darkpool, "r", encoding="utf-8") as f:
                cache = json.load(f)
            for t in (cache.get("intraday") or {}).get("tickers") or []:
                add(t.get("ticker"), f"darkpool:{t.get('direction') or '?'}")
            for t in (cache.get("historical") or {}).get("tickers") or []:
                add(t.get("ticker"), "darkpool:historical")
    except Exception as e:
        notes.append(f"dark-pool file unreadable: {e}")

    for t in CATHIE_ETF_UNIVERSE:
        add(t, "etf_universe")

    try:
        import yfinance as yf
        from options_trading_server import MIN_SCREENER_PRICE, SCREENER_QUERIES
        for query in SCREENER_QUERIES:
            try:
                result = yf.screen(query, count=SCREENER_COUNT)
                for q in (result or {}).get("quotes", []):
                    price = q.get("regularMarketPrice")
                    # Screens return individual stocks, so the $100 stock floor applies.
                    if price is not None and price >= max(MIN_SCREENER_PRICE, trade_rules.MIN_STOCK_PRICE):
                        add(q.get("symbol"), query)
            except Exception as e:
                notes.append(f"screener {query} failed: {e}")
    except Exception as e:
        notes.append(f"screeners unavailable: {e}")
    return found, notes


def _quoted(row: dict) -> bool:
    return all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) and row[k] > 0 for k in ("bid", "ask"))


def best_spreads(chain: dict, net_liq: float | None, why=None,
                 check_gap: bool = True) -> list[dict]:
    """The best qualifying bull put and bear call in one ticker's chain (0-2 results).
    Pass a collections.Counter as `why` to count each rule's rejections (--explain)."""
    def no(reason):
        if why is not None:
            why[reason] += 1
    today = dt.date.today()
    best: dict[str, dict] = {}
    for exp, data in (chain.get("expirations") or {}).items():
        try:
            dte = (dt.date.fromisoformat(exp) - today).days
        except ValueError:
            continue
        if not MIN_DTE <= dte <= MAX_DTE:
            no("expiration outside 25-45 days")
            continue
        for spread_type, rows, sign in (("bull_put", data.get("puts") or [], -1), ("bear_call", data.get("calls") or [], 1)):
            by_strike = {r["strike"]: r for r in rows}
            for short in rows:
                delta = short.get("delta")
                if delta is None or abs(delta) > MAX_SHORT_DELTA:
                    continue  # not a short-strike candidate at all; not counted
                if not _quoted(short):
                    no("short leg has no bid or ask")
                    continue
                if (short.get("openInterest") or 0) < MIN_OPEN_INTEREST:
                    no("short leg open interest under 100")
                    continue
                no(f"{spread_type} short strikes with delta <= 0.20 (checked)")
                for width in WIDTHS:
                    long_ = by_strike.get(round(short["strike"] + sign * width, 4))
                    if long_ is None:
                        no(f"no long strike ${width:g} away")
                        continue
                    if not _quoted(long_):
                        no("long leg has no bid or ask")
                        continue
                    if (long_.get("openInterest") or 0) < MIN_OPEN_INTEREST:
                        no("long leg open interest under 100")
                        continue
                    mid = (short["bid"] + short["ask"]) / 2 - (long_["bid"] + long_["ask"]) / 2
                    natural = short["bid"] - long_["ask"]
                    if mid <= 0:
                        no("no credit at the mid")
                        continue
                    if check_gap and trade_rules.wide_market_problem(mid, natural, width):
                        no("market too wide (natural >10% of width below mid)")
                        continue
                    credit = round(mid, 2)
                    premium = round(credit * 100, 2)
                    max_loss = round((width - credit) * 100, 2)
                    if premium < MIN_NET_PREMIUM:
                        no("premium under $50")
                        continue
                    if max_loss <= 0 or max_loss > premium * MAX_RISK_TO_PREMIUM:
                        no("max loss over 5x premium")
                        continue
                    if net_liq is not None and max_loss > net_liq * MAX_RISK_PCT_OF_NET_LIQ:
                        no("max loss over 8% of net liq")
                        continue
                    no("PASSED every rule")
                    cand = {
                        "symbol": chain.get("symbol"), "spread_type": spread_type,
                        "short_strike": short["strike"], "long_strike": long_["strike"],
                        "expiration": exp, "dte": dte, "credit": credit, "natural": round(natural, 2),
                        "premium": premium,
                        "max_loss": max_loss, "reward_to_risk": round(premium / max_loss, 3),
                        "short_delta": round(delta, 3),
                        "open_interest": [short.get("openInterest"), long_.get("openInterest")],
                        "underlying": round(float(chain.get("current_price") or 0), 2),
                    }
                    if cand["reward_to_risk"] > best.get(spread_type, {}).get("reward_to_risk", -1):
                        best[spread_type] = cand
    return list(best.values())


def scan(top_n: int = TOP_N, time_budget: float = TIME_BUDGET_SECONDS) -> dict:
    """Run the full scan. Never raises: problems come back in the report's notes."""
    import schwab_client

    started = time.monotonic()
    report = {"scanned_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "notes": []}
    if not schwab_client.is_configured():
        report["skipped"] = "Schwab market data isn't configured; the scanner needs it."
        return report

    candidates, notes = _candidates()
    report["notes"].extend(notes)

    mode = "simulated"
    net_liq, own_underlyings = None, set()
    try:
        import live_trading
        import schwab_trading
        mode = live_trading.execution_mode()
        account = schwab_trading.get_account(include_positions=True)
        net_liq = schwab_trading.risk_balances(account)["net_liq"]
        if mode == "approve":
            own_underlyings = live_trading.users_own_underlyings(
                schwab_trading.option_positions(account), live_trading.load_ledger())
    except Exception as e:
        report["notes"].append(f"couldn't read the Schwab account ({e}); 8%-of-net-liq cap not applied")
    for sym in own_underlyings:
        candidates.pop(sym, None)

    today = dt.date.today()
    window = (today + dt.timedelta(days=MIN_DTE), today + dt.timedelta(days=MAX_DTE))
    found, failed, scanned, under_price = [], 0, 0, 0
    symbols = list(candidates)

    def one(symbol):
        if time.monotonic() - started > time_budget:
            return symbol, None, "time"
        try:
            chain = schwab_client.get_option_chain(symbol, *window, strike_count=STRIKE_COUNT)
            if trade_rules.price_floor_problem(symbol, chain.get("current_price")):
                return symbol, None, "price"
            return symbol, best_spreads(chain, net_liq), None
        except Exception as e:
            return symbol, None, str(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for symbol, spreads, err in pool.map(one, symbols):
            if err == "time":
                continue
            scanned += 1
            if err == "price":
                under_price += 1
                continue
            if err is not None:
                failed += 1
                continue
            for s in spreads:
                s["sources"] = candidates[symbol]
                found.append(s)
    if scanned < len(symbols):
        report["notes"].append(f"time budget reached: scanned {scanned} of {len(symbols)} tickers "
                               "(watchlist, dark-pool and ETF tickers go first)")

    # Earnings, only for tickers that produced a spread: upcoming (through 20 days past
    # expiration) and recent (reported in the last 5 trading days, trade_rules.py). The
    # named ETFs have no earnings and are exempt; in approve mode an unknown date drops it.
    from options_trading_server import _get_next_earnings_date
    from universe import CATHIE_ETF_UNIVERSE
    tickers = sorted({s["symbol"] for s in found} - set(CATHIE_ETF_UNIVERSE))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        earnings = dict(zip(tickers, pool.map(_get_next_earnings_date, tickers)))
        last_earnings = dict(zip(tickers, pool.map(trade_rules.get_last_earnings_date, tickers)))
    kept, dropped_earnings, dropped_recent = [], 0, 0
    for s in found:
        if s["symbol"] in CATHIE_ETF_UNIVERSE:
            kept.append(s)
            continue
        e, last = earnings.get(s["symbol"]), last_earnings.get(s["symbol"])
        exp = dt.date.fromisoformat(s["expiration"])
        if e is not None and today <= e <= exp + dt.timedelta(days=EARNINGS_BUFFER_DAYS):
            dropped_earnings += 1
            continue
        if trade_rules.recent_earnings_problem(s["symbol"], last, today):
            dropped_recent += 1
            continue
        if e is None or last is None:
            if mode == "approve":
                dropped_earnings += 1
                continue
            s["earnings"] = "unknown"
        kept.append(s)

    kept.sort(key=lambda s: s["reward_to_risk"], reverse=True)
    report.update(
        candidates=len(symbols), scanned=scanned, chain_errors=failed,
        qualified=len(kept), dropped_for_earnings=dropped_earnings,
        dropped_recent_earnings=dropped_recent, under_price_floor=under_price,
        skipped_own_underlyings=sorted(own_underlyings),
        top=kept[:top_n], seconds=round(time.monotonic() - started),
    )
    return report


def summary_line(report: dict) -> str:
    if report.get("skipped"):
        return f"Spread scanner skipped: {report['skipped']}"
    top = ", ".join(f"{s['symbol']} {s['spread_type']} {s['short_strike']:g}/{s['long_strike']:g} "
                    f"{s['expiration']} @{s['credit']:.2f}" for s in report.get("top", [])[:5])
    return (f"Spread scanner: {report.get('scanned')} of {report.get('candidates')} tickers scanned in "
            f"{report.get('seconds')}s, {report.get('qualified')} qualifying spreads"
            + (f"; top: {top}" if top else ""))


def explain(symbols: list[str]) -> None:
    """Why each ticker did or didn't produce a spread: every rule's rejection count,
    plus earnings and whether it's one of the user's own underlyings."""
    import collections
    import schwab_client
    from options_trading_server import _get_next_earnings_date

    import live_trading
    mode = live_trading.execution_mode()
    net_liq, own = None, set()
    try:
        import schwab_trading
        account = schwab_trading.get_account(include_positions=True)
        net_liq = schwab_trading.risk_balances(account)["net_liq"]
        if mode == "approve":
            own = live_trading.users_own_underlyings(schwab_trading.option_positions(account), live_trading.load_ledger())
    except Exception as e:
        print(f"(couldn't read the account: {e}; 8% cap not applied)")
    today = dt.date.today()
    window = (today + dt.timedelta(days=MIN_DTE), today + dt.timedelta(days=MAX_DTE))
    for symbol in symbols:
        symbol = symbol.upper()
        print(f"\n=== {symbol}")
        if symbol in own:
            print("  SKIPPED: you hold your own options on it (approve mode skips these)")
        try:
            chain = schwab_client.get_option_chain(symbol, *window, strike_count=STRIKE_COUNT)
        except Exception as e:
            print(f"  no chain from Schwab: {e}")
            continue
        exps = sorted(e for e in chain.get("expirations", {}) if MIN_DTE <= (dt.date.fromisoformat(e) - today).days <= MAX_DTE)
        print(f"  underlying ${chain.get('current_price', 0):,.2f}; expirations in window: {', '.join(exps) or 'none'}")
        floor = trade_rules.price_floor_problem(symbol, chain.get("current_price"))
        if floor:
            print(f"  SKIPPED: {floor}")
        why = collections.Counter()
        spreads = best_spreads(chain, net_liq, why)
        for reason, n in why.most_common():
            print(f"  {n:>5}  {reason}")
        from universe import CATHIE_ETF_UNIVERSE
        is_etf = symbol in CATHIE_ETF_UNIVERSE
        e = None if is_etf else _get_next_earnings_date(symbol)
        last = None if is_etf else trade_rules.get_last_earnings_date(symbol)
        if is_etf:
            print("  earnings: none (named ETF, exempt)")
        else:
            print(f"  next earnings: {e or 'unknown'}; last reported: {last or 'unknown'}")
            recent = trade_rules.recent_earnings_problem(symbol, last, today)
            if recent:
                print(f"  -> DROPPED: {recent}")
        for s in spreads:
            exp = dt.date.fromisoformat(s["expiration"])
            blocked = not is_etf and e is not None and today <= e <= exp + dt.timedelta(days=EARNINGS_BUFFER_DAYS)
            print(f"  best {s['spread_type']}: {s['short_strike']:g}/{s['long_strike']:g} {s['expiration']} "
                  f"credit {s['credit']:.2f}, max loss ${s['max_loss']:.0f}"
                  + ("  -> DROPPED: earnings within expiration + 20 days" if blocked else "")
                  + ("  -> DROPPED in approve mode: earnings date unknown"
                     if not is_etf and (e is None or last is None) and mode == "approve" else ""))
        if not spreads:
            print("  no spread passed every rule")
        # Nearest miss on the wide-market rule, per direction with no passing spread: the best
        # spread if that rule were off, with its numbers, so the threshold can be judged from
        # real quotes.
        passed = {s["spread_type"] for s in spreads}
        for s in best_spreads(chain, net_liq, check_gap=False):
            if s["spread_type"] not in passed:
                gap = s["credit"] - s["natural"]
                width = abs(s["short_strike"] - s["long_strike"])
                print(f"  nearest miss {s['spread_type']}: {s['short_strike']:g}/{s['long_strike']:g} "
                      f"{s['expiration']} mid {s['credit']:.2f}, natural {s['natural']:.2f} "
                      f"(${gap:.2f} below mid = {gap / width:.1%} of width; limit {trade_rules.MAX_GAP_PCT_OF_WIDTH:.0%}), "
                      f"max loss ${s['max_loss']:.0f}, delta {s['short_delta']}, OI {s['open_interest'][0]}/{s['open_interest'][1]}")


def main():
    import argparse
    import logging
    parser = argparse.ArgumentParser(description="Scan for credit spreads that pass Cathie's hard rules.")
    parser.add_argument("--explain", nargs="+", metavar="TICKER",
                        help="instead of a full scan, show which rule each of these tickers failed")
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per chain call otherwise
    if args.explain:
        explain(args.explain)
        return
    report = scan()
    print(summary_line(report))
    for note in report.get("notes", []):
        print(f"  note: {note}")
    if report.get("skipped_own_underlyings"):
        print(f"  skipped (you hold options on them): {', '.join(report['skipped_own_underlyings'])}")
    print(f"  chain errors: {report.get('chain_errors', 0)}, under ${trade_rules.MIN_STOCK_PRICE:.0f}: "
          f"{report.get('under_price_floor', 0)}, dropped for upcoming/unknown earnings: "
          f"{report.get('dropped_for_earnings', 0)}, reported in the last "
          f"{trade_rules.RECENT_EARNINGS_TRADING_DAYS} trading days: {report.get('dropped_recent_earnings', 0)}")
    print()
    for i, s in enumerate(report.get("top", []), 1):
        print(f"{i:>2}. {s['symbol']:<6} {s['spread_type']:<9} {s['short_strike']:g}/{s['long_strike']:g}  "
              f"exp {s['expiration']} ({s['dte']}d)  credit {s['credit']:.2f} = ${s['premium']:.0f}  "
              f"max loss ${s['max_loss']:.0f}  ratio {s['reward_to_risk']:.2f}  delta {s['short_delta']}  "
              f"OI {s['open_interest'][0]}/{s['open_interest'][1]}  [{', '.join(s['sources'])}]"
              + ("  earnings: unknown" if s.get("earnings") == "unknown" else ""))


if __name__ == "__main__":
    main()
