"""
Data for the dashboard's "Real Schwab account" panel (app.py): every option position
actually held at Schwab, labelled CATHIE (in her cathie_live ledger) or MANUAL (the
user's own), with status and P&L, plus Cathie's recent real trades that aren't open
(staged, sent, closing, closed). Read-only: two Schwab reads (account + recent orders)
per refresh, nothing is ever sent.

P&L comes from Schwab's own position marks (marketValue) and average prices, so no
extra quote calls are needed.
"""
import datetime as dt

COLUMNS = ["Owner", "Sym", "Type", "Strikes", "Qty", "Expiry", "DTE", "Status",
           "Credit", "Now", "P&L", "Note"]
WORKING = {"WORKING", "QUEUED", "PENDING_ACTIVATION", "ACCEPTED", "AWAITING_MANUAL_REVIEW",
           "NEW", "AWAITING_RELEASE_TIME", "PENDING_ACKNOWLEDGEMENT", "AWAITING_PARENT_ORDER",
           "AWAITING_CONDITION", "PENDING_REPLACE"}
SHOW_DONE_DAYS = 14  # Cathie's closed/not-filled/rejected trades from the last two weeks


def _money(x):
    return "" if x is None else (f"+${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}")


def _exp(text):
    try:
        return dt.date.fromisoformat(text).strftime("%m/%d/%y")
    except Exception:
        return text or ""


def _dte(text, today):
    try:
        return (dt.date.fromisoformat(text) - today).days
    except Exception:
        return ""


def _closing_orders(orders: list[dict]) -> dict:
    """Leg symbol -> (status, price) of a working closing order that includes it."""
    out = {}
    for o in orders:
        if o.get("status") not in WORKING:
            continue
        legs = o.get("legs") or []
        if legs and all((leg.get("instruction") or "").endswith("_TO_CLOSE") for leg in legs):
            for leg in legs:
                out[leg.get("symbol")] = (o.get("status"), o.get("price"))
    return out


def _pair_legs(legs: list[dict]) -> list[tuple]:
    """Pair short and long contracts of one underlying/expiration/put-call into spreads:
    each short with the nearest long strike. Returns (short, long, qty) tuples, with
    None for a leg that has no partner (a single option)."""
    shorts = sorted([dict(p, left=-p["quantity"]) for p in legs if p["quantity"] < 0], key=lambda p: p["strike"])
    longs = [dict(p, left=p["quantity"]) for p in legs if p["quantity"] > 0]
    pairs = []
    for sh in shorts:
        while sh["left"] > 0:
            candidates = [lg for lg in longs if lg["left"] > 0]
            if not candidates:
                break
            lg = min(candidates, key=lambda lg: abs(lg["strike"] - sh["strike"]))
            qty = min(sh["left"], lg["left"])
            pairs.append((sh, lg, qty))
            sh["left"] -= qty
            lg["left"] -= qty
    pairs += [(sh, None, sh["left"]) for sh in shorts if sh["left"] > 0]
    pairs += [(None, lg, lg["left"]) for lg in longs if lg["left"] > 0]
    return pairs


def _spread_type(sh, lg, option_type):
    if sh and lg:
        if option_type == "put":
            return "BP" if lg["strike"] < sh["strike"] else "Put debit"
        return "BC" if lg["strike"] > sh["strike"] else "Call debit"
    return ("Short " if sh else "Long ") + option_type


def _manual_note(sh, lg, credit, option_type, qty, auto_close: bool) -> str:
    if not auto_close:
        return "yours; never touched by Cathie"
    clean = (sh and lg and credit is not None and credit > 0 and qty == -sh["quantity"] == lg["quantity"]
             and (lg["strike"] < sh["strike"] if option_type == "put" else lg["strike"] > sh["strike"]))
    if not clean:
        return "yours; not a simple credit spread, so never auto-closed"
    return f"yours; auto-close at <= {credit * 0.25:.2f} (75% profit) only"


def build(account_positions: list[dict], orders: list[dict], ledger, today: dt.date | None = None,
          auto_close: bool = False, hands_off: set | None = None) -> list[dict]:
    """Dashboard rows. account_positions: schwab_trading.option_positions(); orders:
    schwab_trading.recent_orders(); ledger: live_trading.LiveLedger. auto_close: the
    user turned on AUTO_CLOSE_MANUAL_AT_PROFIT."""
    hands_off = hands_off or set()
    today = today or dt.date.today()
    by_symbol = {p["symbol"]: p for p in account_positions}
    closing = _closing_orders(orders)
    rows = []

    # Cathie's spreads that hold (or should hold) contracts at Schwab.
    cathie_syms = set()
    for s in ledger.spreads:
        if s.status not in ("open", "closing"):
            continue
        cathie_syms |= {s.short_symbol, s.long_symbol}
        sh, lg = by_symbol.get(s.short_symbol), by_symbol.get(s.long_symbol)
        credit = s.fill_credit or s.limit_credit
        now = pnl = None
        if sh and lg and sh.get("market_value") is not None and lg.get("market_value") is not None:
            now = -(sh["market_value"] + lg["market_value"]) / (100 * s.contracts)
            pnl = (credit - now) * 100 * s.contracts if credit is not None else None
        if s.status == "closing":
            status = f"CLOSING ({s.close_rule}) @{s.close_limit_debit:.2f}"
        elif not (sh and lg):
            status = "OPEN (legs not found at Schwab)"
        else:
            status = "OPEN"
        target = credit * 0.25 if credit else None
        rows.append({
            "Owner": "CATHIE", "Sym": s.symbol, "Type": "BP" if s.spread_type == "bull_put" else "BC",
            "Strikes": f"{s.short_strike:g}/{s.long_strike:g}", "Qty": s.contracts,
            "Expiry": _exp(s.expiration_date), "DTE": _dte(s.expiration_date, today), "Status": status,
            "Credit": f"{credit:.2f}" if credit else "", "Now": f"{now:.2f}" if now is not None else "",
            "P&L": _money(pnl),
            "Note": ("HANDS OFF (hands_off.txt): never closed automatically; you close it"
                     if s.symbol in hands_off else
                     f"auto-close at <= {target:.2f}, breach of {s.short_strike:g}, or 7 DTE" if target else ""),
        })

    # Everything else at Schwab is the user's own.
    manual = [p for p in account_positions if p["symbol"] not in cathie_syms and p.get("quantity")]
    groups = {}
    for p in manual:
        groups.setdefault((p["underlying"], p["expiration"], p["option_type"]), []).append(p)
    for (underlying, expiration, option_type), legs in sorted(groups.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")):
        for sh, lg, qty in _pair_legs(legs):
            credit = now = pnl = None
            if sh and lg:
                credit = (sh.get("average_price") or 0) - (lg.get("average_price") or 0)
                if sh.get("market_value") is not None and lg.get("market_value") is not None:
                    # Market values cover the whole position; scale to the paired quantity.
                    sh_mv = sh["market_value"] * qty / abs(sh["quantity"])
                    lg_mv = lg["market_value"] * qty / abs(lg["quantity"])
                    now = -(sh_mv + lg_mv) / (100 * qty)
                    pnl = (credit - now) * 100 * qty
                strikes = f"{sh['strike']:g}/{lg['strike']:g}"
            else:
                leg = sh or lg
                strikes = f"{leg['strike']:g}"
                if leg.get("market_value") is not None and leg.get("average_price") is not None:
                    mv = leg["market_value"] * qty / abs(leg["quantity"])
                    cost = leg["average_price"] * 100 * qty
                    pnl = (cost + mv) if sh else (mv - cost)
                    now = abs(mv) / (100 * qty)
                    credit = leg["average_price"]
            working = next((closing[x["symbol"]] for x in (sh, lg) if x and x["symbol"] in closing), None)
            status = f"CLOSING (order {working[0]} @{working[1]})" if working else "OPEN"
            rows.append({
                "Owner": "MANUAL", "Sym": underlying, "Type": _spread_type(sh, lg, option_type),
                "Strikes": strikes, "Qty": int(qty), "Expiry": _exp(expiration), "DTE": _dte(expiration, today),
                "Status": status, "Credit": f"{credit:.2f}" if credit is not None else "",
                "Now": f"{now:.2f}" if now is not None else "", "P&L": _money(pnl),
                "Note": ("yours; HANDS OFF (hands_off.txt): never auto-closed" if auto_close and underlying in hands_off
                         else _manual_note(sh, lg, credit, option_type, qty, auto_close)),
            })

    # Cathie's recent trades that aren't held: waiting, sent, or finished.
    cutoff = today - dt.timedelta(days=SHOW_DONE_DAYS)
    for s in reversed(ledger.spreads):
        if s.status in ("open", "closing"):
            continue
        when = (s.closed_at or s.submitted_at or s.staged_at or "")[:10]
        try:
            if dt.date.fromisoformat(when) < cutoff:
                continue
        except ValueError:
            pass
        credit = s.fill_credit or s.limit_credit or s.staged_credit
        note = s.events[-1][20:] if s.events else ""
        rows.append({
            "Owner": "CATHIE", "Sym": s.symbol, "Type": "BP" if s.spread_type == "bull_put" else "BC",
            "Strikes": f"{s.short_strike:g}/{s.long_strike:g}", "Qty": s.contracts,
            "Expiry": _exp(s.expiration_date), "DTE": _dte(s.expiration_date, today),
            "Status": s.status.upper().replace("_", " ") + (f" ({s.close_rule})" if s.close_rule else ""),
            "Credit": f"{credit:.2f}" if credit else "",
            "Now": f"{s.close_fill_debit:.2f}" if s.close_fill_debit is not None else "",
            "P&L": _money(s.realized_pnl), "Note": note[:120],
        })
    return rows


def to_html(rows: list[dict]) -> str:
    """Compact HTML table for the dashboard, CATHIE and MANUAL rows colour-coded.
    Dark, semi-bold text on a light background: the user found the earlier pale-grey
    text (especially the Note column) hard to read."""
    import html
    if not rows:
        return "<div style='text-align:center;color:#222;font-weight:600;padding:5px;'>No real positions</div>"
    owner_style = {"CATHIE": "background:#6c3483;", "MANUAL": "background:#566573;"}
    status_color = {"OPEN": "#1e8449", "CLOSING": "#b9770e", "CLOSED": "#444", "STAGED": "#1f618d",
                    "PENDING": "#1f618d", "NOT": "#444", "REJECTED": "#444", "EXPIRED": "#444"}
    head = "".join(f"<th style='padding:3px 5px;text-align:left;font-weight:800;'>{c}</th>" for c in COLUMNS)
    body = []
    for r in rows:
        cells = []
        for c in COLUMNS:
            v = html.escape(str(r.get(c, "")))
            style = "padding:2px 5px;white-space:nowrap;"
            if c == "Owner":
                v = f"<span style='{owner_style.get(r['Owner'], '')}color:#fff;border-radius:3px;padding:0 5px;'>{v}</span>"
            elif c == "Status":
                style = (f"padding:2px 5px;max-width:150px;overflow-wrap:anywhere;color:{status_color.get(str(r['Status']).split(' ')[0], '#111')};"
                         "font-weight:800;")
            elif c == "P&L" and v:
                style += "color:#1e8449;font-weight:800;" if v.startswith("+") else "color:#c0392b;font-weight:800;"
            elif c == "Note":
                style = "padding:2px 5px;color:#111;min-width:260px;"
            cells.append(f"<td style='{style}'>{v}</td>")
        body.append("<tr style='border-top:1px solid #bbb;'>" + "".join(cells) + "</tr>")
    return ("<div style='max-height:360px;overflow:auto;font-size:13px;font-weight:600;color:#111;"
            "background:#fafafa;border-radius:4px;'>"
            "<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr style='color:#000;background:#e5e5e5;'>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def summary(rows: list[dict], balances: dict | None, ledger, error: str | None = None) -> dict:
    def pnl_sum(owner, statuses):
        total = 0.0
        for r in rows:
            if r["Owner"] == owner and r["Status"].split(" ")[0] in statuses and r["P&L"]:
                total += float(r["P&L"].replace("$", "").replace(",", ""))
        return total
    import live_trading
    return {
        "funds_available": (balances or {}).get("funds_available"),
        "net_liq": (balances or {}).get("net_liq"),
        "cathie_open": sum(1 for r in rows if r["Owner"] == "CATHIE" and r["Status"].split(" ")[0] in ("OPEN", "CLOSING")),
        "cathie_open_pnl": pnl_sum("CATHIE", ("OPEN", "CLOSING")),
        "cathie_realized": sum(s.realized_pnl or 0 for s in ledger.spreads if s.status == "closed"),
        "manual_open": sum(1 for r in rows if r["Owner"] == "MANUAL"),
        "manual_open_pnl": pnl_sum("MANUAL", ("OPEN", "CLOSING")),
        "manual_auto_closed_pnl": sum(m.realized_pnl or 0 for m in getattr(ledger, "manual_closes", [])
                                      if m.status == "filled"),
        "orders_today": live_trading.orders_sent_today(ledger),
        "daily_limit": live_trading.MAX_ORDERS_PER_DAY,
        "error": error,
    }


def fetch() -> tuple[list[dict], dict]:
    """Read Schwab and the ledger once; (rows, summary). Never raises: errors are
    reported in summary['error'] with whatever could still be shown."""
    import live_trading
    ledger = live_trading.load_ledger()
    positions, orders, balances, errors = [], [], None, []
    try:
        import schwab_trading
        account = schwab_trading.get_account(include_positions=True)
        positions = schwab_trading.option_positions(account)
        balances = schwab_trading.risk_balances(account)
        live_trading.record_account_value(balances.get("net_liq"))  # the top chart's history
    except Exception as e:
        errors.append(f"account: {e}")
    try:
        import schwab_trading
        orders = schwab_trading.recent_orders(days=2)
    except Exception as e:
        errors.append(f"orders: {e}")
    error = "; ".join(errors) or None
    if error and ("invalid_grant" in error or "Refresh token is invalid" in error):
        error = ("Schwab login expired or was revoked: run `uv run schwab_auth_setup.py`. "
                 "This panel reconnects by itself within a minute after that. "
                 "Positions can't be read until then, so OPEN rows may wrongly say legs weren't found.")
    rows = build(positions, orders, ledger, auto_close=live_trading.auto_close_manual(),
                 hands_off=live_trading.hands_off_tickers())
    return rows, summary(rows, balances, ledger, error)
