#!/usr/bin/env python
"""
TradeAlgo dark-pool scraper -- pulls data from YOUR OWN logged-in TradeAlgo
account, at most twice a day, for Cathie's research only.

Permission: TradeAlgo support informally OK'd automated access to data already
available in this account, for a private trading bot, at a once-or-twice-a-day
cadence. That is the scope this script is built to stay inside -- keep a copy
of that support conversation, since it's informal, not a formal agreement:
  - it only ever uses your own logged-in browser session (you log in by hand;
    this script never sees or stores your password),
  - it only reads pages your account can already see,
  - it refuses to fetch more than MAX_FETCHES_PER_DAY times per calendar day,
    or sooner than MIN_HOURS_BETWEEN_FETCHES after the last attempt.

Runs on your own machine (it needs a real browser for the one-time login), from
inside 6_mcp/. One-time setup:
    uv sync
    uv run playwright install chromium

Modes:
    uv run tradealgo_scraper.py login
        Opens a browser window on TradeAlgo (the first time, type your usual
        TradeAlgo address into it -- it's remembered). Log in by hand, then press Enter
        in the terminal. The session is saved to a private browser profile
        (.tradealgo_profile/, gitignored) so later runs stay logged in.

    uv run tradealgo_scraper.py discover
        Opens the same logged-in browser. Navigate to the dark-pool page and
        let it load, then press Enter. Writes tradealgo_discovery.json: which
        data requests the page made and the *field names/types* in each
        response -- no values, no cookies, no tokens, and account-ID-looking
        URL segments masked -- so it's safe to share for building the parser.

    uv run tradealgo_scraper.py fetch
        Headless, no window. Loads TradeAlgo's two dark-pool pages (Intraday
        and Historical -- see FETCH_SOURCES), captures each page's own data
        request, and saves a compact per-ticker summary to
        tradealgo_darkpool.json (the Intraday page's 20 tickers, and the first
        20 on the Historical page). Enforces the twice-a-day cap. Meant to be
        scheduled (e.g. Windows Task Scheduler), not run by the trading floor
        itself -- a scraper problem should never be able to stall a trading
        cycle. Cathie reads the saved file via options_trading_server.py's
        get_dark_pool_activity tool, never TradeAlgo directly.

Deliberately independent of the trading floor: nothing here is imported by
trading_floor.py or any MCP server.
"""
import datetime as dt
import json
import os
import re
import sys
from urllib.parse import urlsplit

from dotenv import load_dotenv

load_dotenv(override=True)

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(HERE, ".tradealgo_profile")
DISCOVERY_PATH = os.path.join(HERE, "tradealgo_discovery.json")
CACHE_PATH = os.path.join(HERE, "tradealgo_darkpool.json")
# Last fetch's unparsed responses, kept only to debug the parser if TradeAlgo changes shape.
RAW_PATH = os.path.join(HERE, ".tradealgo_darkpool_raw.json")
STATE_PATH = os.path.join(HERE, ".tradealgo_fetch_state.json")

# Where `login`/`discover` open the browser. No built-in default -- the one originally
# guessed (app.tradealgo.com) doesn't exist. If unset, the first `login` asks you to
# type your usual TradeAlgo address into the browser window, then remembers it.
TRADEALGO_HOME = os.getenv("TRADEALGO_HOME_URL")
# What `fetch` captures, identified from the user's `discover` run (2026-09-29): both
# dark-pool pages, each loading one JSON request. Env vars override if TradeAlgo moves them.
FETCH_SOURCES = [
    {
        "name": "intraday",  # today's flagged tickers: 10 trending up + 10 trending down
        "page": os.getenv("TRADEALGO_PAGE_URL", "https://dashboard.tradealgo.com/home/Intraday/Auto/Up"),
        "match": os.getenv("TRADEALGO_DATA_URL_CONTAINS", "/ats/darkflow"),
    },
    {
        "name": "historical",  # longer list; includes tickers flagged on earlier days
        "page": os.getenv(
            "TRADEALGO_HISTORY_PAGE_URL", "https://dashboard.tradealgo.com/historical/Auto/Up"
        ),
        "match": os.getenv("TRADEALGO_HISTORY_DATA_URL_CONTAINS", "/ats/historic/daily-darkflow"),
    },
]
HISTORY_LIMIT = 20  # "the first 20 tickers" on the Historical page, as the user asked
# Only for environments where Playwright's own bundled Chromium isn't installed
# (e.g. a sandbox with a system Chromium). Normal use: leave unset and run
# `uv run playwright install chromium` once.
CHROMIUM_PATH = os.getenv("TRADEALGO_CHROMIUM_PATH")

# The agreed cadence with TradeAlgo support: once or twice a day.
MAX_FETCHES_PER_DAY = 2
MIN_HOURS_BETWEEN_FETCHES = 4.0
FETCH_TIMEOUT_MS = 60_000

# Discovery output limits -- enough to see a response's structure, small enough
# to paste back into a chat.
_SHAPE_MAX_DEPTH = 6
_SHAPE_MAX_KEYS = 60


class RateCapError(RuntimeError):
    """This fetch would exceed the agreed twice-a-day cadence."""


class NotLoggedInError(RuntimeError):
    """The saved session is gone or expired -- re-run `login`."""


# ---------------------------------------------------------------- rate cap

def _load_attempts(now: dt.datetime) -> list[dt.datetime]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f).get("attempts", [])
        attempts = [dt.datetime.fromisoformat(t) for t in raw]
    except (OSError, ValueError, TypeError):
        attempts = []
    # Keep only what's still relevant to the checks below.
    return [t for t in attempts if now - t < dt.timedelta(days=2)]


def check_rate_cap(now: dt.datetime | None = None) -> None:
    """Raise RateCapError if another fetch now would break the agreed cadence.

    Counts *attempts*, not just successes: a fetch that reaches TradeAlgo and
    then fails still made a request, so a flaky page can't turn into repeated
    retries against their site."""
    now = now or dt.datetime.now()
    attempts = _load_attempts(now)
    today = [t for t in attempts if t.date() == now.date()]
    if len(today) >= MAX_FETCHES_PER_DAY:
        raise RateCapError(
            f"Already fetched {len(today)} time(s) today (cap is {MAX_FETCHES_PER_DAY}/day, "
            "the cadence agreed with TradeAlgo). Try again tomorrow."
        )
    if attempts:
        since = now - max(attempts)
        if since < dt.timedelta(hours=MIN_HOURS_BETWEEN_FETCHES):
            wait_h = MIN_HOURS_BETWEEN_FETCHES - since.total_seconds() / 3600
            raise RateCapError(
                f"Last fetch attempt was {since.total_seconds() / 3600:.1f}h ago; need at least "
                f"{MIN_HOURS_BETWEEN_FETCHES:.0f}h between fetches. Try again in ~{wait_h:.1f}h."
            )


def record_attempt(now: dt.datetime | None = None) -> None:
    now = now or dt.datetime.now()
    attempts = _load_attempts(now) + [now]
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"attempts": [t.isoformat(timespec="seconds") for t in attempts]}, f)


# ---------------------------------------------------------------- discovery helpers

_ID_SEGMENT = re.compile(
    r"^(\d{4,}|[0-9a-fA-F]{16,}|[0-9a-fA-F-]{32,36}|[^/]*@[^/]*)$"  # long ids, hex, uuids, emails
)


def mask_url(url: str) -> str:
    """scheme://host/path with the query string dropped (tokens often live there)
    and account-ID-looking path segments replaced with {id}."""
    parts = urlsplit(url)
    segments = ["{id}" if _ID_SEGMENT.match(s) else s for s in parts.path.split("/")]
    return f"{parts.scheme}://{parts.netloc}{'/'.join(segments)}"


def json_shape(value, depth: int = 0):
    """Structure of a JSON value -- key names and types only, never values."""
    if depth >= _SHAPE_MAX_DEPTH:
        return "..."
    if isinstance(value, dict):
        keys = list(value)[:_SHAPE_MAX_KEYS]
        shape = {k: json_shape(value[k], depth + 1) for k in keys}
        if len(value) > _SHAPE_MAX_KEYS:
            shape["..."] = f"{len(value) - _SHAPE_MAX_KEYS} more keys"
        return shape
    if isinstance(value, list):
        return {"list_len": len(value), "item": json_shape(value[0], depth + 1) if value else None}
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "null" if value is None else type(value).__name__


# ---------------------------------------------------------------- browser

def _session_file() -> str:
    # Inside the (gitignored) profile dir. Holds your TradeAlgo login cookies --
    # as sensitive as a password: never commit or share it.
    return os.path.join(PROFILE_DIR, "session_state.json")


def _open_context(playwright, headless: bool):
    kwargs = {"headless": headless}
    if CHROMIUM_PATH:
        kwargs["executable_path"] = CHROMIUM_PATH
    # A persistent profile keeps localStorage and long-lived cookies between runs.
    # Chromium locks a profile while it's open, so two runs can't overlap.
    context = playwright.chromium.launch_persistent_context(PROFILE_DIR, **kwargs)
    # Chromium drops *session* cookies (ones with no expiry) when the browser closes,
    # even with a persistent profile -- if the site's login uses one, every later run
    # would look logged out. So cookies are saved explicitly at the end of each run
    # (_save_session) and restored here.
    try:
        with open(_session_file(), "r", encoding="utf-8") as f:
            context.add_cookies(json.load(f).get("cookies", []))
    except (OSError, ValueError):
        pass
    return context


def _home_file() -> str:
    return os.path.join(PROFILE_DIR, "home_url.txt")


def _get_home_url() -> str | None:
    if TRADEALGO_HOME:
        return TRADEALGO_HOME
    try:
        with open(_home_file(), "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _remember_home(page_url: str) -> str:
    parts = urlsplit(page_url)
    origin = f"{parts.scheme}://{parts.netloc}"
    with open(_home_file(), "w", encoding="utf-8") as f:
        f.write(origin)
    return origin


def _is_web_page(url: str) -> bool:
    return urlsplit(url).scheme in ("http", "https")


def _open_start_page(page, url: str | None):
    """Navigate to `url`. If there's no address yet, or it doesn't load, leave the
    window open for you to type the right address yourself instead of crashing.

    Returns (loaded, page) -- `page` may be a fresh tab replacing the original."""
    from playwright.sync_api import Error as PlaywrightError

    if not url:
        print("\nNo TradeAlgo address saved yet. In the browser window, type the address you "
              "normally use to log in to TradeAlgo into the address bar.")
        return False, page
    try:
        page.goto(url)
        return True, page
    except PlaywrightError as e:
        print(f"\nCouldn't open {url} ({str(e).splitlines()[0]}).\n"
              "In the browser window, type the address you normally use for TradeAlgo into "
              "the address bar instead. (To stop seeing this, fix or remove TRADEALGO_HOME_URL "
              "in .env.)")
        # Chromium commits its own error page a moment after a failed load, and that can
        # cut off whatever navigation comes next in the same tab. Swap in a fresh tab so
        # nothing is left pending.
        fresh = page.context.new_page()
        page.close()
        return False, fresh


def _save_session(context) -> None:
    """Persist cookies (including session cookies) for the next run. Called only
    after a run that was actually logged in, so a logged-out run can't overwrite a
    good session with a bad one."""
    context.storage_state(path=_session_file())


_LOGIN_SEGMENTS = {"login", "log-in", "signin", "sign-in", "sign_in", "auth", "sso", "oauth"}


def _looks_logged_out(url: str) -> bool:
    # Whole path segments only (e.g. /login, /auth/callback), not substrings -- a
    # plain substring check for "auth" also matched ordinary logged-in app pages.
    segments = [s for s in urlsplit(url).path.lower().split("/") if s]
    return any(seg in _LOGIN_SEGMENTS or seg.startswith(("login", "signin")) for seg in segments)


def _logged_in_page(context):
    """The most recently opened tab that's on a real, non-login web page -- or None.
    Checks every tab, since the login may happen in a new tab or a popup rather than
    the one this script opened."""
    for page in reversed(context.pages):
        if _is_web_page(page.url) and not _looks_logged_out(page.url):
            return page
    return None


def _describe_tabs(context) -> str:
    urls = [mask_url(p.url) if _is_web_page(p.url) else p.url for p in context.pages]
    return "; ".join(urls) or "(no tabs open)"


def _wait_for_enter(prompt: str, page=None) -> None:
    input(prompt)


def _ask_yes_no(prompt: str, page=None) -> bool:
    return input(prompt).strip().lower() in ("y", "yes")


def _pick_page(context, confirm):
    """The tab to treat as logged in, or None.

    Address patterns are only a hint: some sites keep "signin" (or similar) in the
    address even after you're logged in. You're at the keyboard in `login`/`discover`,
    so when the address looks like a sign-in page, ask rather than refuse outright."""
    page = _logged_in_page(context)
    if page is not None:
        return page
    web_tabs = [p for p in reversed(context.pages) if _is_web_page(p.url)]
    if web_tabs and confirm(
        f"\nThe tab's address ({mask_url(web_tabs[0].url)}) looks like a sign-in page.\n"
        "Are you logged in and looking at your TradeAlgo dashboard/page right now? [y/N] ",
        web_tabs[0],
    ):
        return web_tabs[0]
    return None


def run_login(headless: bool = False, wait=_wait_for_enter, confirm=_ask_yes_no) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        context = _open_context(p, headless)
        page = context.pages[0] if context.pages else context.new_page()
        _, page = _open_start_page(page, _get_home_url())
        wait(
            "\nLog in to TradeAlgo in the browser window by hand (this script never sees\n"
            "your password), wait until your dashboard loads, then press Enter here... ",
            page,
        )
        page = _pick_page(context, confirm)
        if page is None:
            tabs = _describe_tabs(context)
            context.close()
            raise NotLoggedInError(
                "No open tab was on a logged-in TradeAlgo page when you pressed Enter "
                f"(open tabs: {tabs}). Run `login` again and wait for your dashboard before "
                "pressing Enter."
            )
        _save_session(context)
        home = _remember_home(page.url)
        context.close()
    print(f"Saved your TradeAlgo session to {PROFILE_DIR} (TradeAlgo address: {home})")


def run_discover(
    headless: bool = False, wait=_wait_for_enter, start_url: str | None = None, confirm=_ask_yes_no
) -> dict:
    from playwright.sync_api import sync_playwright

    seen: dict[tuple, dict] = {}

    def on_response(resp):
        try:
            if "json" not in (resp.headers.get("content-type") or ""):
                return
            body = resp.json()
        except Exception:
            return  # not parseable JSON, or the body is already gone -- skip
        key = (resp.request.method, mask_url(resp.url))
        if key in seen:
            seen[key]["times_seen"] += 1
            return
        seen[key] = {
            "method": key[0],
            "url": key[1],
            "status": resp.status,
            "times_seen": 1,
            "shape": json_shape(body),
        }

    with sync_playwright() as p:
        context = _open_context(p, headless)
        context.on("response", on_response)
        page = context.pages[0] if context.pages else context.new_page()
        _, page = _open_start_page(page, start_url or _get_home_url())
        wait(
            "\nIn the browser window, go to TradeAlgo's dark-pool page and let it fully load\n"
            "(scroll a little if the table loads more as you go). Then press Enter here... ",
            page,
        )
        page = _pick_page(context, confirm)
        if page is None:
            tabs = _describe_tabs(context)
            context.close()
            raise NotLoggedInError(
                f"No open tab was on a logged-in TradeAlgo page (open tabs: {tabs}). Run "
                "`login` first, then `discover` again."
            )
        # Enter may come before the page's own data requests finish -- give them a moment
        # so they're captured. A page that never goes quiet (live-updating) is fine too.
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass
        final_page_url = mask_url(page.url)
        _save_session(context)
        context.close()

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "note": (
            "Field names and types only -- no values, cookies, or tokens. Query strings "
            "dropped and account-ID-looking URL segments masked as {id}."
        ),
        "page_url": final_page_url,
        "json_responses": sorted(seen.values(), key=lambda r: r["url"]),
    }
    with open(DISCOVERY_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {DISCOVERY_PATH} ({len(seen)} distinct JSON responses captured).")
    return report


# ---------------------------------------------------------------- parsing
# Field names come from the user's discovery output; values were never seen from here,
# so every lookup is defensive (missing/odd fields become None rather than errors).

def _get(d, *path):
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def _num(v, ndigits=2):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return int(round(f)) if ndigits == 0 else round(f, ndigits)


def _flag_fields(rec: dict) -> dict:
    """The compact per-ticker summary shared by both pages. Drops the chart arrays
    (hundreds of numbers per ticker) and duplicated nested ticker/date fields."""
    return {
        "ticker": rec.get("ticker"),
        "name": rec.get("name"),
        "multiplier": rec.get("multiplier"),
        "dollar_value": rec.get("dollar_value"),
        "date_flagged": rec.get("date_flagged"),
        "perf": _num(rec.get("perf"), 4),
        "last_price": _num(rec.get("last_price")),
        "algo_price": _num(rec.get("algo_price")),
        "market_cap": rec.get("market_cap"),
        "dark_pool": {
            "date": _get(rec, "ats", "date"),
            "day_trades": _num(_get(rec, "ats", "current", "day_trades"), 0),
            "day_volume": _num(_get(rec, "ats", "current", "day_volume"), 0),
            "day_dollar_volume": _num(_get(rec, "ats", "current", "day_dollar_volume"), 0),
            "prev_day_dollar_volume": _num(_get(rec, "ats", "previous", "day_dollar_volume"), 0),
            "compared_day_dollar_volume": _num(_get(rec, "ats", "compared", "day_dollar_volume"), 4),
            "compared_day_volume": _num(_get(rec, "ats", "compared", "day_volume"), 4),
        },
        "options_flow": {
            "date": _get(rec, "options", "date"),
            "call_count": _num(_get(rec, "options", "call_count"), 0),
            "call_total_prem": _num(_get(rec, "options", "call_total_prem"), 0),
            "put_count": _num(_get(rec, "options", "put_count"), 0),
            "put_total_prem": _num(_get(rec, "options", "put_total_prem"), 0),
            "put_to_call": _num(_get(rec, "options", "put_to_call"), 3),
            "flow_sentiment": _num(_get(rec, "options", "flow_sentiment"), 3),
        },
        "ai": {
            "sentiment": _get(rec, "ai_score", "sentiment"),
            "bull_score": _num(_get(rec, "ai_score", "bull_score"), 3),
            "bear_score": _num(_get(rec, "ai_score", "bear_score"), 3),
        },
    }


def parse_intraday(payload) -> list[dict]:
    """/ats/darkflow -> [{direction: up|down, ...}], in the page's own order."""
    out = []
    if not isinstance(payload, dict):
        return out
    for key, direction in (("trending_up", "up"), ("trending_down", "down")):
        for rec in payload.get(key) or []:
            if isinstance(rec, dict) and rec.get("ticker"):
                out.append({"direction": direction, **_flag_fields(rec)})
    return out


def _irregular_vol(rec: dict) -> float:
    """The Historical page's "Irregular Vol" column (json_record.multiplier, a numeric
    string such as '3485.38...'). 0 when missing/unparseable, so those sort last."""
    inner = rec.get("json_record") if isinstance(rec.get("json_record"), dict) else {}
    return _num(re.sub(r"[^0-9.]", "", str(inner.get("multiplier") or ""))) or 0.0


def _pct(v):
    """Percent change as a number, whether sent as 0.43, "0.43" or "0.43%"."""
    return _num(re.sub(r"[^0-9.\-]", "", str(v)) if v is not None else None, 4)


def parse_historical(payload, limit: int = HISTORY_LIMIT) -> list[dict]:
    """/ats/historic/daily-darkflow -> the top `limit` rows as TradeAlgo's Historical page
    ("Historic ATS Gainers & Losers", Up view) shows them.

    The request returns every flag/unflag entry for the session, unsorted; the page
    merges each ticker's entries into ONE row -- From = the price when it was first
    flagged, To = the price when it was last unflagged, percent change between those,
    Irregular Vol = the highest of its entries -- shows the Up view as the tickers whose
    whole-session change is positive, and sorts by Irregular Vol. Confirmed against the
    user's live data (2026-09-30): BEKE's page row reads +0.47%, which is its first
    entry's From ($16.865) to its last entry's To ($16.945), while none of its three
    individual entries shows +0.47%."""
    by_ticker: dict[str, list[dict]] = {}
    for rec in payload if isinstance(payload, list) else []:
        if isinstance(rec, dict) and rec.get("ticker"):
            by_ticker.setdefault(rec["ticker"], []).append(rec)

    merged = []
    for ticker, recs in by_ticker.items():
        first = min(recs, key=lambda r: str(r.get("date_added") or ""))
        last = max(recs, key=lambda r: str(r.get("date_remove") or r.get("date_added") or ""))
        top = max(recs, key=_irregular_vol)  # the entry whose dark-pool detail is shown
        from_price, to_price = _num(first.get("added_price"), 4), _num(last.get("removed_price"), 4)
        if from_price and to_price is not None:
            change = round((to_price / from_price - 1) * 100, 2)
        else:  # no usable prices: fall back to the single entry's own figure
            change = _pct(top.get("performance"))
        merged.append((top, first, last, change, from_price, to_price, len(recs)))

    gainers = [m for m in merged if (m[3] or 0) > 0]
    gainers.sort(key=lambda m: _irregular_vol(m[0]), reverse=True)

    out = []
    for top, first, last, change, from_price, to_price, n in gainers[:limit]:
        inner = top.get("json_record") if isinstance(top.get("json_record"), dict) else {}
        summary = _flag_fields({**inner, "ticker": top.get("ticker")})
        summary.update(
            name=top.get("company_name") or summary["name"],
            irregular_vol=round(_irregular_vol(top), 2),
            percent_change=change,
            from_price=from_price,
            from_time=first.get("date_added"),
            to_price=to_price,
            to_time=last.get("date_remove"),
            times_flagged=n,
        )
        out.append(summary)
    return out


_PARSERS = {"intraday": parse_intraday, "historical": parse_historical}


def _write_json_atomic(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)  # a reader never sees a half-written file


def run_fetch(headless: bool = True, now: dt.datetime | None = None) -> dict:
    check_rate_cap(now)

    from playwright.sync_api import sync_playwright

    results, raw, errors = {}, {}, {}
    with sync_playwright() as p:
        context = _open_context(p, headless)
        page = None
        try:
            page = context.pages[0] if context.pages else context.new_page()
            record_attempt(now)  # counted before any request goes out -- see check_rate_cap
            for src in FETCH_SOURCES:
                try:
                    with page.expect_response(
                        lambda r, m=src["match"]: m in r.url, timeout=FETCH_TIMEOUT_MS
                    ) as resp_info:
                        page.goto(src["page"])
                    # The data request arriving is what proves the session is live -- the
                    # page address alone can't (TradeAlgo keeps "signin" in it at times).
                    resp = resp_info.value
                    if resp.status in (401, 403):
                        raise NotLoggedInError(
                            f"TradeAlgo rejected the saved session (HTTP {resp.status}) -- "
                            "run `uv run tradealgo_scraper.py login` again."
                        )
                    if not resp.ok:
                        raise RuntimeError(f"data request returned HTTP {resp.status}")
                    payload = resp.json()
                    raw[src["name"]] = payload
                    results[src["name"]] = {
                        "source_url": mask_url(resp.url),
                        "tickers": _PARSERS[src["name"]](payload),
                    }
                except NotLoggedInError:
                    raise
                except Exception as e:
                    if _looks_logged_out(page.url):
                        raise NotLoggedInError(
                            "TradeAlgo session expired -- run `uv run tradealgo_scraper.py login` again."
                        ) from e
                    # One page failing shouldn't throw away the other page's data.
                    errors[src["name"]] = str(e).splitlines()[0]
            if results:
                _save_session(context)  # cookies may have been rotated/refreshed
        finally:
            context.close()

    if not results:
        raise RuntimeError(f"Neither TradeAlgo page returned data: {errors}")

    cache = {
        "fetched_at": (now or dt.datetime.now()).isoformat(timespec="seconds"),
        **results,
    }
    if errors:
        cache["errors"] = errors
    _write_json_atomic(CACHE_PATH, cache)
    _write_json_atomic(RAW_PATH, raw)
    counts = ", ".join(f"{k}: {len(v['tickers'])} tickers" for k, v in results.items())
    print(f"Saved TradeAlgo data to {CACHE_PATH} ({counts}).")
    for name, err in errors.items():
        print(f"WARNING: {name} page failed: {err}")
    return cache


def main(argv: list[str]) -> int:
    modes = {"login": run_login, "discover": run_discover, "fetch": run_fetch}
    if len(argv) != 2 or argv[1] not in modes:
        print(__doc__)
        return 2
    from playwright.sync_api import Error as PlaywrightError

    try:
        modes[argv[1]]()
    except (RateCapError, NotLoggedInError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1
    except PlaywrightError as e:
        print(f"ERROR: browser problem: {str(e).splitlines()[0]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
