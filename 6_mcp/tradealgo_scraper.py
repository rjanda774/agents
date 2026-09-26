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
        Opens a browser window on TradeAlgo. Log in by hand, then press Enter
        in the terminal. The session is saved to a private browser profile
        (.tradealgo_profile/, gitignored) so later runs stay logged in.

    uv run tradealgo_scraper.py discover
        Opens the same logged-in browser. Navigate to the dark-pool page and
        let it load, then press Enter. Writes tradealgo_discovery.json: which
        data requests the page made and the *field names/types* in each
        response -- no values, no cookies, no tokens, and account-ID-looking
        URL segments masked -- so it's safe to share for building the parser.

    uv run tradealgo_scraper.py fetch
        Headless, no window. Loads TRADEALGO_PAGE_URL, captures the data
        request matching TRADEALGO_DATA_URL_CONTAINS, and saves it to
        tradealgo_darkpool.json. Enforces the twice-a-day cap. Meant to be
        scheduled (e.g. Windows Task Scheduler), not run by the trading floor
        itself -- a scraper problem should never be able to stall a trading
        cycle. Both env vars are set in .env once discovery has identified them.

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
STATE_PATH = os.path.join(HERE, ".tradealgo_fetch_state.json")

TRADEALGO_HOME = os.getenv("TRADEALGO_HOME_URL", "https://app.tradealgo.com")
TRADEALGO_PAGE_URL = os.getenv("TRADEALGO_PAGE_URL")
TRADEALGO_DATA_URL_CONTAINS = os.getenv("TRADEALGO_DATA_URL_CONTAINS")
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


def _save_session(context) -> None:
    """Persist cookies (including session cookies) for the next run. Called only
    after a run that was actually logged in, so a logged-out run can't overwrite a
    good session with a bad one."""
    context.storage_state(path=_session_file())


def _looks_logged_out(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return any(word in path for word in ("login", "signin", "sign-in", "sign_in", "auth"))


def _wait_for_enter(prompt: str) -> None:
    input(prompt)


def run_login(headless: bool = False, wait=_wait_for_enter) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        context = _open_context(p, headless)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(TRADEALGO_HOME)
        wait(
            "\nA browser window opened on TradeAlgo. Log in there by hand (this script\n"
            "never sees your password), wait until your dashboard loads, then press Enter here... "
        )
        if _looks_logged_out(page.url):
            context.close()
            raise NotLoggedInError("Still on a login page -- log in fully, then press Enter.")
        _save_session(context)
        context.close()
    print(f"Saved your TradeAlgo session to {PROFILE_DIR}")


def run_discover(headless: bool = False, wait=_wait_for_enter, start_url: str | None = None) -> dict:
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
        page.goto(start_url or TRADEALGO_HOME)
        if _looks_logged_out(page.url):
            context.close()
            raise NotLoggedInError("Not logged in -- run `uv run tradealgo_scraper.py login` first.")
        wait(
            "\nIn the browser window, go to TradeAlgo's dark-pool page and let it fully load\n"
            "(scroll a little if the table loads more as you go). Then press Enter here... "
        )
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


def run_fetch(headless: bool = True, now: dt.datetime | None = None) -> dict:
    if not (TRADEALGO_PAGE_URL and TRADEALGO_DATA_URL_CONTAINS):
        raise RuntimeError(
            "TRADEALGO_PAGE_URL and TRADEALGO_DATA_URL_CONTAINS aren't set in .env yet -- "
            "run `discover` first so the right page and data request can be identified."
        )
    check_rate_cap(now)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        context = _open_context(p, headless)
        page = None
        try:
            page = context.pages[0] if context.pages else context.new_page()
            record_attempt(now)  # counted before the request goes out -- see check_rate_cap
            with page.expect_response(
                lambda r: TRADEALGO_DATA_URL_CONTAINS in r.url, timeout=FETCH_TIMEOUT_MS
            ) as resp_info:
                page.goto(TRADEALGO_PAGE_URL)
            resp = resp_info.value
            if _looks_logged_out(page.url):
                raise NotLoggedInError(
                    "TradeAlgo session expired -- run `uv run tradealgo_scraper.py login` again."
                )
            if not resp.ok:
                raise RuntimeError(f"Data request returned HTTP {resp.status}.")
            payload = resp.json()
            _save_session(context)  # cookies may have been rotated/refreshed
        except Exception as e:
            if page is not None and _looks_logged_out(page.url):
                raise NotLoggedInError(
                    "TradeAlgo session expired -- run `uv run tradealgo_scraper.py login` again."
                ) from e
            raise
        finally:
            context.close()

    cache = {
        "fetched_at": (now or dt.datetime.now()).isoformat(timespec="seconds"),
        "source_url": mask_url(resp.url),
        # Parsed into normalized records once discovery shows the real field names;
        # until then the raw response is kept so nothing fetched is wasted.
        "raw": payload,
    }
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_PATH)  # atomic: a reader never sees a half-written file
    print(f"Saved TradeAlgo data to {CACHE_PATH}.")
    return cache


def main(argv: list[str]) -> int:
    modes = {"login": run_login, "discover": run_discover, "fetch": run_fetch}
    if len(argv) != 2 or argv[1] not in modes:
        print(__doc__)
        return 2
    try:
        modes[argv[1]]()
    except (RateCapError, NotLoggedInError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
