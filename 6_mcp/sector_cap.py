"""
Sector concentration cap for real trades, added 2026-10-15 at the user's request.

Why: a semiconductor selloff hit the account hard. Cathie's AVGO/MU/CRDO spreads and the
user's own AMD/MRVL/SNDK/CIEN spreads looked like many independent trades, but they move
together. The per-trade limits (8% of net liq, 5x premium, 2 orders a day, 1 contract)
bound each trade, not the correlated sum of them.

The rule: at most MAX_PER_SECTOR (default 2) different underlyings in one group, counting
Cathie's spreads AND the user's own option positions at Schwab. A second spread on a name
already held doesn't add a name. Applied at staging, at approval, and in the spread scanner
(live_trading.py, spread_scanner.py). Real trading only; there is no paper path any more.

Groups, in order:
  1. the user's own overrides, `sectors.txt` ("TICKER group_name" per line);
  2. the built-in correlated clusters below (semiconductors and the AI-hardware names that
     trade with them: memory/storage, optical, networking silicon);
  3. Cathie's named ETFs: each its own group (never capped; they're broad);
  4. yfinance's industry/sector for anything else, cached in `.sector_cache.json`;
  5. unknown -> the ticker is its own group, with a warning (fails OPEN: yfinance is flaky
     and the hand-kept clusters above cover the names that actually hurt).
Group names are lower case with underscores.
"""
import concurrent.futures
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OVERRIDES_PATH = os.path.join(HERE, "sectors.txt")
CACHE_PATH = os.path.join(HERE, ".sector_cache.json")

DEFAULT_MAX_PER_SECTOR = 2

# Correlated cluster, deliberately wider than "semiconductor" by GICS: SNDK/WDC/STX (memory,
# storage), CIEN/COHR/LITE (optical), CRDO/ALAB (AI connectivity) all sold off with the
# chip names. Edit by overriding in sectors.txt rather than here.
SEMICONDUCTORS = (
    "NVDA AMD AVGO MU MRVL INTC QCOM TXN ADI MCHP ON NXPI SWKS QRVO MPWR ARM TSM ASML "
    "AMAT LRCX KLAC TER ENTG ONTO CRDO ALAB SNDK WDC STX CIEN COHR LITE GFS UMC "
    "SMH SOXX SOXL SOXS"
).split()
DEFAULT_GROUPS = {t: "semiconductors" for t in SEMICONDUCTORS}

# yfinance industries that belong in the semiconductor cluster even if unlisted above.
SEMI_INDUSTRIES = {"semiconductors", "semiconductor equipment & materials"}


def max_per_sector() -> int:
    """MAX_PER_SECTOR in .env (re-read each call). 0 turns the cap off."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    raw = (os.getenv("MAX_PER_SECTOR") or "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_MAX_PER_SECTOR
    except ValueError:
        return DEFAULT_MAX_PER_SECTOR


def _norm(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", " ").split())


def load_overrides(path: str | None = None) -> dict[str, str]:
    """`TICKER group_name` per line, '#' comments. Missing file = no overrides."""
    out: dict[str, str] = {}
    try:
        with open(path or OVERRIDES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split("#", 1)[0].split(None, 1)
                if len(parts) == 2:
                    out[parts[0].upper()] = _norm(parts[1])
    except OSError:
        pass
    return out


def _read_cache() -> dict[str, str]:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(cache: dict[str, str]):
    try:
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=0, sort_keys=True)
        os.replace(tmp, CACHE_PATH)
    except OSError:
        pass


def _yfinance_group(symbol: str, timeout: float = 10) -> str | None:
    def _fetch():
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        industry = str(info.get("industry") or "").strip().lower()
        if industry in SEMI_INDUSTRIES:
            return "semiconductors"
        sector = info.get("sector")
        return _norm(sector) if sector else None

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(_fetch).result(timeout=timeout)
    except Exception:
        return None
    finally:
        pool.shutdown(wait=False)


def group_of(symbol: str, lookup: bool = True) -> tuple[str, str | None]:
    """(group, warning). `lookup=False` never touches the network (map and cache only)."""
    from universe import CATHIE_ETF_UNIVERSE
    sym = symbol.upper()
    overrides = load_overrides()
    if sym in overrides:
        return overrides[sym], None
    if sym in DEFAULT_GROUPS:
        return DEFAULT_GROUPS[sym], None
    if sym in CATHIE_ETF_UNIVERSE:
        return sym.lower(), None
    cache = _read_cache()
    if sym in cache:
        return cache[sym], None
    if lookup:
        found = _yfinance_group(sym)
        if found:
            cache[sym] = found
            _write_cache(cache)
            return found, None
    return sym.lower(), f"sector for {sym} is unknown; the sector cap treated it as its own group"


def assess(symbol: str, held: set[str] | list[str], lookup: bool = True) -> dict:
    """Would adding `symbol` break the cap, given the underlyings already `held`?
    {problem: str|None, group, peers: [other held names in the group], warning: str|None}.
    Never raises."""
    sym = symbol.upper()
    try:
        cap = max_per_sector()
        group, warning = group_of(sym, lookup)
        peers = sorted({h.upper() for h in held if h.upper() != sym
                        and group_of(h, lookup)[0] == group})
        problem = None
        if cap and len(peers) >= cap:
            problem = (f"sector cap: {sym} is in '{group}', where {len(peers)} other underlyings are "
                       f"already held ({', '.join(peers)}); the limit is {cap}. "
                       "Close one of them, or pick a different sector")
        return {"problem": problem, "group": group, "peers": peers, "warning": warning}
    except Exception as e:  # never let a lookup problem break staging
        return {"problem": None, "group": sym.lower(), "peers": [],
                "warning": f"sector cap check failed ({e}); not applied"}


def exposure(held: set[str] | list[str]) -> dict[str, list[str]]:
    """{group: sorted underlyings} for the names that share a group with another one.
    Map and cache only (no network), for the dashboard."""
    groups: dict[str, set[str]] = {}
    for h in held:
        groups.setdefault(group_of(h, lookup=False)[0], set()).add(h.upper())
    return {g: sorted(names) for g, names in sorted(groups.items()) if len(names) >= 2}
