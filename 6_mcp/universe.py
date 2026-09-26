"""Cathie's named options-eligible underlying universe -- single source of truth.

Previously this list of ~13 tickers was duplicated by hand across three places
(reset.py's strategy text, templates.py's trader_instructions, and templates.py's
new_trades_message). The two templates.py copies had already drifted apart:
new_trades_message was silently missing XLI, XLU, EEM, and EFA, so Cathie's actual
per-cycle candidate-selection prompt was quietly narrower than what she was told her
universe was elsewhere. Consolidating to one list here makes that class of drift
impossible going forward.

Broadened from the original 13 at the same time, alongside anti-repetition guidance
added in templates.py -- a wider, more diverse named list gives Cathie's Researcher
(which starts fresh every cycle with no built-in "avoid repeating recent picks"
mechanism beyond its own persistent per-trader memory) more genuinely different
candidates to consider instead of converging on the same handful of large, liquid
names every cycle. Note this is still only the *named* list offered as a starting
point -- Cathie's instructions already allow any liquid stock/ETF meeting the
price/open-interest bar, named or not.
"""
import os

CATHIE_ETF_UNIVERSE = [
    # Broad market / large-cap
    "SPY", "QQQ", "IWM", "DIA",
    # Fixed income / macro hedges
    "GLD", "TLT", "SLV",
    # Sector SPDRs
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLU", "XLY", "XLP", "XLB",
    # International / broad exposure
    "EEM", "EFA",
]

CATHIE_ETF_UNIVERSE_TEXT = ", ".join(CATHIE_ETF_UNIVERSE)


# Hand-edited by whoever runs this: a plain text file (one ticker per line, '#'
# comments allowed) the user can freely add to/prune without touching any code.
# Deliberately separate from CATHIE_ETF_UNIVERSE above -- that list is a curated,
# code-reviewed set of broad ETFs; this one is the user's own ad hoc candidate
# list and is expected to change often, hence a plain text file rather than a
# Python literal. See options_trading_server.py: get_custom_watchlist, the MCP
# tool that exposes this to Cathie.
WATCHLIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.txt")


def load_custom_watchlist(path: str = WATCHLIST_PATH) -> list[str]:
    """Parse the user's hand-edited watchlist file into a deduped ticker list.

    Reads the file fresh on every call (no caching) so edits take effect on
    Cathie's very next cycle without restarting the trading floor. Returns an
    empty list -- never raises -- if the file is missing or empty; the caller
    (get_custom_watchlist) is responsible for turning that into a friendly
    message rather than a bare empty result.
    """
    if not os.path.exists(path):
        return []

    tickers: list[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()  # strip full-line and inline comments
            if not line:
                continue
            ticker = line.split()[0].upper()  # first whitespace-separated token on the line
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers
