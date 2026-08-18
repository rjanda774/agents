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
