#!/usr/bin/env python
"""
One-time (well -- recurring every ~7 days) interactive Schwab OAuth login.

Run this from a machine with a real web browser to authorize Cathie's options
tools to pull real market data (option chains, quotes) from your Schwab
account. It does NOT enable order placement -- there is no order-placement
code anywhere in this project; sell_credit_spread/close_credit_spread only
ever write to the local simulated `cathie_options` ledger in accounts.db.
This login only grants read access to market data.

Prerequisites:
  1. A developer.schwab.com account with an approved "Trader API - Individual"
     app. Approval is manual on Schwab's end and is not instant.
  2. That app's App Key and App Secret, and the exact callback URL you
     registered for it (must match byte-for-byte, including scheme, port, and
     any trailing slash -- Schwab's login will fail otherwise).

Set these in your .env before running (see CLAUDE.md for the full list):
  SCHWAB_APP_KEY=...
  SCHWAB_APP_SECRET=...
  SCHWAB_CALLBACK_URL=https://127.0.0.1:8182     # whatever you registered
  SCHWAB_TOKEN_PATH=./schwab_token.json          # optional, this is the default

This walks you through a manual copy-paste login (open a URL, log into Schwab,
approve the app, paste the resulting redirect URL back into this terminal) --
no local HTTPS server required, which keeps this usable even on a locked-down
machine as long as you have a browser somewhere to complete the login.

Once this succeeds, options_trading_server.py picks up the token file
automatically; schwab-py refreshes the short-lived access token in the
background on every use. The refresh token itself is only valid for 7 days
(a Schwab platform limit, not a bug here) -- when Schwab calls in the trading
floor's logs start showing a "yfinance (schwab fallback)" data_source again,
that's your cue to re-run this script.
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

APP_KEY = os.getenv("SCHWAB_APP_KEY")
APP_SECRET = os.getenv("SCHWAB_APP_SECRET")
CALLBACK_URL = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
TOKEN_PATH = os.getenv(
    "SCHWAB_TOKEN_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "schwab_token.json"),
)


def main():
    if not APP_KEY or not APP_SECRET:
        print("ERROR: SCHWAB_APP_KEY and/or SCHWAB_APP_SECRET are not set in your .env.")
        print("Get these from your app's page at https://developer.schwab.com once it's approved.")
        sys.exit(1)

    try:
        import schwab.auth
    except ImportError:
        print("ERROR: the 'schwab-py' package isn't installed.")
        print("Run `uv sync` in 6_mcp/ first (it's declared in pyproject.toml).")
        sys.exit(1)

    print(f"Callback URL:  {CALLBACK_URL}")
    print(f"Token path:    {TOKEN_PATH}")
    print()
    print("Follow the prompts below -- you'll be asked to open a URL in your browser,")
    print("log into Schwab, approve the app, then paste the URL you land on back here.")
    print()

    schwab.auth.client_from_manual_flow(
        api_key=APP_KEY,
        app_secret=APP_SECRET,
        callback_url=CALLBACK_URL,
        token_path=TOKEN_PATH,
    )

    print()
    print(f"Success -- token saved to {TOKEN_PATH}.")
    print("Cathie's options tools will now use real Schwab market data (falling back to")
    print("yfinance automatically if Schwab is ever unavailable -- check the data_source")
    print("field in tool responses / logs to see which one actually served each call).")
    print("Remember: this refresh token expires in ~7 days. Re-run this script when it does.")


if __name__ == "__main__":
    main()
