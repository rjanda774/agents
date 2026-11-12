# Running the trading floor on a cloud server

Runs `trading_floor.py` as a background service on a small always-on Linux
server (Ubuntu or Debian), so it keeps going when your PC sleeps or restarts.

## The one rule: everything that touches Cathie's trades runs on the server

`trading_floor.py`, `approve_orders.py`, `app.py`, `breach_watch.py` and
`schwab_auth_setup.py` all read and write the same two files in `6_mcp/`:
`accounts.db` (Cathie's real-trade ledger) and `schwab_token.json`. They only
work together when they use the same copies. So you still run those commands
from a terminal, but the terminal is an SSH session on the server, not your
PC's own checkout. A trade Cathie stages on the server never appears in an
`approve_orders.py` run on your PC.

Once the server is running, do not start `trading_floor.py` on the PC again.

## One-time setup

1. Create the server (1 GB of memory is enough) and log in with SSH.
2. Get the code:

   ```bash
   git clone https://github.com/rjanda774/agents.git
   ```

3. On your PC, stop the trading floor, then copy your private files up
   (PowerShell, from the PC's `6_mcp` folder; replace `you@server`):

   ```bash
   scp .env accounts.db schwab_token.json watchlist.txt hands_off.txt sectors.txt you@server:agents/6_mcp/
   ```

   Leave out any of the last three you don't have. To keep the Researcher's
   memory too: `scp memory/*.db you@server:agents/6_mcp/memory/` after step 4.
   If `.env` sets `SCHWAB_TOKEN_PATH` to a Windows path, change it on the
   server (or remove the line to use the default).

4. On the server:

   ```bash
   bash agents/6_mcp/deploy/setup_server.sh
   ```

5. Start it:

   ```bash
   sudo systemctl start cathie-floor
   ```

## Day to day (all on the server, over SSH, in `agents/6_mcp`)

| What | Command |
|---|---|
| Watch the floor's output | `journalctl -u cathie-floor -f` |
| Is it running? | `systemctl status cathie-floor` |
| Stop / start / restart | `sudo systemctl stop cathie-floor` (or `start`, `restart`) |
| Approve trades | `uv run approve_orders.py` |
| Schwab login (every 7 days) | `uv run schwab_auth_setup.py` (open the printed link in your PC's browser, paste the redirect address back) |
| Breach alerts | `uv run breach_watch.py` (see below to keep it running) |
| Dashboard | `uv run app.py`, see below |
| After a code change | `git pull && uv sync && sudo systemctl restart cathie-floor` |

Edits to `.env` need a restart; `watchlist.txt`, `hands_off.txt`,
`sectors.txt` and `STOP_TRADING` take effect on the next cycle as before.

**Dashboard.** It listens only on the server itself. From your PC, connect with
a tunnel and leave that window open, then browse to http://localhost:7860:

```bash
ssh -L 7860:localhost:7860 you@server
```

**Keeping `breach_watch.py` or `app.py` running after you disconnect.** A
command started in an SSH window stops when the window closes. Start it inside
`tmux` (`sudo apt-get install -y tmux`, then `tmux new -s watch`, run the
command, press Ctrl+B then D to leave it running; `tmux attach -t watch` to
come back).

## What stays on the PC

The TradeAlgo scraper (`tradealgo_scraper.py`) and its scheduled task. To give
Cathie its data, copy the result up after each fetch:

```bash
scp tradealgo_darkpool.json you@server:agents/6_mcp/
```

Without that, `get_dark_pool_activity` reports the data as stale.

## Things to check in the first days

- **Yahoo data.** The earnings and sector checks use yfinance, and Yahoo
  sometimes blocks cloud servers. Run `uv run spread_scanner.py --explain AMD`
  on the server: if the earnings dates come back empty, real trades will be
  refused (unknown earnings), and the server's provider or region needs to
  change.
- **Server security.** The server holds keys that can place real orders. Use
  SSH keys, turn off password login, and open no ports besides SSH.
