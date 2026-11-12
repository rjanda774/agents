#!/usr/bin/env bash
# One-time setup of a Linux server (Ubuntu/Debian) to run the Cathie trading
# floor as a background service. Run it on the server, from anywhere:
#
#     bash 6_mcp/deploy/setup_server.sh
#
# It installs uv and Node.js if missing, syncs the Python environment, sets
# the server clock to Eastern time, and installs (but does not start) the
# cathie-floor service. Safe to run again. See deploy/README.md.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../6_mcp
RUN_USER="$(id -un)"
SERVICE=cathie-floor

if [ "$RUN_USER" = "root" ]; then
    echo "Run this as your normal login user, not root (it uses sudo where needed)." >&2
    exit 1
fi

echo "== Trading floor directory: $DIR (service will run as $RUN_USER)"

# Node.js: the Researcher's Brave-search and memory MCP servers run through npx.
if ! command -v npx >/dev/null 2>&1; then
    echo "== Installing Node.js"
    sudo apt-get update -y
    sudo apt-get install -y nodejs npm
fi

# uv: runs the floor from the pinned uv.lock.
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "== Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
UVDIR="$(dirname "$UV")"

echo "== Syncing Python environment"
(cd "$DIR" && "$UV" sync)
mkdir -p "$DIR/memory" "$DIR/logs"

# Same clock as the PC the floor ran on, so dates and log times read the same.
echo "== Setting server time zone to America/New_York"
sudo timedatectl set-timezone America/New_York

echo "== Installing $SERVICE.service"
sed -e "s|__USER__|$RUN_USER|g" -e "s|__DIR__|$DIR|g" \
    -e "s|__UVDIR__|$UVDIR|g" -e "s|__UV__|$UV|g" \
    "$DIR/deploy/cathie-floor.service.template" \
    | sudo tee "/etc/systemd/system/$SERVICE.service" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE" >/dev/null

echo
echo "Done. The service is installed but NOT started."
missing=0
for f in .env schwab_token.json accounts.db; do
    if [ ! -e "$DIR/$f" ]; then
        echo "  MISSING: $DIR/$f  (copy it from your PC first)"
        missing=1
    fi
done
[ "$missing" = 0 ] && echo "  .env, schwab_token.json and accounts.db are in place."
echo "Start it with:   sudo systemctl start $SERVICE"
echo "Watch it with:   journalctl -u $SERVICE -f"
