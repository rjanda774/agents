#!/usr/bin/env bash
# One-time setup of a Linux server (Ubuntu/Debian or AlmaLinux/Rocky) to run the Cathie trading
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

# Ubuntu/Debian use apt-get; AlmaLinux/Rocky/RHEL use dnf.
if command -v apt-get >/dev/null 2>&1; then
    PKG=apt
elif command -v dnf >/dev/null 2>&1; then
    PKG=dnf
else
    echo "Neither apt-get nor dnf found: this script supports Ubuntu/Debian and AlmaLinux/Rocky/RHEL." >&2
    exit 1
fi

# Node.js: the Researcher's Brave-search and memory MCP servers run through npx.
if ! command -v npx >/dev/null 2>&1; then
    echo "== Installing Node.js"
    if [ "$PKG" = apt ]; then
        sudo apt-get update -y
        sudo apt-get install -y nodejs npm curl
    else
        # The default Node.js on AlmaLinux 9 is too old; ask for version 20.
        sudo dnf module reset -y nodejs || true
        sudo dnf module enable -y nodejs:20 || true
        sudo dnf install -y nodejs npm curl tar
    fi
fi
NODE_MAJOR="$(node -v 2>/dev/null | sed -e 's/^v//' -e 's/\..*//' || true)"
if [ -z "$NODE_MAJOR" ] || [ "$NODE_MAJOR" -lt 18 ]; then
    echo "WARNING: Node.js 18 or newer is needed (found: $(node -v 2>/dev/null || echo none))." >&2
    echo "         The Researcher's search and memory tools may not start." >&2
fi

# uv: runs the floor from the pinned uv.lock. Installed system-wide, in
# /usr/local/bin, so the service can run it on servers with SELinux
# (AlmaLinux), which blocks services from starting programs kept in a home
# folder.
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
else
    echo "== Installing uv into /usr/local/bin"
    curl -LsSf https://astral.sh/uv/install.sh \
        | sudo env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
    UV=/usr/local/bin/uv
    if [ ! -x "$UV" ]; then
        echo "uv did not end up in /usr/local/bin; stop here and report this message." >&2
        exit 1
    fi
    if command -v restorecon >/dev/null 2>&1; then
        sudo restorecon /usr/local/bin/uv /usr/local/bin/uvx 2>/dev/null || true
    fi
fi
UVDIR="$(dirname "$UV")"
case "$UV" in
    /home/*|/root/*)
        if command -v getenforce >/dev/null 2>&1 && [ "$(getenforce)" = "Enforcing" ]; then
            echo "WARNING: uv is in a home folder ($UV) and SELinux is on;" >&2
            echo "         the service may fail to start (status=203/EXEC)." >&2
        fi ;;
esac

echo "== Syncing Python environment"
# Pin the Python version: left to itself uv picks the newest Python, and some
# packages (lxml) have no ready-built files for it yet and fail to build.
(cd "$DIR" && "$UV" sync --python 3.13)
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
