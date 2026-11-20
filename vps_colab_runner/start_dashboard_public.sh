#!/usr/bin/env bash
# Start the monitoring dashboard + a public Cloudflare quick tunnel (https://*.trycloudflare.com).
# The dashboard is protected by an access key stored in ~/.colab_dash_token (never in git).
# Re-running this script restarts both and prints a fresh public link (quick tunnel URLs change).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${DASH_PORT:-8765}"
BIN_DIR="$HOME/.local/bin"
CF="$BIN_DIR/cloudflared"
TOKEN_FILE="$HOME/.colab_dash_token"
CF_LOG="$HOME/cloudflared.log"

mkdir -p "$BIN_DIR"
if [ ! -x "$CF" ]; then
  case "$(uname -m)" in
    aarch64|arm64) ARCH=arm64 ;;
    x86_64|amd64) ARCH=amd64 ;;
    *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
  esac
  echo "Downloading cloudflared ($ARCH)..."
  curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$ARCH" -o "$CF"
  chmod +x "$CF"
fi

if [ ! -s "$TOKEN_FILE" ]; then
  python3 -c "import secrets; print(secrets.token_urlsafe(18))" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi
TOKEN="$(cat "$TOKEN_FILE")"

tmux kill-session -t dashboard 2>/dev/null || true
tmux kill-session -t tunnel 2>/dev/null || true
tmux new-session -d -s dashboard "DASH_TOKEN='$TOKEN' python3 '$SCRIPT_DIR/dashboard.py' --port $PORT"
: > "$CF_LOG"
tmux new-session -d -s tunnel "'$CF' tunnel --no-autoupdate --url http://127.0.0.1:$PORT > '$CF_LOG' 2>&1"

URL=""
for _ in $(seq 1 45); do
  URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$CF_LOG" | head -n 1 || true)"
  [ -n "$URL" ] && break
  sleep 2
done
if [ -z "$URL" ]; then
  echo "Tunnel URL not found yet. Check: tail -n 30 $CF_LOG" >&2
  exit 1
fi
echo "PUBLIC_URL=$URL/?key=$TOKEN"
