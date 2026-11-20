#!/usr/bin/env bash
# Temporary Oracle Cloud / Linux VPS One-Time Setup for Google Colab CLI
# Supports: Ubuntu/Debian (apt) & Oracle Linux/RHEL (dnf/yum), ARM64 & x86_64
# Sizing: Verified for 2-4 OCPU Oracle Free Tier / standard Linux VPS
set -euo pipefail

VENV_DIR="${COLAB_VENV_DIR:-$HOME/.colab-venv}"
CRED_PATH="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}"
SSH_KEY_PATH="${COLAB_SSH_KEY:-$HOME/.ssh/id_ed25519}"

log() {
  printf '[SETUP %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    log "ERROR: Root or sudo is required to run: $*"
    exit 1
  fi
}

log "Detected architecture: $(uname -m) | CPUs: $(nproc 2>/dev/null || echo unknown)"

if command -v apt-get >/dev/null 2>&1; then
  log "Installing system dependencies via apt-get..."
  run_root apt-get update -y
  run_root apt-get install -y git tmux openssh-client python3 python3-venv python3-pip curl ca-certificates
elif command -v dnf >/dev/null 2>&1; then
  log "Installing system dependencies via dnf..."
  run_root dnf install -y git tmux openssh-clients python3 python3-pip curl ca-certificates
elif command -v yum >/dev/null 2>&1; then
  log "Installing system dependencies via yum..."
  run_root yum install -y git tmux openssh-clients python3 python3-pip curl ca-certificates
else
  log "WARN: Unsupported package manager; ensure git, tmux, ssh, python3, and pip are installed."
fi

if [ -f /etc/gai.conf ]; then
  if ! grep -qE '^[[:space:]]*precedence[[:space:]]+::ffff:0:0/96[[:space:]]+100' /etc/gai.conf; then
    log "Configuring IPv4 preference in /etc/gai.conf..."
    printf '\n# Prefer IPv4 for Google Colab CLI proxy connections\nprecedence ::ffff:0:0/96  100\n' | run_root tee -a /etc/gai.conf >/dev/null
  else
    log "IPv4 preference already active in /etc/gai.conf."
  fi
else
  log "Creating /etc/gai.conf with IPv4 preference..."
  printf 'precedence ::ffff:0:0/96  100\n' | run_root tee /etc/gai.conf >/dev/null
fi

if [ ! -d "$VENV_DIR" ]; then
  log "Creating Python virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

log "Installing/upgrading google-colab-cli in $VENV_DIR..."
"$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
"$VENV_DIR/bin/pip" install --upgrade google-colab-cli || \
  "$VENV_DIR/bin/pip" install --upgrade "git+https://github.com/googlecolab/google-colab-cli.git"

mkdir -p "$(dirname "$SSH_KEY_PATH")"
chmod 700 "$(dirname "$SSH_KEY_PATH")"
if [ ! -f "$SSH_KEY_PATH" ]; then
  log "Generating Ed25519 SSH key at $SSH_KEY_PATH..."
  ssh-keygen -q -t ed25519 -N "" -f "$SSH_KEY_PATH"
  chmod 600 "$SSH_KEY_PATH"
else
  log "SSH key already exists at $SSH_KEY_PATH."
fi

mkdir -p "$(dirname "$CRED_PATH")"
chmod 700 "$(dirname "$CRED_PATH")"
if [ -f "$CRED_PATH" ]; then
  chmod 600 "$CRED_PATH"
  log "Verified ADC credentials at $CRED_PATH (chmod 600)."
  log "Testing Colab authentication and compute balance (no GPU VM created)..."
  GOOGLE_APPLICATION_CREDENTIALS="$CRED_PATH" "$VENV_DIR/bin/colab" --auth adc usage
else
  log "NOTICE: ADC credentials file not found yet at: $CRED_PATH"
  log "Copy your JSON file from local machine using scp, then run:"
  log "  chmod 600 \"$CRED_PATH\""
  log "  GOOGLE_APPLICATION_CREDENTIALS=\"$CRED_PATH\" \"$VENV_DIR/bin/colab\" --auth adc usage"
fi

log "VPS setup complete. Colab CLI binary: $VENV_DIR/bin/colab"
