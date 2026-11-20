"""Account/password + API-key authentication for the Colab VPS dashboard (stdlib only).

Credentials live in ~/.colab_dash_auth.json (chmod 600, never in git): the password is stored as a
PBKDF2 hash. Create or reset them with:
    python3 dash_auth.py --init [--user NAME] [--reset-password]
"""

import argparse
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

AUTH_FILE = Path(os.environ.get("DASH_AUTH_FILE", str(Path.home() / ".colab_dash_auth.json")))
SESSION_TTL = 30 * 24 * 3600
MAX_FAILS = 5
LOCK_SECONDS = 600
ITERATIONS = 200_000
CREDS = {}
_SESSIONS = {}
_FAILS = {}


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()
    return salt, digest


def configure(user=None, password=None, api_key=""):
    """Set credentials in memory (used by load() and tests). configure(None) disables auth."""
    CREDS.clear()
    _SESSIONS.clear()
    _FAILS.clear()
    if user and password:
        salt, digest = hash_password(password)
        CREDS.update(user=user, salt=salt, hash=digest, api_key=api_key or "")


def load():
    """Load credentials from env (DASH_USER/DASH_PASS/DASH_API_KEY) or AUTH_FILE."""
    if os.environ.get("DASH_USER") and os.environ.get("DASH_PASS"):
        configure(os.environ["DASH_USER"], os.environ["DASH_PASS"], os.environ.get("DASH_API_KEY", ""))
        return True
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        if data.get("user") and data.get("salt") and data.get("hash"):
            CREDS.clear()
            CREDS.update(user=data["user"], salt=data["salt"], hash=data["hash"], api_key=data.get("api_key", ""))
            return True
    except (OSError, ValueError):
        pass
    return False


def configured():
    return bool(CREDS)


def verify_login(user, password):
    if not CREDS:
        return False
    _salt, digest = hash_password(password or "", CREDS["salt"])
    user_ok = hmac.compare_digest((user or "").encode(), CREDS["user"].encode())
    return hmac.compare_digest(digest, CREDS["hash"]) and user_ok


def api_key_ok(value):
    key = CREDS.get("api_key", "")
    return bool(key) and hmac.compare_digest((value or "").encode(), key.encode())


def new_session():
    now = time.time()
    for tok in [t for t, exp in _SESSIONS.items() if exp < now]:
        _SESSIONS.pop(tok, None)
    token = secrets.token_urlsafe(32)
    _SESSIONS[token] = now + SESSION_TTL
    return token


def session_valid(token):
    exp = _SESSIONS.get(token or "")
    return bool(exp and exp > time.time())


def drop_session(token):
    _SESSIONS.pop(token or "", None)


def locked(ip):
    return len([t for t in _FAILS.get(ip, []) if t > time.time() - LOCK_SECONDS]) >= MAX_FAILS


def record_fail(ip):
    _FAILS[ip] = [t for t in _FAILS.get(ip, []) if t > time.time() - LOCK_SECONDS] + [time.time()]


def clear_fails(ip):
    _FAILS.pop(ip, None)


def init_credentials(user="colab", reset=False):
    """Create the credential file if missing (or reset). Returns (user, plaintext_password_or_None, api_key)."""
    if AUTH_FILE.exists() and not reset:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        return data["user"], None, data.get("api_key", "")
    password, api_key = secrets.token_urlsafe(12), secrets.token_urlsafe(24)
    salt, digest = hash_password(password)
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps({"user": user, "salt": salt, "hash": digest, "api_key": api_key}), encoding="utf-8")
    os.chmod(AUTH_FILE, 0o600)
    return user, password, api_key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true", help="create credentials if missing")
    ap.add_argument("--user", default="colab")
    ap.add_argument("--reset-password", action="store_true")
    args = ap.parse_args()
    if not args.init:
        ap.print_help()
        return
    user, password, api_key = init_credentials(args.user, args.reset_password)
    print(f"DASH_USER={user}")
    print(f"DASH_PASSWORD={password if password else '(unchanged, use --reset-password to issue a new one)'}")
    print(f"DASH_API_KEY={api_key}")


if __name__ == "__main__":
    main()
