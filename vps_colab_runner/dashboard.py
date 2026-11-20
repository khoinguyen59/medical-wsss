#!/usr/bin/env python3
"""Temporary Colab-on-VPS monitoring dashboard (stdlib only, light blue/white theme).

Login with account + password (see dash_auth.py). Bots/scripts may use the X-API-Key header instead.
Binds to 127.0.0.1 by default; expose it with a Cloudflare tunnel (start_dashboard_public.sh).
"""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import dash_auth

RUNNER_DIR = Path(__file__).resolve().parent
RUNS_DIR = RUNNER_DIR / "runs"
RUNNER_SH = RUNNER_DIR / "run_colab_vps.sh"
CRED = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(Path.home() / ".config/gcloud/application_default_credentials.json"),
)
NOISE = re.compile(
    r"Exception ignored|Traceback \(most|_shutdown_workers|is_alive\(|dataloader\.py|"
    r"_parent_pid|can only test a child|^\s*self\._shutdown|^\s*if w\.|^\s*assert self|^\s*File \""
)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH_RE = re.compile(r"\[EPOCH\s+(\d+)\]\s*(.*)")
METRIC_RE = re.compile(r"([A-Za-z][A-Za-z0-9_() ]*?)\s*[=:]\s*(-?\d+\.\d+|nan)")
EVENT_KEYS = (
    "Verified atomic sync", "WATCHDOG", "ALARM", "Attempt", "CLEANUP", "SUCCESS", "STOPPED",
    "Halting", "WARN", "CHECKPOINT", "Uploading", "Notebook execution", "Connection was lost",
)
SYNC_RE = re.compile(r"\[VPS-RUNNER ([\d\- :]+)\] Verified atomic sync OK: (\S+) \(SHA256=(\w+)\)")
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
STATE = {"usage": {}, "sessions": "", "ts": 0.0, "stop_msg": ""}
LOCK = threading.Lock()


def colab_bin():
    cand = os.environ.get("COLAB_BIN") or str(Path.home() / ".colab-venv/bin/colab")
    return cand if Path(cand).exists() else "colab"


def run_colab(*args, timeout=40):
    env = dict(os.environ, GOOGLE_APPLICATION_CREDENTIALS=CRED)
    try:
        out = subprocess.run([colab_bin(), "--auth", "adc", *args], capture_output=True,
                             text=True, timeout=timeout, env=env, check=False)
        return (out.stdout + out.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"ERROR: {exc}"


def parse_usage(text):
    keys = {"Current balance": "balance", "Usage rate": "rate", "Active assignments": "active"}
    return {v: m.group(1).strip() for k, v in keys.items()
            if (m := re.search(rf"{k}:\s*(.+)", text))}


def refresh_colab():
    usage = run_colab("usage")
    sessions = run_colab("sessions")
    with LOCK:
        STATE.update(usage=parse_usage(usage), usage_raw=usage, sessions=sessions, ts=time.time())


def refresher(interval):
    while True:
        refresh_colab()
        time.sleep(interval)


def clean_lines(text):
    lines = [ANSI.sub("", x).rstrip() for x in re.split(r"[\r\n]+", text)]
    return [x for x in lines if x.strip() and not NOISE.search(x)]


def tail_text(path, max_bytes=300_000):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - max_bytes))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def parse_epochs(lines):
    by_epoch = {}
    for line in lines:
        m = EPOCH_RE.search(line)
        if m:
            by_epoch[int(m.group(1))] = {k.strip(): float(v) if v != "nan" else None
                                         for k, v in METRIC_RE.findall(m.group(2))}
    return [{"epoch": e, "metrics": by_epoch[e]} for e in sorted(by_epoch)]


def list_sessions():
    if not RUNS_DIR.is_dir():
        return []
    dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    return [d.name for d in sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)]


def read_meta(session):
    meta = {}
    try:
        for line in (RUNS_DIR / session / ".state" / "meta.env").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.lower()] = v
    except OSError:
        return meta
    for key in ("attempt", "stop_reason"):
        p = RUNS_DIR / session / ".state" / key
        meta[key] = p.read_text().strip() if p.exists() else ""
    return meta


def checkpoints(session, log_text):
    syncs = {}
    for ts, path, sha in SYNC_RE.findall(log_text):
        syncs[Path(path).name] = {"synced_at": ts, "sha": sha[:12]}
    rows = []
    for sub in ("checkpoints", "artifacts"):
        d = RUNS_DIR / session / sub
        for f in sorted(d.glob("*")) if d.is_dir() else []:
            if f.is_file() and ".tmp." not in f.name:
                st = f.stat()
                info = syncs.get(f.name, {})
                rows.append({"name": f.name, "kind": sub, "size_mb": round(st.st_size / 1e6, 1),
                             "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                             "age_s": int(time.time() - st.st_mtime), "sha": info.get("sha", "")})
    return rows


def runner_pids():
    pids = []
    proc = Path("/proc")
    if not proc.is_dir():
        return pids
    for p in proc.iterdir():
        if p.name.isdigit() and int(p.name) != os.getpid():
            try:
                cmd = (p / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
            except OSError:
                continue
            joined = " ".join(cmd)
            if "run_colab_vps.sh" in joined and not any(f in joined for f in ("--stop-all", "--status", "--dry-run")):
                pids.append(int(p.name))
    return pids


def system_stats():
    stats = {}
    try:
        stats["load"] = Path("/proc/loadavg").read_text().split()[:3]
        mem = dict(x.replace(" kB", "").split(":", 1) for x in Path("/proc/meminfo").read_text().splitlines() if ":" in x)
        total, avail = int(mem["MemTotal"]) / 1e6, int(mem["MemAvailable"]) / 1e6
        stats["mem"] = f"{total - avail:.1f} / {total:.1f} GB"
    except (OSError, KeyError, ValueError):
        pass
    du = shutil.disk_usage(RUNNER_DIR)
    stats["disk"] = f"{du.used / 1e9:.1f} / {du.total / 1e9:.1f} GB"
    return stats


def parse_params(log_text, lines):
    """Extract the run's main configuration from notebook log lines."""
    rules = [
        ("Thiết bị", r"\[READY\] Compute Device: (.+)"), ("AMP", r"AMP Dtype: (\S+)"),
        ("Dữ liệu train", r"Train batches/epoch: (\d+ \(.*?\))"), ("Val", r"Val batches/epoch: (\d+ \(.*?\))"),
        ("Hàm loss", r"\[PARITY\] Loss=(.*)"), ("Epoch / LR / Steps", r"\[PARITY\] (Epochs=.*)"),
        ("Pretrained", r"Pretrained: (.+)"), ("Mô hình", r"\[MODEL\] (Stage \d[^:]*|DeepLab[^:]*)"),
    ]
    params = []
    for label, pat in rules:
        for line in reversed(lines):
            m = re.search(pat, line)
            if m:
                params.append([label, m.group(1).strip()])
                break
    cur = re.findall(r"Epoch\s+(\d+)/(\d+)", log_text)
    return params, (int(cur[-1][0]), int(cur[-1][1])) if cur else (0, 0)


def progress(epochs, cur_total, meta, running):
    cur, total = cur_total
    names = [k for e in epochs for k in e["metrics"]]
    pick = next((k for k in names if "PM_mIoU" in k), None) or next((k for k in names if "mIoU" in k), None)
    best = None
    if pick:
        vals = [(e["metrics"][pick], e["epoch"]) for e in epochs if e["metrics"].get(pick) is not None]
        best = {"name": pick, "value": max(vals)[0], "epoch": max(vals)[1]} if vals else None
    start = int(meta["start_epoch"]) if meta.get("start_epoch", "").isdigit() else 0
    elapsed = int(time.time()) - start if start and running else None
    done = len(epochs)
    eta = int(elapsed / done * (total - done)) if elapsed and done and total > done else None
    return {"epoch": cur, "total": total, "done": done, "best": best, "elapsed_s": elapsed, "eta_s": eta,
            "last": epochs[-1] if epochs else None}


def build_status(session=None):
    sessions = list_sessions()
    session = session if session in sessions else (sessions[0] if sessions else "")
    log_text = tail_text(RUNS_DIR / session / "run.log", 2_000_000) if session else ""
    lines = clean_lines(log_text)
    meta = read_meta(session) if session else {}
    deadline = int(meta["deadline_epoch"]) if meta.get("deadline_epoch", "").isdigit() else 0
    with LOCK:
        usage, sess_raw, ts, stop_msg = dict(STATE["usage"]), STATE["sessions"], STATE["ts"], STATE["stop_msg"]
    pids = runner_pids()
    epochs = parse_epochs(lines)
    params, cur_total = parse_params(log_text, lines)
    return {
        "now": time.strftime("%Y-%m-%d %H:%M:%S"), "colab_ts": int(ts), "usage": usage, "colab_sessions": sess_raw,
        "runner_running": bool(pids), "runner_pids": pids, "sessions": sessions, "session": session, "meta": meta,
        "remaining_s": max(0, deadline - int(time.time())) if deadline and pids else None,
        "checkpoints": checkpoints(session, log_text) if session else [],
        "epochs": epochs, "params": params, "progress": progress(epochs, cur_total, meta, bool(pids)),
        "events": [x for x in lines if any(k in x for k in EVENT_KEYS)][-40:],
        "log": lines[-150:], "system": system_stats(), "stop_msg": stop_msg,
    }


def do_stop():
    for pid in runner_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    time.sleep(3)
    msg = subprocess.run(["bash", str(RUNNER_SH), "--stop-all"], capture_output=True, text=True,
                         timeout=120, cwd=RUNNER_DIR.parent, check=False)
    text = (msg.stdout + msg.stderr).strip()[-600:]
    refresh_colab()
    with LOCK:
        STATE["stop_msg"] = f"{time.strftime('%H:%M:%S')} Đã gửi lệnh tắt GPU. {text}"
    return STATE["stop_msg"]


def do_start(payload):
    """Launch a resilient runner inside tmux. Raises ValueError with a user-facing reason."""
    nb, gpu = str(payload.get("notebook", "")), str(payload.get("gpu", "L4"))
    try:
        minutes = int(payload.get("max_minutes") or (210 if nb == "nb1" else 240))
    except (TypeError, ValueError):
        raise ValueError("max_minutes không hợp lệ") from None
    if nb not in ("nb1", "nb2") or gpu not in ("L4", "T4") or not 10 <= minutes <= 300:
        raise ValueError("Tham số không hợp lệ (nb1|nb2, L4|T4, 10-300 phút)")
    if runner_pids():
        raise ValueError("Runner đang chạy, hãy tắt phiên hiện tại trước")
    with LOCK:
        active = str(STATE["usage"].get("active", "0"))
    if active not in ("0", ""):
        raise ValueError(f"Colab đang có {active} GPU hoạt động, không khởi chạy thêm")
    if not shutil.which("tmux"):
        raise ValueError("VPS chưa cài tmux")
    name = f"colab_{nb}"
    subprocess.run(["tmux", "kill-session", "-t", name], capture_output=True, check=False)
    cmd = f"bash {shlex.quote(str(RUNNER_SH))} --notebook {nb} --gpu {gpu} --max-minutes {minutes}"
    subprocess.run(["tmux", "new-session", "-d", "-s", name, cmd], cwd=RUNNER_DIR.parent, check=True)
    return f"Đã khởi chạy {nb} ({gpu}, giới hạn {minutes} phút) trong tmux '{name}'."


def read_form(handler, limit=4096):
    length = min(int(handler.headers.get("Content-Length") or 0), limit)
    return {k: v[0] for k, v in parse_qs(handler.rfile.read(length).decode("utf-8", "replace")).items()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def _host_ok(self):
        """Auth disabled: accept only loopback Host headers (DNS-rebinding guard)."""
        host = self.headers.get("Host") or ""
        host = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        return host in ALLOWED_HOSTS

    def _cookie(self, name):
        jar = {k.strip(): v for k, _, v in (c.partition("=") for c in (self.headers.get("Cookie") or "").split(";"))}
        return jar.get(name, "")

    def _ip(self):
        return self.headers.get("CF-Connecting-IP") or self.client_address[0]

    def _auth_kind(self):
        """Return 'session', 'apikey', 'open' (auth disabled, loopback) or '' when unauthorized."""
        if not dash_auth.configured():
            return "open" if self._host_ok() else ""
        if dash_auth.session_valid(self._cookie("dash_session")):
            return "session"
        return "apikey" if dash_auth.api_key_ok(self.headers.get("X-API-Key")) else ""

    def _send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _login_page(self, code=200, error=""):
        html = (RUNNER_DIR / "login.html").read_text(encoding="utf-8")
        note = f'<div class="err">{error}</div>' if error else ""
        return self._send(code, html.replace("<!--ERR-->", note), "text/html; charset=utf-8")

    def _do_login(self):
        ip = self._ip()
        if dash_auth.locked(ip):
            return self._login_page(429, "Sai quá nhiều lần. Thử lại sau 10 phút.")
        form = read_form(self)
        if not dash_auth.verify_login(form.get("username", ""), form.get("password", "")):
            dash_auth.record_fail(ip)
            return self._login_page(401, "Sai tài khoản hoặc mật khẩu.")
        dash_auth.clear_fails(ip)
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
        cookie = f"dash_session={dash_auth.new_session()}; Path=/; HttpOnly; SameSite=Lax; Max-Age={dash_auth.SESSION_TTL}{secure}"
        return self._send(302, b"", headers={"Location": "/", "Set-Cookie": cookie})

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/login" and dash_auth.configured():
            return self._login_page()
        if url.path == "/logout":
            dash_auth.drop_session(self._cookie("dash_session"))
            return self._send(302, b"", headers={"Location": "/login", "Set-Cookie": "dash_session=; Path=/; Max-Age=0"})
        if not self._auth_kind():
            if not dash_auth.configured():
                return self._send(403, '{"error":"bad host"}')
            if url.path.startswith("/api/"):
                return self._send(401, '{"error":"unauthorized"}')
            return self._send(302, b"", headers={"Location": "/login"})
        if url.path == "/api/status":
            sess = parse_qs(url.query).get("session", [None])[0]
            return self._send(200, json.dumps(build_status(sess)))
        if url.path in ("/", "/index.html"):
            html = (RUNNER_DIR / "dashboard.html").read_text(encoding="utf-8")
            return self._send(200, html, "text/html; charset=utf-8")
        return self._send(404, '{"error":"not found"}')

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/login" and dash_auth.configured():
            return self._do_login()
        kind = self._auth_kind()
        if not kind:
            return self._send(401, '{"error":"unauthorized"}')
        if kind != "apikey" and self.headers.get("X-Requested-With") != "dashboard":
            return self._send(403, '{"error":"forbidden"}')
        if path == "/api/stop":
            return self._send(200, json.dumps({"message": do_stop()}))
        if path == "/api/start":
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 2048)) or b"{}")
                return self._send(200, json.dumps({"message": do_start(body)}))
            except (ValueError, subprocess.SubprocessError) as exc:
                return self._send(400, json.dumps({"error": str(exc)}))
        return self._send(404, '{"error":"not found"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--colab-refresh", type=int, default=20, help="seconds between colab usage/sessions polls")
    args = ap.parse_args()
    dash_auth.load()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not dash_auth.configured():
        raise SystemExit("Refusing to bind a non-loopback address without credentials (python3 dash_auth.py --init)")
    threading.Thread(target=refresher, args=(args.colab_refresh,), daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    mode = "login required" if dash_auth.configured() else "loopback only, no login"
    print(f"Dashboard on http://{args.host}:{args.port} ({mode})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
