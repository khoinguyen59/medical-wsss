#!/usr/bin/env python3
"""Temporary Colab-on-VPS monitoring dashboard (stdlib only, light blue/white theme)."""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

RUNNER_DIR = Path(__file__).resolve().parent
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import dash_auth  # noqa: E402
import session_guard  # noqa: E402

RUNS_DIR, RUNNER_SH = RUNNER_DIR / "runs", RUNNER_DIR / "run_colab_vps.sh"
CRED = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", str(Path.home() / ".config/gcloud/application_default_credentials.json"))
NOISE = re.compile(r"Exception ignored|Traceback \(most|_shutdown_workers|is_alive\(|dataloader\.py|_parent_pid|can only test a child|^\s*self\._shutdown|^\s*if w\.|^\s*assert self|^\s*File \"")
ANSI, EPOCH_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]"), re.compile(r"\[EPOCH\s+(\d+)\]\s*(.*)")
METRIC_RE = re.compile(r"([A-Za-z][A-Za-z0-9_() ]*?)\s*[=:]\s*(-?\d+\.\d+|nan)")
EVENT_KEYS = ("Verified atomic sync", "WATCHDOG", "ALARM", "Attempt", "CLEANUP", "SUCCESS", "STOPPED", "Halting", "WARN", "CHECKPOINT", "Uploading", "Notebook execution", "Connection was lost")
SYNC_RE = re.compile(r"\[VPS-RUNNER ([\d\- :]+)\] Verified atomic sync OK: (\S+) \(SHA256=(\w+)\)")
PRESET_RE, ENV_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$"), re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
STATE = {"usage": {}, "sessions": "", "ts": 0.0, "stop_msg": ""}
LOCK = threading.Lock()
_START_LOCK = threading.Lock()
_PENDING_STARTS: dict[str, float] = {}


def max_owned_concurrent():
    try:
        return max(1, int(os.environ.get("COLAB_MAX_OWNED_CONCURRENT", "1")))
    except ValueError:
        return 1


def colab_bin():
    cand = os.environ.get("COLAB_BIN") or str(Path.home() / ".colab-venv/bin/colab")
    return cand if Path(cand).exists() else "colab"


def run_colab(*args, timeout=40):
    ro_cfg = session_guard.resolve_state_dir() / ".readonly_sessions.json"
    ro_cfg.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = subprocess.run([colab_bin(), "--config", str(ro_cfg), "--auth", "adc", *args], capture_output=True, text=True, timeout=timeout, env=dict(os.environ, GOOGLE_APPLICATION_CREDENTIALS=CRED), check=False)
        return (out.stdout + out.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"ERROR: {exc}"


def parse_usage(text):
    keys = {"Current balance": "balance", "Usage rate": "rate", "Active assignments": "active"}
    return {v: m.group(1).strip() for k, v in keys.items() if (m := re.search(rf"{k}:\s*(.+)", text))}


def refresh_colab():
    usage, sessions = run_colab("usage"), run_colab("sessions")
    with LOCK:
        STATE.update(usage=parse_usage(usage), usage_raw=usage, sessions=sessions, ts=time.time())


def refresher(interval):
    while True:
        refresh_colab()
        time.sleep(interval)


def clean_lines(text):
    return [x for x in (ANSI.sub("", ln).rstrip() for ln in re.split(r"[\r\n]+", text)) if x.strip() and not NOISE.search(x)]


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
        if m := EPOCH_RE.search(line):
            by_epoch[int(m.group(1))] = {k.strip(): float(v) if v != "nan" else None for k, v in METRIC_RE.findall(m.group(2))}
    return [{"epoch": e, "metrics": by_epoch[e]} for e in sorted(by_epoch)]


def list_sessions():
    return [d.name for d in sorted((d for d in RUNS_DIR.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)] if RUNS_DIR.is_dir() else []


def read_meta(session):
    meta = {}
    try:
        meta = {k.lower(): v for ln in (RUNS_DIR / session / ".state" / "meta.env").read_text().splitlines() if "=" in ln for k, v in [ln.split("=", 1)]}
    except OSError:
        return meta
    for key in ("attempt", "stop_reason"):
        p = RUNS_DIR / session / ".state" / key
        meta[key] = p.read_text().strip() if p.exists() else ""
    return meta


def checkpoints(session, log_text):
    syncs = {Path(path).name: sha[:12] for _, path, sha in SYNC_RE.findall(log_text)}
    rows = []
    for sub in ("checkpoints", "artifacts"):
        d = RUNS_DIR / session / sub
        for f in sorted(d.glob("*")) if d.is_dir() else []:
            if f.is_file() and ".tmp." not in f.name:
                st = f.stat()
                rows.append({"name": f.name, "kind": sub, "size_mb": round(st.st_size / 1e6, 1), "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)), "age_s": int(time.time() - st.st_mtime), "sha": syncs.get(f.name, "")})
    return rows


def runner_pids():
    return session_guard.scan_runner_pids()


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
    rules = (("Thiết bị", r"\[READY\] Compute Device: (.+)"), ("AMP", r"AMP Dtype: (\S+)"), ("Dữ liệu train", r"Train batches/epoch: (\d+ \(.*?\))"), ("Val", r"Val batches/epoch: (\d+ \(.*?\))"), ("Hàm loss", r"\[PARITY\] Loss=(.*)"), ("Epoch / LR / Steps", r"\[PARITY\] (Epochs=.*)"), ("Pretrained", r"Pretrained: (.+)"), ("Mô hình", r"\[MODEL\] (Stage \d[^:]*|DeepLab[^:]*)"))
    params = [[lbl, m.group(1).strip()] for lbl, pat in rules if (m := next((m for ln in reversed(lines) if (m := re.search(pat, ln))), None))]
    cur = re.findall(r"Epoch\s+(\d+)/(\d+)", log_text)
    return params, (int(cur[-1][0]), int(cur[-1][1])) if cur else (0, 0)


def progress(epochs, cur_total, meta, running):
    cur, total = cur_total
    names = [k for e in epochs for k in e["metrics"]]
    pick = next((k for k in names if "PM_mIoU" in k), None) or next((k for k in names if "mIoU" in k), None)
    vals = [(e["metrics"][pick], e["epoch"]) for e in epochs if pick and e["metrics"].get(pick) is not None]
    best = {"name": pick, "value": max(vals)[0], "epoch": max(vals)[1]} if vals else None
    start = int(meta["start_epoch"]) if meta.get("start_epoch", "").isdigit() else 0
    elapsed = int(time.time()) - start if start and running else None
    done = len(epochs)
    eta = int(elapsed / done * (total - done)) if elapsed and done and total > done else None
    return {"epoch": cur, "total": total, "done": done, "best": best, "elapsed_s": elapsed, "eta_s": eta, "last": epochs[-1] if epochs else None}


def summarize_sessions(sessions, pids_map):
    rows, now_ts = [], int(time.time())
    for s in sessions:
        m, txt = read_meta(s), tail_text(RUNS_DIR / s / "run.log", 200_000)
        lines = clean_lines(txt)
        spids = pids_map.get(s, [])
        prog = progress(parse_epochs(lines), parse_params(txt, lines)[1], m, bool(spids))
        dl = int(m["deadline_epoch"]) if m.get("deadline_epoch", "").isdigit() else 0
        rows.append({"session": s, "notebook": m.get("notebook", "").split("/")[-1], "gpu": m.get("gpu", ""), "high_mem": m.get("high_mem", "") == "1", "preset": m.get("preset", ""), "running": bool(spids), "pids": spids, "epoch": prog["epoch"], "total": prog["total"], "best": prog["best"], "remaining_s": max(0, dl - now_ts) if dl and spids else None, "stop_reason": m.get("stop_reason", ""), "attempt": m.get("attempt", "")})
    return rows


def build_status(session=None):
    sessions = list_sessions()
    session = session if session in sessions else (sessions[0] if sessions else "")
    log_text = tail_text(RUNS_DIR / session / "run.log", 2_000_000) if session else ""
    lines, meta = clean_lines(log_text), read_meta(session) if session else {}
    deadline = int(meta["deadline_epoch"]) if meta.get("deadline_epoch", "").isdigit() else 0
    with LOCK:
        usage, sess_raw, ts, stop_msg = dict(STATE["usage"]), STATE["sessions"], STATE["ts"], STATE["stop_msg"]
    pids_map = runner_pids()
    sess_pids = pids_map.get(session, []) if session else []
    epochs, (params, cur_total) = parse_epochs(lines), parse_params(log_text, lines)
    assigns = session_guard.classify_colab_sessions(sess_raw, cred_path=CRED)
    return {
        "now": time.strftime("%Y-%m-%d %H:%M:%S"), "colab_ts": int(ts), "usage": usage, "colab_sessions": sess_raw, "colab_assignments": assigns,
        "owned_active_count": assigns["owned_active_count"], "foreign_active_count": assigns["foreign_active_count"], "max_owned_concurrent": max_owned_concurrent(),
        "runner_running": bool(sess_pids), "runner_pids": sess_pids, "all_runner_pids": pids_map, "sessions": sessions, "session": session,
        "all_sessions_summary": summarize_sessions(sessions, pids_map), "meta": meta, "remaining_s": max(0, deadline - int(time.time())) if deadline and sess_pids else None,
        "checkpoints": checkpoints(session, log_text) if session else [], "epochs": epochs, "params": params, "progress": progress(epochs, cur_total, meta, bool(sess_pids)),
        "events": [x for x in lines if any(k in x for k in EVENT_KEYS)][-40:], "log": lines[-150:], "system": system_stats(), "stop_msg": stop_msg,
    }


def _kill_pids(pid_list):
    signaled = False
    for pid in pid_list:
        try:
            os.kill(pid, signal.SIGTERM)
            signaled = True
        except OSError:
            pass
    return signaled


def do_stop(payload=None):
    payload = payload if isinstance(payload, dict) else {}
    target = str(payload.get("session") or "").strip()
    scope = str(payload.get("scope") or ("session" if "session" in payload else "owned_all")).strip()
    pids_map = runner_pids()
    if scope == "session":
        if not session_guard.is_valid_session_name(target):
            raise ValueError(f"Tên phiên không hợp lệ: {target}")
        killed, stop_args = _kill_pids(pids_map.get(target, [])), ["--stop-session", target]
    elif scope == "owned_all":
        killed, stop_args = _kill_pids([p for plist in pids_map.values() for p in plist]), ["--stop-owned"]
    else:
        raise ValueError(f"Scope không hợp lệ: {scope}")
    if killed:
        time.sleep(3)
    msg = subprocess.run(["bash", str(RUNNER_SH), *stop_args], capture_output=True, text=True, timeout=120, cwd=RUNNER_DIR.parent, check=False)
    refresh_colab()
    out, rc = (msg.stdout + "\n" + msg.stderr).strip(), getattr(msg, "returncode", 0)
    if rc != 0 or "REFUSED" in out:
        with LOCK:
            STATE["stop_msg"] = f"{time.strftime('%H:%M:%S')} Từ chối/lỗi dừng phiên ({target or 'owned_all'}): {out[-600:]}"
        raise RuntimeError(out[-600:] or f"Dừng phiên thất bại (exit {rc})")
    with _START_LOCK:
        _PENDING_STARTS.pop(target, None) if scope == "session" else _PENDING_STARTS.clear()
    with LOCK:
        STATE["stop_msg"] = f"{time.strftime('%H:%M:%S')} Đã dừng phiên owned ({target or 'owned_all'}). {out[-600:]}"
    return STATE["stop_msg"]


def _parse_env_items(raw_env):
    if isinstance(raw_env, dict):
        pairs = raw_env.items()
    elif isinstance(raw_env, list):
        pairs = [x.split("=", 1) for x in raw_env if isinstance(x, str) and "=" in x]
    else:
        raise ValueError("Tham số env phải là dict hoặc danh sách KEY=VALUE")
    items = []
    for k, v in pairs:
        ks, vs = str(k).strip(), str(v)
        if not ENV_KEY_RE.match(ks) or any(c in vs for c in ("\x00", "\r", "\n")):
            raise ValueError(f"Biến môi trường không hợp lệ: {ks}")
        items.append(f"{ks}={vs}")
    return items


def do_start(payload):
    """Launch a resilient owned runner inside tmux. Raises ValueError with a user-facing reason."""
    nb, gpu = str(payload.get("notebook", "")), str(payload.get("gpu", "L4")).upper()
    try:
        minutes = int(payload.get("max_minutes") or (210 if nb == "nb1" else 240))
    except (TypeError, ValueError):
        raise ValueError("max_minutes không hợp lệ") from None
    if nb not in ("nb1", "nb2") or gpu not in session_guard.ALLOWED_GPUS or not 10 <= minutes <= 720:
        raise ValueError("Tham số không hợp lệ (nb1|nb2, T4|L4|G4|A100|H100, 10-720 phút)")
    raw_hm = payload.get("high_mem")
    high_mem = False if gpu == "L4" else ((gpu in ("H100", "G4")) if raw_hm is None else bool(raw_hm))
    session = str(payload.get("session") or "").strip() or f"se374_{nb}_{gpu.lower()}_{time.strftime('%H%M%S')}"
    if not session_guard.is_valid_session_name(session):
        raise ValueError("Tên phiên phải khớp ^se374_[a-zA-Z0-9_-]+$ (hoặc nb1/nb2)")
    preset = str(payload.get("preset") or "").strip()
    if preset and not PRESET_RE.match(preset):
        raise ValueError("Tên preset không hợp lệ")
    fb = str(payload.get("gpu_fallback") or "").strip().upper()
    if fb and any(t not in session_guard.ALLOWED_GPUS for t in fb.split(",") if t):
        raise ValueError("gpu_fallback chứa GPU không hợp lệ")
    env_items = _parse_env_items(payload.get("env") or {})
    if not shutil.which("tmux"):
        raise ValueError("VPS chưa cài tmux")
    tmux_name = f"colab_{session}"
    parts = ["exec", "bash", shlex.quote(str(RUNNER_SH)), "--notebook", nb, "--gpu", gpu, "--session", shlex.quote(session), "--max-minutes", str(minutes), "--high-mem" if high_mem else "--no-high-mem"]
    parts += (["--preset", shlex.quote(preset)] if preset else []) + (["--gpu-fallback", shlex.quote(fb)] if fb else []) + [x for kv in env_items for x in ("--env", shlex.quote(kv))]
    with _START_LOCK:
        now, pids_map = time.time(), runner_pids()
        for k in [s for s, exp in _PENDING_STARTS.items() if s in pids_map or exp <= now]:
            _PENDING_STARTS.pop(k, None)
        if pids_map.get(session) or session in _PENDING_STARTS:
            raise ValueError(f"Phiên '{session}' đang chạy")
        with LOCK:
            sess_raw = STATE["sessions"]
        assigns = session_guard.classify_colab_sessions(sess_raw, cred_path=CRED)
        owned_active = max(len(set(pids_map) | set(_PENDING_STARTS)), assigns["owned_active_count"])
        if owned_active >= max_owned_concurrent():
            raise ValueError(f"Đã đạt giới hạn phiên owned đồng thời ({owned_active}/{max_owned_concurrent()})")
        subprocess.run(["tmux", "kill-session", "-t", tmux_name], capture_output=True, check=False)
        subprocess.run(["tmux", "new-session", "-d", "-s", tmux_name, " ".join(parts)], cwd=RUNNER_DIR.parent, check=True)
        _PENDING_STARTS[session] = now + 15.0
    return f"Đã khởi chạy {session} ({nb}, {gpu}, high_mem={high_mem}, {minutes} phút) trong tmux '{tmux_name}'."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def _cookie(self, name):
        return {k.strip(): v for k, _, v in (c.partition("=") for c in (self.headers.get("Cookie") or "").split(";"))}.get(name, "")

    def _auth_kind(self):
        if not dash_auth.configured():
            host = self.headers.get("Host") or ""
            host = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
            return "open" if host in ALLOWED_HOSTS else ""
        if dash_auth.session_valid(self._cookie("dash_session")):
            return "session"
        return "apikey" if dash_auth.api_key_ok(self.headers.get("X-API-Key")) else ""

    def _send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        for k, v in {"Content-Type": ctype, "Content-Length": str(len(data)), "Cache-Control": "no-store", **(headers or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _login_page(self, code=200, error=""):
        html = (RUNNER_DIR / "login.html").read_text(encoding="utf-8")
        return self._send(code, html.replace("<!--ERR-->", f'<div class="err">{error}</div>' if error else ""), "text/html; charset=utf-8")

    def _do_login(self):
        ip = self.headers.get("CF-Connecting-IP") or self.client_address[0]
        if dash_auth.locked(ip):
            return self._login_page(429, "Sai quá nhiều lần. Thử lại sau 10 phút.")
        length = min(int(self.headers.get("Content-Length") or 0), 4096)
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode("utf-8", "replace")).items()}
        if not dash_auth.verify_login(form.get("username", ""), form.get("password", "")):
            dash_auth.record_fail(ip)
            return self._login_page(401, "Sai tài khoản hoặc mật khẩu.")
        dash_auth.clear_fails(ip)
        sec = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
        return self._send(302, b"", headers={"Location": "/", "Set-Cookie": f"dash_session={dash_auth.new_session()}; Path=/; HttpOnly; SameSite=Lax; Max-Age={dash_auth.SESSION_TTL}{sec}"})

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
            return self._send(401, '{"error":"unauthorized"}') if url.path.startswith("/api/") else self._send(302, b"", headers={"Location": "/login"})
        if url.path == "/api/status":
            return self._send(200, json.dumps(build_status(parse_qs(url.query).get("session", [None])[0])))
        if url.path in ("/", "/index.html"):
            return self._send(200, (RUNNER_DIR / "dashboard.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
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
        try:
            raw = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 2048))
            body = json.loads(raw) if raw else {}
        except ValueError:
            return self._send(400, '{"error":"invalid json"}')
        if path in ("/api/stop", "/api/start"):
            try:
                fn = do_stop if path == "/api/stop" else do_start
                return self._send(200, json.dumps({"message": fn(body)}))
            except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                return self._send(400, json.dumps({"error": str(exc)}))
        return self._send(404, '{"error":"not found"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--colab-refresh", type=int, default=60, help="seconds between colab usage/sessions polls")
    args = ap.parse_args()
    dash_auth.load()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not dash_auth.configured():
        raise SystemExit("Refusing to bind a non-loopback address without credentials (python3 dash_auth.py --init)")
    threading.Thread(target=refresher, args=(args.colab_refresh,), daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard on http://{args.host}:{args.port} ({'login required' if dash_auth.configured() else 'loopback only, no login'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
