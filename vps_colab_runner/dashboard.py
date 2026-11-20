#!/usr/bin/env python3
"""Temporary Colab-on-VPS monitoring dashboard (stdlib only, light blue/white theme).

Binds to 127.0.0.1 by default. Access it through an SSH tunnel, no public port needed:
    ssh -L 8765:127.0.0.1:8765 -i <key> ubuntu@<vps-ip>   then open http://127.0.0.1:8765
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


def build_status(session=None):
    sessions = list_sessions()
    session = session if session in sessions else (sessions[0] if sessions else "")
    log_text = tail_text(RUNS_DIR / session / "run.log") if session else ""
    lines = clean_lines(log_text)
    meta = read_meta(session) if session else {}
    deadline = int(meta["deadline_epoch"]) if meta.get("deadline_epoch", "").isdigit() else 0
    with LOCK:
        usage, sess_raw, ts, stop_msg = dict(STATE["usage"]), STATE["sessions"], STATE["ts"], STATE["stop_msg"]
    pids = runner_pids()
    return {
        "now": time.strftime("%Y-%m-%d %H:%M:%S"), "colab_ts": int(ts), "usage": usage, "colab_sessions": sess_raw,
        "runner_running": bool(pids), "runner_pids": pids, "sessions": sessions, "session": session, "meta": meta,
        "remaining_s": max(0, deadline - int(time.time())) if deadline and pids else None,
        "checkpoints": checkpoints(session, log_text) if session else [],
        "epochs": parse_epochs(lines),
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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def _host_ok(self):
        """Reject unexpected Host headers (DNS-rebinding guard); port part is ignored."""
        host = self.headers.get("Host") or ""
        host = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        return host in ALLOWED_HOSTS

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, '{"error":"bad host"}')
        url = urlparse(self.path)
        if url.path == "/api/status":
            sess = parse_qs(url.query).get("session", [None])[0]
            return self._send(200, json.dumps(build_status(sess)))
        if url.path in ("/", "/index.html"):
            html = (RUNNER_DIR / "dashboard.html").read_text(encoding="utf-8")
            return self._send(200, html, "text/html; charset=utf-8")
        return self._send(404, '{"error":"not found"}')

    def do_POST(self):
        if not self._host_ok() or self.headers.get("X-Requested-With") != "dashboard":
            return self._send(403, '{"error":"forbidden"}')
        if urlparse(self.path).path == "/api/stop":
            return self._send(200, json.dumps({"message": do_stop()}))
        return self._send(404, '{"error":"not found"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--colab-refresh", type=int, default=20, help="seconds between colab usage/sessions polls")
    args = ap.parse_args()
    threading.Thread(target=refresher, args=(args.colab_refresh,), daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard on http://{args.host}:{args.port} (SSH tunnel recommended)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
