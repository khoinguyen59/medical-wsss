#!/usr/bin/env python3
"""P0 ComfyUI-safe session guard and owned-session registry (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

fcntl = None if os.name == "nt" else __import__("fcntl")
DEFAULT_STATE_DIR = Path(__file__).resolve().parent / ".state"
DEFAULT_CRED = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", str(Path.home() / ".config/gcloud/application_default_credentials.json"))
ALLOWED_GPUS = ("T4", "L4", "G4", "A100", "H100")
SESSION_RE = re.compile(r"^(se374_[a-zA-Z0-9_-]+|nb1|nb2)$")
SESS_LINE_RE = re.compile(r"^\[([^\]]+)\]\s+(\S+)\s+\|\s*(.*)$")
ERR_RE = re.compile(r"(?:^|\n)\s*(?:ERROR:|Traceback|.*(?:Unauthorized|invalid_grant|(?<![\w-])403(?![\w-])|Forbidden|Exception|No valid default credentials|gcloud auth application-default login|Warning: failed|EOF when reading a line))", re.I)
_THREAD_LOCK = threading.RLock()

def is_valid_session_name(name: str) -> bool:
    return bool(name and SESSION_RE.match(name))
def resolve_state_dir(state_dir: str | Path | None = None) -> Path:
    return Path(state_dir or os.environ.get("COLAB_GUARD_STATE_DIR") or DEFAULT_STATE_DIR)
def resolve_account_name(cred_path: str | Path | None = None) -> str:
    if override := os.environ.get("COLAB_ACCOUNT_NAME", "").strip():
        return re.sub(r"[^a-zA-Z0-9_-]", "_", override)[:48] or "default"
    path = Path(cred_path or DEFAULT_CRED).expanduser()
    if isinstance(data := _load_opt_json(path), dict) and (ident := "|".join(str(data.get(k, "")) for k in ("client_email", "refresh_" + "token", "client_id", "quota_project_id"))).strip("|"):
        return "acct_" + hashlib.sha256(ident.encode("utf-8")).hexdigest()[:12]
    return "acct_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
def _paths(state_dir: Path, cred_path: str | Path | None) -> tuple[str, Path, Path, Path]:
    acct = resolve_account_name(cred_path)
    return acct, state_dir / f"{acct}_protected_endpoints.json", state_dir / f"{acct}_owned_sessions.json", state_dir / f"{acct}.lock"
def provision_lock_path(state_dir: Path, cred_path: str | Path | None = None) -> Path:
    return state_dir / f"{resolve_account_name(cred_path)}_provision.lock"
@contextmanager
def _guard_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK, open(lock_path, "a+", encoding="utf-8") as fh:
        fcntl and fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl and fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
def _read_json(path: Path, default: dict) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(json.dumps(default))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Corrupt guard state {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid guard state root in {path}")
    return data
def _require_prot(prot_path: Path, acct: str, msg: str) -> dict:
    if not prot_path.is_file():
        raise RuntimeError(msg)
    return _read_json(prot_path, {"account": acct, "endpoints": {}})
def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
def parse_colab_sessions(text: str) -> list[dict]:
    return [{"name": m.group(1).strip(), "endpoint": m.group(2).strip(), "raw": ln} for raw in (text or "").splitlines() if (m := SESS_LINE_RE.match(ln := raw.strip()))]
def _load_opt_json(cfg_path: str | Path | None) -> object:
    try:
        return json.loads(Path(cfg_path).read_text(encoding="utf-8")) if cfg_path and Path(cfg_path).is_file() else None
    except (OSError, ValueError):
        return None
def extract_endpoint_from_config(cfg_path: str | Path | None, session_name: str) -> str | None:
    data = _load_opt_json(cfg_path)
    if isinstance(data, dict):
        node = data.get(session_name) or (data["sessions"].get(session_name) if isinstance(data.get("sessions"), dict) else None)
        return str(node["endpoint"]) if isinstance(node, dict) and node.get("endpoint") else None
    return next((str(x["endpoint"]) for x in data if isinstance(x, dict) and x.get("name") == session_name and x.get("endpoint")), None) if isinstance(data, list) else None
def _run_colab_global(colab_bin: str, cred_path: str, state_dir: Path | None, *args: str) -> tuple[int, str]:
    ro_cfg = resolve_state_dir(state_dir) / ".readonly_sessions.json"
    ro_cfg.parent.mkdir(parents=True, exist_ok=True)
    try:
        res = subprocess.run([colab_bin, "--config", str(ro_cfg), "--auth", "adc", *args], capture_output=True, text=True, timeout=45, env=dict(os.environ, GOOGLE_APPLICATION_CREDENTIALS=cred_path), check=False)
        return res.returncode, (res.stdout + "\n" + res.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"ERROR: {exc}"
def _colab_stop(colab_bin: str, cred_path: str, cfg_str: str, session_name: str) -> tuple[bool, int, str]:
    res = subprocess.run([colab_bin, "--config", cfg_str, "--auth", "adc", "stop", "-s", session_name], capture_output=True, text=True, timeout=60, env=dict(os.environ, GOOGLE_APPLICATION_CREDENTIALS=cred_path), check=False)
    out = (res.stdout + res.stderr).strip()
    ok = res.returncode == 0 or "not found" in out.lower() or "no active session" in out.lower()
    return ok, (0 if ok else (res.returncode or 1)), out
def _prune_store(cfg_path: str | Path | None, session_name: str) -> None:
    if isinstance(data := _load_opt_json(cfg_path), dict) and ((data.pop(session_name, None) is not None) | (isinstance(data.get("sessions"), dict) and data["sessions"].pop(session_name, None) is not None)):
        _write_json_atomic(Path(cfg_path), data)
def snapshot_protected(state_dir: Path, cred_path: str, colab_bin: str, sessions_text: str | None = None) -> dict:
    acct, prot_path, owned_path, lock_path = _paths(state_dir, cred_path)
    if sessions_text is None:
        rc, out = _run_colab_global(colab_bin, cred_path, state_dir, "sessions")
        if rc != 0 or ERR_RE.search(out):
            raise RuntimeError(f"Failed read-only `colab sessions` snapshot (fail-closed): {out[:240]}")
        sessions_text = out
    now = int(time.time())
    with _guard_lock(lock_path):
        prot, owned = _read_json(prot_path, {"account": acct, "endpoints": {}}), _read_json(owned_path, {"account": acct, "sessions": {}})
        active_eps = {r.get("endpoint") for r in owned.get("sessions", {}).values() if isinstance(r, dict) and r.get("status") != "stopped"}
        eps = prot.setdefault("endpoints", {})
        for item in parse_colab_sessions(sessions_text):
            ep = item["endpoint"]
            if ep in eps:
                eps[ep]["last_seen_at"] = now
            elif ep not in active_eps:
                eps[ep] = {"first_seen_at": now, "last_seen_at": now, "name": item["name"], "raw": item["raw"], "source": "pre_existing_snapshot"}
        _write_json_atomic(prot_path, prot)
        return prot
def assert_can_create(session_name: str, state_dir: Path, cred_path: str, colab_config_path: str | Path | None = None) -> None:
    if not is_valid_session_name(session_name):
        raise ValueError(f"Invalid session name '{session_name}': must match ^se374_[a-zA-Z0-9_-]+$ (or nb1/nb2)")
    acct, prot_path, owned_path, lock_path = _paths(state_dir, cred_path)
    with _guard_lock(lock_path):
        prot = _require_prot(prot_path, acct, "Missing protected_endpoints snapshot; run snapshot-protected before creating a session")
        rec = _read_json(owned_path, {"account": acct, "sessions": {}}).get("sessions", {}).get(session_name)
        if isinstance(rec, dict) and rec.get("status") != "stopped":
            raise RuntimeError(f"Owned session '{session_name}' is already active (endpoint={rec.get('endpoint')})")
        if (cfg_ep := extract_endpoint_from_config(colab_config_path, session_name)) and cfg_ep in prot.get("endpoints", {}):
            raise RuntimeError(f"Config store endpoint {cfg_ep} is in protected_endpoints")
def register_owned(session_name: str, colab_config_path: str | Path, state_dir: Path, cred_path: str, gpu: str, high_mem: bool, pid: int, tmux_name: str = "") -> dict:
    if not is_valid_session_name(session_name):
        raise ValueError(f"Invalid session name '{session_name}'")
    cfg_resolved = str(Path(colab_config_path).resolve())
    if not (endpoint := extract_endpoint_from_config(cfg_resolved, session_name)):
        raise RuntimeError(f"Could not find endpoint for '{session_name}' in {cfg_resolved}")
    acct, prot_path, owned_path, lock_path = _paths(state_dir, cred_path)
    with _guard_lock(lock_path):
        if endpoint in _require_prot(prot_path, acct, "Missing protected_endpoints snapshot before register-owned").get("endpoints", {}):
            raise RuntimeError(f"REFUSED: endpoint {endpoint} is in protected_endpoints and cannot be owned")
        owned = _read_json(owned_path, {"account": acct, "sessions": {}})
        rec = {"session_name": session_name, "endpoint": endpoint, "account": acct, "gpu": gpu, "high_mem": bool(high_mem), "pid": int(pid), "tmux_name": tmux_name, "created_at": int(time.time()), "colab_config_path": cfg_resolved, "status": "running"}
        owned.setdefault("sessions", {})[session_name] = rec
        _write_json_atomic(owned_path, owned)
        return rec
def stop_session(session_name: str, state_dir: Path, cred_path: str, colab_bin: str, colab_config_path: str | Path | None = None) -> dict:
    if not is_valid_session_name(session_name):
        raise ValueError(f"REFUSED: invalid owned session name '{session_name}'")
    acct, prot_path, owned_path, lock_path = _paths(state_dir, cred_path)
    with _guard_lock(lock_path):
        prot = _require_prot(prot_path, acct, "REFUSED: missing protected_endpoints snapshot")
        rec = _read_json(owned_path, {"account": acct, "sessions": {}}).get("sessions", {}).get(session_name)
        if not isinstance(rec, dict):
            raise RuntimeError(f"REFUSED: '{session_name}' is not an active owned session")
        if not (ep := rec.get("endpoint", "")) or ep in prot.get("endpoints", {}):
            raise RuntimeError(f"REFUSED: endpoint '{ep}' is protected or missing")
        if rec.get("status") == "stopped":
            return {"session": session_name, "endpoint": ep, "returncode": 0, "output": "already stopped"}
        rec_cfg, cli_cfg = str(rec.get("colab_config_path") or "").strip(), str(colab_config_path or "").strip()
        cfg_str = rec_cfg if (rec_cfg and Path(rec_cfg).is_file()) else (cli_cfg if (cli_cfg and Path(cli_cfg).is_file()) else "")
        store_ep = extract_endpoint_from_config(cfg_str, session_name) if cfg_str else None
        if rec.get("session_name") != session_name or not store_ep or store_ep != ep:
            raise RuntimeError(f"REFUSED: dual-match failed for '{session_name}' (owned={ep}, store={store_ep})")
    ok, rc, out = _colab_stop(colab_bin, cred_path, cfg_str, session_name)
    if ok:
        with _guard_lock(lock_path):
            owned = _read_json(owned_path, {"account": acct, "sessions": {}})
            if session_name in owned.get("sessions", {}):
                owned["sessions"][session_name].update(status="stopped", stopped_at=int(time.time()))
                _write_json_atomic(owned_path, owned)
            _prune_store(cfg_str, session_name)
    return {"session": session_name, "endpoint": ep, "returncode": rc, "output": out}
def stop_owned(state_dir: Path, cred_path: str, colab_bin: str) -> list[dict]:
    acct, _, owned_path, lock_path = _paths(state_dir, cred_path)
    with _guard_lock(lock_path):
        names = [k for k, v in _read_json(owned_path, {"account": acct, "sessions": {}}).get("sessions", {}).items() if isinstance(v, dict) and v.get("status") != "stopped"]
    results, errs = [], []
    for name in names:
        try:
            results.append(stop_session(name, state_dir, cred_path, colab_bin))
        except RuntimeError as exc:
            errs.append(str(exc))
    if errs:
        raise RuntimeError("; ".join(errs))
    return results
def cleanup_partial(session_name: str, colab_config_path: str | Path, state_dir: Path, cred_path: str, colab_bin: str) -> dict:
    if not is_valid_session_name(session_name):
        raise ValueError(f"REFUSED: invalid session name '{session_name}'")
    cfg_str = str(colab_config_path or "").strip()
    if not (ep := extract_endpoint_from_config(cfg_str, session_name) if cfg_str else None):
        return {"session": session_name, "cleaned": False, "returncode": 0}
    acct, prot_path, owned_path, lock_path = _paths(state_dir, cred_path)
    with _guard_lock(lock_path):
        if ep in _require_prot(prot_path, acct, "REFUSED: missing protected_endpoints snapshot").get("endpoints", {}):
            raise RuntimeError(f"REFUSED: partial endpoint '{ep}' is in protected_endpoints")
        rec = _read_json(owned_path, {"account": acct, "sessions": {}}).get("sessions", {}).get(session_name)
        is_owned_active = isinstance(rec, dict) and rec.get("status") != "stopped" and rec.get("endpoint") == ep
    if is_owned_active:
        return stop_session(session_name, state_dir, cred_path, colab_bin, cfg_str)
    ok, rc, out = _colab_stop(colab_bin, cred_path, cfg_str, session_name)
    if ok:
        with _guard_lock(lock_path):
            _prune_store(cfg_str, session_name)
    return {"session": session_name, "endpoint": ep, "cleaned": ok, "returncode": rc, "output": out}
def scan_runner_pids(proc_root: Path = Path("/proc")) -> dict[str, list[int]]:
    by_sess, skip = {}, ("--stop-all", "--stop-owned", "--stop-session", "--status", "--dry-run", "--help")
    for p in (proc_root.iterdir() if proc_root.is_dir() else ()):
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = [x for x in (p / "cmdline").read_bytes().decode("utf-8", "replace").split("\0") if x]
        except OSError:
            continue
        if not argv or (Path(argv[0]).name in ("sh", "dash", "bash", "zsh") and argv[1:2] in (["-c"], ["-lc"])) or "run_colab_vps.sh" not in " ".join(argv) or any(f in argv for f in skip):
            continue
        sess = next((argv[i + 1] for i, t in enumerate(argv) if t == "--session" and i + 1 < len(argv)), "") or next((t.split("=", 1)[1] for t in argv if t.startswith("--session=")), "") or next((argv[i + 1] for i, t in enumerate(argv) if t == "--notebook" and i + 1 < len(argv)), "nb1")
        by_sess.setdefault(sess, []).append(int(p.name))
    return by_sess
def classify_colab_sessions(sessions_stdout: str, state_dir: Path | None = None, cred_path: str | None = None) -> dict:
    acct, prot_path, owned_path, _ = _paths(resolve_state_dir(state_dir), cred_path)
    prot = _read_json(prot_path, {"endpoints": {}}) if prot_path.is_file() else {"endpoints": {}}
    owned = _read_json(owned_path, {"sessions": {}}) if owned_path.is_file() else {"sessions": {}}
    active = {v["endpoint"]: k for k, v in owned.get("sessions", {}).items() if isinstance(v, dict) and v.get("status") != "stopped" and v.get("endpoint")}
    prot_eps = set(prot.get("endpoints", {}).keys())
    out_owned = [{**x, "owned": True, "session_name": active[x["endpoint"]]} for x in parse_colab_sessions(sessions_stdout) if x["endpoint"] in active and x["endpoint"] not in prot_eps]
    out_foreign = [{**x, "owned": False, "protected": x["endpoint"] in prot_eps} for x in parse_colab_sessions(sessions_stdout) if not (x["endpoint"] in active and x["endpoint"] not in prot_eps)]
    return {"account": acct, "owned": out_owned, "foreign": out_foreign, "owned_active_count": len(out_owned), "foreign_active_count": len(out_foreign), "protected_count": len(prot_eps)}
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P0 ComfyUI-safe Colab session guard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate-session", "snapshot-protected", "assert-can-create", "register-owned", "stop-session", "stop-owned", "classify", "provision-lock-path", "cleanup-partial"):
        p = sub.add_parser(name)
        for flag, kw in (("--state-dir", {}), ("--cred-path", {"default": DEFAULT_CRED}), ("--colab-bin", {"default": os.environ.get("COLAB_BIN", "colab")}), ("--session", {"default": ""}), ("--colab-config", {"default": ""}), ("--gpu", {"default": "L4"}), ("--high-mem", {"type": int, "default": 0}), ("--pid", {"type": int, "default": os.getpid()}), ("--tmux-name", {"default": ""})):
            p.add_argument(flag, **kw)
    a = ap.parse_args(argv)
    sdir = resolve_state_dir(a.state_dir)
    if a.cmd == "validate-session":
        return 0 if is_valid_session_name(a.session) else (print(f"[ERROR] Invalid session '{a.session}'", file=sys.stderr) or 2)
    if a.cmd in ("assert-can-create", "provision-lock-path"):
        return (assert_can_create(a.session, sdir, a.cred_path, a.colab_config or None) if a.cmd == "assert-can-create" else print(provision_lock_path(sdir, a.cred_path))) or 0
    dispatch = {"snapshot-protected": lambda: snapshot_protected(sdir, a.cred_path, a.colab_bin), "register-owned": lambda: register_owned(a.session, a.colab_config, sdir, a.cred_path, a.gpu, bool(a.high_mem), a.pid, a.tmux_name), "stop-session": lambda: stop_session(a.session, sdir, a.cred_path, a.colab_bin, a.colab_config or None), "stop-owned": lambda: stop_owned(sdir, a.cred_path, a.colab_bin), "cleanup-partial": lambda: cleanup_partial(a.session, a.colab_config, sdir, a.cred_path, a.colab_bin), "classify": lambda: classify_colab_sessions(sys.stdin.read(), sdir, a.cred_path)}
    out = dispatch[a.cmd]()
    print(json.dumps(out))
    return int(out["returncode"]) if isinstance(out, dict) and out.get("returncode", 0) != 0 else (1 if isinstance(out, list) and any(isinstance(r, dict) and r.get("returncode", 0) != 0 for r in out) else 0)
if __name__ == "__main__":
    raise SystemExit(main())
