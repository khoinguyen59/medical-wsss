"""Standalone & adversarial P0 tests for vps_colab_runner/."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER_DIR = PROJECT_ROOT / "vps_colab_runner"
sys.path.insert(0, str(RUNNER_DIR))
import session_guard  # noqa: E402

FORBIDDEN_DASHES = ("\u2014", "\u2013", "\u2015", "\u2212")
TRACKED_FILES = ["setup_vps_colab.sh", "run_colab_vps.sh", "start_dashboard_public.sh", "session_guard.py", "dashboard.py", "dashboard.html", "dash_auth.py", "login.html", "README.md", "test_vps_colab_runner.py", "test_dashboard.py"]


@pytest.mark.parametrize("name", TRACKED_FILES)
def test_vps_files_exist_ascii_dashes_and_line_cap(name: str) -> None:
    """All vps_colab_runner files stay <= 400 lines (session_guard <= 250), LF on .sh, ASCII '-' only."""
    path = RUNNER_DIR / name
    assert path.is_file(), f"Missing expected file: {path}"
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8")
    if path.suffix == ".sh":
        assert b"\r\n" not in raw_bytes, f"{path.name} contains CRLF line endings"
    lines = text.splitlines()
    cap = 250 if name == "session_guard.py" else 400
    assert len(lines) <= cap, f"{path.name} has {len(lines)} lines (> {cap} cap)"
    for bad in FORBIDDEN_DASHES:
        assert bad not in text, f"Forbidden non-ASCII dash {bad!r} found in {path.name}"
    for secret_marker in ("refresh_" + "token", "client_" + "secret", "hf" + "_", "AI" + "za"):
        assert secret_marker not in text, f"Potential secret marker {secret_marker!r} in {path.name}"


def test_runner_script_has_p0_guards_and_no_blind_stop() -> None:
    """Runner script enforces session_guard, isolated --config, notebook copy, and safe stop aliases."""
    text = (RUNNER_DIR / "run_colab_vps.sh").read_text(encoding="utf-8")
    for snippet in (
        "trap cleanup EXIT INT TERM", "DEADLINE_EPOCH", "WATCHDOG_TIMEOUT", "CALIBRATION COLLAPSE", "Loss: nan|Non-finite epoch loss",
        "sha256sum", "snapshot-protected", "assert-can-create", "register-owned", "stop-session", "stop-owned", "provision-lock-path",
        "cleanup-partial", "${BASHPID:-$$}", "_descendants_bottom_up", "no such file or directory", "colab_session_t",
        '--config "$SESS_COLAB_CFG"', "run_notebook.ipynb", "stage1_best_cls.pth", "--high-mem", "--preset",
        r"\[STAGE[12] ERROR\]", "OutOfMemoryError", "FileNotFoundError", "STAGE1_LOCAL_CKPT",
    ):
        assert snippet in text, f"Missing mandatory guard snippet: {snippet}"
    assert "grep -Eq" not in text
    assert "colab stop --all" not in text
    assert 'for sname in "$SESSION_NAME" nb1 nb2' not in text


def test_session_guard_protects_comfyui_and_enforces_dual_match(tmp_path: Path, monkeypatch) -> None:
    """P0: foreign ComfyUI A100 endpoint can NEVER be registered or stopped, and dual-match is required."""
    cred = tmp_path / "acc11.json"
    cred.write_text(json.dumps({"client_id": "cid", "quota_project_id": "p11"}), encoding="utf-8")
    sdir = tmp_path / "state"
    stop_calls = []

    def fake_run(cmd, **_kw):
        stop_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="stopped", stderr="")

    monkeypatch.setattr(session_guard.subprocess, "run", fake_run)

    # 1. Fail-closed before snapshot
    with pytest.raises(RuntimeError, match="snapshot"):
        session_guard.assert_can_create("se374_arm_a", sdir, str(cred))

    # 2. Fail-closed when snapshot command fails
    monkeypatch.setattr(session_guard, "_run_colab_global", lambda *_a: (1, "ERROR: Unauthorized"))
    with pytest.raises(RuntimeError, match="fail-closed"):
        session_guard.snapshot_protected(sdir, str(cred), "colab")

    # 3. Snapshot captures ComfyUI A100 as protected (even if named nb1 on server)
    sessions_out = "[?] m-s-comfyui-a100 | Hardware: A100 | Shape: HIGH_RAM | Variant: GPU\n[nb1] m-s-comfyui-legacy | Hardware: A100 | Shape: HIGH_RAM | Variant: GPU\n"
    prot = session_guard.snapshot_protected(sdir, str(cred), "colab", sessions_text=sessions_out)
    assert "m-s-comfyui-a100" in prot["endpoints"] and "m-s-comfyui-legacy" in prot["endpoints"]

    # 4. Invalid session name rejected
    with pytest.raises(ValueError, match="Invalid session name"):
        session_guard.assert_can_create("foreign_comfy", sdir, str(cred))

    # 5. Refuses registering a protected endpoint as owned
    cfg_comfy = tmp_path / "comfy_cfg.json"
    cfg_comfy.write_text(json.dumps({"nb1": {"endpoint": "m-s-comfyui-a100"}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="protected_endpoints"):
        session_guard.register_owned("nb1", cfg_comfy, sdir, str(cred), "A100", True, 123)

    # 6. Even if manually tampered into owned_sessions, stop_session REFUSES protected endpoint
    acct = session_guard.resolve_account_name(str(cred))
    owned_path = sdir / f"{acct}_owned_sessions.json"
    owned_path.write_text(json.dumps({"account": acct, "sessions": {"nb1": {"session_name": "nb1", "endpoint": "m-s-comfyui-a100", "colab_config_path": str(cfg_comfy), "status": "running"}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="protected"):
        session_guard.stop_session("nb1", sdir, str(cred), "colab")
    with pytest.raises(RuntimeError, match="protected"):
        session_guard.stop_owned(sdir, str(cred), "colab")
    assert stop_calls == []

    # 7. Valid owned session registration + dual-match mismatch check + valid teardown
    owned_path.unlink()
    cfg_a = tmp_path / "arm_a_cfg.json"
    cfg_a.write_text(json.dumps({"se374_arm_a": {"endpoint": "m-s-owned-1"}}), encoding="utf-8")
    session_guard.assert_can_create("se374_arm_a", sdir, str(cred), cfg_a)
    rec = session_guard.register_owned("se374_arm_a", cfg_a, sdir, str(cred), "A100", True, 456, "colab_se374_arm_a")
    assert rec["endpoint"] == "m-s-owned-1"
    prot2 = session_guard.snapshot_protected(sdir, str(cred), "colab", sessions_text=sessions_out + "[se374_arm_a] m-s-owned-1 | Hardware: A100 | Shape: HIGH_RAM | Variant: GPU\n")
    assert "m-s-owned-1" not in prot2["endpoints"] and "m-s-comfyui-a100" in prot2["endpoints"]

    # Endpoint mismatch in store refuses stop
    cfg_a.write_text(json.dumps({"se374_arm_a": {"endpoint": "m-s-other-999"}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="dual-match"):
        session_guard.stop_session("se374_arm_a", sdir, str(cred), "colab")
    assert stop_calls == []

    # Restore matching endpoint -> stop succeeds with explicit --config and -s
    cfg_a.write_text(json.dumps({"se374_arm_a": {"endpoint": "m-s-owned-1"}}), encoding="utf-8")
    res = session_guard.stop_session("se374_arm_a", sdir, str(cred), "colab")
    assert res["endpoint"] == "m-s-owned-1"
    assert stop_calls == [["colab", "--config", str(cfg_a), "--auth", "adc", "stop", "-s", "se374_arm_a"]]

    # 8. Missing protected_endpoints snapshot fails closed on stop_session
    stop_calls.clear()
    (sdir / f"{acct}_protected_endpoints.json").unlink()
    owned_path.write_text(json.dumps({"account": acct, "sessions": {"se374_arm_a": {"session_name": "se374_arm_a", "endpoint": "m-s-owned-1", "colab_config_path": str(cfg_a), "status": "running"}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="REFUSED: missing protected_endpoints snapshot"):
        session_guard.stop_session("se374_arm_a", sdir, str(cred), "colab")
    assert stop_calls == []

    # 9. ERR_RE catches 403/Forbidden/Exception on rc=0 while ignoring valid endpoints containing 403
    monkeypatch.setattr(session_guard, "_run_colab_global", lambda *_a: (0, "HTTP 403 Forbidden"))
    with pytest.raises(RuntimeError, match="fail-closed"):
        session_guard.snapshot_protected(sdir, str(cred), "colab")
    monkeypatch.setattr(session_guard, "_run_colab_global", lambda *_a: (0, "SomeException: auth expired"))
    with pytest.raises(RuntimeError, match="fail-closed"):
        session_guard.snapshot_protected(sdir, str(cred), "colab")
    ok_403 = session_guard.snapshot_protected(sdir, str(cred), "colab", sessions_text="[?] m-s-403-abc | Hardware: A100\n")
    assert "m-s-403-abc" in ok_403["endpoints"]


def test_scan_runner_pids_parses_per_session(tmp_path: Path, monkeypatch) -> None:
    """scan_runner_pids maps each live runner process to its exact --session name and skips sh -c wrappers."""
    proc = tmp_path / "proc"
    for pid, argv in (
        ("2000", ["sh", "-c", "bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --session se374_arm_a"]),
        ("2001", ["bash", "vps_colab_runner/run_colab_vps.sh", "--notebook", "nb1", "--session", "se374_arm_a"]),
        ("2002", ["bash", "vps_colab_runner/run_colab_vps.sh", "--notebook", "nb1", "--session=se374_arm_b"]),
        ("2003", ["bash", "vps_colab_runner/run_colab_vps.sh", "--stop-session", "se374_arm_a"]),
    ):
        d = proc / pid
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(("\0".join(argv) + "\0").encode("utf-8"))
    monkeypatch.setattr(session_guard.os, "getpid", lambda: 99999)
    assert session_guard.scan_runner_pids(proc) == {"se374_arm_a": [2001], "se374_arm_b": [2002]}


def _to_wsl(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}" if len(s) >= 2 and s[1] == ":" else s


def _wsl(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    wsl = shutil.which("wsl")
    if wsl is None:
        pytest.skip("WSL not available on this host")
    return subprocess.run([wsl, "-d", "Ubuntu", "--", *args], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout, check=False)


def _setup_fake_colab(tmp_path: Path):
    cred, state_dir, calls, cnt_f = tmp_path / "acc11.json", tmp_path / "gs", tmp_path / "calls.log", tmp_path / "exec_cnt"
    cred.write_text('{"client_id":"c11","quota_project_id":"p11"}', encoding="utf-8")
    rem = tmp_path / "rem"
    (rem / "content" / "checkpoints").mkdir(parents=True)
    (rem / "content" / "stage1_results").mkdir(parents=True)
    for rel_f, data in (("content/checkpoints/stage1_retrained_best.pth", b"best"), ("content/checkpoints/stage1_latest.pth", b"latest"), ("content/checkpoints/stage1_best_cls.pth", b"cls"), ("content/stage1_results/STAGE1_TRAINING_METRICS.json", b'{"best":0.71}'), ("content/stage1_results/stage1_training_curves.png", b"png"), ("content/SE374_Stage1_Training_Results.zip", b"zip")):
        (rem / rel_f).write_bytes(data)
    fake = tmp_path / "fake_colab.sh"
    fake.write_bytes(f"""#!/usr/bin/env bash
set -euo pipefail
CFG=""; while [ $# -gt 0 ]; do case "$1" in --config) CFG="$2"; shift 2;; --auth) shift 2;; *) break;; esac; done
SUB="${{1:-}}"; shift || true; echo "CFG=$CFG CMD=$SUB $*" >> "{_to_wsl(calls)}"
case "$SUB" in
  usage) echo "Current balance: 250.00 compute units"; echo "Usage rate: 5.00/hr"; echo "Active assignments: 1" ;;
  sessions) [ "${{FAKE_SNAPSHOT_FAIL:-0}}" = "1" ] && {{ echo "ERROR: Unauthorized" >&2; exit 1; }}; echo "[?] m-s-comfyui-a100 | Hardware: A100 | Shape: HIGH_RAM | Variant: GPU" ;;
  new) S="${{FAKE_SESS:-se374_arm_a}}"; mkdir -p "$(dirname "$CFG")"
    if [ "${{FAIL_A100:-0}}" = "1" ] && [[ "$*" == *"A100"* ]]; then printf '{{"%s": {{"endpoint": "m-s-part-a100"}}}}\\n' "$S" > "$CFG"; exit 1; fi
    printf '{{"%s": {{"endpoint": "%s"}}}}\\n' "$S" "${{FAKE_EP:-m-s-owned-a1}}" > "$CFG" ;;
  exec) N=$(( $(cat "{_to_wsl(cnt_f)}" 2>/dev/null || echo 0) + 1 )); echo "$N" > "{_to_wsl(cnt_f)}"
    NB=""; A=("$@"); for ((i=0;i<${{#A[@]}};i++)); do [ "${{A[$i]}}" = "-f" ] && NB="${{A[$((i+1))]}}"; done
    if [ -n "${{FAKE_NB_ERR:-}}" ] && [ -n "$NB" ]; then
      EV="transient error"; [ "$FAKE_NB_ERR" = "fatal" ] && EV="STAGE1_GIT_REF checkout failed"; [ "$FAKE_NB_ERR" = "stage2" ] && EV="[STAGE2 ERROR] missing ckpt"
      printf '{{"cells":[{{"cell_type":"code","outputs":[{{"output_type":"error","ename":"RuntimeError","evalue":"%s","traceback":["%s"]}}]}}]}}\n' "$EV" "$EV" > "${{NB%.ipynb}}_output.ipynb"
      exit 0
    fi
    [ "${{FAKE_COLLAPSE:-0}}" = "1" ] && {{ echo "[CALIBRATION COLLAPSE ALARM] val_mAP < 0.65"; sleep 20; exit 1; }}
    [ "${{FAIL_FIRST_EXEC:-0}}" = "1" ] && [ "$N" -eq 1 ] && exit 1
    echo "[EARLY STOP PATIENCE] Clean patience stop, proceeding to Cell 8 export." ;;
  ssh) shift 3; CMD="$1"; shift
    if [ "$CMD" = "mkdir" ]; then mkdir -p "{_to_wsl(rem)}/up"
    elif [ "$CMD" = "sh" ]; then cat > "{_to_wsl(rem)}/up/$(basename "$(printf '%s' "$2" | awk -F"'" '{{print $(NF-1)}}')")"
    elif [ "$CMD" = "stat" ]; then [ -n "${{FAKE_STAT_ONCE:-}}" ] && [ ! -f "$FAKE_STAT_ONCE" ] && [[ "$3" == *"stage1_retrained_best.pth"* ]] && {{ touch "$FAKE_STAT_ONCE"; echo "HTTP 429" >&2; exit 1; }}; stat -c '%Y %s' "{_to_wsl(rem)}$3"
    elif [ "$CMD" = "sha256sum" ]; then [ -f "{_to_wsl(rem)}/up/$(basename "$1")" ] && sha256sum "{_to_wsl(rem)}/up/$(basename "$1")" || sha256sum "{_to_wsl(rem)}$1"
    elif [ "$CMD" = "cat" ]; then cat "{_to_wsl(rem)}$1"; fi ;;
  log) echo "session log" > "$4" ;;
  stop) [ "${{FAIL_STOP:-0}}" = "1" ] && exit 1; echo "stopped $*" ;;
esac
""".encode("utf-8"))
    env_b = f"chmod +x '{_to_wsl(fake)}' && COLAB_BIN='{_to_wsl(fake)}' COLAB_GUARD_STATE_DIR='{_to_wsl(state_dir)}' GOOGLE_APPLICATION_CREDENTIALS='{_to_wsl(cred)}' "
    return env_b, calls, state_dir, rem, cnt_f


def test_wsl_syntax_dry_runs_and_e2e_guards(tmp_path: Path) -> None:
    """WSL: validate bash -n, multi-GPU --dry-run (HIGH_MEM=0 default), ComfyUI guard, early-stop vs collapse."""
    for rel in ("vps_colab_runner/setup_vps_colab.sh", "vps_colab_runner/run_colab_vps.sh", "vps_colab_runner/start_dashboard_public.sh"):
        res = _wsl(["bash", "-n", rel], timeout=30)
        assert res.returncode == 0, f"bash -n failed for {rel}: {res.stderr}"
    for gpu in ("T4", "L4", "G4", "A100", "H100"):
        dry_def = _wsl(["bash", "vps_colab_runner/run_colab_vps.sh", "--notebook", "nb1", "--gpu", gpu, "--session", f"se374_{gpu.lower()}", "--dry-run"], timeout=30)
        assert dry_def.returncode == 0 and f"GPU / High-Mem  : {gpu} / 0" in dry_def.stdout
        dry_hm = _wsl(["bash", "vps_colab_runner/run_colab_vps.sh", "--notebook", "nb1", "--gpu", gpu, "--high-mem", "--preset", "arm_a_polysgd", "--session", f"se374_{gpu.lower()}", "--env", "STAGE1_SEED=42", "--dry-run"], timeout=30)
        assert dry_hm.returncode == 0 and "Validation PASS. Zero Colab sessions created." in dry_hm.stdout
        if gpu in ("G4", "H100"):
            assert f"GPU / High-Mem  : {gpu} / 1" in dry_hm.stdout

    env_base, colab_calls, _, _, _ = _setup_fake_colab(tmp_path)
    out_dir = tmp_path / "run_arm_a"
    r_fail = _wsl(["bash", "-c", env_base + f"FAKE_SNAPSHOT_FAIL=1 bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --gpu A100 --session se374_arm_a --output-dir '{_to_wsl(out_dir)}'"], timeout=30)
    assert r_fail.returncode != 0 and "CMD=new" not in colab_calls.read_text() and "CMD=stop" not in colab_calls.read_text()

    colab_calls.unlink()
    r_ok = _wsl(["bash", "-c", env_base + f"bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --gpu A100 --high-mem --preset arm_a_polysgd --session se374_arm_a --env STAGE1_SEED=42 --output-dir '{_to_wsl(out_dir)}'"], timeout=45)
    assert r_ok.returncode == 0, f"E2E failed: {r_ok.stdout}\n{r_ok.stderr}"
    calls_txt = colab_calls.read_text(encoding="utf-8")
    assert "CMD=new -s se374_arm_a --gpu A100 --high-mem" in calls_txt and "STAGE1_EXP_PRESET=arm_a_polysgd" in calls_txt and "STAGE1_SEED=42" in calls_txt
    assert all("CFG=" in ln and "CFG= " not in ln and ".readonly_sessions.json" in ln for ln in calls_txt.splitlines() if "CMD=usage" in ln or "CMD=sessions" in ln)
    assert "run_notebook.ipynb" in calls_txt and "01_Train_Stage1_Classifier_L4_output.ipynb" not in calls_txt
    assert "CMD=stop -s se374_arm_a" in calls_txt and "m-s-comfyui-a100" not in calls_txt
    assert (out_dir / "checkpoints" / "stage1_best_cls.pth").read_bytes() == b"cls"
    assert (out_dir / "artifacts" / "STAGE1_TRAINING_METRICS.json").read_bytes() == b'{"best":0.71}'
    assert (out_dir / ".state" / "stop_reason").read_text(encoding="utf-8") == "EARLY_STOP_PATIENCE"

    colab_calls.unlink()
    for flag in ("--stop-owned", "--stop-all"):
        assert _wsl(["bash", "-c", env_base + f"bash vps_colab_runner/run_colab_vps.sh {flag}"], timeout=30).returncode == 0
    assert "CMD=stop" not in colab_calls.read_text(encoding="utf-8")

    colab_calls.unlink()
    out_collapse = tmp_path / "run_collapse"
    r_col = _wsl(["bash", "-c", env_base + f"FAKE_COLLAPSE=1 bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --gpu A100 --session se374_arm_a --max-retries 0 --output-dir '{_to_wsl(out_collapse)}'"], timeout=45)
    assert r_col.returncode != 0 and (out_collapse / ".state" / "stop_reason").read_text(encoding="utf-8") == "CALIBRATION_COLLAPSE_ALARM"
    assert "m-s-comfyui-a100" not in colab_calls.read_text(encoding="utf-8")


def test_idempotent_stop_resilient_stop_owned_and_config_precedence(tmp_path: Path, monkeypatch) -> None:
    """Idempotent stop_session, store pruning, resilient stop_owned partial race, and config precedence."""
    cred, sdir, stop_calls = tmp_path / "c.json", tmp_path / "st", []
    cred.write_text('{"client_id":"c11","quota_project_id":"p11"}', encoding="utf-8")
    monkeypatch.setattr(session_guard.subprocess, "run", lambda cmd, **_k: stop_calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "ok", ""))
    session_guard.snapshot_protected(sdir, str(cred), "colab", sessions_text="[?] m-s-comfy | Hardware: A100\n")
    other = tmp_path / "other_cwd"
    other.mkdir()
    monkeypatch.chdir(tmp_path)
    cfg_rel = Path("rel_dir") / "sess.json"
    cfg_rel.parent.mkdir()
    cfg_rel.write_text(json.dumps({"se374_arm_a": {"endpoint": "m-s-a"}, "se374_arm_b": {"endpoint": "m-s-b"}}), encoding="utf-8")
    rec_a = session_guard.register_owned("se374_arm_a", cfg_rel, sdir, str(cred), "A100", False, 10)
    assert Path(rec_a["colab_config_path"]).is_absolute()
    session_guard.register_owned("se374_arm_b", cfg_rel, sdir, str(cred), "L4", False, 11)

    monkeypatch.chdir(other)
    wrong_cfg = tmp_path / "wrong.json"
    wrong_cfg.write_text(json.dumps({"se374_arm_a": {"endpoint": "m-s-wrong"}}), encoding="utf-8")
    r1 = session_guard.stop_session("se374_arm_a", sdir, str(cred), "colab", wrong_cfg)
    assert r1["returncode"] == 0 and len(stop_calls) == 1
    assert "se374_arm_a" not in json.loads((tmp_path / "rel_dir" / "sess.json").read_text(encoding="utf-8"))
    assert session_guard.stop_session("se374_arm_a", sdir, str(cred), "colab") == {"session": "se374_arm_a", "endpoint": "m-s-a", "returncode": 0, "output": "already stopped"}
    assert len(stop_calls) == 1
    with pytest.raises(RuntimeError, match="REFUSED"):
        session_guard.stop_session("se374_unknown", sdir, str(cred), "colab")

    acct = session_guard.resolve_account_name(str(cred))
    owned_p = sdir / f"{acct}_owned_sessions.json"
    owned_data = json.loads(owned_p.read_text(encoding="utf-8"))
    owned_data["sessions"]["se374_bad"] = {"session_name": "se374_bad", "endpoint": "m-s-comfy", "colab_config_path": "", "status": "running"}
    owned_data["sessions"]["se374_nocfg"] = {"session_name": "se374_nocfg", "endpoint": "m-s-x", "colab_config_path": "", "status": "running"}
    owned_p.write_text(json.dumps(owned_data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="dual-match"):
        session_guard.stop_session("se374_nocfg", sdir, str(cred), "colab", "")
    with pytest.raises(RuntimeError, match="protected"):
        session_guard.stop_owned(sdir, str(cred), "colab")
    assert len(stop_calls) == 2 and "se374_arm_b" in stop_calls[-1]
    assert json.loads(owned_p.read_text(encoding="utf-8"))["sessions"]["se374_arm_b"]["status"] == "stopped"


def test_wsl_retry_resume_cls_fallback_partial_cleanup_flock_and_stop_abort(tmp_path: Path) -> None:
    """WSL E2E: primary-first A100 HIGH_MEM=0 + partial cleanup, resume cls upload, flock, stop-failure abort, stop log."""
    env_b, calls, state_dir, rem, cnt_f = _setup_fake_colab(tmp_path)
    out1 = tmp_path / "r1"
    r = _wsl(["bash", "-c", env_b + f"FAKE_SESS=se374_fb FAKE_EP=m-s-l4 FAIL_A100=1 FAIL_FIRST_EXEC=1 bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --gpu A100 --gpu-fallback L4 --session se374_fb --sync-interval 999 --output-dir '{_to_wsl(out1)}'"], timeout=90)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    ctxt = calls.read_text(encoding="utf-8")
    assert "CMD=new -s se374_fb --gpu A100\n" in ctxt and "--high-mem" not in ctxt
    assert "CMD=stop -s se374_fb" in ctxt and "m-s-part-a100" not in next(state_dir.glob("*_protected_endpoints.json")).read_text()
    assert (rem / "up" / "stage1_best_cls.pth").read_bytes() == b"cls"
    meta = (out1 / ".state" / "meta.env").read_text()
    assert "ASSIGNED_GPU=L4" in meta and "ASSIGNED_HMEM=0" in meta and any(state_dir.glob("*_provision.lock"))

    calls.unlink()
    cnt_f.unlink()
    r_abort = _wsl(["bash", "-c", env_b + f"FAKE_SESS=se374_fb FAIL_FIRST_EXEC=1 FAIL_STOP=1 bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --gpu A100 --session se374_fb --max-retries 2 --output-dir '{_to_wsl(tmp_path / 'r2')}'"])
    assert r_abort.returncode != 0 and calls.read_text().count("CMD=new ") == 1

    r_nb1_log = PROJECT_ROOT / "vps_colab_runner" / "runs" / "nb1" / "run.log"
    nb1_before = r_nb1_log.read_text() if r_nb1_log.exists() else ""
    _wsl(["bash", "-c", env_b + "bash vps_colab_runner/run_colab_vps.sh --stop-session se374_fb"], timeout=30)
    assert (r_nb1_log.read_text() if r_nb1_log.exists() else "") == nb1_before


def test_adc_fail_closed_and_readonly_config(tmp_path: Path, monkeypatch) -> None:
    """ADC rc=0 failures fail-closed without writing protected_endpoints, and _run_colab_global passes --config."""
    cred = tmp_path / "acc11.json"
    cred.write_text('{"client_id":"c11","quota_project_id":"p11"}', encoding="utf-8")
    seen = []
    monkeypatch.setattr(session_guard.subprocess, "run", lambda cmd, **_kw: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    rc, _ = session_guard._run_colab_global("colab", str(cred), tmp_path / "gs0", "sessions")
    assert rc == 0 and seen[-1] == ["colab", "--config", str(tmp_path / "gs0" / ".readonly_sessions.json"), "--auth", "adc", "sessions"]
    for idx, msg in enumerate(("No valid default credentials", "Please run:\n  gcloud auth application-default login", "Warning: failed to load", "EOF when reading a line")):
        sdir = tmp_path / f"s_{idx}"
        monkeypatch.setattr(session_guard, "_run_colab_global", lambda *_a, m=msg: (0, m))
        with pytest.raises(RuntimeError, match="fail-closed"):
            session_guard.snapshot_protected(sdir, str(cred), "colab")
        assert not list(sdir.glob("*_protected_endpoints.json"))


def test_wsl_notebook_cell_error_and_nb2_stage1_upload(tmp_path: Path) -> None:
    """WSL E2E: nb2 --env STAGE1_LOCAL_CKPT, fail-fast missing ckpt, G4 HIGH_MEM=0, and [STAGE2 ERROR] 0 retries."""
    env_b, calls, _, rem, cnt_f = _setup_fake_colab(tmp_path)
    out_miss = tmp_path / "nb_miss"
    r_m1 = _wsl(["bash", "-c", env_b + f"bash vps_colab_runner/run_colab_vps.sh --notebook nb2 --gpu G4 --session se374_m1 --env STAGE1_LOCAL_CKPT='/no/such/ckpt.pth' --output-dir '{_to_wsl(out_miss)}'"])
    assert r_m1.returncode == 2 and not calls.exists()
    r_m2 = _wsl(["bash", "-c", env_b + f"bash vps_colab_runner/run_colab_vps.sh --notebook nb2 --gpu G4 --session se374_m2 --env STAGE1_CKPT_SOURCE=retrained --output-dir '{_to_wsl(out_miss)}'"])
    assert r_m2.returncode == 2 and not calls.exists()

    s1_ckpt = tmp_path / "winner_s1.pth"
    s1_ckpt.write_bytes(b"stage1_winner_weights")
    out_fatal = tmp_path / "nb_fatal"
    r_fatal = _wsl(["bash", "-c", env_b + f"FAKE_SESS=se374_nb2 FAKE_NB_ERR=stage2 bash vps_colab_runner/run_colab_vps.sh --notebook nb2 --gpu G4 --session se374_nb2 --max-retries 2 --env STAGE1_LOCAL_CKPT='{_to_wsl(s1_ckpt)}' --output-dir '{_to_wsl(out_fatal)}'"])
    assert r_fatal.returncode != 0
    ctxt = calls.read_text(encoding="utf-8")
    assert "CMD=new -s se374_nb2 --gpu G4\n" in ctxt and "--high-mem" not in ctxt
    exec_ln = next(ln for ln in ctxt.splitlines() if "CMD=exec " in ln)
    assert "STAGE1_CKPT_SOURCE=retrained" in exec_ln and "STAGE1_LOCAL_CKPT" not in exec_ln
    assert ctxt.count("CMD=exec ") == 1
    assert (out_fatal / ".state" / "stop_reason").read_text(encoding="utf-8") == "NOTEBOOK_CELL_ERROR"
    assert "Notebook cell exception:" in (out_fatal / "run.log").read_text(encoding="utf-8")
    assert (rem / "up" / "stage1_retrained_best.pth").read_bytes() == b"stage1_winner_weights"

    calls.unlink()
    cnt_f.unlink()
    r_retry = _wsl(["bash", "-c", env_b + f"FAKE_SESS=se374_nb1_err FAKE_NB_ERR=retry bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --gpu L4 --session se374_nb1_err --max-retries 1 --output-dir '{_to_wsl(tmp_path / 'nb_retry')}'"])
    assert r_retry.returncode != 0 and calls.read_text().count("CMD=exec ") == 2


def test_wsl_sync_retry_enoent_order_and_stop_bg(tmp_path: Path) -> None:
    """WSL: sync retries transient 429, skips ENOENT with 0 retries/WARN, syncs latest last, stop_bg kills tree."""
    env_b, calls, _, rem, _ = _setup_fake_colab(tmp_path)
    (rem / "content" / "stage1_results" / "STAGE1_TRAINING_METRICS.json").unlink()
    out_d, flag_f = tmp_path / "run_retry", tmp_path / "stat_once.flag"
    r = _wsl(["bash", "-c", env_b + f"FAKE_SESS=se374_sync FAKE_STAT_ONCE='{_to_wsl(flag_f)}' bash vps_colab_runner/run_colab_vps.sh --notebook nb1 --gpu A100 --session se374_sync --sync-interval 999 --output-dir '{_to_wsl(out_d)}'"])
    assert r.returncode == 0 and (out_d / "checkpoints" / "stage1_retrained_best.pth").read_bytes() == b"best"
    stat_lns = [ln for ln in calls.read_text(encoding="utf-8").splitlines() if "CMD=ssh -s se374_sync -- stat " in ln]
    assert sum("STAGE1_TRAINING_METRICS.json" in ln for ln in stat_lns) == 2 and sum("stage1_retrained_best.pth" in ln for ln in stat_lns) >= 3
    assert stat_lns.index(next(x for x in stat_lns if "STAGE1_TRAINING_METRICS.json" in x)) < stat_lns.index(next(x for x in stat_lns if "stage1_latest.pth" in x))
    log_txt = (out_d / "run.log").read_text(encoding="utf-8")
    assert "[WARN] stat failed for /content/checkpoints/stage1_retrained_best.pth" in log_txt
    assert not any("STAGE1_TRAINING_METRICS.json" in ln for ln in log_txt.splitlines() if "[WARN]" in ln)
    src_sh = (RUNNER_DIR / "run_colab_vps.sh").read_text(encoding="utf-8")
    funcs = src_sh[src_sh.index("_descendants_bottom_up()"):src_sh.index("\nstop_this_owned_session()")]
    (tmp_path / "tree.sh").write_text(f"set -euo pipefail\n{funcs}\nD='{_to_wsl(tmp_path)}'\n( bash -c 'sleep 60 & echo $! > \"'\"$D\"'/gc.pid\"; wait' & echo $! > \"$D/c.pid\"; wait ) & P=$!\nfor _ in 1 2 3 4 5 6 7 8 9 10; do [ -s \"$D/gc.pid\" ] && break; sleep 0.2; done\nC=$(cat \"$D/c.pid\"); GC=$(cat \"$D/gc.pid\"); stop_bg \"$P\"\n! kill -0 \"$P\" 2>/dev/null && ! kill -0 \"$C\" 2>/dev/null && ! kill -0 \"$GC\" 2>/dev/null\n", encoding="utf-8", newline="\n")
    assert _wsl(["bash", _to_wsl(tmp_path / "tree.sh")], timeout=30).returncode == 0
