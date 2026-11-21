"""Tests for the temporary Colab VPS monitoring dashboard (stdlib only)."""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dash_auth  # noqa: E402
import dashboard  # noqa: E402

FORBIDDEN_DASHES = ("\u2014", "\u2013", "\u2015", "\u2212")
LOG = (
    "[VPS-RUNNER 2026-11-20 10:00:00] Starting resilient Colab run\n"
    "Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x1>\n"
    "  File \"/usr/lib/python3.13/dataloader.py\", line 1706, in __del__\n"
    "[EPOCH 01] Train Loss=0.2952 EM=0.5900 | Val mAP=0.8932 AUROC=0.9189 GatedPM_mIoU=0.6511 (Dice=0.78)\n"
    "[EPOCH 02] Train Loss=0.2503 EM=0.6471 | Val mAP=0.8841 AUROC=0.9177 GatedPM_mIoU=0.6652 (Dice=0.79)\n"
    "[VPS-RUNNER 2026-11-20 10:05:00] Verified atomic sync OK: runs/nb1/checkpoints/stage1_retrained_best.pth (SHA256=abcdef0123456789abcdef)\n"
    "[VPS-RUNNER 2026-11-20 10:06:00] WATCHDOG TIMEOUT: Reached hard limit\n"
)


def test_parse_usage_and_noise_filter_and_epochs() -> None:
    usage = dashboard.parse_usage("Current balance: 258.77 compute units\nUsage rate: 1.94/hr\nActive assignments: 1")
    assert usage == {"balance": "258.77 compute units", "rate": "1.94/hr", "active": "1"}
    lines = dashboard.clean_lines(LOG)
    assert not any("Exception ignored" in x or "dataloader.py" in x for x in lines)
    epochs = dashboard.parse_epochs(lines)
    assert [e["epoch"] for e in epochs] == [1, 2]
    assert epochs[1]["metrics"]["GatedPM_mIoU"] == pytest.approx(0.6652)
    assert epochs[0]["metrics"]["Val mAP"] == pytest.approx(0.8932)


@pytest.fixture()
def server(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "nb1"
    (run / "checkpoints").mkdir(parents=True)
    (run / ".state").mkdir()
    (run / "run.log").write_text(LOG, encoding="utf-8")
    (run / "checkpoints" / "stage1_retrained_best.pth").write_bytes(b"x" * 2048)
    (run / ".state" / "meta.env").write_text(
        "START_EPOCH=1\nDEADLINE_EPOCH=9999999999\nNOTEBOOK=notebooks/01_Train_Stage1_Classifier_L4.ipynb\nGPU=L4\nSESSION=nb1\nMAX_ATTEMPTS=4\n"
    )
    (run / ".state" / "attempt").write_text("2")
    monkeypatch.setattr(dashboard, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(dashboard, "runner_pids", lambda: {})
    monkeypatch.setenv("COLAB_GUARD_STATE_DIR", str(tmp_path / "guard_state"))
    monkeypatch.setenv("COLAB_ACCOUNT_NAME", "acc11_test")
    dash_auth.configure(None)
    dashboard._PENDING_STARTS.clear()
    dashboard.STATE.update(usage={"balance": "1 compute units", "rate": "0.00/hr", "active": "0"}, sessions="none", ts=1.0)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_status_endpoint_reports_checkpoint_and_meta(server) -> None:
    data = json.load(urllib.request.urlopen(server + "/api/status?session=nb1"))
    assert data["session"] == "nb1"
    assert data["meta"]["gpu"] == "L4" and data["meta"]["attempt"] == "2"
    ckpt = {c["name"]: c for c in data["checkpoints"]}
    assert ckpt["stage1_retrained_best.pth"]["size_mb"] >= 0
    assert ckpt["stage1_retrained_best.pth"]["sha"] == "abcdef012345"
    assert any("WATCHDOG" in e for e in data["events"])
    assert len(data["epochs"]) == 2
    assert data["usage"]["active"] == "0"
    assert data["max_owned_concurrent"] == 1


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _open(url, headers=None, data=None, method=None):
    req = urllib.request.Request(url, headers=headers or {}, data=data, method=method)
    return urllib.request.build_opener(_NoRedirect).open(req)


def _code(url, **kw):
    try:
        return _open(url, **kw).status
    except urllib.error.HTTPError as err:
        return err.code


@pytest.fixture()
def secured(server):
    dash_auth.configure("colab", "p@ss-Word1", api_key="api-key-123")
    yield server
    dash_auth.configure(None)


def test_login_flow_session_cookie_and_logout(secured) -> None:
    host = {"Host": "random-words.trycloudflare.com", "X-Forwarded-Proto": "https"}
    assert _code(secured + "/api/status", headers=host) == 401
    assert _code(secured + "/", headers=host) == 302
    assert "Đăng nhập" in _open(secured + "/login", headers=host).read().decode("utf-8")
    assert _code(secured + "/login", headers=host, data=b"username=colab&password=wrong", method="POST") == 401
    with pytest.raises(urllib.error.HTTPError) as redirect:
        _open(secured + "/login", headers=host, data=b"username=colab&password=p%40ss-Word1", method="POST")
    assert redirect.value.code == 302
    cookie = redirect.value.headers["Set-Cookie"]
    assert "dash_session=" in cookie and "HttpOnly" in cookie and "Secure" in cookie
    sess = cookie.split(";")[0]
    data = json.load(_open(secured + "/api/status?session=nb1", headers={**host, "Cookie": sess}))
    assert data["session"] == "nb1"
    assert _code(secured + "/api/stop", headers={**host, "Cookie": sess}, data=b"", method="POST") == 403
    assert _code(secured + "/logout", headers={**host, "Cookie": sess}) == 302
    assert _code(secured + "/api/status", headers={**host, "Cookie": sess}) == 401


def test_api_key_and_login_lockout(secured) -> None:
    assert json.load(_open(secured + "/api/status", headers={"X-API-Key": "api-key-123"}))["usage"]["active"] == "0"
    assert _code(secured + "/api/status", headers={"X-API-Key": "nope"}) == 401
    bad = b"username=colab&password=bad"
    codes = [_code(secured + "/login", headers={"CF-Connecting-IP": "9.9.9.9"}, data=bad, method="POST") for _ in range(7)]
    assert codes[:5] == [401] * 5 and codes[5:] == [429, 429]


def test_start_supports_multi_gpu_and_ignores_foreign_active(secured, monkeypatch) -> None:
    """P0: do_start supports A100/H100/G4/L4/T4 + preset, and succeeds when foreign ComfyUI usage.active == '1'."""
    calls = []
    monkeypatch.setattr(dashboard.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    monkeypatch.setattr(dashboard.shutil, "which", lambda _name: "/usr/bin/tmux")
    hdr = {"X-API-Key": "api-key-123", "Content-Type": "application/json"}

    def start(body):
        return _code(secured + "/api/start", headers=hdr, data=json.dumps(body).encode(), method="POST")

    assert start({"notebook": "nb9"}) == 400
    assert start({"notebook": "nb1", "gpu": "V100"}) == 400
    assert start({"notebook": "nb1", "session": "bad_foreign"}) == 400
    assert start({"notebook": "nb1", "max_minutes": 9999}) == 400
    assert calls == []

    # Foreign ComfyUI A100 active on acc11 must NOT block owned launch
    dashboard.STATE["usage"]["active"] = "1"
    dashboard.STATE["sessions"] = "[?] m-s-comfyui-a100 | Hardware: A100 | Shape: HIGH_RAM | Variant: GPU"
    assert start({
        "notebook": "nb1", "gpu": "A100", "session": "se374_arm_a",
        "preset": "arm_a_polysgd", "max_minutes": 240, "env": {"STAGE1_SEED": "42"},
    }) == 200
    assert calls[-1][:5] == ["tmux", "new-session", "-d", "-s", "colab_se374_arm_a"]
    launch_cmd = calls[-1][5]
    assert launch_cmd.startswith("exec bash ")
    assert "--gpu A100" in launch_cmd and "--no-high-mem" in launch_cmd
    assert "--preset arm_a_polysgd" in launch_cmd and "STAGE1_SEED=42" in launch_cmd

    # Explicit high_mem=True on A100 passes --high-mem; H100 defaults to --high-mem
    monkeypatch.setenv("COLAB_MAX_OWNED_CONCURRENT", "3")
    assert start({"notebook": "nb1", "gpu": "A100", "high_mem": True, "session": "se374_a100_hm"}) == 200
    assert "--high-mem" in calls[-1][5]

    # _PENDING_STARTS TOCTOU guard: 2 pending launches block a 3rd launch at limit 2
    monkeypatch.setenv("COLAB_MAX_OWNED_CONCURRENT", "2")
    assert start({"notebook": "nb1", "gpu": "H100", "session": "se374_arm_b"}) == 400
    dashboard._PENDING_STARTS.clear()

    # Owned concurrency guard (runner_pids): 1 blocks second owned runner, env=2 allows parallel
    monkeypatch.setenv("COLAB_MAX_OWNED_CONCURRENT", "1")
    monkeypatch.setattr(dashboard, "runner_pids", lambda: {"se374_arm_a": [1234]})
    assert start({"notebook": "nb1", "gpu": "H100", "session": "se374_arm_b"}) == 400
    monkeypatch.setenv("COLAB_MAX_OWNED_CONCURRENT", "2")
    assert start({"notebook": "nb1", "gpu": "H100", "session": "se374_arm_b"}) == 200
    assert calls[-1][:5] == ["tmux", "new-session", "-d", "-s", "colab_se374_arm_b"]
    assert "--high-mem" in calls[-1][5]


def test_scoped_stop_and_per_session_pids(secured, monkeypatch) -> None:
    """P0: /api/stop targets explicit session or owned_all, never touching foreign sessions or other PIDs."""
    killed, runs = [], []
    monkeypatch.setattr(dashboard, "runner_pids", lambda: {"se374_arm_a": [1111], "se374_arm_b": [2222]})
    monkeypatch.setattr(dashboard.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(dashboard, "refresh_colab", lambda: None)

    class _Res:
        stdout, stderr = "ok", ""

    monkeypatch.setattr(dashboard.subprocess, "run", lambda cmd, **_k: runs.append(cmd) or _Res())
    monkeypatch.setattr(dashboard.time, "sleep", lambda _s: None)

    status = json.load(_open(secured + "/api/status?session=nb1", headers={"X-API-Key": "api-key-123"}))
    assert status["runner_running"] is False and status["runner_pids"] == []

    hdr = {"X-API-Key": "api-key-123", "Content-Type": "application/json"}
    assert _code(secured + "/api/stop", headers=hdr, data=json.dumps({"session": "se374_arm_a"}).encode(), method="POST") == 200
    assert [p for p, _ in killed] == [1111]
    assert runs[-1][-2:] == ["--stop-session", "se374_arm_a"]

    killed.clear()
    runs.clear()
    assert _code(secured + "/api/stop", headers={"X-API-Key": "api-key-123"}, data=b"", method="POST") == 200
    assert sorted(p for p, _ in killed) == [1111, 2222]
    assert runs[-1][-1] == "--stop-owned"

    # Non-zero exit from session_guard / runner must return HTTP 400
    class _Refused:
        returncode = 1
        stdout, stderr = "", "RuntimeError: REFUSED: protected endpoint"

    monkeypatch.setattr(dashboard.subprocess, "run", lambda cmd, **_k: _Refused())
    assert _code(secured + "/api/stop", headers=hdr, data=json.dumps({"session": "se374_arm_a"}).encode(), method="POST") == 400


def test_stop_rejects_empty_session_scope_and_pops_pending_on_idempotent_stop(secured, monkeypatch) -> None:
    """Empty session with scope='session' returns 400 (never --stop-owned) and idempotent stop clears _PENDING_STARTS."""
    runs = []
    monkeypatch.setattr(dashboard, "runner_pids", lambda: {})
    monkeypatch.setattr(dashboard, "refresh_colab", lambda: None)

    class _Idem:
        returncode = 0
        stdout, stderr = '{"session": "se374_arm_a", "returncode": 0, "output": "already stopped"}', ""

    monkeypatch.setattr(dashboard.subprocess, "run", lambda cmd, **_k: runs.append(cmd) or _Idem())
    hdr = {"X-API-Key": "api-key-123", "Content-Type": "application/json"}
    assert _code(secured + "/api/stop", headers=hdr, data=json.dumps({"scope": "session", "session": ""}).encode(), method="POST") == 400
    assert _code(secured + "/api/stop", headers=hdr, data=json.dumps({"scope": "session"}).encode(), method="POST") == 400
    assert _code(secured + "/api/stop", headers=hdr, data=json.dumps({"session": ""}).encode(), method="POST") == 400
    assert runs == []
    assert _code(secured + "/api/stop", headers=hdr, data=b"{}", method="POST") == 200
    assert runs[-1][-1] == "--stop-owned"

    dashboard._PENDING_STARTS["se374_arm_a"] = 9999999999.0
    assert _code(secured + "/api/stop", headers=hdr, data=json.dumps({"scope": "session", "session": "se374_arm_a"}).encode(), method="POST") == 200
    assert "se374_arm_a" not in dashboard._PENDING_STARTS


def test_simultaneous_post_start_race_respects_max_concurrent(secured, monkeypatch) -> None:
    """Two simultaneous POST /api/start requests with MAX=1 allow exactly 1 launch (200) and reject 1 (400)."""
    tmux_new_calls = []

    def fake_run(cmd, **_k):
        if cmd[:2] == ["tmux", "new-session"]:
            tmux_new_calls.append(cmd)

    monkeypatch.setattr(dashboard.subprocess, "run", fake_run)
    monkeypatch.setattr(dashboard.shutil, "which", lambda _name: "/usr/bin/tmux")
    monkeypatch.setenv("COLAB_MAX_OWNED_CONCURRENT", "1")
    hdr = {"X-API-Key": "api-key-123", "Content-Type": "application/json"}
    barrier = threading.Barrier(2)
    results = []

    def fire(sess_name):
        barrier.wait()
        code = _code(
            secured + "/api/start",
            headers=hdr,
            data=json.dumps({"notebook": "nb1", "gpu": "A100", "session": sess_name}).encode(),
            method="POST",
        )
        results.append(code)

    t1 = threading.Thread(target=fire, args=("se374_race_1",))
    t2 = threading.Thread(target=fire, args=("se374_race_2",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert sorted(results) == [200, 400]
    assert len(tmux_new_calls) == 1


def test_reset_password_keeps_api_key_and_accepts_custom_password(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dash_auth, "AUTH_FILE", tmp_path / "auth.json")
    user, pw, key = dash_auth.init_credentials("colab")
    assert pw and key and dash_auth.init_credentials("colab") == (user, None, key)
    user2, pw2, key2 = dash_auth.init_credentials("colab", reset=True, password="admin123")
    assert (pw2, key2) == ("admin123", key)
    monkeypatch.delenv("DASH_USER", raising=False)
    dash_auth.CREDS.clear()
    assert dash_auth.load() and dash_auth.verify_login("colab", "admin123") and not dash_auth.verify_login("colab", pw)
    dash_auth.configure(None)


def test_params_and_progress_are_parsed() -> None:
    log = (
        "[READY] Compute Device: cuda (NVIDIA A100)\n"
        "[PARITY] Loss=MultiLabelSoftMarginLoss (LOSS_TYPE=soft_margin) | norm=zero_one\n"
        "[PARITY] Epochs=20 | LR=0.0015 | Steps/Epoch=731 | Total Steps=14,620\n"
        "[EPOCH 01] Train Loss=0.2952 | Val mAP=0.89 GatedPM_mIoU=0.6511\n"
        "[EPOCH 02] Train Loss=0.2503 | Val mAP=0.88 GatedPM_mIoU=0.6652\n"
    )
    lines = dashboard.clean_lines(log)
    params, cur_total = dashboard.parse_params(log, lines)
    assert dict(params)["Thiết bị"] == "cuda (NVIDIA A100)"
    prog = dashboard.progress(dashboard.parse_epochs(lines), cur_total, {}, False)
    assert prog["best"]["name"] == "GatedPM_mIoU" and prog["best"]["epoch"] == 2


def test_index_serves_light_blue_html_and_stop_requires_header(server) -> None:
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "Colab VPS Monitor" in html and "#1565c0" in html and "TẮT GPU NGAY" in html
    assert '{ scope: "session", session: current }' in html
    assert "Thông số chính của lần chạy" in html
    assert "default=60" in (Path(__file__).resolve().parent / "dashboard.py").read_text(encoding="utf-8")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(urllib.request.Request(server + "/api/stop", method="POST"))
    assert err.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as err2:
        urllib.request.urlopen(urllib.request.Request(server + "/api/status", headers={"Host": "evil.example.com"}))
    assert err2.value.code == 403


def test_run_colab_passes_isolated_readonly_config(tmp_path, monkeypatch) -> None:
    seen = []
    monkeypatch.setenv("COLAB_GUARD_STATE_DIR", str(tmp_path / "gs"))

    def fake_run(cmd, **_kw):
        seen.append(cmd)
        return type("R", (), {"stdout": "ok", "stderr": ""})()

    monkeypatch.setattr(dashboard.subprocess, "run", fake_run)
    for sub in ("usage", "sessions"):
        assert dashboard.run_colab(sub) == "ok"
        cfg_idx = seen[-1].index("--config")
        assert seen[-1][cfg_idx + 1] == str(tmp_path / "gs" / ".readonly_sessions.json")
        assert seen[-1][cfg_idx + 2 : cfg_idx + 5] == ["--auth", "adc", sub]
