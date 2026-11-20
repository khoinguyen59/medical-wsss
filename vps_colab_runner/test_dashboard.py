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
    monkeypatch.setattr(dashboard, "runner_pids", lambda: [])
    dash_auth.configure(None)
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
    assert _code(secured + "/", headers=host) == 302  # redirect to /login
    assert "Đăng nhập" in _open(secured + "/login", headers=host).read().decode("utf-8")

    form = b"username=colab&password=wrong"
    assert _code(secured + "/login", headers=host, data=form, method="POST") == 401

    good = b"username=colab&password=p%40ss-Word1"
    with pytest.raises(urllib.error.HTTPError) as redirect:
        _open(secured + "/login", headers=host, data=good, method="POST")
    assert redirect.value.code == 302
    cookie = redirect.value.headers["Set-Cookie"]
    assert "dash_session=" in cookie and "HttpOnly" in cookie and "Secure" in cookie
    sess = cookie.split(";")[0]

    data = json.load(_open(secured + "/api/status?session=nb1", headers={**host, "Cookie": sess}))
    assert data["session"] == "nb1"
    assert _code(secured + "/api/stop", headers={**host, "Cookie": sess}, data=b"", method="POST") == 403  # CSRF header
    assert _code(secured + "/logout", headers={**host, "Cookie": sess}) == 302
    assert _code(secured + "/api/status", headers={**host, "Cookie": sess}) == 401


def test_api_key_and_login_lockout(secured) -> None:
    assert json.load(_open(secured + "/api/status", headers={"X-API-Key": "api-key-123"}))["usage"]["active"] == "0"
    assert _code(secured + "/api/status", headers={"X-API-Key": "nope"}) == 401

    bad = b"username=colab&password=bad"
    codes = [_code(secured + "/login", headers={"CF-Connecting-IP": "9.9.9.9"}, data=bad, method="POST") for _ in range(7)]
    assert codes[:5] == [401] * 5 and codes[5:] == [429, 429]


def test_start_validates_input_and_launches_tmux(secured, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dashboard.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    monkeypatch.setattr(dashboard.shutil, "which", lambda _name: "/usr/bin/tmux")
    hdr = {"X-API-Key": "api-key-123", "Content-Type": "application/json"}

    def start(body):
        return _code(secured + "/api/start", headers=hdr, data=json.dumps(body).encode(), method="POST")

    assert start({"notebook": "nb9"}) == 400
    assert start({"notebook": "nb1", "gpu": "H100"}) == 400
    assert start({"notebook": "nb1", "max_minutes": 9999}) == 400
    assert calls == []
    assert start({"notebook": "nb1", "gpu": "L4", "max_minutes": 120}) == 200
    assert calls[-1][:5] == ["tmux", "new-session", "-d", "-s", "colab_nb1"]
    assert "--max-minutes 120" in calls[-1][5]

    monkeypatch.setattr(dashboard, "runner_pids", lambda: [1234])
    assert start({"notebook": "nb2"}) == 400  # runner already running
    monkeypatch.setattr(dashboard, "runner_pids", lambda: [])
    dashboard.STATE["usage"]["active"] = "1"
    assert start({"notebook": "nb2"}) == 400  # GPU already active
    dashboard.STATE["usage"]["active"] = "0"


def test_params_and_progress_are_parsed() -> None:
    log = (
        "[READY] Compute Device: cuda (NVIDIA L4)\n"
        "[PARITY] Loss=MultiLabelSoftMarginLoss (LOSS_TYPE=soft_margin) | norm=zero_one\n"
        "[PARITY] Epochs=25 | LR=0.00014 | Steps/Epoch=731 | Total Steps=18,275\n"
        "[DATALOADER] Train batches/epoch: 731 (BS=32, Samples=23,422, norm=zero_one)\n"
        "[MODEL] Stage 1 ResNet38d Net_CAM: 105,087,296 params | Pretrained: domain_pretrained:luad (228/228 layers)\n"
        "Epoch 03/25 [PDA=True, gamma=1.00]:   0%|\n"
        "[EPOCH 01] Train Loss=0.2952 | Val mAP=0.89 GatedPM_mIoU=0.6511\n"
        "[EPOCH 02] Train Loss=0.2503 | Val mAP=0.88 GatedPM_mIoU=0.6652\n"
    )
    lines = dashboard.clean_lines(log)
    params, cur_total = dashboard.parse_params(log, lines)
    labels = dict(params)
    assert labels["Thiết bị"] == "cuda (NVIDIA L4)"
    assert "LR=0.00014" in labels["Epoch / LR / Steps"]
    assert labels["Hàm loss"].startswith("MultiLabelSoftMarginLoss")
    assert cur_total == (3, 25)
    prog = dashboard.progress(dashboard.parse_epochs(lines), cur_total, {}, False)
    assert prog["best"]["name"] == "GatedPM_mIoU" and prog["best"]["epoch"] == 2
    assert prog["done"] == 2 and prog["total"] == 25


def test_index_serves_light_blue_html_and_stop_requires_header(server) -> None:
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "Colab VPS Monitor" in html and "#1565c0" in html and "TẮT GPU NGAY" in html
    assert "Thông số chính của lần chạy" in html

    req = urllib.request.Request(server + "/api/stop", method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 403

    bad_host = urllib.request.Request(server + "/api/status", headers={"Host": "evil.example.com"})
    with pytest.raises(urllib.error.HTTPError) as err2:
        urllib.request.urlopen(bad_host)
    assert err2.value.code == 403


@pytest.mark.parametrize("name", ["dashboard.py", "dashboard.html", "dash_auth.py", "login.html"])
def test_dashboard_files_line_cap_and_ascii_dashes(name) -> None:
    path = Path(__file__).resolve().parent / name
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 400
    for bad in FORBIDDEN_DASHES:
        assert bad not in text
