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


def test_index_serves_light_blue_html_and_stop_requires_header(server) -> None:
    html = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "Colab VPS Monitor" in html and "#1565c0" in html and "TẮT GPU NGAY" in html

    req = urllib.request.Request(server + "/api/stop", method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 403

    bad_host = urllib.request.Request(server + "/api/status", headers={"Host": "evil.example.com"})
    with pytest.raises(urllib.error.HTTPError) as err2:
        urllib.request.urlopen(bad_host)
    assert err2.value.code == 403


@pytest.mark.parametrize("name", ["dashboard.py", "dashboard.html"])
def test_dashboard_files_line_cap_and_ascii_dashes(name) -> None:
    path = Path(__file__).resolve().parent / name
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 400
    for bad in FORBIDDEN_DASHES:
        assert bad not in text
