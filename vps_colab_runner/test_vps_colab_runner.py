"""Standalone tests for the temporary isolated vps_colab_runner/ folder."""

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER_DIR = PROJECT_ROOT / "vps_colab_runner"
SETUP_SCRIPT = RUNNER_DIR / "setup_vps_colab.sh"
RUNNER_SCRIPT = RUNNER_DIR / "run_colab_vps.sh"
README_DOC = RUNNER_DIR / "README.md"
PUBLIC_SCRIPT = RUNNER_DIR / "start_dashboard_public.sh"
FORBIDDEN_DASHES = ("\u2014", "\u2013", "\u2015", "\u2212")


@pytest.mark.parametrize("path", [SETUP_SCRIPT, RUNNER_SCRIPT, README_DOC, PUBLIC_SCRIPT])
def test_vps_files_exist_ascii_dashes_and_line_cap(path: Path) -> None:
    """All temporary VPS runner files must stay <= 400 lines, use LF on .sh, and obey Invariant-12."""
    assert path.is_file(), f"Missing expected file: {path}"
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8")

    if path.suffix == ".sh":
        assert b"\r\n" not in raw_bytes, f"{path.name} contains Windows CRLF line endings"

    lines = text.splitlines()
    assert len(lines) <= 400, f"{path.name} has {len(lines)} lines (> 400 cap)"

    for bad in FORBIDDEN_DASHES:
        assert bad not in text, f"Forbidden non-ASCII dash {bad!r} found in {path.name}"

    for secret_marker in ("refresh_token", "client_secret", "hf_", "AIza"):
        assert secret_marker not in text, f"Potential secret marker {secret_marker!r} in {path.name}"


def test_runner_script_has_mandatory_cost_and_resume_guards() -> None:
    """Runner script must enforce trap auto-stop, wall-clock watchdog, alarm fail-fast, and SHA256 sync."""
    text = RUNNER_SCRIPT.read_text(encoding="utf-8")
    required_snippets = [
        "trap cleanup EXIT INT TERM",
        "DEADLINE_EPOCH",
        "WATCHDOG_TIMEOUT",
        "EARLY REGRESSION ALARM",
        "Loss: nan|Non-finite epoch loss",
        "sha256sum",
        'mv -f "$tmp_file" "$local_path"',
        "upload_checkpoint_for_resume",
        'colab_cmd stop -s "$SESSION_NAME"',
        "--stop-all",
        "--dry-run",
        "MAX_RETRIES",
        "vps_colab_runner/runs/$SESSION_NAME",
    ]
    for snippet in required_snippets:
        assert snippet in text, f"Missing mandatory guard snippet in run_colab_vps.sh: {snippet}"


def test_setup_script_supports_oracle_and_ubuntu_ipv4() -> None:
    """Setup script must support apt/dnf, IPv4 gai.conf preference, venv, and ed25519 key."""
    text = SETUP_SCRIPT.read_text(encoding="utf-8")
    for snippet in (
        "apt-get",
        "dnf",
        "precedence ::ffff:0:0/96  100",
        "google-colab-cli",
        "ssh-keygen -q -t ed25519",
        "chmod 600",
    ):
        assert snippet in text, f"Missing expected setup step in setup_vps_colab.sh: {snippet}"


def test_bash_syntax_and_dry_run_in_wsl() -> None:
    """Validate bash syntax (`bash -n`) and `--dry-run` execution if WSL Ubuntu is available."""
    wsl = shutil.which("wsl")
    if wsl is None:
        pytest.skip("WSL not available on this host")

    for rel_script in (
        "vps_colab_runner/setup_vps_colab.sh",
        "vps_colab_runner/run_colab_vps.sh",
        "vps_colab_runner/start_dashboard_public.sh",
    ):
        res = subprocess.run(
            [wsl, "-d", "Ubuntu", "--", "bash", "-n", rel_script],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert res.returncode == 0, f"bash -n failed for {rel_script}: {res.stderr}"

    for preset, expected_nb in (
        ("nb1", "notebooks/01_Train_Stage1_Classifier_L4.ipynb"),
        ("nb2", "notebooks/02_Pseudo_Masks_Train_Stage2_L4.ipynb"),
    ):
        dry = subprocess.run(
            [
                wsl,
                "-d",
                "Ubuntu",
                "--",
                "bash",
                "vps_colab_runner/run_colab_vps.sh",
                "--notebook",
                preset,
                "--gpu",
                "L4",
                "--dry-run",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert dry.returncode == 0, f"--dry-run {preset} failed: {dry.stdout}\n{dry.stderr}"
        assert "Validation PASS. Zero Colab sessions created." in dry.stdout
        assert expected_nb in dry.stdout
        assert f"vps_colab_runner/runs/{preset}" in dry.stdout
