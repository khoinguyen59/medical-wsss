"""
Specialized Audit Suite 6: Downloader Security, Archive Validation & Multi-Seed Comparative Runner.

Focused, non-overlapping verification for Pillar 8 (Audit Team 6):
1. URL scheme security gate (_require_https) in src/colab_pkg/downloader.py:
   - Blocks file://, http://, ftp://, javascript:, and empty schemes.
   - Allows valid https:// URLs.
2. Archive and checkpoint integrity validators (is_valid_zip_archive, is_valid_torch_checkpoint):
   - Rejects truncated files, HTML quota-exceeded pages masquerading as .zip/.pth, and empty files.
   - Accepts genuine zip archives and PyTorch state_dict checkpoints.
3. Comparative benchmark runner (src/pipeline/comparative_runner.py) multi-seed dry-run execution
   on real sample patches from data/sample_dataset/, verifying JSON and Markdown report generation.

Strictly adheres to Invariant 12 (standard ASCII hyphen '-' 0x2D only).
"""

import json
import zipfile
from pathlib import Path

import pytest
import torch

from src.colab_pkg.downloader import (
    _require_https,
    is_valid_torch_checkpoint,
    is_valid_zip_archive,
)
from src.pipeline.comparative_runner import run_comparative_benchmark


class TestDownloaderSecurityAndArchiveIntegrity:
    """Verifies URL scheme security checks and corrupted archive/checkpoint detection."""

    def test_require_https_blocks_insecure_and_local_file_schemes(self):
        """Verifies _require_https blocks file://, http://, ftp://, and custom schemes."""
        # Valid HTTPS URLs must pass
        _require_https("https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip")

        # Insecure or local file schemes must raise ValueError
        forbidden_urls = [
            "http://huggingface.co/datasets/BCSS-WSSS.zip",
            "file:///etc/passwd",
            "file:///C:/Windows/System32/drivers/etc/hosts",
            "ftp://mirror.example.com/dataset.zip",
            "javascript:alert(1)",
            "",
        ]
        for bad_url in forbidden_urls:
            with pytest.raises(ValueError, match="Only https URLs are allowed"):
                _require_https(bad_url)

    def test_is_valid_zip_archive_rejects_html_quota_page_and_accepts_real_zip(self, tmp_path: Path):
        """Verifies Google Drive HTML quota pages saved as .zip are rejected by is_valid_zip_archive."""
        # 1. Fake HTML quota page masquerading as .zip
        fake_zip = tmp_path / "quota_exceeded.zip"
        fake_zip.write_text("<html><body>Google Drive - Quota Exceeded</body></html>" * 100, encoding="utf-8")
        assert is_valid_zip_archive(fake_zip, min_valid_bytes=100) is False

        # 2. Genuine non-empty zip archive
        real_zip = tmp_path / "valid_archive.zip"
        with zipfile.ZipFile(real_zip, "w") as zf:
            zf.writestr("patch_01.png", b"\x89PNG\r\n\x1a\n" + b"0" * 512)
        assert is_valid_zip_archive(real_zip, min_valid_bytes=100) is True
        # Fails if min_valid_bytes is larger than file size
        assert is_valid_zip_archive(real_zip, min_valid_bytes=10 * 1024 * 1024) is False

    def test_is_valid_torch_checkpoint_rejects_corrupt_and_accepts_state_dict(self, tmp_path: Path):
        """Verifies is_valid_torch_checkpoint rejects corrupt files and accepts valid .pth dicts."""
        corrupt_ckpt = tmp_path / "corrupt.pth"
        corrupt_ckpt.write_bytes(b"<!DOCTYPE html><html>Error</html>" * 50)
        assert is_valid_torch_checkpoint(corrupt_ckpt, min_valid_bytes=64) is False

        valid_ckpt = tmp_path / "valid_model.pth"
        torch.save({"conv1.weight": torch.randn(8, 3, 3, 3), "conv1.bias": torch.zeros(8)}, valid_ckpt)
        assert is_valid_torch_checkpoint(valid_ckpt, min_valid_bytes=64) is True


class TestComparativeBenchmarkRunnerEndToEnd:
    """Verifies run_comparative_benchmark multi-seed execution and artifact output."""

    def test_run_comparative_benchmark_multi_seed_on_sample_dataset(self, tmp_path: Path):
        """Runs run_comparative_benchmark across 2 seeds and 2 configurations on data/sample_dataset."""
        out_dir = tmp_path / "benchmarks"
        results = run_comparative_benchmark(
            data_root="data/sample_dataset",
            output_dir=str(out_dir),
            split="valid",
            seeds=[42, 100],
            configurations=["baseline", "proposed"],
            dry_run_samples=2,
            device="cpu",
        )

        assert "baseline" in results and "proposed" in results
        for cfg in ("baseline", "proposed"):
            assert "mIoU" in results[cfg]
            assert "mean" in results[cfg]["mIoU"]
            assert "std" in results[cfg]["mIoU"]
            assert 0.0 <= results[cfg]["mIoU"]["mean"] <= 100.0

        json_path = out_dir / "benchmark_results.json"
        md_path = out_dir / "BENCHMARK_REPORT.md"
        assert json_path.is_file(), "Benchmark JSON log was not created"
        assert md_path.is_file(), "Benchmark Markdown summary table was not created"

        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        assert "aggregated" in parsed
        assert "baseline" in parsed["aggregated"] and "proposed" in parsed["aggregated"]
