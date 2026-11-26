import ast
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import pytest

IGNORED_DASH_PREFIXES = (
    "repos\\", "repos/",
    "graphify-out\\", "graphify-out/",
    "graphify-out-all\\", "graphify-out-all/",
    "temp_graphify_all\\", "temp_graphify_all/",
    ".pytest_cache\\", ".pytest_cache/",
    ".ruff_cache\\", ".ruff_cache/",
    ".git\\", ".git/",
)

# Exact byte-for-byte copy of instructor's lab notebook and user-uploaded reference documents
EXEMPT_EXTERNAL_DOCS = {
    os.path.normpath("notebooks/Lab1.ipynb"),
    os.path.normpath("Lab1.ipynb"),
    os.path.normpath("docs/references/Quy Trình Tester Y Khoa.md"),
}

BAD_DASH_CHARS = ['\u2014', '\u2013', '\u2012', '\u2015', '\u2212']

SECRET_TOKEN_REGEXES = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
]


def _is_exempt_file(fpath: str) -> bool:
    if fpath.startswith(IGNORED_DASH_PREFIXES):
        return True
    norm_p = os.path.normpath(fpath)
    return norm_p in EXEMPT_EXTERNAL_DOCS


def _scan_files_for_bad_dashes(patterns):
    target_files = []
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            if not _is_exempt_file(f):
                target_files.append(f)

    violations = []
    for fpath in target_files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f, start=1):
                for bad_c in BAD_DASH_CHARS:
                    if bad_c in line:
                        violations.append(f"{fpath}:{line_idx} contains bad dash (U+{ord(bad_c):04X})")
    return violations


def test_no_ai_em_dash_in_markdown():
    """Verify zero non-ASCII dashes across all markdown files (Invariant 12)."""
    violations = _scan_files_for_bad_dashes(["**/*.md"])
    assert len(violations) == 0, (
        f"Found {len(violations)} non-ASCII dash violations in markdown:\n"
        + "\n".join(violations[:10])
    )


def test_no_ai_em_dash_in_python_and_notebooks():
    """Verify zero non-ASCII dashes across all project Python files and Jupyter notebooks (Invariant 12)."""
    violations = _scan_files_for_bad_dashes(["**/*.py", "**/*.ipynb"])
    assert len(violations) == 0, (
        f"Found {len(violations)} non-ASCII dash violations in code/notebooks:\n"
        + "\n".join(violations[:10])
    )


def test_no_leaked_api_tokens_in_project_files():
    """Verify no hardcoded HuggingFace, GitHub, or Google API tokens exist in project code/notebooks."""
    leaks = []
    for pat in ["src/**/*.py", "scripts/**/*.py", "notebooks/**/*.ipynb"]:
        for fpath in glob.glob(pat, recursive=True):
            if _is_exempt_file(fpath):
                continue
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, start=1):
                    for rgx in SECRET_TOKEN_REGEXES:
                        if rgx.search(line):
                            leaks.append(f"{fpath}:{line_idx} matches secret token pattern {rgx.pattern}")
    assert len(leaks) == 0, "Detected leaked API tokens in project files:\n" + "\n".join(leaks)


def test_no_ai_em_dash_in_html_workstation():
    """Verify zero non-ASCII dashes in src/visualization/static/*.html (Invariant 12)."""
    violations = _scan_files_for_bad_dashes(["src/visualization/static/*.html"])
    assert len(violations) == 0, (
        "Found non-ASCII dash violations in workstation HTML:\n" + "\n".join(violations[:10])
    )


def test_src_and_drive_bundle_ast_and_content_sync():
    """Verify every Python module in src/ is synchronized with drive_upload_ready/SE374_WSSS_Drive_Code/src/."""
    src_root = Path("src")
    bundle_src_root = Path("drive_upload_ready/SE374_WSSS_Drive_Code/src")
    assert src_root.is_dir(), "src/ directory must exist"
    if not bundle_src_root.is_dir():
        pytest.skip("Gitignored Drive bundle drive_upload_ready/SE374_WSSS_Drive_Code/src is not present on clean CI checkout.")

    src_py_files = sorted(src_root.rglob("*.py"))
    assert len(src_py_files) >= 25, "Expected at least 25 Python modules in src/"

    mismatches = []
    for src_file in src_py_files:
        rel_path = src_file.relative_to(src_root)
        bundle_file = bundle_src_root / rel_path
        if not bundle_file.is_file():
            mismatches.append(f"Missing in Drive bundle: {rel_path}")
            continue
        src_text = src_file.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        bundle_text = bundle_file.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        # Verify AST syntax validity of both
        ast.parse(src_text, filename=str(src_file))
        ast.parse(bundle_text, filename=str(bundle_file))
        if hashlib.sha256(src_text.encode("utf-8")).hexdigest() != hashlib.sha256(bundle_text.encode("utf-8")).hexdigest():
            mismatches.append(f"Out-of-sync content between src/{rel_path} and Drive bundle")

    assert len(mismatches) == 0, "Drive bundle out of sync with src/:\n" + "\n".join(mismatches)


def test_teacher_lab1_notebook_copy_untouched_and_valid_json():
    """Verify that if notebooks/Lab1.ipynb exists, it is valid nbformat v4 JSON and untouched."""
    lab1_path = Path("notebooks/Lab1.ipynb")
    if lab1_path.is_file():
        data = json.loads(lab1_path.read_text(encoding="utf-8"))
        assert data.get("nbformat") == 4
        assert len(data.get("cells", [])) >= 10


def test_invariants_24_and_25_defined_in_agents_md():
    """Verify that Invariant 24 (Scientific visuals) and Invariant 25 (Auditor gate) exist in AGENTS.md."""
    agents_md = Path("AGENTS.md")
    if not agents_md.is_file():
        pytest.skip("Gitignored AGENTS.md is not present on clean CI checkout.")
    content = agents_md.read_text(encoding="utf-8")
    assert "[INVARIANT-24]" in content, "INVARIANT-24 must be defined in AGENTS.md"
    assert "[INVARIANT-25]" in content, "INVARIANT-25 must be defined in AGENTS.md"
    assert "deliverable_auditor.py" in content, "deliverable_auditor.py must be documented in AGENTS.md"


def test_no_raw_phone_photos_in_product_directory():
    """Verify that no camera/phone photos (media_*, IMG_*, PXL_*) exist in product/."""
    forbidden_prefixes = ("media_", "img_", "pxl_", "screenshot", "dcm_")
    for root, _, files in os.walk("product"):
        for fname in files:
            lower = fname.lower()
            for pref in forbidden_prefixes:
                assert not lower.startswith(pref), f"Raw photo/screenshot detected in product: {os.path.join(root, fname)}"


def test_no_private_vps_runner_or_account_leaks_in_tracked_repo():
    """Verify zero private account IDs, runner tools, or personal local paths leak in tracked text or OOXML/PDF files."""
    import subprocess
    import zipfile

    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf

    git_check = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=False)
    if git_check.returncode != 0:
        pytest.skip("Not inside a git working tree.")

    tracked_files = [f.strip() for f in git_check.stdout.splitlines() if f.strip()]
    binary_exts = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".docx", ".pptx", ".zip", ".pth", ".npz", ".ico", ".xlsx"}
    exempt = {".gitignore", ".github/workflows/ci.yml", "tests/test_governance_invariants.py"}
    forbidden_rgx = re.compile(
        r"\bacc[0-9]{2}\b|\bacc[0-9]+_adc\b"
        r"|\b(?:vps_colab_runner|run_colab_vps|session_guard|dash_auth|setup_vps_colab|VPS_COLAB_RUNBOOK)\b"
        r"|(?:file:///)?[A-Za-z]:[/\\]{1,2}Users[/\\]{1,2}",
        re.IGNORECASE,
    )
    personal_binary_rgx = re.compile(
        r"(?i)file:///|[A-Za-z]:[/\\]{1,2}Users[/\\]{1,2}|/Users/[^/\s]+/"
    )
    xml_tag_rgx = re.compile(r"<[^>]+>")

    violations = []
    for rel in tracked_files:
        norm = rel.replace("\\", "/")
        p = Path(rel)
        if not p.is_file():
            continue
        suf = p.suffix.lower()

        if suf in {".docx", ".pptx"}:
            with zipfile.ZipFile(p, "r") as zf:
                for member in zf.namelist():
                    if not member.endswith((".xml", ".rels")):
                        continue
                    xml_text = zf.read(member).decode("utf-8", errors="ignore")
                    m = personal_binary_rgx.search(xml_text) or personal_binary_rgx.search(
                        xml_tag_rgx.sub("", xml_text)
                    )
                    if m:
                        violations.append(f"{norm}::{member} contains personal path token '{m.group(0)}'")
                        break
            continue

        if suf == ".pdf":
            with pymupdf.open(p) as doc:
                hit = False
                for page_idx, page in enumerate(doc, start=1):
                    m = personal_binary_rgx.search(page.get_text())
                    if m:
                        violations.append(f"{norm}#p{page_idx} contains personal path token '{m.group(0)}'")
                        hit = True
                        break
                    for lnk in page.get_links():
                        uri = str(lnk.get("uri", ""))
                        m_uri = personal_binary_rgx.search(uri)
                        if m_uri:
                            violations.append(f"{norm}#p{page_idx} link contains personal path '{m_uri.group(0)}'")
                            hit = True
                            break
                    if hit:
                        break
                if not hit:
                    m_meta = personal_binary_rgx.search(str(doc.metadata))
                    if m_meta:
                        violations.append(f"{norm} metadata contains personal path token '{m_meta.group(0)}'")
            continue

        if norm in exempt or suf in binary_exts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        for idx, line in enumerate(text.splitlines(), start=1):
            m = forbidden_rgx.search(line)
            if m:
                violations.append(f"{norm}:{idx} contains forbidden private token '{m.group(0)}'")

    assert not violations, "Private account / runner / personal path leak detected in tracked academic repo:\n" + "\n".join(
        violations[:20]
    )

