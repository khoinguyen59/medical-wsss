import os
import glob

def test_no_ai_em_dash_in_markdown():
    bad_chars = ['\u2014', '\u2013', '\u2012', '\u2015', '\u2212']
    
    ignored_prefixes = (
        "repos\\", "repos/",
        "graphify-out\\", "graphify-out/",
        "graphify-out-all\\", "graphify-out-all/",
        "temp_graphify_all\\", "temp_graphify_all/",
        ".pytest_cache\\", ".pytest_cache/"
    )
    
    md_files = glob.glob("**/*.md", recursive=True)
    target_files = [
        f for f in md_files 
        if not f.startswith(ignored_prefixes)
    ]
    
    violations = []
    for fpath in target_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                for bad_c in bad_chars:
                    if bad_c in line:
                        violations.append(f"{fpath}:{line_idx} contains bad dash (U+{ord(bad_c):04X})")
    
    assert len(violations) == 0, f"Found {len(violations)} non-ASCII dash violations:\n" + "\n".join(violations[:10])


def test_invariants_24_and_25_defined_in_agents_md():
    """Verify that Invariant 24 (Scientific visuals) and Invariant 25 (Auditor gate) exist in AGENTS.md."""
    with open("AGENTS.md", "r", encoding="utf-8") as f:
        content = f.read()
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
