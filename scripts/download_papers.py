import json
import os
import sys
import time
import urllib.request
import requests

PAPERS = [
    ("DBDC_WSL_MICCAI2025.pdf", "https://papers.miccai.org/miccai-2025/paper/2287_paper.pdf"),
    ("D_CAM_MICCAI2025.pdf", "https://papers.miccai.org/miccai-2025/paper/0830_paper.pdf"),
    ("MambaMIL_arxiv_2403.06800.pdf", "https://arxiv.org/pdf/2403.06800"),
    ("DualProtoSeg_arxiv_2512.10314.pdf", "https://arxiv.org/pdf/2512.10314"),
    ("QFKF_BMVC2025.pdf", "https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_218/paper.pdf"),
    ("PFM_DenseBench_arxiv_2602.03887.pdf", "https://arxiv.org/pdf/2602.03887"),
    ("KMG_Mamba_MICCAI2025.pdf", "https://papers.miccai.org/miccai-2025/paper/0569_paper.pdf"),
    ("LPD_arxiv_2512.18342.pdf", "https://arxiv.org/pdf/2512.18342"),
]

def download_with_playwright(url, dest_path):
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            page = context.new_page()
            response = page.goto(url, wait_until="domcontentloaded")
            body = response.body()
            if len(body) > 10000:
                with open(dest_path, "wb") as f:
                    f.write(body)
                browser.close()
                return True
            browser.close()
    except Exception as e:
        print(f"Playwright error for {url}: {e}", flush=True)
    return False

def main():
    target_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "papers")
    os.makedirs(target_dir, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    
    for filename, url in PAPERS:
        dest_path = os.path.join(target_dir, filename)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
            print(f"[EXISTS] {filename} ({os.path.getsize(dest_path):,} bytes)", flush=True)
            continue
            
        print(f"[DOWNLOADING] {filename} from {url}...", flush=True)
        try:
            r = requests.get(url, headers=headers, timeout=25)
            if r.status_code == 200 and len(r.content) > 10000:
                with open(dest_path, "wb") as f:
                    f.write(r.content)
                print(f"[SUCCESS] {filename} ({len(r.content):,} bytes)", flush=True)
            else:
                print(f"[HTTP {r.status_code}] Failed for {filename}", flush=True)
        except Exception as e:
            print(f"[ERROR] {e} for {filename}", flush=True)
            
    # XWIL via OpenReview / Playwright
    xwil_dest = os.path.join(target_dir, "XWIL_OpenReview2026.pdf")
    if not (os.path.exists(xwil_dest) and os.path.getsize(xwil_dest) > 10000):
        print("[DOWNLOADING PLAYWRIGHT] XWIL_OpenReview2026.pdf...", flush=True)
        download_with_playwright("https://openreview.net/pdf/faba0cd741f00eba5a26d4d7dea02a0d99fbc3f7.pdf", xwil_dest)


def restore_markdown_docs_from_transcript():
    """Restores any 0-byte Markdown files in docs/ from all brain transcripts and git HEAD, sanitizing ASCII hyphens."""
    brain_root = r"C:\Users\Nguyen Trong Khoi\.gemini\antigravity\brain"
    bad_dashes = ("\u2013", "\u2014", "\u2012", "\u2015", "\u2212")

    def clean_hyphens(text: str) -> str:
        for ch in bad_dashes:
            text = text.replace(ch, "-")
        return text

    # Map duplicate copies in docs/ root to docs/research/
    docs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
    for dup_name in (
        "GITHUB_REPOSITORIES_DEEP_ANALYSIS.md",
        "PROPOSAL_STRATEGY_4_DIFFERENTIATED_TOPICS.md",
        "PROPOSAL_TOPIC_RECOMMENDATION.md",
        "SE374_TOPIC6_WEAKLY_SUPERVISED_WSI_RESEARCH_AND_ROADMAP.md",
        "SE374_TOPIC_MATRIX_AND_GUIDE.md",
    ):
        src_p = os.path.join(docs_root, dup_name)
        dst_p = os.path.join(docs_root, "research", dup_name)
        if os.path.exists(src_p) and os.path.getsize(src_p) > 0 and os.path.exists(dst_p) and os.path.getsize(dst_p) == 0:
            with open(src_p, "r", encoding="utf-8") as rf:
                content = clean_hyphens(rf.read())
            with open(dst_p, "w", encoding="utf-8") as wf:
                wf.write(content)
            print(f"[COPIED FROM DOCS ROOT] {os.path.relpath(dst_p)} ({len(content):,} chars)", flush=True)

    # Scan all brain transcripts for write_file / write_to_file calls targeting Y_KHOA\docs
    restored_contents = {}
    if os.path.exists(brain_root):
        for conv_id in os.listdir(brain_root):
            tf_path = os.path.join(brain_root, conv_id, ".system_generated", "logs", "transcript_full.jsonl")
            if not os.path.exists(tf_path):
                continue
            try:
                with open(tf_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if ".md" not in line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        for tc in obj.get("tool_calls", []):
                            args = tc.get("args") or tc.get("arguments") or {}
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    continue
                            path = args.get("TargetFile") or args.get("Path") or ""
                            content = args.get("CodeContent") or args.get("Content") or ""
                            if path and content and path.endswith(".md"):
                                bname = os.path.basename(path.replace("\\", "/"))
                                if len(content.strip()) > len(restored_contents.get(bname, "")):
                                    restored_contents[bname] = content
                        # Also extract from view_file / read_file GENERIC tool responses before step 622!
                        if obj.get("type") == "GENERIC":
                            c_str = str(obj.get("content", ""))
                            if ".md" in c_str and ("File Path:" in c_str or "Showing lines" in c_str):
                                try:
                                    bname = None
                                    if "File Path: `file:///" in c_str:
                                        fp_part = c_str.split("File Path: `file:///")[1].split("`")[0]
                                        bname = os.path.basename(fp_part.replace("%20", " ").replace("\\", "/"))
                                    if bname:
                                        if "The following code has been modified" in c_str:
                                            body_part = c_str.split("The following code has been modified to include a line number before every line, in the format: <line_number>: <original_line>. Please note that any changes targeting the original code should remove the line number, colon, and leading space.\n")[1]
                                        elif "Showing lines" in c_str:
                                            body_part = c_str.split("\n", 4)[-1]
                                        else:
                                            body_part = ""
                                        if body_part:
                                            lines_extracted = []
                                            for bline in body_part.splitlines():
                                                if ": " in bline and bline.split(": ", 1)[0].strip().isdigit():
                                                    lines_extracted.append(bline.split(": ", 1)[1])
                                                else:
                                                    lines_extracted.append(bline)
                                            extracted_text = "\n".join(lines_extracted)
                                            if len(extracted_text.strip()) > len(restored_contents.get(bname, "")):
                                                restored_contents[bname] = extracted_text
                                except Exception:
                                    pass
                        for target_rem in ("CLAM_arxiv_2004.09666.md", "HoVerNet_arxiv_1812.06499.md", "PathMamba_MICCAI2024.md", "UNI_arxiv_2308.05814.md", "Virchow_arxiv_2309.07778.md", "deep-research-report.md", "Digital_Pathology_WSI_Resources.md", "Gigapixel_WSI_Segmentation_Survey.md", "Nghien_Cuu_AI_WSI.md"):
                            if target_rem in line and ("write" in line or "Showing lines" in line or "File Path" in line or "open(" in line):
                                print(f"[TRACE {target_rem} in {conv_id} step {obj.get('step_index')}] type={obj.get('type')} preview={str(obj.get('content', ''))[:180]!r}", flush=True)
            except Exception:
                continue

    restored_count = 0
    for root, _, files in os.walk(docs_root):
        for fname in files:
            if fname.endswith(".md"):
                fpath = os.path.join(root, fname)
                if os.path.getsize(fpath) == 0 and fname in restored_contents and len(restored_contents[fname].strip()) > 0:
                    cleaned = clean_hyphens(restored_contents[fname])
                    with open(fpath, "w", encoding="utf-8") as wf:
                        wf.write(cleaned)
                    restored_count += 1
                    print(f"[RESTORED BY BASENAME] {os.path.relpath(fpath)} ({len(cleaned):,} chars)", flush=True)

    # Also sanitize any non-empty markdown files in docs/ safely (read first, then write)
    for root, _, files in os.walk(docs_root):
        for fname in files:
            if fname.endswith(".md"):
                fpath = os.path.join(root, fname)
                if os.path.getsize(fpath) > 0:
                    with open(fpath, "r", encoding="utf-8") as rf:
                        raw = rf.read()
                    cleaned = clean_hyphens(raw)
                    if cleaned != raw:
                        with open(fpath, "w", encoding="utf-8") as wf:
                            wf.write(cleaned)
                        print(f"[SANITIZED HYPHENS] {os.path.relpath(fpath)}", flush=True)

    # List any remaining 0-byte .md files in docs/
    zero_files = []
    for root, _, files in os.walk(docs_root):
        for fname in files:
            if fname.endswith(".md"):
                fpath = os.path.join(root, fname)
                if os.path.getsize(fpath) == 0:
                    zero_files.append(os.path.relpath(fpath))
    print(f"Total restored files: {restored_count}, Remaining 0-byte .md files: {len(zero_files)}", flush=True)
    for zf in zero_files:
        print(f"  [STILL 0 BYTES] {zf}", flush=True)


if __name__ == "__main__":
    if "--restore-md" in sys.argv:
        restore_markdown_docs_from_transcript()
    else:
        main()
