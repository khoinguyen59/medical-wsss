import json
import glob
import re
from pathlib import Path

def refactor_notebook(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    # 1. Update Cell 0 header markdown
    cell0_text = "".join(nb["cells"][0]["source"])
    cell0_text = re.sub(
        r"\* \*\*Checkpoints:\*\*.*",
        "* **Checkpoints:** Direct high-speed NVMe SSD storage on Colab VM disk (`/content/checkpoints/`) with instant one-click download (Zero Google Drive dependency).",
        cell0_text
    )
    nb["cells"][0]["source"] = [line + "\n" for line in cell0_text.split("\n")[:-1]] if cell0_text.endswith("\n") else [line + "\n" for line in cell0_text.split("\n")]
    
    # 2. Iterate through all cells
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        
        # Replace Drive paths with Colab VM local disk paths
        if "/content/drive/MyDrive/SE374_WSSS_Checkpoints" in src:
            src = src.replace("/content/drive/MyDrive/SE374_WSSS_Checkpoints", "/content/checkpoints")
        if "/content/drive/MyDrive" in src:
            src = src.replace("/content/drive/MyDrive", "/content")
            
        # Replace step titles
        src = src.replace("Mount Google Drive & Configure Checkpoint Storage (Dual-Path Resilience)", "Configure High-Speed Colab Local Storage (/content/checkpoints/) - Zero Drive Required")
        src = src.replace("Mount Google Drive & Auto-Discover Codebase (Dual-Path Resilience)", "Auto-Clone from GitHub & Configure Colab Local Storage - Zero Drive Required")
        src = src.replace("Automatically mounts Google Drive if executing in Google Colab, or falls back to local directory `./checkpoints/`.", "Configures high-speed local NVMe SSD storage at `/content/checkpoints/` directly on the Colab VM disk.")
        src = src.replace("Mounts Google Drive when running in Colab, or falls back to local `./checkpoints/` directory.", "Configures high-speed local NVMe SSD storage at `/content/checkpoints/` directly on the Colab VM disk.")
        src = src.replace("Configures checkpoint storage in `/content/drive/MyDrive/SE374_WSSS_Checkpoints/` or local fallback `./checkpoints/`.", "Configures high-speed local NVMe SSD storage at `/content/checkpoints/` directly on the Colab VM disk.")
        src = src.replace("Automatically mounts Google Drive if available and resolves `src` directory without manual path configuration.", "Clones official repository from GitHub and configures local NVMe SSD storage at `/content/checkpoints/`.")
        src = src.replace("from `/content/drive/MyDrive/SE374_WSSS_Checkpoints/stage2_latest.pth`", "from `/content/checkpoints/stage2_latest.pth`")
        src = src.replace("from `/content/drive/MyDrive/SE374_WSSS_Checkpoints/medsam_latest.pth`", "from `/content/checkpoints/medsam_latest.pth`")
        src = src.replace("to Google Drive or local workspace", "directly on the Colab VM disk")
        src = src.replace("with Google Drive fallback", "with high-speed mirror")
        
        # Replace Python drive.mount logic with pure Colab local disk
        old_drive_block = """if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        CHECKPOINT_DIR = Path("/content/drive/MyDrive/SE374_WSSS_Checkpoints")
        print("[DRIVE] Google Drive mounted successfully at /content/drive")
    except Exception as e:
        print(f"[WARN] Google Drive mount failed: {e}. Using /content/checkpoints")
        CHECKPOINT_DIR = Path("/content/checkpoints")
else:
    CHECKPOINT_DIR = Path("./checkpoints")"""
        
        new_drive_block = """# Pure Colab Local NVMe Storage (Pure GitHub + Colab, Zero Drive Popup)
if IN_COLAB:
    CHECKPOINT_DIR = Path("/content/checkpoints")
    print("[STORAGE] Using high-speed Colab VM local disk at /content/checkpoints")
else:
    CHECKPOINT_DIR = Path("./checkpoints")"""

        if "drive.mount(" in src:
            # General replacement for drive.mount block
            src = re.sub(
                r'if IN_COLAB:\s+try:\s+from google\.colab import drive.*?(else:\s+CHECKPOINT_DIR = Path\("\./checkpoints"\))',
                new_drive_block,
                src,
                flags=re.DOTALL
            )
            
        # Clean candidates to prioritize GitHub repo in /content/
        if 'MYDRIVE = Path("/content/drive/MyDrive")' in src or 'MYDRIVE = Path("/content")' in src:
            src = re.sub(
                r'MYDRIVE = Path\(.*?\)\s+CANDIDATES = \[.*?\]',
                '# Pure GitHub Codebase Discovery\nCANDIDATES = [\n    Path("/content/medical_wsss_repo"),\n    Path("/content/medical-wsss"),\n    Path.cwd(),\n    Path.cwd().parent,\n]',
                src,
                flags=re.DOTALL
            )
            
        cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] if src.endswith("\n") else [line + "\n" for line in src.split("\n")]
        
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Refactored successfully:", nb_path)

if __name__ == "__main__":
    for nb in sorted(glob.glob("notebooks/0*.ipynb")):
        refactor_notebook(nb)
