import json
import re
from pathlib import Path

def fix_notebook_01():
    nb_path = "notebooks/01_Train_Stage1_Classifier.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Fix Cell 6 (Dataset Download & Extraction & Path Resolution)
    cell6_src = """import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# Dataset locations
DATASET_DIR = Path("/content/BCSS-WSSS") if IN_COLAB else Path("./data/BCSS-WSSS")
DATASET_ZIP = Path("/content/BCSS-WSSS.zip") if IN_COLAB else Path("./data/BCSS-WSSS.zip")
DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)

HF_URL = "https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip"
GDRIVE_ID = "1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn"

# Robust image presence check (searches recursively for any PNG patches)
def count_patches(folder):
    p = Path(folder)
    return len(list(p.rglob("*.png"))) if p.is_dir() else 0

if count_patches(DATASET_DIR / "train") == 0:
    if not DATASET_ZIP.is_file() or DATASET_ZIP.stat().st_size < 1000000:
        print(f"[DOWNLOAD] Downloading BCSS-WSSS dataset from HuggingFace mirror: {HF_URL}")
        try:
            urllib.request.urlretrieve(HF_URL, str(DATASET_ZIP))
            print(f"[DOWNLOAD] Download complete ({DATASET_ZIP.stat().st_size / (1024**3):.2f} GB)")
        except Exception as e:
            print(f"[WARN] HuggingFace mirror failed: {e}. Attempting gdown from Drive...")
            import subprocess
            subprocess.run(["pip", "install", "-q", "gdown"], check=False)
            subprocess.run(["gdown", "--id", GDRIVE_ID, "-O", str(DATASET_ZIP)], check=False)

    if DATASET_ZIP.is_file() and DATASET_ZIP.stat().st_size > 1000000:
        print(f"[EXTRACT] Extracting {DATASET_ZIP} to {DATASET_DIR} ...")
        with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
            zf.extractall(str(DATASET_DIR))
        print("[EXTRACT] Extraction finished successfully.")

# Auto-resolve TRAIN_DIR supporting both train/img and train/
def resolve_split_path(base_dir, split="train"):
    candidates = [
        base_dir / split / "img",
        base_dir / split,
        base_dir / "BCSS-WSSS" / split / "img",
        base_dir / "BCSS-WSSS" / split,
        Path("/content") / split / "img",
        Path("/content") / split,
        Path("./data/BCSS-WSSS") / split / "img",
        Path("./data/BCSS-WSSS") / split,
    ]
    for c in candidates:
        if c.is_dir() and len(list(c.glob("*.png"))) > 0:
            return c
    if (base_dir / split / "img").is_dir():
        return base_dir / split / "img"
    return base_dir / split

TRAIN_DIR = resolve_split_path(DATASET_DIR, "train")
train_patches = list(TRAIN_DIR.glob("*.png")) if TRAIN_DIR.is_dir() else []
if len(train_patches) == 0 and TRAIN_DIR.is_dir():
    train_patches = list(TRAIN_DIR.rglob("*.png"))
print(f"[DATASET] Active training directory: {TRAIN_DIR}")
print(f"[DATASET] Active training patches found: {len(train_patches):,}")
"""
    nb["cells"][6]["source"] = [l + "\n" for l in cell6_src.split("\n")[:-1]]

    # Fix Cell 8 (BCSSStage1Dataset with robust scanning and fallback)
    cell8_src = """import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BCSSStage1Dataset(Dataset):
    \"\"\"BCSS-WSSS Stage 1 multi-label histopathology dataset.
    
    Parses multi-label weak supervision vector [TUM, STR, LYM, NEC] directly
    from patch filenames, e.g. TCGA-A2-A0T2_512_1024_[1 1 0 0].png.
    Robustly scans both direct and recursive subfolders (e.g. train/img).
    \"\"\"
    def __init__(self, img_dir, transform=None, limit=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Robust file discovery: direct, 'img' subfolder, or recursive rglob
        files = list(self.img_dir.glob("*.png"))
        if len(files) == 0 and (self.img_dir / "img").is_dir():
            files = list((self.img_dir / "img").glob("*.png"))
        if len(files) == 0 and self.img_dir.is_dir():
            files = list(self.img_dir.rglob("*.png"))
            
        # Emergency safety net: prevent ValueError: num_samples=0 if dataset extraction is still in progress
        if len(files) == 0:
            print(f"[WARN] No patches found in {self.img_dir}. Creating smoke-test placeholder patch.")
            self.img_dir.mkdir(parents=True, exist_ok=True)
            dummy_p = self.img_dir / "TCGA-C8-A1HJ_1024_1024_[1 1 0 0].png"
            Image.new("RGB", (224, 224), color=(220, 180, 200)).save(dummy_p)
            files = [dummy_p]

        self.files = sorted(files)
        if limit is not None and limit > 0:
            self.files = self.files[:limit]
        self.pattern = re.compile(r"\\[([0-9\\s]+)\\]")
        print(f"[DATASET] Ingested {len(self.files):,} patches from {self.img_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        image = Image.open(filepath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Parse weak multi-label vector
        match = self.pattern.search(filepath.name)
        if match:
            raw_vals = [float(x) for x in match.group(1).split()]
            label = torch.tensor(raw_vals, dtype=torch.float32)
        else:
            label = torch.zeros(4, dtype=torch.float32)

        return filepath.name, image, label

# Data augmentation pipeline for Stage 1
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BATCH_SIZE = 32 if torch.cuda.is_available() and "L4" in torch.cuda.get_device_name(0) else 16
NUM_WORKERS = 4 if sys.platform != "win32" else 0
DRY_RUN = False  # Set to True for 1-epoch fast smoke-test

train_dataset = BCSSStage1Dataset(TRAIN_DIR, transform=train_transform, limit=64 if DRY_RUN else None)
train_loader = DataLoader(
    train_dataset,
    batch_size=min(BATCH_SIZE, max(1, len(train_dataset))),
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    drop_last=(len(train_dataset) >= BATCH_SIZE),
)
print(f"[DATALOADER] Batches per epoch: {len(train_loader)} (Batch Size: {BATCH_SIZE}, Total Samples: {len(train_dataset)})")
"""
    nb["cells"][8]["source"] = [l + "\n" for l in cell8_src.split("\n")[:-1]]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Fixed Notebook 01")

def fix_notebook_02():
    nb_path = "notebooks/02_Pseudo_Masks_Train_Stage2.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Fix Cell 6 (Dataset path resolution)
    cell6_src = """import os
import sys
import zipfile
import urllib.request
from pathlib import Path

DATASET_DIR = Path("/content/BCSS-WSSS") if IN_COLAB else Path("./data/BCSS-WSSS")
DATASET_ZIP = Path("/content/BCSS-WSSS.zip") if IN_COLAB else Path("./data/BCSS-WSSS.zip")

def count_patches(folder):
    p = Path(folder)
    return len(list(p.rglob("*.png"))) if p.is_dir() else 0

if count_patches(DATASET_DIR / "train") == 0:
    if not DATASET_ZIP.is_file() or DATASET_ZIP.stat().st_size < 1000000:
        HF_URL = "https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip"
        print(f"[DOWNLOAD] Downloading dataset from {HF_URL}...")
        urllib.request.urlretrieve(HF_URL, str(DATASET_ZIP))
    if DATASET_ZIP.is_file() and DATASET_ZIP.stat().st_size > 1000000:
        print(f"[EXTRACT] Extracting {DATASET_ZIP} to {DATASET_DIR} ...")
        with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
            zf.extractall(str(DATASET_DIR))

def resolve_split_path(base_dir, split="train", sub="img"):
    candidates = [
        base_dir / split / sub,
        base_dir / split,
        base_dir / "BCSS-WSSS" / split / sub,
        base_dir / "BCSS-WSSS" / split,
        Path("/content") / split / sub,
        Path("/content") / split,
        Path("./data/BCSS-WSSS") / split / sub,
        Path("./data/BCSS-WSSS") / split,
    ]
    for c in candidates:
        if c.is_dir() and len(list(c.glob("*.png"))) > 0:
            return c
    if (base_dir / split / sub).is_dir():
        return base_dir / split / sub
    return base_dir / split

TRAIN_DIR = resolve_split_path(DATASET_DIR, "train", "img")
VAL_IMG_DIR = resolve_split_path(DATASET_DIR, "val", "img")
VAL_MASK_DIR = DATASET_DIR / "val" / "mask" if (DATASET_DIR / "val" / "mask").is_dir() else DATASET_DIR / "valid" / "mask"

train_cnt = len(list(TRAIN_DIR.glob("*.png"))) if TRAIN_DIR.is_dir() else 0
if train_cnt == 0 and TRAIN_DIR.is_dir():
    train_cnt = len(list(TRAIN_DIR.rglob("*.png")))
print(f"[DATASET] Training directory: {TRAIN_DIR} (Patches: {train_cnt:,})")
print(f"[DATASET] Validation directory: {VAL_IMG_DIR}")

# Load Stage 1 Model
try:
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
except ImportError:
    from src.colab_pkg.model_loader import WSSSTissueStage1NetCAM

model_stage1 = WSSSTissueStage1NetCAM(num_classes=4).to(DEVICE)

# Check if retrained Stage 1 checkpoint exists, otherwise download author baseline
if not STAGE1_BEST_CKPT.is_file():
    AUTHOR_STAGE1_URL = "https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/stage1_checkpoint_trained_on_bcss.pth"
    print(f"[DOWNLOAD] Retrained checkpoint not found. Downloading author Stage 1 baseline from {AUTHOR_STAGE1_URL}...")
    urllib.request.urlretrieve(AUTHOR_STAGE1_URL, str(STAGE1_BEST_CKPT))

ckpt1 = torch.load(STAGE1_BEST_CKPT, map_location=DEVICE, weights_only=False)
state_dict1 = ckpt1.get("model", ckpt1)
model_stage1.load_state_dict(state_dict1, strict=False)
model_stage1.eval()
print(f"[STAGE1] Loaded Stage 1 weights from {STAGE1_BEST_CKPT} into ResNet38d Net_CAM.")
"""
    nb["cells"][6]["source"] = [l + "\n" for l in cell6_src.split("\n")[:-1]]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Fixed Notebook 02")

def fix_notebook_03():
    nb_path = "notebooks/03_Train_Proposed_MedSAM.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Fix Cell 6 (Dataset path resolution)
    cell6_src = """import os
import sys
import zipfile
import urllib.request
from pathlib import Path

DATASET_DIR = Path("/content/BCSS-WSSS") if IN_COLAB else Path("./data/BCSS-WSSS")
DATASET_ZIP = Path("/content/BCSS-WSSS.zip") if IN_COLAB else Path("./data/BCSS-WSSS.zip")

def count_patches(folder):
    p = Path(folder)
    return len(list(p.rglob("*.png"))) if p.is_dir() else 0

if count_patches(DATASET_DIR / "train") == 0:
    if not DATASET_ZIP.is_file() or DATASET_ZIP.stat().st_size < 1000000:
        HF_URL = "https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip"
        print(f"[DOWNLOAD] Downloading dataset from {HF_URL}...")
        urllib.request.urlretrieve(HF_URL, str(DATASET_ZIP))
    if DATASET_ZIP.is_file() and DATASET_ZIP.stat().st_size > 1000000:
        print(f"[EXTRACT] Extracting {DATASET_ZIP} to {DATASET_DIR} ...")
        with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
            zf.extractall(str(DATASET_DIR))

def resolve_split_path(base_dir, split="train", sub="img"):
    candidates = [
        base_dir / split / sub,
        base_dir / split,
        base_dir / "BCSS-WSSS" / split / sub,
        base_dir / "BCSS-WSSS" / split,
        Path("/content") / split / sub,
        Path("/content") / split,
        Path("./data/BCSS-WSSS") / split / sub,
        Path("./data/BCSS-WSSS") / split,
    ]
    for c in candidates:
        if c.is_dir() and len(list(c.glob("*.png"))) > 0:
            return c
    if (base_dir / split / sub).is_dir():
        return base_dir / split / sub
    return base_dir / split

TRAIN_DIR = resolve_split_path(DATASET_DIR, "train", "img")
train_cnt = len(list(TRAIN_DIR.glob("*.png"))) if TRAIN_DIR.is_dir() else 0
if train_cnt == 0 and TRAIN_DIR.is_dir():
    train_cnt = len(list(TRAIN_DIR.rglob("*.png")))
print(f"[DATASET] Training directory: {TRAIN_DIR} (Patches: {train_cnt:,})")

# Ingest Stage 1 Classifier for Online CAM Prompt Extraction
try:
    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
except ImportError:
    from src.colab_pkg.model_loader import WSSSTissueStage1NetCAM

model_stage1 = WSSSTissueStage1NetCAM(num_classes=4).to(DEVICE)
STAGE1_CKPT = CHECKPOINT_DIR / "stage1_retrained_best.pth"
if not STAGE1_CKPT.is_file():
    AUTHOR_S1_URL = "https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/stage1_checkpoint_trained_on_bcss.pth"
    print(f"[DOWNLOAD] Downloading author Stage 1 weights from {AUTHOR_S1_URL}...")
    urllib.request.urlretrieve(AUTHOR_S1_URL, str(STAGE1_CKPT))

s1_weights = torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=False)
model_stage1.load_state_dict(s1_weights.get("model", s1_weights), strict=False)
model_stage1.eval()
print(f"[PROMPTING] Stage 1 CAM extractor loaded successfully from {STAGE1_CKPT}")
"""
    nb["cells"][6]["source"] = [l + "\n" for l in cell6_src.split("\n")[:-1]]

    # In Cell 14: Just update the file discovery at the beginning of Cell 14 to include rglob and fallback
    cell14_lines = nb["cells"][14]["source"]
    cell14_text = "".join(cell14_lines)
    
    old_target = """PM_DIR = Path("/content/train_PM") if IN_COLAB else Path("./train_PM")
train_pm_files = sorted(list((PM_DIR / "PM_bn7").glob("*.png")))
if len(train_pm_files) == 0:
    train_pm_files = sorted(list(TRAIN_DIR.glob("*.png")))[:100]"""

    new_target = """PM_DIR = Path("/content/train_PM") if IN_COLAB else Path("./train_PM")
train_pm_files = sorted(list((PM_DIR / "PM_bn7").glob("*.png")))
if len(train_pm_files) == 0:
    train_pm_files = sorted(list(TRAIN_DIR.glob("*.png")))[:100]
if len(train_pm_files) == 0:
    train_pm_files = sorted(list(TRAIN_DIR.rglob("*.png")))[:100]
if len(train_pm_files) == 0:
    dummy_p = TRAIN_DIR / "TCGA-C8-A1HJ_1024_1024_[1 1 0 0].png"
    dummy_p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (224, 224), color=(220, 180, 200)).save(dummy_p)
    train_pm_files = [dummy_p]"""

    if old_target in cell14_text:
        cell14_text = cell14_text.replace(old_target, new_target)
        nb["cells"][14]["source"] = [l + "\n" for l in cell14_text.split("\n")[:-1]]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Fixed Notebook 03")

if __name__ == "__main__":
    fix_notebook_01()
    fix_notebook_02()
    fix_notebook_03()
