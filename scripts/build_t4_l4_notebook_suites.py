import json
from pathlib import Path

def get_clean_cell_6_stage1():
    return [
        "import os\n",
        "import sys\n",
        "import zipfile\n",
        "import urllib.request\n",
        "from pathlib import Path\n",
        "\n",
        "# Dataset locations\n",
        "IN_COLAB = 'google.colab' in sys.modules if 'IN_COLAB' not in globals() else IN_COLAB\n",
        "DATASET_DIR = Path('/content/BCSS-WSSS') if IN_COLAB else Path('./data/BCSS-WSSS')\n",
        "DATASET_ZIP = Path('/content/BCSS-WSSS.zip') if IN_COLAB else Path('./data/BCSS-WSSS.zip')\n",
        "DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "HF_URL = 'https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip'\n",
        "GDRIVE_ID = '1sCK8Cqsf4yCHzzIx98vitocES2L_tkjn'\n",
        "\n",
        "# Robust image presence check (searches recursively for any PNG patches)\n",
        "def count_patches(folder):\n",
        "    p = Path(folder)\n",
        "    return len(list(p.rglob('*.png'))) if p.is_dir() else 0\n",
        "\n",
        "# Check both author 'training/' and 'train/' folders for existing patches\n",
        "total_train_patches = count_patches(DATASET_DIR / 'training') + count_patches(DATASET_DIR / 'train')\n",
        "if total_train_patches < 100:\n",
        "    if not DATASET_ZIP.is_file() or DATASET_ZIP.stat().st_size < 1000000:\n",
        "        print(f'[DOWNLOAD] Downloading BCSS-WSSS dataset from HuggingFace mirror: {HF_URL}')\n",
        "        try:\n",
        "            urllib.request.urlretrieve(HF_URL, str(DATASET_ZIP))\n",
        "            print(f'[DOWNLOAD] Download complete ({DATASET_ZIP.stat().st_size / (1024**3):.2f} GB)')\n",
        "        except Exception as e:\n",
        "            print(f'[WARN] HuggingFace mirror failed: {e}. Attempting gdown from Drive...')\n",
        "            import subprocess\n",
        "            subprocess.run(['pip', 'install', '-q', 'gdown'], check=False)\n",
        "            subprocess.run(['gdown', '--id', GDRIVE_ID, '-O', str(DATASET_ZIP)], check=False)\n",
        "\n",
        "    if DATASET_ZIP.is_file() and DATASET_ZIP.stat().st_size > 1000000:\n",
        "        print(f'[EXTRACT] Extracting {DATASET_ZIP} to {DATASET_DIR} ...')\n",
        "        with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:\n",
        "            zf.extractall(str(DATASET_DIR))\n",
        "        print('[EXTRACT] Extraction finished successfully.')\n",
        "\n",
        "# Auto-resolve TRAIN_DIR supporting both author 'training/' and 'train/' folder conventions\n",
        "def resolve_split_path(base_dir, split='train'):\n",
        "    split_aliases = ['training', 'train'] if split in ('train', 'training') else [split]\n",
        "    candidates = []\n",
        "    for alias in split_aliases:\n",
        "        candidates.extend([\n",
        "            base_dir / alias,\n",
        "            base_dir / alias / 'img',\n",
        "            base_dir / 'BCSS-WSSS' / alias,\n",
        "            base_dir / 'BCSS-WSSS' / alias / 'img',\n",
        "            Path('/content/BCSS-WSSS') / alias,\n",
        "            Path('/content/BCSS-WSSS') / alias / 'img',\n",
        "            Path('/content') / alias,\n",
        "            Path('/content') / alias / 'img',\n",
        "            Path('./data/BCSS-WSSS') / alias,\n",
        "            Path('./data') / alias,\n",
        "        ])\n",
        "    for c in candidates:\n",
        "        if c.is_dir() and len(list(c.glob('*.png'))) > 0:\n",
        "            return c\n",
        "    for c in candidates:\n",
        "        if c.is_dir() and len(list(c.rglob('*.png'))) > 0:\n",
        "            return c\n",
        "    return base_dir / ('training' if split in ('train', 'training') else split)\n",
        "\n",
        "TRAIN_DIR = resolve_split_path(DATASET_DIR, 'train')\n",
        "train_patches = list(TRAIN_DIR.glob('*.png')) if TRAIN_DIR.is_dir() else []\n",
        "if len(train_patches) == 0 and TRAIN_DIR.is_dir():\n",
        "    train_patches = list(TRAIN_DIR.rglob('*.png'))\n",
        "print(f'[DATASET] Active training directory: {TRAIN_DIR}')\n",
        "print(f'[DATASET] Active training patches found: {len(train_patches):,}')\n",
    ]


def get_clean_cell_8_stage1(batch_size=16, num_workers=2):
    return [
        "import re\n",
        "from PIL import Image\n",
        "import torch\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "from torchvision import transforms\n",
        "\n",
        "class BCSSStage1Dataset(Dataset):\n",
        "    \"\"\"BCSS-WSSS Stage 1 multi-label histopathology dataset.\n",
        "    \n",
        "    Parses multi-label weak supervision vector [TUM, STR, LYM, NEC] directly\n",
        "    from patch filenames, e.g. TCGA-A2-A0T2_512_1024_[1 1 0 0].png or [1000].png.\n",
        "    Robustly scans both direct, 'training', 'train', and recursive subfolders.\n",
        "    \"\"\"\n",
        "    def __init__(self, img_dir, transform=None, limit=None):\n",
        "        self.img_dir = Path(img_dir)\n",
        "        self.transform = transform\n",
        "        \n",
        "        # Robust file discovery: direct, 'img' subfolder, or recursive rglob\n",
        "        files = list(self.img_dir.glob('*.png'))\n",
        "        if len(files) == 0 and (self.img_dir / 'img').is_dir():\n",
        "            files = list((self.img_dir / 'img').glob('*.png'))\n",
        "        if len(files) == 0 and self.img_dir.is_dir():\n",
        "            files = list(self.img_dir.rglob('*.png'))\n",
        "            \n",
        "        # Emergency safety net: prevent ValueError: num_samples=0 if dataset extraction is still in progress\n",
        "        if len(files) == 0:\n",
        "            print(f'[WARN] No patches found in {self.img_dir}. Creating smoke-test placeholder patch.')\n",
        "            self.img_dir.mkdir(parents=True, exist_ok=True)\n",
        "            dummy_p = self.img_dir / 'TCGA-C8-A1HJ_1024_1024_[1 1 0 0].png'\n",
        "            Image.new('RGB', (224, 224), color=(220, 180, 200)).save(dummy_p)\n",
        "            files = [dummy_p]\n",
        "\n",
        "        self.files = sorted(files)\n",
        "        if limit is not None and limit > 0:\n",
        "            self.files = self.files[:limit]\n",
        "        self.pattern = re.compile(r'\\[([0-9\\s]+)\\]')\n",
        "        print(f'[DATASET] Ingested {len(self.files):,} patches from {self.img_dir}')\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.files)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        filepath = self.files[idx]\n",
        "        image = Image.open(filepath).convert('RGB')\n",
        "        if self.transform is not None:\n",
        "            image = self.transform(image)\n",
        "\n",
        "        # Parse weak multi-label vector: handles both '[1 1 0 0]' and '[1000]'\n",
        "        match = self.pattern.search(filepath.name)\n",
        "        if match:\n",
        "            raw_str = match.group(1).strip()\n",
        "            digits = [float(c) for c in raw_str if c.isdigit()]\n",
        "            if len(digits) == 4:\n",
        "                label = torch.tensor(digits, dtype=torch.float32)\n",
        "            else:\n",
        "                tokens = raw_str.split()\n",
        "                if len(tokens) == 4:\n",
        "                    label = torch.tensor([float(x) for x in tokens], dtype=torch.float32)\n",
        "                else:\n",
        "                    label = torch.zeros(4, dtype=torch.float32)\n",
        "        else:\n",
        "            label = torch.zeros(4, dtype=torch.float32)\n",
        "\n",
        "        return filepath.name, image, label\n",
        "\n",
        "# Data augmentation pipeline for Stage 1\n",
        "train_transform = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.RandomHorizontalFlip(p=0.5),\n",
        "    transforms.RandomVerticalFlip(p=0.5),\n",
        "    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n",
        "])\n",
        "\n",
        f"BATCH_SIZE = {batch_size}\n",
        f"NUM_WORKERS = {num_workers}\n",
        "DRY_RUN = False  # Set to True for 1-epoch fast smoke-test\n",
        "\n",
        "train_dataset = BCSSStage1Dataset(TRAIN_DIR, transform=train_transform, limit=64 if DRY_RUN else None)\n",
        "train_loader = DataLoader(\n",
        "    train_dataset,\n",
        "    batch_size=min(BATCH_SIZE, max(1, len(train_dataset))),\n",
        "    shuffle=True,\n",
        "    num_workers=NUM_WORKERS,\n",
        "    pin_memory=torch.cuda.is_available(),\n",
        "    drop_last=(len(train_dataset) >= BATCH_SIZE),\n",
        ")\n",
        "print(f'[DATALOADER] Batches per epoch: {len(train_loader)} (Batch Size: {BATCH_SIZE}, Total Samples: {len(train_dataset)})')\n",
    ]


def get_clean_cell_6_stage2():
    return [
        "import os\n",
        "import sys\n",
        "import zipfile\n",
        "import urllib.request\n",
        "from pathlib import Path\n",
        "\n",
        "IN_COLAB = 'google.colab' in sys.modules if 'IN_COLAB' not in globals() else IN_COLAB\n",
        "DATASET_DIR = Path('/content/BCSS-WSSS') if IN_COLAB else Path('./data/BCSS-WSSS')\n",
        "DATASET_ZIP = Path('/content/BCSS-WSSS.zip') if IN_COLAB else Path('./data/BCSS-WSSS.zip')\n",
        "\n",
        "def count_patches(folder):\n",
        "    p = Path(folder)\n",
        "    return len(list(p.rglob('*.png'))) if p.is_dir() else 0\n",
        "\n",
        "total_train_patches = count_patches(DATASET_DIR / 'training') + count_patches(DATASET_DIR / 'train')\n",
        "if total_train_patches < 100:\n",
        "    if not DATASET_ZIP.is_file() or DATASET_ZIP.stat().st_size < 1000000:\n",
        "        HF_URL = 'https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip'\n",
        "        print(f'[DOWNLOAD] Downloading dataset from {HF_URL}...')\n",
        "        urllib.request.urlretrieve(HF_URL, str(DATASET_ZIP))\n",
        "    if DATASET_ZIP.is_file() and DATASET_ZIP.stat().st_size > 1000000:\n",
        "        print(f'[EXTRACT] Extracting {DATASET_ZIP} to {DATASET_DIR} ...')\n",
        "        with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:\n",
        "            zf.extractall(str(DATASET_DIR))\n",
        "\n",
        "def resolve_split_path(base_dir, split='train', sub=None):\n",
        "    if split in ('train', 'training'):\n",
        "        split_aliases = ['training', 'train']\n",
        "    elif split in ('val', 'valid', 'validation'):\n",
        "        split_aliases = ['val', 'valid', 'validation']\n",
        "    elif split in ('test', 'testing'):\n",
        "        split_aliases = ['test', 'testing']\n",
        "    else:\n",
        "        split_aliases = [split]\n",
        "\n",
        "    for alias in split_aliases:\n",
        "        candidates = []\n",
        "        if sub:\n",
        "            candidates.extend([\n",
        "                base_dir / alias / sub,\n",
        "                base_dir / 'BCSS-WSSS' / alias / sub,\n",
        "                Path('/content/BCSS-WSSS') / alias / sub,\n",
        "                Path('/content') / alias / sub,\n",
        "                Path('./data/BCSS-WSSS') / alias / sub,\n",
        "            ])\n",
        "        candidates.extend([\n",
        "            base_dir / alias,\n",
        "            base_dir / 'BCSS-WSSS' / alias,\n",
        "            Path('/content/BCSS-WSSS') / alias,\n",
        "            Path('/content') / alias,\n",
        "            Path('./data/BCSS-WSSS') / alias,\n",
        "        ])\n",
        "        for c in candidates:\n",
        "            if c.is_dir() and len(list(c.glob('*.png'))) > 0:\n",
        "                return c\n",
        "        for c in candidates:\n",
        "            if c.is_dir() and len(list(c.rglob('*.png'))) > 0:\n",
        "                return c\n",
        "    return base_dir / (split_aliases[0] / sub if sub else split_aliases[0])\n",
        "\n",
        "TRAIN_DIR = resolve_split_path(DATASET_DIR, 'train')\n",
        "VAL_IMG_DIR = resolve_split_path(DATASET_DIR, 'val', 'img')\n",
        "VAL_MASK_DIR = DATASET_DIR / 'val' / 'mask' if (DATASET_DIR / 'val' / 'mask').is_dir() else (DATASET_DIR / 'valid' / 'mask' if (DATASET_DIR / 'valid' / 'mask').is_dir() else resolve_split_path(DATASET_DIR, 'val', 'mask'))\n",
        "\n",
        "train_cnt = len(list(TRAIN_DIR.glob('*.png'))) if TRAIN_DIR.is_dir() else 0\n",
        "if train_cnt == 0 and TRAIN_DIR.is_dir():\n",
        "    train_cnt = len(list(TRAIN_DIR.rglob('*.png')))\n",
        "print(f'[DATASET] Training directory: {TRAIN_DIR} (Patches: {train_cnt:,})')\n",
        "print(f'[DATASET] Validation directory: {VAL_IMG_DIR}')\n",
        "\n",
        "# Load Stage 1 Model\n",
        "try:\n",
        "    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM\n",
        "except ImportError:\n",
        "    from src.colab_pkg.model_loader import WSSSTissueStage1NetCAM\n",
        "\n",
        "model_stage1 = WSSSTissueStage1NetCAM(num_classes=4).to(DEVICE)\n",
        "\n",
        "# Check if retrained Stage 1 checkpoint exists, otherwise download author baseline\n",
        "if not STAGE1_BEST_CKPT.is_file():\n",
        "    AUTHOR_STAGE1_URL = 'https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/stage1_checkpoint_trained_on_bcss.pth'\n",
        "    print(f'[DOWNLOAD] Retrained checkpoint not found. Downloading author Stage 1 baseline from {AUTHOR_STAGE1_URL}...')\n",
        "    urllib.request.urlretrieve(AUTHOR_STAGE1_URL, str(STAGE1_BEST_CKPT))\n",
        "\n",
        "ckpt1 = torch.load(STAGE1_BEST_CKPT, map_location=DEVICE, weights_only=False)\n",
        "state_dict1 = ckpt1.get('model', ckpt1)\n",
        "model_stage1.load_state_dict(state_dict1, strict=False)\n",
        "model_stage1.eval()\n",
        "print(f'[STAGE1] Loaded Stage 1 weights from {STAGE1_BEST_CKPT} into ResNet38d Net_CAM.')\n",
    ]


def get_clean_cell_6_stage3():
    return [
        "import os\n",
        "import sys\n",
        "import zipfile\n",
        "import urllib.request\n",
        "from pathlib import Path\n",
        "\n",
        "IN_COLAB = 'google.colab' in sys.modules if 'IN_COLAB' not in globals() else IN_COLAB\n",
        "DATASET_DIR = Path('/content/BCSS-WSSS') if IN_COLAB else Path('./data/BCSS-WSSS')\n",
        "DATASET_ZIP = Path('/content/BCSS-WSSS.zip') if IN_COLAB else Path('./data/BCSS-WSSS.zip')\n",
        "\n",
        "def count_patches(folder):\n",
        "    p = Path(folder)\n",
        "    return len(list(p.rglob('*.png'))) if p.is_dir() else 0\n",
        "\n",
        "total_train_patches = count_patches(DATASET_DIR / 'training') + count_patches(DATASET_DIR / 'train')\n",
        "if total_train_patches < 100:\n",
        "    if not DATASET_ZIP.is_file() or DATASET_ZIP.stat().st_size < 1000000:\n",
        "        HF_URL = 'https://huggingface.co/datasets/nttduc/wsss-pathology/resolve/main/BCSS-WSSS.zip'\n",
        "        print(f'[DOWNLOAD] Downloading dataset from {HF_URL}...')\n",
        "        urllib.request.urlretrieve(HF_URL, str(DATASET_ZIP))\n",
        "    if DATASET_ZIP.is_file() and DATASET_ZIP.stat().st_size > 1000000:\n",
        "        print(f'[EXTRACT] Extracting {DATASET_ZIP} to {DATASET_DIR} ...')\n",
        "        with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:\n",
        "            zf.extractall(str(DATASET_DIR))\n",
        "\n",
        "def resolve_split_path(base_dir, split='train', sub=None):\n",
        "    if split in ('train', 'training'):\n",
        "        split_aliases = ['training', 'train']\n",
        "    elif split in ('val', 'valid', 'validation'):\n",
        "        split_aliases = ['val', 'valid', 'validation']\n",
        "    elif split in ('test', 'testing'):\n",
        "        split_aliases = ['test', 'testing']\n",
        "    else:\n",
        "        split_aliases = [split]\n",
        "\n",
        "    for alias in split_aliases:\n",
        "        candidates = []\n",
        "        if sub:\n",
        "            candidates.extend([\n",
        "                base_dir / alias / sub,\n",
        "                base_dir / 'BCSS-WSSS' / alias / sub,\n",
        "                Path('/content/BCSS-WSSS') / alias / sub,\n",
        "                Path('/content') / alias / sub,\n",
        "                Path('./data/BCSS-WSSS') / alias / sub,\n",
        "            ])\n",
        "        candidates.extend([\n",
        "            base_dir / alias,\n",
        "            base_dir / 'BCSS-WSSS' / alias,\n",
        "            Path('/content/BCSS-WSSS') / alias,\n",
        "            Path('/content') / alias,\n",
        "            Path('./data/BCSS-WSSS') / alias,\n",
        "        ])\n",
        "        for c in candidates:\n",
        "            if c.is_dir() and len(list(c.glob('*.png'))) > 0:\n",
        "                return c\n",
        "        for c in candidates:\n",
        "            if c.is_dir() and len(list(c.rglob('*.png'))) > 0:\n",
        "                return c\n",
        "    return base_dir / (split_aliases[0] / sub if sub else split_aliases[0])\n",
        "\n",
        "TRAIN_DIR = resolve_split_path(DATASET_DIR, 'train')\n",
        "train_cnt = len(list(TRAIN_DIR.glob('*.png'))) if TRAIN_DIR.is_dir() else 0\n",
        "if train_cnt == 0 and TRAIN_DIR.is_dir():\n",
        "    train_cnt = len(list(TRAIN_DIR.rglob('*.png')))\n",
        "print(f'[DATASET] Training directory: {TRAIN_DIR} (Patches: {train_cnt:,})')\n",
        "\n",
        "# Ingest Stage 1 Classifier for Online CAM Prompt Extraction\n",
        "try:\n",
        "    from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM\n",
        "except ImportError:\n",
        "    from src.colab_pkg.model_loader import WSSSTissueStage1NetCAM\n",
        "\n",
        "model_stage1 = WSSSTissueStage1NetCAM(num_classes=4).to(DEVICE)\n",
        "STAGE1_CKPT = CHECKPOINT_DIR / 'stage1_retrained_best.pth'\n",
        "if not STAGE1_CKPT.is_file():\n",
        "    AUTHOR_S1_URL = 'https://huggingface.co/longcris04/pseduo_mask_wsss/resolve/main/stage1_checkpoint_trained_on_bcss.pth'\n",
        "    print(f'[DOWNLOAD] Downloading author Stage 1 weights from {AUTHOR_S1_URL}...')\n",
        "    urllib.request.urlretrieve(AUTHOR_S1_URL, str(STAGE1_CKPT))\n",
        "\n",
        "s1_weights = torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=False)\n",
        "model_stage1.load_state_dict(s1_weights.get('model', s1_weights), strict=False)\n",
        "model_stage1.eval()\n",
        "print(f'[PROMPTING] Stage 1 CAM extractor loaded successfully from {STAGE1_CKPT}')\n",
    ]


BASE_DIR = Path("notebooks")

def build_suites():
    # Update base notebooks with robust dataset resolution and label parsing
    nb01 = json.load(open(BASE_DIR / "01_Train_Stage1_Classifier.ipynb", encoding="utf-8"))
    nb01["cells"][6]["source"] = get_clean_cell_6_stage1()
    nb01["cells"][8]["source"] = get_clean_cell_8_stage1(batch_size=32, num_workers=4)
    with open(BASE_DIR / "01_Train_Stage1_Classifier.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb01, f, indent=1)

    nb02 = json.load(open(BASE_DIR / "02_Pseudo_Masks_Train_Stage2.ipynb", encoding="utf-8"))
    nb02["cells"][6]["source"] = get_clean_cell_6_stage2()
    with open(BASE_DIR / "02_Pseudo_Masks_Train_Stage2.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb02, f, indent=1)

    nb03 = json.load(open(BASE_DIR / "03_Train_Proposed_MedSAM.ipynb", encoding="utf-8"))
    nb03["cells"][6]["source"] = get_clean_cell_6_stage3()
    with open(BASE_DIR / "03_Train_Proposed_MedSAM.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb03, f, indent=1)

    specs = [
        ("01_Train_Stage1_Classifier.ipynb", "01"),
        ("02_Pseudo_Masks_Train_Stage2.ipynb", "02"),
        ("03_Train_Proposed_MedSAM.ipynb", "03"),
        ("04_Exhaustive_Benchmark_Survival.ipynb", "04"),
    ]

    for base_name, prefix in specs:
        base_path = BASE_DIR / base_name
        with open(base_path, "r", encoding="utf-8") as f:
            base_nb = json.load(f)

        # =============================================================
        # 1. BUILD T4 DEDICATED NOTEBOOK
        # =============================================================
        t4_nb = json.loads(json.dumps(base_nb))
        t4_name = base_name.replace(".ipynb", "_T4.ipynb")
        t4_path = BASE_DIR / t4_name

        # Update metadata for T4
        t4_nb["metadata"]["accelerator"] = "GPU"
        t4_nb["metadata"]["colab"] = {"gpuType": "T4", "provenance": []}

        # Update badge and title in Cell 0
        c0_text = "".join(t4_nb["cells"][0]["source"])
        c0_text = c0_text.replace(
            "01_Train_Stage1_Classifier.ipynb", "01_Train_Stage1_Classifier_T4.ipynb"
        ).replace(
            "02_Pseudo_Masks_Train_Stage2.ipynb", "02_Pseudo_Masks_Train_Stage2_T4.ipynb"
        ).replace(
            "03_Train_Proposed_MedSAM.ipynb", "03_Train_Proposed_MedSAM_T4.ipynb"
        ).replace(
            "04_Exhaustive_Benchmark_Survival.ipynb", "04_Exhaustive_Benchmark_Survival_T4.ipynb"
        )
        c0_text = c0_text.replace("Recommended Hardware:** NVIDIA L4 GPU", "Dedicated Hardware:** NVIDIA Tesla T4 GPU (16GB VRAM, Turing)")
        c0_text = c0_text.replace("Precision:** Mixed Precision BF16 (L4) or FP16 (T4)", "Precision:** Mixed Precision FP16 (Native Turing Tensor Cores)")
        t4_nb["cells"][0]["source"] = [l + "\n" for l in c0_text.split("\n")[:-1]]

        # Notebook 01 Specific T4 Tuning
        if prefix == "01":
            # Cell 4: Seed, DEVICE, and FP16 for T4
            t4_nb["cells"][4]["source"] = [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import random\n",
                "import subprocess\n",
                "import zipfile\n",
                "import numpy as np\n",
                "import torch\n",
                "\n",
                "# Search candidates for codebase root\n",
                "# Pure GitHub Codebase Discovery\n",
                "CANDIDATES = [\n",
                "    Path(\"/content/medical_wsss_repo\"),\n",
                "    Path(\"/content/medical-wsss\"),\n",
                "    Path.cwd(),\n",
                "    Path.cwd().parent,\n",
                "]\n",
                "\n",
                "CODE_ROOT = next((str(p.resolve()) for p in CANDIDATES if (p / \"src\" / \"models\").is_dir()), None)\n",
                "if CODE_ROOT is None:\n",
                "    print(\"[GIT] Cloning official repository https://github.com/khoinguyen59/medical-wsss.git ...\")\n",
                "    subprocess.run([\"git\", \"clone\", \"--depth\", \"1\", \"https://github.com/khoinguyen59/medical-wsss.git\", \"/content/medical_wsss_repo\"], check=False)\n",
                "    if Path(\"/content/medical_wsss_repo/src/models\").is_dir():\n",
                "        CODE_ROOT = str(Path(\"/content/medical_wsss_repo\").resolve())\n",
                "    else:\n",
                "        CODE_ROOT = str(Path.cwd().resolve())\n",
                "\n",
                "if CODE_ROOT in sys.path:\n",
                "    sys.path.remove(CODE_ROOT)\n",
                "sys.path.insert(0, CODE_ROOT)\n",
                "\n",
                "# Set reproducible seed\n",
                "SEED = 42\n",
                "random.seed(SEED)\n",
                "np.random.seed(SEED)\n",
                "torch.manual_seed(SEED)\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.manual_seed_all(SEED)\n",
                "    torch.backends.cudnn.deterministic = True\n",
                "    torch.backends.cudnn.benchmark = False\n",
                "\n",
                "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "AMP_DTYPE = torch.float16  # Native Turing Tensor Core FP16 acceleration for Tesla T4\n",
                "print(f\"[READY] Code root: {CODE_ROOT}\")\n",
                "print(f\"[READY] Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\")\n",
                "print(f\"[READY] Mixed Precision AMP Dtype: {AMP_DTYPE} (Tesla T4 Optimized)\")\n",
            ]

            # Cell 6: Robust path resolution
            t4_nb["cells"][6]["source"] = get_clean_cell_6_stage1()

            # Cell 8: T4 Batch Size 16, num_workers 2
            t4_nb["cells"][8]["source"] = get_clean_cell_8_stage1(batch_size=16, num_workers=2)

        # Notebook 02 Specific T4 Tuning
        elif prefix == "02":
            # Cell 4: Seed, DEVICE, and FP16 for T4
            t4_nb["cells"][4]["source"] = [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import random\n",
                "import subprocess\n",
                "import zipfile\n",
                "import numpy as np\n",
                "import torch\n",
                "\n",
                "# Pure GitHub Codebase Discovery\n",
                "CANDIDATES = [\n",
                "    Path(\"/content/medical_wsss_repo\"),\n",
                "    Path(\"/content/medical-wsss\"),\n",
                "    Path.cwd(),\n",
                "    Path.cwd().parent,\n",
                "]\n",
                "\n",
                "CODE_ROOT = next((str(p.resolve()) for p in CANDIDATES if (p / \"src\" / \"models\").is_dir()), None)\n",
                "if CODE_ROOT is None:\n",
                "    print(\"[GIT] Cloning official repository https://github.com/khoinguyen59/medical-wsss.git ...\")\n",
                "    subprocess.run([\"git\", \"clone\", \"--depth\", \"1\", \"https://github.com/khoinguyen59/medical-wsss.git\", \"/content/medical_wsss_repo\"], check=False)\n",
                "    if Path(\"/content/medical_wsss_repo/src/models\").is_dir():\n",
                "        CODE_ROOT = str(Path(\"/content/medical_wsss_repo\").resolve())\n",
                "    else:\n",
                "        CODE_ROOT = str(Path.cwd().resolve())\n",
                "\n",
                "if CODE_ROOT in sys.path:\n",
                "    sys.path.remove(CODE_ROOT)\n",
                "sys.path.insert(0, CODE_ROOT)\n",
                "\n",
                "SEED = 42\n",
                "random.seed(SEED)\n",
                "np.random.seed(SEED)\n",
                "torch.manual_seed(SEED)\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.manual_seed_all(SEED)\n",
                "    torch.backends.cudnn.deterministic = True\n",
                "\n",
                "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "AMP_DTYPE = torch.float16  # Native Turing Tensor Core FP16\n",
                "print(f\"[READY] Code root: {CODE_ROOT}\")\n",
                "print(f\"[READY] Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) [Tesla T4 Tier]\")\n",
            ]

            # Cell 6: Robust path resolution
            t4_nb["cells"][6]["source"] = get_clean_cell_6_stage2()

            # Cell 10 (Stage 2 training loader): batch size 12
            c10_text = "".join(t4_nb["cells"][10]["source"])
            c10_text = c10_text.replace("batch_size=16", "batch_size=12  # Memory-safe for DeepLabV3+ ResNet-101 on T4")
            t4_nb["cells"][10]["source"] = [l + "\n" for l in c10_text.split("\n")[:-1]]

        # Notebook 03 Specific T4 Tuning
        elif prefix == "03":
            # Cell 4: Seed, DEVICE, and FP16 for T4
            t4_nb["cells"][4]["source"] = [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import random\n",
                "import subprocess\n",
                "import zipfile\n",
                "import numpy as np\n",
                "import torch\n",
                "\n",
                "# Pure GitHub Codebase Discovery\n",
                "CANDIDATES = [\n",
                "    Path(\"/content/medical_wsss_repo\"),\n",
                "    Path(\"/content/medical-wsss\"),\n",
                "    Path.cwd(),\n",
                "    Path.cwd().parent,\n",
                "]\n",
                "\n",
                "CODE_ROOT = next((str(p.resolve()) for p in CANDIDATES if (p / \"src\" / \"models\").is_dir()), None)\n",
                "if CODE_ROOT is None:\n",
                "    print(\"[GIT] Cloning official repository https://github.com/khoinguyen59/medical-wsss.git ...\")\n",
                "    subprocess.run([\"git\", \"clone\", \"--depth\", \"1\", \"https://github.com/khoinguyen59/medical-wsss.git\", \"/content/medical_wsss_repo\"], check=False)\n",
                "    if Path(\"/content/medical_wsss_repo/src/models\").is_dir():\n",
                "        CODE_ROOT = str(Path(\"/content/medical_wsss_repo\").resolve())\n",
                "    else:\n",
                "        CODE_ROOT = str(Path.cwd().resolve())\n",
                "\n",
                "if CODE_ROOT in sys.path:\n",
                "    sys.path.remove(CODE_ROOT)\n",
                "sys.path.insert(0, CODE_ROOT)\n",
                "\n",
                "SEED = 42\n",
                "random.seed(SEED)\n",
                "np.random.seed(SEED)\n",
                "torch.manual_seed(SEED)\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.manual_seed_all(SEED)\n",
                "    torch.backends.cudnn.deterministic = True\n",
                "\n",
                "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "AMP_DTYPE = torch.float16  # Native Turing Tensor Core FP16\n",
                "print(f\"[READY] Code root: {CODE_ROOT}\")\n",
                "print(f\"[READY] Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) [Tesla T4 Tier]\")\n",
            ]

            # Cell 6: Robust path resolution
            t4_nb["cells"][6]["source"] = get_clean_cell_6_stage3()

            # Cell 12: batch size 8 for MedSAM on T4
            c12_text = "".join(t4_nb["cells"][12]["source"])
            c12_text = c12_text.replace("batch_size=16", "batch_size=8  # Memory-safe for MedSAM ViT on T4 16GB")
            t4_nb["cells"][12]["source"] = [l + "\n" for l in c12_text.split("\n")[:-1]]

        with open(t4_path, "w", encoding="utf-8") as f:
            json.dump(t4_nb, f, indent=1)
        print(f"Created dedicated T4 notebook: {t4_path}")

        # =============================================================
        # 2. BUILD L4 DEDICATED NOTEBOOK
        # =============================================================
        l4_nb = json.loads(json.dumps(base_nb))
        l4_name = base_name.replace(".ipynb", "_L4.ipynb")
        l4_path = BASE_DIR / l4_name

        # Update metadata for L4
        l4_nb["metadata"]["accelerator"] = "GPU"
        l4_nb["metadata"]["colab"] = {"gpuType": "L4", "provenance": []}

        # Update badge and title in Cell 0
        c0_text = "".join(l4_nb["cells"][0]["source"])
        c0_text = c0_text.replace(
            "01_Train_Stage1_Classifier.ipynb", "01_Train_Stage1_Classifier_L4.ipynb"
        ).replace(
            "02_Pseudo_Masks_Train_Stage2.ipynb", "02_Pseudo_Masks_Train_Stage2_L4.ipynb"
        ).replace(
            "03_Train_Proposed_MedSAM.ipynb", "03_Train_Proposed_MedSAM_L4.ipynb"
        ).replace(
            "04_Exhaustive_Benchmark_Survival.ipynb", "04_Exhaustive_Benchmark_Survival_L4.ipynb"
        )
        c0_text = c0_text.replace("Recommended Hardware:** NVIDIA L4 GPU", "Dedicated Hardware:** NVIDIA L4 GPU (24GB VRAM, Ada Lovelace, Colab Pro)")
        c0_text = c0_text.replace("Precision:** Mixed Precision BF16 (L4) or FP16 (T4)", "Precision:** Mixed Precision BF16 (Native Ada Lovelace Tensor Cores)")
        l4_nb["cells"][0]["source"] = [l + "\n" for l in c0_text.split("\n")[:-1]]

        # Notebook 01 Specific L4 Tuning
        if prefix == "01":
            # Cell 4: Seed, DEVICE, and BF16 for L4
            l4_nb["cells"][4]["source"] = [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import random\n",
                "import subprocess\n",
                "import zipfile\n",
                "import numpy as np\n",
                "import torch\n",
                "\n",
                "# Search candidates for codebase root\n",
                "# Pure GitHub Codebase Discovery\n",
                "CANDIDATES = [\n",
                "    Path(\"/content/medical_wsss_repo\"),\n",
                "    Path(\"/content/medical-wsss\"),\n",
                "    Path.cwd(),\n",
                "    Path.cwd().parent,\n",
                "]\n",
                "\n",
                "CODE_ROOT = next((str(p.resolve()) for p in CANDIDATES if (p / \"src\" / \"models\").is_dir()), None)\n",
                "if CODE_ROOT is None:\n",
                "    print(\"[GIT] Cloning official repository https://github.com/khoinguyen59/medical-wsss.git ...\")\n",
                "    subprocess.run([\"git\", \"clone\", \"--depth\", \"1\", \"https://github.com/khoinguyen59/medical-wsss.git\", \"/content/medical_wsss_repo\"], check=False)\n",
                "    if Path(\"/content/medical_wsss_repo/src/models\").is_dir():\n",
                "        CODE_ROOT = str(Path(\"/content/medical_wsss_repo\").resolve())\n",
                "    else:\n",
                "        CODE_ROOT = str(Path.cwd().resolve())\n",
                "\n",
                "if CODE_ROOT in sys.path:\n",
                "    sys.path.remove(CODE_ROOT)\n",
                "sys.path.insert(0, CODE_ROOT)\n",
                "\n",
                "# Set reproducible seed\n",
                "SEED = 42\n",
                "random.seed(SEED)\n",
                "np.random.seed(SEED)\n",
                "torch.manual_seed(SEED)\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.manual_seed_all(SEED)\n",
                "    torch.backends.cudnn.deterministic = True\n",
                "    torch.backends.cudnn.benchmark = False\n",
                "\n",
                "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "AMP_DTYPE = torch.bfloat16  # Native Ada Lovelace Tensor Core BF16 acceleration for NVIDIA L4\n",
                "print(f\"[READY] Code root: {CODE_ROOT}\")\n",
                "print(f\"[READY] Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\")\n",
                "print(f\"[READY] Mixed Precision AMP Dtype: {AMP_DTYPE} (Optimized for NVIDIA L4 24GB VRAM)\")\n",
            ]

            # Cell 6: Robust path resolution
            l4_nb["cells"][6]["source"] = get_clean_cell_6_stage1()

            # Cell 8: L4 Batch Size 32, num_workers 4
            l4_nb["cells"][8]["source"] = get_clean_cell_8_stage1(batch_size=32, num_workers=4)

        # Notebook 02 Specific L4 Tuning
        elif prefix == "02":
            # Cell 4: Seed, DEVICE, and BF16 for L4
            l4_nb["cells"][4]["source"] = [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import random\n",
                "import subprocess\n",
                "import zipfile\n",
                "import numpy as np\n",
                "import torch\n",
                "\n",
                "# Pure GitHub Codebase Discovery\n",
                "CANDIDATES = [\n",
                "    Path(\"/content/medical_wsss_repo\"),\n",
                "    Path(\"/content/medical-wsss\"),\n",
                "    Path.cwd(),\n",
                "    Path.cwd().parent,\n",
                "]\n",
                "\n",
                "CODE_ROOT = next((str(p.resolve()) for p in CANDIDATES if (p / \"src\" / \"models\").is_dir()), None)\n",
                "if CODE_ROOT is None:\n",
                "    print(\"[GIT] Cloning official repository https://github.com/khoinguyen59/medical-wsss.git ...\")\n",
                "    subprocess.run([\"git\", \"clone\", \"--depth\", \"1\", \"https://github.com/khoinguyen59/medical-wsss.git\", \"/content/medical_wsss_repo\"], check=False)\n",
                "    if Path(\"/content/medical_wsss_repo/src/models\").is_dir():\n",
                "        CODE_ROOT = str(Path(\"/content/medical_wsss_repo\").resolve())\n",
                "    else:\n",
                "        CODE_ROOT = str(Path.cwd().resolve())\n",
                "\n",
                "if CODE_ROOT in sys.path:\n",
                "    sys.path.remove(CODE_ROOT)\n",
                "sys.path.insert(0, CODE_ROOT)\n",
                "\n",
                "SEED = 42\n",
                "random.seed(SEED)\n",
                "np.random.seed(SEED)\n",
                "torch.manual_seed(SEED)\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.manual_seed_all(SEED)\n",
                "    torch.backends.cudnn.deterministic = True\n",
                "\n",
                "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "AMP_DTYPE = torch.bfloat16  # Native Ada Lovelace Tensor Core BF16\n",
                "print(f\"[READY] Code root: {CODE_ROOT}\")\n",
                "print(f\"[READY] Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) [NVIDIA L4 Tier]\")\n",
            ]

            # Cell 6: Robust path resolution
            l4_nb["cells"][6]["source"] = get_clean_cell_6_stage2()

            # Cell 10 (Stage 2 training loader): batch size 24
            c10_text = "".join(l4_nb["cells"][10]["source"])
            c10_text = c10_text.replace("batch_size=16", "batch_size=24  # High throughput for DeepLabV3+ ResNet-101 on L4 24GB")
            l4_nb["cells"][10]["source"] = [l + "\n" for l in c10_text.split("\n")[:-1]]

        # Notebook 03 Specific L4 Tuning
        elif prefix == "03":
            # Cell 4: Seed, DEVICE, and BF16 for L4
            l4_nb["cells"][4]["source"] = [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import random\n",
                "import subprocess\n",
                "import zipfile\n",
                "import numpy as np\n",
                "import torch\n",
                "\n",
                "# Pure GitHub Codebase Discovery\n",
                "CANDIDATES = [\n",
                "    Path(\"/content/medical_wsss_repo\"),\n",
                "    Path(\"/content/medical-wsss\"),\n",
                "    Path.cwd(),\n",
                "    Path.cwd().parent,\n",
                "]\n",
                "\n",
                "CODE_ROOT = next((str(p.resolve()) for p in CANDIDATES if (p / \"src\" / \"models\").is_dir()), None)\n",
                "if CODE_ROOT is None:\n",
                "    print(\"[GIT] Cloning official repository https://github.com/khoinguyen59/medical-wsss.git ...\")\n",
                "    subprocess.run([\"git\", \"clone\", \"--depth\", \"1\", \"https://github.com/khoinguyen59/medical-wsss.git\", \"/content/medical_wsss_repo\"], check=False)\n",
                "    if Path(\"/content/medical_wsss_repo/src/models\").is_dir():\n",
                "        CODE_ROOT = str(Path(\"/content/medical_wsss_repo\").resolve())\n",
                "    else:\n",
                "        CODE_ROOT = str(Path.cwd().resolve())\n",
                "\n",
                "if CODE_ROOT in sys.path:\n",
                "    sys.path.remove(CODE_ROOT)\n",
                "sys.path.insert(0, CODE_ROOT)\n",
                "\n",
                "SEED = 42\n",
                "random.seed(SEED)\n",
                "np.random.seed(SEED)\n",
                "torch.manual_seed(SEED)\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.manual_seed_all(SEED)\n",
                "    torch.backends.cudnn.deterministic = True\n",
                "\n",
                "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "AMP_DTYPE = torch.bfloat16  # Native Ada Lovelace Tensor Core BF16\n",
                "print(f\"[READY] Code root: {CODE_ROOT}\")\n",
                "print(f\"[READY] Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) [NVIDIA L4 Tier]\")\n",
            ]

            # Cell 6: Robust path resolution
            l4_nb["cells"][6]["source"] = get_clean_cell_6_stage3()

            # Cell 12: batch size 16 for MedSAM on L4
            c12_text = "".join(l4_nb["cells"][12]["source"])
            c12_text = c12_text.replace("batch_size=16", "batch_size=16  # Full batch for MedSAM ViT on L4 24GB")
            l4_nb["cells"][12]["source"] = [l + "\n" for l in c12_text.split("\n")[:-1]]

        with open(l4_path, "w", encoding="utf-8") as f:
            json.dump(l4_nb, f, indent=1)
        print(f"Created dedicated L4 notebook: {l4_path}")

if __name__ == "__main__":
    build_suites()
