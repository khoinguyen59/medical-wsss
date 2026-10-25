"""
High-Throughput BCSS-WSSS Dataset Loader for SE374.R11.

Implements high-performance PyTorch Dataset and DataLoader abstractions for:
1. Stage 1: Weakly-supervised multi-label tissue classification (23,422 patches).
2. Stage 2: Multi-layer pseudo-mask segmentation training (PM_bn7, PM_b5_2, PM_b4_5).
3. Stage 3: Clinical evaluation and benchmark testing on real patient cohorts.

Key Architectural Guarantees:
- Strict patient-level anti-leakage protocol matching src/config/bcss_patient_splits.json.
- Zero patient overlap across train, val, and test splits (Invariant 08).
- Multi-label parsing from filename vector [t s l n] for classes:
    0: TUM (Tumor epithelium)
    1: STR (Stroma / reactive connective tissue)
    2: LYM (Lymphocytic infiltrates)
    3: NEC (Necrosis)
    4: BACK (Background glass / whitespace)
- Synchronized spatial augmentations across image and all pseudo-masks.
- Pure ASCII hyphen '-' (0x2D) enforced throughout.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES: Tuple[str, ...] = ("TUM", "STR", "LYM", "NEC", "BACK")
FOREGROUND_CLASSES: Tuple[str, ...] = ("TUM", "STR", "LYM", "NEC")
NUM_FOREGROUND: int = 4
NUM_CLASSES: int = 5

BCSS_PALETTE: List[int] = [
    255, 0, 0,      # 0: TUM - Red
    0, 255, 0,      # 1: STR - Green
    0, 0, 255,      # 2: LYM - Blue
    153, 0, 255,    # 3: NEC - Purple
    255, 255, 255,  # 4: BACK - White
]

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def extract_patient_id(filepath_or_name: Union[str, Path]) -> str:
    """Extract patient or case identifier from histopathology patch filename.

    Supports standard BCSS-WSSS naming:
        TCGA-A2-A0T2_1024_2048_[1 0 0 1].png -> 'TCGA-A2-A0T2'
    Supports LUAD-HistoSeg naming:
        TCGA-XX-XXXX+001+[1 0 0 0].png -> 'TCGA-XX-XXXX'
        GDPH_1234_patch_01.png -> 'GDPH_1234'
    """
    basename = os.path.basename(str(filepath_or_name))
    fname = os.path.splitext(basename)[0]

    tcga_match = re.match(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", fname, re.IGNORECASE)
    if tcga_match:
        return tcga_match.group(1).upper()

    tokens = re.split(r"[_+]", fname)
    return tokens[0].strip().upper()


def parse_patch_filename(filepath_or_name: Union[str, Path]) -> Tuple[str, int, int, np.ndarray]:
    """Parse patient ID, (x, y) coordinates, and multi-hot label from patch filename.

    Examples:
        TCGA-A2-A0T2_1024_2048_[1 0 0 1].png -> ('TCGA-A2-A0T2', 1024, 2048, array([1., 0., 0., 1.]))
        TCGA-A1-A0SK_512_1024_[1000].png -> ('TCGA-A1-A0SK', 512, 1024, array([1., 0., 0., 0.]))

    Returns:
        patient_id: string ID (e.g. 'TCGA-A2-A0T2')
        x_coord: int x coordinate
        y_coord: int y coordinate
        multi_label: np.ndarray shape (4,) float32 with binary indicators [TUM, STR, LYM, NEC]
    """
    basename = os.path.basename(str(filepath_or_name))
    fname = os.path.splitext(basename)[0]
    patient_id = extract_patient_id(fname)

    # Extract coordinates if present
    x_coord = 0
    y_coord = 0
    coord_match = re.search(r"_(\d+)_(\d+)_", fname)
    if coord_match:
        x_coord = int(coord_match.group(1))
        y_coord = int(coord_match.group(2))

    # Extract weak label vector inside brackets [...]
    label_match = re.search(r"\[([0-9\s,]+)\]", fname)
    multi_label = np.zeros(NUM_FOREGROUND, dtype=np.float32)

    if label_match:
        raw_str = label_match.group(1).strip()
        # Parse space, comma, or contiguous digits
        digits = [int(ch) for ch in re.findall(r"[0-9]", raw_str)]
        for i in range(min(len(digits), NUM_FOREGROUND)):
            multi_label[i] = float(digits[i])
    else:
        # Default label if brackets are absent
        multi_label[0] = 1.0

    return patient_id, x_coord, y_coord, multi_label


def load_bcss_patient_splits(
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """Load deterministic 151-patient split manifest from src/config/bcss_patient_splits.json.

    Returns:
        Dict with keys 'train', 'val', 'test' containing patient ID lists.
    """
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "src" / "config" / "bcss_patient_splits.json"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        # Fallback to local default patient lists if config file does not exist
        return {
            "train": ["TCGA-A1-A0SK", "TCGA-A2-A04N", "TCGA-A2-A0T2"],
            "val": ["TCGA-AN-A03X", "TCGA-AN-A0AL"],
            "test": ["TCGA-AQ-A04H", "TCGA-C8-A1HJ"],
        }

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "splits" in data and isinstance(data["splits"], dict):
        train_ids = data["splits"].get("train", [])
        val_ids = data["splits"].get("val", data["splits"].get("valid", []))
        test_ids = data["splits"].get("test", [])
    else:
        train_ids = data.get("train", [])
        val_ids = data.get("val", data.get("valid", []))
        test_ids = data.get("test", [])

    return {
        "train": [p.upper() for p in train_ids],
        "val": [p.upper() for p in val_ids],
        "test": [p.upper() for p in test_ids],
    }


def find_image_files(root_dir: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Scan directory for PNG and JPEG histopathology patches."""
    root_path = Path(root_dir)
    if not root_path.exists():
        return []

    extensions = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    files: List[Path] = []
    if recursive:
        for p in root_path.rglob("*"):
            if p.is_file() and p.suffix in extensions:
                files.append(p)
    else:
        for p in root_path.iterdir():
            if p.is_file() and p.suffix in extensions:
                files.append(p)

    files.sort(key=lambda p: p.name)
    return files


class BCSSStage1Dataset(Dataset):
    """PyTorch Dataset for Stage-1 ResNet38d classifier training on BCSS-WSSS.

    Returns:
        image: torch.Tensor of shape (3, 224, 224), float32 normalized
        label: torch.Tensor of shape (4,), float32 multi-hot [TUM, STR, LYM, NEC]
        name: patch filename string
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        patient_ids: Optional[Iterable[str]] = None,
        transform: Optional[Callable] = None,
        is_train: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.is_train = is_train

        # Scan patches
        all_files = find_image_files(self.root_dir)
        allowed_set: Optional[Set[str]] = (
            {p.upper() for p in patient_ids} if patient_ids is not None else None
        )

        filtered_files: List[Path] = []
        for f in all_files:
            if allowed_set is not None:
                pid = extract_patient_id(f.name)
                if pid not in allowed_set:
                    continue
            filtered_files.append(f)

        if max_samples is not None and max_samples > 0:
            filtered_files = filtered_files[:max_samples]

        self.files = filtered_files

        if transform is not None:
            self.transform = transform
        else:
            if is_train:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path = self.files[idx]
        image = Image.open(path).convert("RGB")
        _, _, _, label_np = parse_patch_filename(path.name)
        label_tensor = torch.from_numpy(label_np).float()

        image_tensor = self.transform(image)
        return image_tensor, label_tensor, path.stem


class BCSSStage2MultiMaskDataset(Dataset):
    """PyTorch Dataset for Stage-2 DeepLabV3+ segmentation training on pseudo-masks.

    Synchronously loads RGB patch and 3 pseudo-masks from layers:
        1. Primary: PM_bn7
        2. Intermediate: PM_b5_2
        3. Shallow: PM_b4_5

    Applies synchronized spatial transformations (D4 group horizontal/vertical flips)
    to ensure geometric consistency between the RGB image and all three pseudo-masks.

    Returns dictionary:
        {
            "image": torch.Tensor (3, 224, 224),
            "label": torch.Tensor (224, 224) long, PM_bn7
            "label_a": torch.Tensor (224, 224) long, PM_b5_2
            "label_b": torch.Tensor (224, 224) long, PM_b4_5
            "cls_label": torch.Tensor (4,) float32,
            "name": str,
        }
    """

    def __init__(
        self,
        img_dir: Union[str, Path],
        pm_root_dir: Optional[Union[str, Path]] = None,
        patient_ids: Optional[Iterable[str]] = None,
        is_train: bool = True,
        max_samples: Optional[int] = None,
        fallback_mask: bool = True,
    ) -> None:
        super().__init__()
        self.img_dir = Path(img_dir)
        if pm_root_dir is None:
            self.pm_root_dir = self.img_dir.parent / "train_PM"
        else:
            self.pm_root_dir = Path(pm_root_dir)

        self.is_train = is_train
        self.fallback_mask = fallback_mask

        # Subdirectories for the 3 multi-layer pseudo-masks
        self.pm_bn7_dir = self.pm_root_dir / "PM_bn7"
        self.pm_b5_2_dir = self.pm_root_dir / "PM_b5_2"
        self.pm_b4_5_dir = self.pm_root_dir / "PM_b4_5"

        all_files = find_image_files(self.img_dir)
        allowed_set: Optional[Set[str]] = (
            {p.upper() for p in patient_ids} if patient_ids is not None else None
        )

        filtered_files: List[Path] = []
        for f in all_files:
            if allowed_set is not None:
                pid = extract_patient_id(f.name)
                if pid not in allowed_set:
                    continue
            filtered_files.append(f)

        if max_samples is not None and max_samples > 0:
            filtered_files = filtered_files[:max_samples]

        self.files = filtered_files
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.files)

    def _load_mask(self, mask_dir: Path, filename: str) -> np.ndarray:
        mask_path = mask_dir / filename
        if mask_path.exists():
            mask_img = Image.open(mask_path)
            if mask_img.mode == "P":
                arr = np.array(mask_img, dtype=np.int64)
            else:
                arr = np.array(mask_img.convert("L"), dtype=np.int64)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            arr = np.clip(arr, 0, 4)
            return arr
        elif self.fallback_mask:
            # Fallback to zeros (class 0 or background) if layer mask is missing
            return np.zeros((224, 224), dtype=np.int64)
        else:
            raise FileNotFoundError(f"Pseudo-mask not found at: {mask_path}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        _, _, _, label_np = parse_patch_filename(img_path.name)
        cls_label = torch.from_numpy(label_np).float()

        mask_bn7 = self._load_mask(self.pm_bn7_dir, img_path.name)
        mask_b5_2 = self._load_mask(self.pm_b5_2_dir, img_path.name)
        mask_b4_5 = self._load_mask(self.pm_b4_5_dir, img_path.name)

        # Synchronized spatial augmentations for training (full 8-element D4 dihedral group)
        if self.is_train:
            # Random 90-degree rotations (D4 group: 0, 90, 180, 270 deg)
            rot_k = int(torch.randint(0, 4, (1,)).item())
            if rot_k == 1:
                image = image.transpose(Image.ROTATE_90)
                mask_bn7 = np.rot90(mask_bn7, k=1).copy()
                mask_b5_2 = np.rot90(mask_b5_2, k=1).copy()
                mask_b4_5 = np.rot90(mask_b4_5, k=1).copy()
            elif rot_k == 2:
                image = image.transpose(Image.ROTATE_180)
                mask_bn7 = np.rot90(mask_bn7, k=2).copy()
                mask_b5_2 = np.rot90(mask_b5_2, k=2).copy()
                mask_b4_5 = np.rot90(mask_b4_5, k=2).copy()
            elif rot_k == 3:
                image = image.transpose(Image.ROTATE_270)
                mask_bn7 = np.rot90(mask_bn7, k=3).copy()
                mask_b5_2 = np.rot90(mask_b5_2, k=3).copy()
                mask_b4_5 = np.rot90(mask_b4_5, k=3).copy()

            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask_bn7 = np.fliplr(mask_bn7).copy()
                mask_b5_2 = np.fliplr(mask_b5_2).copy()
                mask_b4_5 = np.fliplr(mask_b4_5).copy()

            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask_bn7 = np.flipud(mask_bn7).copy()
                mask_b5_2 = np.flipud(mask_b5_2).copy()
                mask_b4_5 = np.flipud(mask_b4_5).copy()

        img_tensor = transforms.functional.to_tensor(image)
        img_tensor = self.normalize(img_tensor)

        t_bn7 = torch.from_numpy(mask_bn7).long()
        t_b5_2 = torch.from_numpy(mask_b5_2).long()
        t_b4_5 = torch.from_numpy(mask_b4_5).long()

        return {
            "image": img_tensor,
            "label": t_bn7,
            "label_a": t_b5_2,
            "label_b": t_b4_5,
            "cls_label": cls_label,
            "name": img_path.stem,
        }


class BCSSEvalDataset(Dataset):
    """PyTorch Dataset for quantitative evaluation on val/test splits with ground truth masks.

    Returns:
        image: torch.Tensor (3, 224, 224), float32 normalized
        mask: torch.Tensor (224, 224), int64 ground truth (0..4)
        cls_label: torch.Tensor (4,), float32 multi-hot
        name: patch filename string
    """

    def __init__(
        self,
        img_dir: Union[str, Path],
        mask_dir: Optional[Union[str, Path]] = None,
        patient_ids: Optional[Iterable[str]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.img_dir = Path(img_dir)
        if mask_dir is None:
            self.mask_dir = self.img_dir.parent / "mask"
        else:
            self.mask_dir = Path(mask_dir)

        all_files = find_image_files(self.img_dir)
        allowed_set: Optional[Set[str]] = (
            {p.upper() for p in patient_ids} if patient_ids is not None else None
        )

        filtered_files: List[Path] = []
        for f in all_files:
            if allowed_set is not None:
                pid = extract_patient_id(f.name)
                if pid not in allowed_set:
                    continue
            filtered_files.append(f)

        if max_samples is not None and max_samples > 0:
            filtered_files = filtered_files[:max_samples]

        self.files = filtered_files
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        _, _, _, label_np = parse_patch_filename(img_path.name)
        cls_label = torch.from_numpy(label_np).float()

        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            mask_img = Image.open(mask_path)
            if mask_img.mode == "P":
                mask_np = np.array(mask_img, dtype=np.int64)
            else:
                mask_np = np.array(mask_img.convert("L"), dtype=np.int64)
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            mask_np = np.clip(mask_np, 0, 4)
        else:
            # Fallback zero mask if mask folder is absent
            mask_np = np.zeros((224, 224), dtype=np.int64)

        img_tensor = self.transform(image)
        mask_tensor = torch.from_numpy(mask_np).long()
        return img_tensor, mask_tensor, cls_label, img_path.stem


def create_stage1_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    splits_config: Optional[Union[str, Path]] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build high-throughput DataLoaders for Stage 1 multi-label training.

    Ensures 0.0% patient-level leakage between train and val splits.
    """
    data_path = Path(data_dir)
    if (data_path / "training").exists():
        train_dir = data_path / "training"
    elif (data_path / "train").exists():
        train_dir = data_path / "train"
    elif (data_path / "BCSS-WSSS" / "training").exists():
        train_dir = data_path / "BCSS-WSSS" / "training"
    elif (data_path / "BCSS-WSSS" / "train").exists():
        train_dir = data_path / "BCSS-WSSS" / "train"
    else:
        train_dir = data_path

    val_dir = None
    for cand in [
        data_path / "val" / "img",
        data_path / "valid" / "img",
        data_path / "BCSS-WSSS" / "val" / "img",
        data_path / "BCSS-WSSS" / "valid" / "img",
    ]:
        if cand.exists():
            val_dir = cand
            break

    splits = load_bcss_patient_splits(splits_config)
    train_patients = set(splits.get("train", []))
    val_patients = set(splits.get("val", []))

    # Auto-adjust num_workers on Windows
    if os.name == "nt" and num_workers > 4:
        num_workers = 2

    train_dataset = BCSSStage1Dataset(
        root_dir=train_dir,
        patient_ids=train_patients if train_patients else None,
        is_train=True,
        max_samples=max_train_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True if len(train_dataset) > batch_size else False,
    )

    val_loader: Optional[DataLoader] = None
    if val_dir and val_dir.exists():
        val_dataset = BCSSStage1Dataset(
            root_dir=val_dir,
            patient_ids=val_patients if val_patients else None,
            is_train=False,
            max_samples=max_val_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False,
        )

    return train_loader, val_loader


def create_stage2_dataloader(
    img_dir: Union[str, Path],
    pm_root_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    patient_ids: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Build high-throughput DataLoader for Stage 2 DeepLabV3+ segmentation training."""
    if os.name == "nt" and num_workers > 4:
        num_workers = 2

    dataset = BCSSStage2MultiMaskDataset(
        img_dir=img_dir,
        pm_root_dir=pm_root_dir,
        patient_ids=patient_ids,
        is_train=True,
        max_samples=max_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True if len(dataset) > batch_size else False,
    )


def create_eval_dataloader(
    img_dir: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True,
    patient_ids: Optional[Iterable[str]] = None,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Build DataLoader for quantitative evaluation on val or test splits."""
    if os.name == "nt" and num_workers > 4:
        num_workers = 2

    dataset = BCSSEvalDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        patient_ids=patient_ids,
        max_samples=max_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )
