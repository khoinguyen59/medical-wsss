import glob
import os
import re
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.patient_split import extract_patient_id


class BCSSDataset(Dataset):
    """BCSS-WSSS histopathology patch dataset for weakly-supervised semantic segmentation.
    
    Tissue classes (5 classes):
        0: TUM (Tumor)
        1: STR (Stroma)
        2: LYM (Lymphocyte)
        3: NEC (Necrosis)
        4: BACK (Background / Other)
    """

    CLASSES: List[str] = ["TUM", "STR", "LYM", "NEC", "BACK"]
    NUM_CLASSES: int = 5
    NUM_FOREGROUND_CLASSES: int = 4

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        assert split in ["train", "valid", "test"], f"Invalid split: {split}"
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.image_paths: List[str] = []
        self.mask_paths: List[str] = []
        self.labels: List[np.ndarray] = []
        self.patient_ids: List[str] = []

        self._load_dataset()

    def _load_dataset(self) -> None:
        if self.split == "train":
            train_dir = os.path.join(self.root_dir, "train")
            pattern = os.path.join(train_dir, "*.png")
            self.image_paths = sorted(glob.glob(pattern))

            for path in self.image_paths:
                patient_id = extract_patient_id(path)
                self.patient_ids.append(patient_id)
                label = self._extract_weak_label(path)
                self.labels.append(label)
        else:
            img_dir = os.path.join(self.root_dir, self.split, "img")
            mask_dir = os.path.join(self.root_dir, self.split, "mask")
            pattern = os.path.join(img_dir, "*.png")
            self.image_paths = sorted(glob.glob(pattern))

            for path in self.image_paths:
                patient_id = extract_patient_id(path)
                self.patient_ids.append(patient_id)
                fname = os.path.basename(path)
                mask_path = os.path.join(mask_dir, fname)
                self.mask_paths.append(mask_path)

    @staticmethod
    def _extract_weak_label(path: str) -> np.ndarray:
        """Parse multi-hot foreground label [y0 y1 y2 y3] from filename."""
        basename = os.path.basename(path)
        bracket_match = re.search(r"\[([0-9\s]+)\]", basename)
        if bracket_match:
            tokens = bracket_match.group(1).strip().split()
            if len(tokens) == 4:
                return np.array([int(t) for t in tokens], dtype=np.float32)
            elif len(tokens) == 1 and len(tokens[0]) == 4:
                return np.array([int(c) for c in tokens[0]], dtype=np.float32)
        return np.zeros(4, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            res = self.transform(img)
            if isinstance(res, torch.Tensor):
                img_tensor = res
            elif isinstance(res, np.ndarray):
                img_tensor = torch.from_numpy(res).permute(2, 0, 1) if res.ndim == 3 else torch.from_numpy(res)
            elif isinstance(res, Image.Image):
                img_np = np.array(res, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            else:
                img_tensor = res
        else:
            img_np = np.array(img, dtype=np.float32) / 255.0
            # Transpose HWC -> CHW
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        if self.split == "train":
            label_np = self.labels[index]
            label_tensor = torch.from_numpy(label_np)
            return img_tensor, label_tensor, os.path.basename(img_path)
        else:
            mask_path = self.mask_paths[index]
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask_np = np.array(mask, dtype=np.int64)
                mask_tensor = torch.from_numpy(mask_np)
            else:
                mask_tensor = torch.zeros((img_np.shape[0], img_np.shape[1]), dtype=torch.int64)
            return img_tensor, mask_tensor, os.path.basename(img_path)
