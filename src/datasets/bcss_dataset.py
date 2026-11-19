import glob
import os
import re
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.patient_split import extract_patient_id, load_patient_splits


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
        strict_masks: bool = False,
        enforce_patient_split: bool = False,
    ) -> None:
        super().__init__()
        assert split in ["train", "valid", "test", "val", "training"], f"Invalid split: {split}"
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.strict_masks = strict_masks
        self.enforce_patient_split = enforce_patient_split

        self.image_paths: List[str] = []
        self.mask_paths: List[str] = []
        self.labels: List[np.ndarray] = []
        self.patient_ids: List[str] = []

        self._load_dataset()

    def _load_dataset(self) -> None:
        is_train_split = self.split in ["train", "training"]
        if is_train_split:
            candidates = [
                os.path.join(self.root_dir, "training"),
                os.path.join(self.root_dir, "train", "img"),
                os.path.join(self.root_dir, "train"),
                os.path.join(self.root_dir, "BCSS-WSSS", "training"),
                os.path.join(self.root_dir, "BCSS-WSSS", "train", "img"),
                os.path.join(self.root_dir, "BCSS-WSSS", "train"),
                self.root_dir,
            ]
            for cand in candidates:
                if os.path.isdir(cand):
                    found = sorted(glob.glob(os.path.join(cand, "*.png")))
                    if found:
                        self.image_paths = found
                        break

            if self.enforce_patient_split:
                splits = load_patient_splits()
                allowed_pids = set(splits.get("train", []))
                if allowed_pids:
                    self.image_paths = [p for p in self.image_paths if extract_patient_id(p) in allowed_pids]

            for path in self.image_paths:
                patient_id = extract_patient_id(path)
                self.patient_ids.append(patient_id)
                label = self._extract_weak_label(path)
                self.labels.append(label)
        else:
            split_aliases = ["val", "valid"] if self.split in ["val", "valid"] else [self.split]
            mask_dir = os.path.join(self.root_dir, split_aliases[0], "mask")

            found_paths: List[str] = []
            for sp in split_aliases:
                for sub_root in [self.root_dir, os.path.join(self.root_dir, "BCSS-WSSS")]:
                    cand_img = os.path.join(sub_root, sp, "img")
                    cand_mask = os.path.join(sub_root, sp, "mask")
                    if os.path.isdir(cand_img):
                        matches = sorted(glob.glob(os.path.join(cand_img, "*.png")))
                        if matches:
                            mask_dir = cand_mask
                            found_paths = matches
                            break
                    cand_flat = os.path.join(sub_root, sp)
                    if os.path.isdir(cand_flat):
                        matches = sorted(glob.glob(os.path.join(cand_flat, "*.png")))
                        if matches:
                            mask_dir = cand_mask
                            found_paths = matches
                            break
                if found_paths:
                    break

            self.image_paths = found_paths
            if self.enforce_patient_split:
                splits = load_patient_splits()
                split_key = "val" if self.split in ["val", "valid"] else "test"
                allowed_pids = set(splits.get(split_key, []))
                if allowed_pids:
                    self.image_paths = [p for p in self.image_paths if extract_patient_id(p) in allowed_pids]
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
        bracket_match = re.search(r"\[([0-9\s,]+)\]", basename)
        if bracket_match:
            raw_tokens = bracket_match.group(1).replace(",", " ").strip().split()
            if len(raw_tokens) == 4:
                return np.array([int(t) for t in raw_tokens], dtype=np.float32)
            elif len(raw_tokens) == 1 and len(raw_tokens[0]) == 4:
                return np.array([int(c) for c in raw_tokens[0]], dtype=np.float32)
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

        if self.split in ["train", "training"]:
            label_np = self.labels[index]
            label_tensor = torch.from_numpy(label_np)
            return img_tensor, label_tensor, os.path.basename(img_path)
        else:
            mask_path = self.mask_paths[index]
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                if mask.mode == "P":
                    mask_np = np.array(mask, dtype=np.int64)
                else:
                    mask_np = np.array(mask.convert("L"), dtype=np.int64)
                if mask_np.ndim == 3:
                    mask_np = mask_np[:, :, 0]
                ignore_mask = mask_np == 255
                mask_np = np.clip(mask_np, 0, 4)
                mask_np[ignore_mask] = 255
                mask_tensor = torch.from_numpy(mask_np)
            else:
                if self.strict_masks:
                    raise FileNotFoundError(f"Missing ground-truth segmentation mask: {mask_path}")
                h_t = int(img_tensor.shape[-2]) if isinstance(img_tensor, torch.Tensor) and img_tensor.ndim >= 2 else img.height
                w_t = int(img_tensor.shape[-1]) if isinstance(img_tensor, torch.Tensor) and img_tensor.ndim >= 2 else img.width
                mask_tensor = torch.zeros((h_t, w_t), dtype=torch.int64)
            return img_tensor, mask_tensor, os.path.basename(img_path)
