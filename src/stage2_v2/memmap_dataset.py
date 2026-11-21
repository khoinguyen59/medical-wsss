"""
Stage-2 v2 Disk-Backed uint8 Soft Pseudo-Label Memmap Generator & Dataset.

Features:
- generate_bn7_soft_memmap: Generates uint8 (N, 4, 224, 224) soft probability memmap
  using Champion Stage-1 bn7 6-view MS-TTA [0.75, 1.0, 1.25] + HFlip (0% b5_2/b4_5),
  with 0 ms single-label short-circuit and soft_mode in ('raw_l1', 'softmax_t', 'sg_dap_soft').
- BCSSSoftMemmapDataset: Top-level picklable Dataset for spawn DataLoader workers;
  lazily opens np.memmap(..., mode='r') inside __getitem__, enforces strict patient-level
  anti-leakage, applies synchronized D4 geometric transforms to (img, q_soft), photometric
  jitter to img only, and down-weights pure white glass on single-class patches.
- generate_round2_ema_memmap: Runs EMA teacher 4-view D4 TTA for Arm 3 Round 2 and writes
  blended soft maps to a separate pm_soft_r2.dat (never overwrites pm_soft_s1.dat).

Pure ASCII hyphen '-' only (Invariant 12).
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from src.models.sg_dap import StainGuidedAffinityRefiner
from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
from src.stage2_v2.config import VALID_SOFT_MODES
from src.training.dataset_loader import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_FOREGROUND,
    extract_patient_id,
    parse_patch_filename,
)
from src.training.train_stage2 import prepare_stage1_input


def _discover_pngs(img_dir: Path) -> List[Path]:
    files = sorted(img_dir.glob("*.png"))
    if not files:
        files = sorted(img_dir.rglob("*.png"))
    return files


def _filter_by_patients(
    files: Sequence[Path],
    patient_ids: Optional[Sequence[str]],
    strict_patient_filter: bool,
    context_label: str,
) -> List[Path]:
    if patient_ids is None:
        return list(files)
    allowed = {str(p).upper() for p in patient_ids}
    filtered = [f for f in files if extract_patient_id(f.name).upper() in allowed]
    if strict_patient_filter and len(filtered) == 0:
        raise RuntimeError(
            f"[STAGE2 ANTI-LEAKAGE ERROR] {context_label}: 0 images matched "
            f"{len(allowed)} allowed patient IDs."
        )
    return filtered


def _load_rgb_01(path: Path, size_hw: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        if rgb.size != (size_hw[1], size_hw[0]):
            rgb = rgb.resize((size_hw[1], size_hw[0]), resample=Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1))).contiguous()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@torch.no_grad()
def _extract_bn7_6view_global_cam(
    stage1_model: WSSSTissueStage1NetCAM,
    imgs_s1: torch.Tensor,
    target_hw: Tuple[int, int] = (224, 224),
    scales: Tuple[float, ...] = (0.75, 1.0, 1.25),
) -> torch.Tensor:
    """Compute 6-view MS-TTA bn7 raw CAM upsampled to target_hw."""
    fc8_w = stage1_model.fc8.weight.float()
    b, _, h_in, w_in = imgs_s1.shape
    cam_sum = torch.zeros(
        (b, NUM_FOREGROUND, target_hw[0], target_hw[1]),
        device=imgs_s1.device,
        dtype=torch.float32,
    )
    count = 0
    for sc in scales:
        if abs(sc - 1.0) > 1e-4:
            nh = max(8, int(round(h_in * sc)))
            nw = max(8, int(round(w_in * sc)))
            x_s = F.interpolate(imgs_s1, size=(nh, nw), mode="bilinear", align_corners=False)
        else:
            x_s = imgs_s1
        for do_flip in (False, True):
            x_in = torch.flip(x_s, dims=[-1]) if do_flip else x_s
            conv6 = stage1_model.forward_backbone(x_in).float()
            cam = F.relu(F.conv2d(conv6, fc8_w))
            if do_flip:
                cam = torch.flip(cam, dims=[-1])
            cam_up = F.interpolate(cam, size=target_hw, mode="bilinear", align_corners=False)
            cam_sum = cam_sum + cam_up
            count += 1
    return cam_sum / float(max(count, 1))


def _cams_to_soft_probs(
    cam_ms: torch.Tensor,
    y_gt: torch.Tensor,
    rgb_01: torch.Tensor,
    soft_mode: str,
    soft_temp: float,
    sg_dap: Optional[StainGuidedAffinityRefiner] = None,
) -> torch.Tensor:
    """Convert raw multi-scale bn7 CAMs to label-gated soft probabilities (B, 4, H, W)."""
    b = cam_ms.size(0)
    gate = (y_gt.float() > 0.5).float().view(b, NUM_FOREGROUND, 1, 1)
    cam_gated = torch.clamp(cam_ms.float(), min=0.0) * gate
    gmin = cam_gated.view(b, -1).min(dim=-1)[0].view(b, 1, 1, 1)
    gmax = cam_gated.view(b, -1).max(dim=-1)[0].view(b, 1, 1, 1)
    cam_norm = ((cam_gated - gmin) / (gmax - gmin).clamp(min=1e-7)) * gate
    uniform_gt = gate / gate.sum(dim=1, keepdim=True).clamp(min=1.0)

    mode = str(soft_mode).strip().lower()
    if mode == "raw_l1":
        cam_sum = cam_norm.sum(dim=1, keepdim=True)
        q_soft = torch.where(
            cam_sum > 1e-6,
            cam_norm / cam_sum.clamp(min=1e-6),
            uniform_gt,
        )
    elif mode == "softmax_t":
        temp = max(float(soft_temp), 1e-4)
        logits_s = torch.where(
            gate > 0.5,
            cam_norm / temp,
            torch.full_like(cam_norm, -1e9),
        )
        q_soft = torch.softmax(logits_s, dim=1) * gate
        q_sum = q_soft.sum(dim=1, keepdim=True)
        q_soft = torch.where(
            q_sum > 1e-6,
            q_soft / q_sum.clamp(min=1e-6),
            uniform_gt,
        )
    elif mode == "sg_dap_soft":
        if sg_dap is None:
            sg_dap = StainGuidedAffinityRefiner(
                kernel_size=5, num_iters=10, use_deep_affinity=False
            ).to(cam_norm.device).eval()
        q_refined, _ = sg_dap(
            rgb_img=rgb_01.to(cam_norm.device),
            deep_feats=None,
            init_cam=cam_norm,
            return_ignore_mask=False,
            active_labels=y_gt.to(cam_norm.device),
        )
        q_refined = q_refined * gate
        q_sum = q_refined.sum(dim=1, keepdim=True)
        q_soft = torch.where(
            q_sum > 1e-6,
            q_refined / q_sum.clamp(min=1e-6),
            uniform_gt,
        )
    else:
        raise ValueError(f"Unsupported soft_mode: {soft_mode!r}. Expected one of {VALID_SOFT_MODES}.")

    return q_soft.clamp(0.0, 1.0)


@torch.no_grad()
def generate_bn7_soft_memmap(
    stage1_model: WSSSTissueStage1NetCAM,
    train_img_dir: Union[str, Path],
    output_dir: Union[str, Path],
    patient_ids: Optional[Sequence[str]] = None,
    stage1_norm_mode: str = "zero_one",
    stage1_sha256: str = "",
    soft_mode: str = "softmax_t",
    soft_temp: float = 0.25,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None,
    strict_patient_filter: bool = True,
    min_free_gb: float = 18.0,
    dat_filename: str = "pm_soft_s1.dat",
    manifest_filename: str = "pm_soft_manifest.json",
    force: bool = False,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Generate uint8 memmap (N, 4, 224, 224) + manifest JSON using bn7 6-view MS-TTA."""
    mode = str(soft_mode).strip().lower()
    if mode not in VALID_SOFT_MODES:
        raise ValueError(f"Unsupported soft_mode={soft_mode!r}. Valid: {VALID_SOFT_MODES}.")

    img_dir_p = Path(train_img_dir)
    out_dir_p = Path(output_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    dat_path = out_dir_p / dat_filename
    manifest_path = out_dir_p / manifest_filename

    all_files = _discover_pngs(img_dir_p)
    files = _filter_by_patients(all_files, patient_ids, strict_patient_filter, "generate_bn7_soft_memmap")
    if max_samples is not None:
        files = files[: int(max_samples)]
    n_patches = len(files)
    if n_patches == 0:
        raise RuntimeError(f"No PNG patches found in {img_dir_p}.")

    filenames = [f.name for f in files]
    expected_bytes = n_patches * NUM_FOREGROUND * 224 * 224

    if not force and dat_path.is_file() and manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                cached.get("stage1_sha256") == stage1_sha256
                and cached.get("soft_mode") == mode
                and abs(float(cached.get("soft_temp", -1.0)) - float(soft_temp)) < 1e-6
                and int(cached.get("num_patches", -1)) == n_patches
                and cached.get("filenames") == filenames
                and dat_path.stat().st_size == expected_bytes
            ):
                return dat_path, manifest_path, cached
        except Exception:
            pass

    if n_patches > 256:
        free_gb = shutil.disk_usage(out_dir_p).free / float(1024**3)
        if free_gb < float(min_free_gb):
            raise RuntimeError(
                f"[STAGE2 ERROR] Insufficient NVMe disk space in {out_dir_p}: "
                f"{free_gb:.2f} GB free (< {min_free_gb:.1f} GB required)."
            )

    dev = device or next(stage1_model.parameters()).device
    stage1_model = stage1_model.to(dev).eval()
    sg_dap = (
        StainGuidedAffinityRefiner(kernel_size=5, num_iters=10, use_deep_affinity=False)
        .to(dev)
        .eval()
        if mode == "sg_dap_soft"
        else None
    )

    manifest_path.unlink(missing_ok=True)
    mm = np.memmap(dat_path, dtype=np.uint8, mode="w+", shape=(n_patches, NUM_FOREGROUND, 224, 224))
    multi_indices: List[int] = []
    multi_labels: List[np.ndarray] = []
    single_count = 0

    try:
        for idx, fpath in enumerate(files):
            _, _, _, y_gt = parse_patch_filename(fpath.name)
            if int((y_gt > 0.5).sum()) == 1:
                y_norm = y_gt / max(float(y_gt.sum()), 1.0)
                one_hot = np.clip(np.rint(y_norm * 255.0), 0, 255).astype(np.uint8)
                mm[idx] = np.broadcast_to(one_hot[:, None, None], (NUM_FOREGROUND, 224, 224))
                single_count += 1
            else:
                multi_indices.append(idx)
                multi_labels.append(y_gt)

        bs = max(1, int(batch_size))
        for start in range(0, len(multi_indices), bs):
            batch_idxs = multi_indices[start : start + bs]
            batch_lbls = multi_labels[start : start + bs]
            rgb_batch = torch.stack([_load_rgb_01(files[i]) for i in batch_idxs], dim=0).to(dev)
            y_batch = torch.from_numpy(np.stack(batch_lbls, axis=0)).float().to(dev)
            s1_inp = prepare_stage1_input(
                rgb_batch, stage1_norm_mode=stage1_norm_mode, images_are_imagenet_norm=False
            )
            cam_ms = _extract_bn7_6view_global_cam(stage1_model, s1_inp, target_hw=(224, 224))
            q_soft = _cams_to_soft_probs(
                cam_ms=cam_ms,
                y_gt=y_batch,
                rgb_01=rgb_batch,
                soft_mode=mode,
                soft_temp=float(soft_temp),
                sg_dap=sg_dap,
            )
            q_u8 = np.clip(np.rint(q_soft.cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
            for local_i, mem_i in enumerate(batch_idxs):
                mm[mem_i] = q_u8[local_i]

        mm.flush()
    finally:
        del mm

    manifest: Dict[str, Any] = {
        "filenames": filenames,
        "stage1_sha256": str(stage1_sha256),
        "soft_mode": mode,
        "soft_temp": float(soft_temp),
        "num_patches": n_patches,
        "single_label_short_circuit": single_count,
        "multi_label_computed": len(multi_indices),
        "shape": [n_patches, NUM_FOREGROUND, 224, 224],
        "dtype": "uint8",
    }
    _atomic_write_json(manifest_path, manifest)
    return dat_path, manifest_path, manifest


@torch.no_grad()
def generate_round2_ema_memmap(
    ema_model: nn.Module,
    train_img_dir: Union[str, Path],
    s1_memmap_path: Union[str, Path],
    s1_manifest_path: Union[str, Path],
    output_dat_path: Union[str, Path],
    output_manifest_path: Union[str, Path],
    blend: float = 0.5,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Generate Arm 3 Round-2 self-trained memmap in a separate file (never overwrites s1)."""
    s1_dat_p = Path(s1_memmap_path).resolve()
    out_dat_p = Path(output_dat_path).resolve()
    if s1_dat_p == out_dat_p:
        raise ValueError("Round-2 memmap output path MUST differ from Stage-1 memmap path.")

    s1_man = json.loads(Path(s1_manifest_path).read_text(encoding="utf-8"))
    filenames: List[str] = list(s1_man["filenames"])
    n_patches = len(filenames)
    if n_patches == 0:
        raise RuntimeError("Empty manifest for Round-2 EMA memmap generation.")

    img_dir_p = Path(train_img_dir)
    file_lookup = {p.name: p for p in _discover_pngs(img_dir_p)}
    missing = [fn for fn in filenames if fn not in file_lookup]
    if missing:
        raise RuntimeError(f"Round-2 missing {len(missing)} training files, e.g. {missing[0]}")

    dev = device or next(ema_model.parameters()).device
    ema_model.eval()
    mean_t = torch.tensor(IMAGENET_MEAN, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, device=dev, dtype=torch.float32).view(1, 3, 1, 1)

    out_man_p = Path(output_manifest_path)
    out_dat_p.parent.mkdir(parents=True, exist_ok=True)
    out_man_p.unlink(missing_ok=True)
    mm_s1 = np.memmap(s1_dat_p, dtype=np.uint8, mode="r", shape=(n_patches, NUM_FOREGROUND, 224, 224))
    mm_r2 = np.memmap(out_dat_p, dtype=np.uint8, mode="w+", shape=(n_patches, NUM_FOREGROUND, 224, 224))

    alpha = float(blend)
    bs = max(1, int(batch_size))
    try:
        for start in range(0, n_patches, bs):
            end = min(n_patches, start + bs)
            batch_fns = filenames[start:end]
            rgb_01 = torch.stack([_load_rgb_01(file_lookup[fn]) for fn in batch_fns], dim=0).to(dev)
            imgs_norm = (rgb_01 - mean_t) / std_t
            y_gt = torch.from_numpy(
                np.stack([parse_patch_filename(fn)[3] for fn in batch_fns], axis=0)
            ).float().to(dev)
            gate = (y_gt > 0.5).float().view(-1, NUM_FOREGROUND, 1, 1)

            # 4-view D4 TTA: orig, hflip, vflip, rot90
            views = (
                imgs_norm,
                torch.flip(imgs_norm, dims=[-1]),
                torch.flip(imgs_norm, dims=[-2]),
                torch.rot90(imgs_norm, k=1, dims=[-2, -1]),
            )
            probs_sum = torch.zeros(
                (imgs_norm.size(0), NUM_FOREGROUND, 224, 224),
                device=dev,
                dtype=torch.float32,
            )
            for v_idx, x_v in enumerate(views):
                out_v = ema_model(x_v)
                lg_v = out_v["seg_logits"] if isinstance(out_v, dict) else out_v
                pr_v = torch.softmax(lg_v[:, :NUM_FOREGROUND].float(), dim=1)
                if v_idx == 1:
                    pr_v = torch.flip(pr_v, dims=[-1])
                elif v_idx == 2:
                    pr_v = torch.flip(pr_v, dims=[-2])
                elif v_idx == 3:
                    pr_v = torch.rot90(pr_v, k=3, dims=[-2, -1])
                probs_sum = probs_sum + pr_v
            p_ema = (probs_sum / 4.0) * gate
            uniform_gt = gate / gate.sum(dim=1, keepdim=True).clamp(min=1.0)
            p_sum = p_ema.sum(dim=1, keepdim=True)
            p_ema_norm = torch.where(p_sum > 1e-6, p_ema / p_sum.clamp(min=1e-6), uniform_gt)

            q_s1 = torch.from_numpy(np.array(mm_s1[start:end], dtype=np.float32) / 255.0).to(dev)
            q_r2 = (1.0 - alpha) * q_s1 + alpha * p_ema_norm
            q_r2 = q_r2 * gate
            r2_sum = q_r2.sum(dim=1, keepdim=True)
            q_r2 = torch.where(r2_sum > 1e-6, q_r2 / r2_sum.clamp(min=1e-6), uniform_gt)

            mm_r2[start:end] = np.clip(np.rint(q_r2.cpu().numpy() * 255.0), 0, 255).astype(np.uint8)

        mm_r2.flush()
    finally:
        del mm_s1
        del mm_r2

    r2_manifest = dict(s1_man)
    r2_manifest["round2_blend"] = alpha
    r2_manifest["source_memmap"] = str(s1_dat_p.name)
    out_man_p = Path(output_manifest_path)
    _atomic_write_json(out_man_p, r2_manifest)
    return out_dat_p, out_man_p, r2_manifest


class BCSSSoftMemmapDataset(Dataset):
    """Picklable lazy-memmap Dataset with synchronized D4 augmentation & anti-leakage guard."""

    def __init__(
        self,
        img_dir: Union[str, Path],
        memmap_path: Union[str, Path],
        manifest_path: Union[str, Path],
        patient_ids: Optional[Sequence[str]] = None,
        strict_patient_filter: bool = True,
        is_train: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.memmap_path = Path(memmap_path)
        self.manifest_path = Path(manifest_path)
        self.is_train = bool(is_train)
        self._memmap: Optional[np.memmap] = None

        if not self.memmap_path.is_file():
            raise FileNotFoundError(f"Soft memmap file not found: {self.memmap_path}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Soft memmap manifest not found: {self.manifest_path}")

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest_fns: List[str] = list(manifest.get("filenames", []))
        self.memmap_len = int(manifest.get("num_patches", len(manifest_fns)))
        expected_bytes = self.memmap_len * NUM_FOREGROUND * 224 * 224
        if self.memmap_path.stat().st_size < expected_bytes:
            raise RuntimeError(
                f"Memmap size {self.memmap_path.stat().st_size} < expected {expected_bytes}."
            )

        file_map = {p.name: p for p in _discover_pngs(self.img_dir)}
        aligned_items: List[Tuple[Path, int]] = []
        for idx, fn in enumerate(manifest_fns):
            if fn in file_map:
                aligned_items.append((file_map[fn], idx))

        if len(aligned_items) == 0:
            raise RuntimeError(
                f"Manifest filename alignment failed: 0 files in {self.img_dir} matched {self.manifest_path}."
            )

        if patient_ids is not None:
            allowed = {str(p).upper() for p in patient_ids}
            aligned_items = [
                (p, m_idx)
                for p, m_idx in aligned_items
                if extract_patient_id(p.name).upper() in allowed
            ]
            if strict_patient_filter and len(aligned_items) == 0:
                raise RuntimeError(
                    "[STAGE2 ANTI-LEAKAGE ERROR] BCSSSoftMemmapDataset: 0 samples matched "
                    f"{len(allowed)} allowed patient IDs."
                )

        if max_samples is not None:
            aligned_items = aligned_items[: int(max_samples)]
        self.samples: List[Tuple[Path, int]] = aligned_items

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_memmap"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._memmap = None

    def _get_memmap(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(
                self.memmap_path,
                dtype=np.uint8,
                mode="r",
                shape=(self.memmap_len, NUM_FOREGROUND, 224, 224),
            )
        return self._memmap

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, mem_idx = self.samples[idx]
        img_01 = _load_rgb_01(img_path, size_hw=(224, 224))
        mm = self._get_memmap()
        q_soft = torch.from_numpy(np.array(mm[mem_idx], dtype=np.float32) / 255.0)
        _, _, _, y_gt_np = parse_patch_filename(img_path.name)
        y_gt = torch.from_numpy(np.ascontiguousarray(y_gt_np)).float()

        if self.is_train:
            # Synchronized D4 geometric transforms on BOTH img_01 and q_soft
            k_rot = random.randint(0, 3)
            if k_rot > 0:
                img_01 = torch.rot90(img_01, k=k_rot, dims=[-2, -1])
                q_soft = torch.rot90(q_soft, k=k_rot, dims=[-2, -1])
            if random.random() < 0.5:
                img_01 = torch.flip(img_01, dims=[-1])
                q_soft = torch.flip(q_soft, dims=[-1])
            if random.random() < 0.5:
                img_01 = torch.flip(img_01, dims=[-2])
                q_soft = torch.flip(q_soft, dims=[-2])

        # Compute white-glass pixel weight from D4-aligned image BEFORE photometric jitter/blur
        if int((y_gt > 0.5).sum().item()) <= 1:
            mean_rgb = img_01.mean(dim=0)
            c_max = img_01.max(dim=0).values
            c_min = img_01.min(dim=0).values
            sat = (c_max - c_min) / c_max.clamp(min=1e-6)
            is_white_glass = (mean_rgb >= 0.90) & (sat <= 0.08)
            pixel_weight = torch.where(
                is_white_glass,
                torch.full_like(mean_rgb, 0.25),
                torch.ones_like(mean_rgb),
            )
        else:
            pixel_weight = torch.ones((img_01.shape[1], img_01.shape[2]), dtype=torch.float32)

        if self.is_train:
            # Mild photometric jitter ONLY on img_01 (never q_soft or pixel_weight)
            if random.random() < 0.7:
                img_01 = TF.adjust_brightness(img_01, random.uniform(0.90, 1.10))
                img_01 = TF.adjust_contrast(img_01, random.uniform(0.90, 1.10))
                img_01 = TF.adjust_saturation(img_01, random.uniform(0.90, 1.10))
                img_01 = img_01.clamp(0.0, 1.0)
            if random.random() < 0.15:
                img_01 = TF.gaussian_blur(img_01, kernel_size=[3, 3], sigma=[0.1, 1.0]).clamp(0.0, 1.0)

        # Re-mask and re-normalize q_soft strictly onto active GT classes
        gate = (y_gt > 0.5).float().view(NUM_FOREGROUND, 1, 1)
        q_soft = q_soft * gate
        q_sum = q_soft.sum(dim=0, keepdim=True)
        uniform_gt = gate / gate.sum(dim=0, keepdim=True).clamp(min=1.0)
        q_soft = torch.where(q_sum > 1e-6, q_soft / q_sum.clamp(min=1e-6), uniform_gt)

        mean_t = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        img_norm = (img_01 - mean_t) / std_t

        return img_norm, q_soft, y_gt, pixel_weight
