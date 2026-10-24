"""
Multi-Layer Pseudo-Mask Generation and Affinity Matting Service for SE374.R11 WSSS.

Generates high-fidelity 5-class discrete pseudo-masks (TUM, STR, LYM, NEC, BACK) for all
BCSS-WSSS training patches using:
1. Multi-Layer Class Activation Map (CAM) extraction from ResNet-38d:
   - Deep layer: bn7 (4096 channels, high-level semantic context)
   - Intermediate layer: b5_2 (1024 channels, structural tissue layout)
   - Shallow layer: b4_5 (512 channels, cellular boundary fidelity)
2. Ground-truth label gating: mathematically zeros out absent tissue types (y_c = 0).
3. Pure-PyTorch GPU-accelerated Bilateral Affinity Propagation / Random Walk.
4. Clinical Otsu tissue gating in HSV saturation space to extinguish glass background false positives.
5. Multi-directory palette PNG export to train_PM/ (PM_bn7, PM_b5_2, PM_b4_5).
6. Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src.datasets.otsu_filter import compute_otsu_tissue_mask
from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
from src.training.dataset_loader import (
    BCSS_PALETTE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_FOREGROUND,
    find_image_files,
    parse_patch_filename,
)


def bilateral_affinity_propagation_gpu(
    cams: torch.Tensor,
    images: torch.Tensor,
    radius: int = 3,
    sigma_color: float = 0.15,
    sigma_pos: float = 3.0,
    alpha: float = 0.6,
    iterations: int = 2,
) -> torch.Tensor:
    """Pure-PyTorch GPU-accelerated Bilateral Affinity Propagation.

    Propagates CAM activations locally based on color similarity and spatial proximity,
    aligning activation boundaries with histological tissue structures without CPU CRF bottlenecks.

    Args:
        cams: (B, C, H, W) normalized CAM activations on GPU.
        images: (B, 3, H, W) RGB image tensor in [0, 1] on GPU.
        radius: Neighborhood radius (window size K = 2 * radius + 1).
        sigma_color: Bandwidth for color Gaussian kernel.
        sigma_pos: Bandwidth for spatial Gaussian kernel.
        alpha: Propagation blend factor in [0, 1].
        iterations: Number of diffusion steps.

    Returns:
        refined_cams: (B, C, H, W) refined CAM tensor on GPU.
    """
    b, c, h, w = cams.shape
    k = 2 * radius + 1
    device = cams.device

    # Precompute spatial distance weights: (1, 1, K*K, 1, 1)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device, dtype=torch.float32),
        torch.arange(-radius, radius + 1, device=device, dtype=torch.float32),
        indexing="ij",
    )
    spatial_dist_sq = (x_coords ** 2 + y_coords ** 2).view(1, 1, k * k, 1, 1)
    spatial_weight = torch.exp(-spatial_dist_sq / (2.0 * sigma_pos ** 2))

    # Unfold image RGB patches: (B, 3, K, K, H, W) -> (B, 3, K*K, H, W)
    unfolded_img = F.unfold(images, kernel_size=k, padding=radius).view(b, 3, k * k, h, w)
    center_img = images.unsqueeze(2)  # (B, 3, 1, H, W)

    color_dist_sq = ((unfolded_img - center_img) ** 2).sum(dim=1, keepdim=True)  # (B, 1, K*K, H, W)
    color_weight = torch.exp(-color_dist_sq / (2.0 * sigma_color ** 2))

    # Bilateral affinity matrix: (B, 1, K*K, H, W)
    affinity = spatial_weight * color_weight
    affinity = affinity / (affinity.sum(dim=2, keepdim=True) + 1e-7)

    # Iterative diffusion
    refined_cams = cams.clone()
    for _ in range(iterations):
        # Unfold cams: (B, C, K*K, H, W)
        unfolded_cams = F.unfold(refined_cams, kernel_size=k, padding=radius).view(b, c, k * k, h, w)
        diffused = (unfolded_cams * affinity).sum(dim=2)  # (B, C, H, W)
        refined_cams = (1.0 - alpha) * cams + alpha * diffused

    return refined_cams


class MultiLayerCAMExtractor:
    """Extracts CAMs from bn7, b5_2, and b4_5 layers of WSSSTissueStage1NetCAM."""

    def __init__(
        self,
        model: WSSSTissueStage1NetCAM,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.features: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.handles: List[Any] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        def get_activation(name: str):
            def hook(module: nn.Module, inp: Any, output: Any):
                if isinstance(output, tuple):
                    output = output[0]
                self.features[name] = output
            return hook

        def get_gradient(name: str):
            def hook(module: nn.Module, grad_in: Any, grad_out: Any):
                if isinstance(grad_out, tuple):
                    grad_out = grad_out[0]
                self.gradients[name] = grad_out
            return hook

        # Register forward and backward hooks for b4_5 and b5_2
        if hasattr(self.model, "b4_5"):
            self.handles.append(self.model.b4_5.register_forward_hook(get_activation("b4_5")))
            self.handles.append(self.model.b4_5.register_full_backward_hook(get_gradient("b4_5")))

        if hasattr(self.model, "b5_2"):
            self.handles.append(self.model.b5_2.register_forward_hook(get_activation("b5_2")))
            self.handles.append(self.model.b5_2.register_full_backward_hook(get_gradient("b5_2")))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def extract_cams(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-layer CAMs for active classes.

        Args:
            images: (B, 3, H, W) normalized image tensor.
            labels: (B, 4) binary ground truth indicators.

        Returns:
            Dictionary with keys 'bn7', 'b5_2', 'b4_5', each tensor shape (B, 4, H, W).
        """
        b, _, h, w = images.shape
        images = images.to(self.device).requires_grad_(True)
        labels = labels.to(self.device)

        # Clear hooks
        self.features.clear()
        self.gradients.clear()

        # Forward pass
        conv6 = self.model.forward_backbone(images)  # bn7 feature: (B, 4096, 28, 28)
        pool_feat = self.model.pool(self.model.dropout7(conv6))
        logits = self.model.fc8(pool_feat).view(b, -1)  # (B, 4)

        # Layer 1: bn7 direct projection CAM
        cam_bn7 = F.relu(F.conv2d(conv6, self.model.fc8.weight))  # (B, 4, 28, 28)
        cam_bn7 = F.interpolate(cam_bn7, size=(h, w), mode="bilinear", align_corners=False)

        # Layers 2 & 3: b5_2 and b4_5 via Grad-CAM
        cam_b5_2 = torch.zeros((b, NUM_FOREGROUND, h, w), device=self.device)
        cam_b4_5 = torch.zeros((b, NUM_FOREGROUND, h, w), device=self.device)

        # Compute gradients for each active foreground class
        for c in range(NUM_FOREGROUND):
            if labels[:, c].sum() == 0:
                continue

            self.model.zero_grad()
            score = (logits[:, c] * labels[:, c]).sum()
            score.backward(retain_graph=True)

            if "b5_2" in self.features and "b5_2" in self.gradients:
                feat5 = self.features["b5_2"]
                grad5 = self.gradients["b5_2"]
                w5 = grad5.mean(dim=(2, 3), keepdim=True)
                cam5 = F.relu((w5 * feat5).sum(dim=1, keepdim=True))
                cam5 = F.interpolate(cam5, size=(h, w), mode="bilinear", align_corners=False)
                cam_b5_2[:, c : c + 1] = cam5

            if "b4_5" in self.features and "b4_5" in self.gradients:
                feat4 = self.features["b4_5"]
                grad4 = self.gradients["b4_5"]
                w4 = grad4.mean(dim=(2, 3), keepdim=True)
                cam4 = F.relu((w4 * feat4).sum(dim=1, keepdim=True))
                cam4 = F.interpolate(cam4, size=(h, w), mode="bilinear", align_corners=False)
                cam_b4_5[:, c : c + 1] = cam4

        # Fallback if b4_5 or b5_2 hooks were not active
        if cam_b5_2.sum() == 0:
            cam_b5_2 = cam_bn7.clone()
        if cam_b4_5.sum() == 0:
            cam_b4_5 = cam_bn7.clone()

        # Label gating: extinguish inactive classes
        gate = labels.view(b, NUM_FOREGROUND, 1, 1)
        cam_bn7 = cam_bn7 * gate
        cam_b5_2 = cam_b5_2 * gate
        cam_b4_5 = cam_b4_5 * gate

        # Normalize per-image min-max
        def normalize_cam(cam_tensor: torch.Tensor) -> torch.Tensor:
            min_val = cam_tensor.view(b, NUM_FOREGROUND, -1).min(dim=-1)[0].view(b, NUM_FOREGROUND, 1, 1)
            max_val = cam_tensor.view(b, NUM_FOREGROUND, -1).max(dim=-1)[0].view(b, NUM_FOREGROUND, 1, 1)
            denom = (max_val - min_val).clamp(min=1e-7)
            return ((cam_tensor - min_val) / denom) * gate

        return {
            "bn7": normalize_cam(cam_bn7).detach(),
            "b5_2": normalize_cam(cam_b5_2).detach(),
            "b4_5": normalize_cam(cam_b4_5).detach(),
        }


def assemble_pseudo_mask(
    cams: torch.Tensor,
    raw_images_np: np.ndarray,
    background_threshold: float = 0.20,
) -> np.ndarray:
    """Combine 4-class CAM scores with clinical Otsu tissue gating into a 5-class mask.

    Args:
        cams: (B, 4, H, W) normalized CAM tensor on CPU.
        raw_images_np: (B, H, W, 3) uint8 RGB image batch.
        background_threshold: Base sensitivity threshold for tissue vs background.

    Returns:
        discrete_masks: (B, H, W) int64 array with pixel values in {0, 1, 2, 3, 4}.
    """
    b, c, h, w = cams.shape
    cams_np = cams.numpy()  # (B, 4, H, W)
    discrete_masks = np.zeros((b, h, w), dtype=np.int64)

    for i in range(b):
        patch_img = raw_images_np[i]
        patch_cam = cams_np[i]  # (4, H, W)

        # Detect glass background using clinical Otsu in HSV space
        tissue_binary = compute_otsu_tissue_mask(patch_img)  # 1 for tissue, 0 for glass

        # Background score channel
        bg_score = np.full((1, h, w), fill_value=background_threshold, dtype=np.float32)
        bg_score[0, tissue_binary == 0] = 1.0  # Force 1.0 on glass

        # Concatenate 4 foreground CAMs + 1 background channel: (5, H, W)
        volume = np.concatenate([patch_cam, bg_score], axis=0)
        label_map = np.argmax(volume, axis=0)

        # Enforce glass as class 4
        label_map[tissue_binary == 0] = 4
        discrete_masks[i] = label_map

    return discrete_masks


def save_palette_mask(mask: np.ndarray, save_path: Union[str, Path]) -> None:
    """Save 2D discrete segmentation mask as 8-bit paletted PNG."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(mask.astype(np.uint8), mode="P")
    img.putpalette(BCSS_PALETTE)
    img.save(str(save_path), format="PNG")


def generate_pseudo_masks_pipeline(
    data_dir: Union[str, Path],
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    dry_run: bool = False,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Orchestrate multi-layer pseudo-mask generation for training dataset."""
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_path = Path(data_dir)
    train_dir = data_path / "train" if (data_path / "train").exists() else data_path

    if output_dir is None:
        pm_root = train_dir.parent / "train_PM"
    else:
        pm_root = Path(output_dir)

    pm_bn7_dir = pm_root / "PM_bn7"
    pm_b5_2_dir = pm_root / "PM_b5_2"
    pm_b4_5_dir = pm_root / "PM_b4_5"

    for d in (pm_bn7_dir, pm_b5_2_dir, pm_b4_5_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)

    extractor = MultiLayerCAMExtractor(model, device=device)

    # Scan patches
    all_files = find_image_files(train_dir)
    if dry_run:
        all_files = all_files[:4]

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    processed_count = 0
    num_batches = int(np.ceil(len(all_files) / max(batch_size, 1)))

    for batch_idx in range(num_batches):
        batch_files = all_files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        if not batch_files:
            continue

        raw_imgs = []
        tensors = []
        labels = []

        for f in batch_files:
            img = Image.open(f).convert("RGB")
            raw_imgs.append(np.array(img, dtype=np.uint8))

            t = transforms.functional.to_tensor(img)
            tensors.append(normalize(t))

            _, _, _, l_np = parse_patch_filename(f.name)
            labels.append(l_np)

        img_batch = torch.stack(tensors, dim=0).to(device)
        label_batch = torch.from_numpy(np.stack(labels, axis=0)).float().to(device)
        raw_imgs_np = np.stack(raw_imgs, axis=0)

        # Extract multi-layer CAMs
        cams_dict = extractor.extract_cams(img_batch, label_batch)

        # Pure-PyTorch GPU Affinity Matting refinement
        raw_img_tensors = torch.from_numpy(raw_imgs_np).permute(0, 3, 1, 2).float().to(device) / 255.0
        refined_bn7 = bilateral_affinity_propagation_gpu(cams_dict["bn7"], raw_img_tensors)
        refined_b5_2 = bilateral_affinity_propagation_gpu(cams_dict["b5_2"], raw_img_tensors)
        refined_b4_5 = bilateral_affinity_propagation_gpu(cams_dict["b4_5"], raw_img_tensors)

        # Assemble discrete pseudo-masks
        masks_bn7 = assemble_pseudo_mask(refined_bn7.cpu(), raw_imgs_np)
        masks_b5_2 = assemble_pseudo_mask(refined_b5_2.cpu(), raw_imgs_np)
        masks_b4_5 = assemble_pseudo_mask(refined_b4_5.cpu(), raw_imgs_np)

        # Save to disk
        for i, f in enumerate(batch_files):
            save_palette_mask(masks_bn7[i], pm_bn7_dir / f.name)
            save_palette_mask(masks_b5_2[i], pm_b5_2_dir / f.name)
            save_palette_mask(masks_b4_5[i], pm_b4_5_dir / f.name)
            processed_count += 1

    extractor.remove_hooks()

    return {
        "processed_count": processed_count,
        "pm_root": str(pm_root),
        "pm_bn7_dir": str(pm_bn7_dir),
        "pm_b5_2_dir": str(pm_b5_2_dir),
        "pm_b4_5_dir": str(pm_b4_5_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Multi-Layer Pseudo-Masks with GPU Affinity Matting")
    parser.add_argument("--data-dir", type=str, default="data/BCSS-WSSS", help="Dataset directory")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/stage1_retrained_best.pth", help="Stage 1 checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for train_PM")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Process only 4 patches for testing")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    res = generate_pseudo_masks_pipeline(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        device_str=args.device,
    )
    print("Pseudo-Mask Generation Complete.")
    print(f"Processed: {res['processed_count']} patches")
    print(f"Saved to: {res['pm_root']}")


if __name__ == "__main__":
    main()
