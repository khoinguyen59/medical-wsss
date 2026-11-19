"""
Multi-Layer Pseudo-Mask Generation and Affinity Matting Service for Histopathology WSSS.

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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root is in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.datasets.otsu_filter import apply_otsu_gating_to_prediction
from src.models.pda_generator import ProgressiveDropoutAttention
from src.models.sg_dap import StainGuidedAffinityRefiner
from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
from src.training.dataset_loader import (
    BCSS_PALETTE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_FOREGROUND,
    find_image_files,
    parse_patch_filename,
)
from src.training.train_stage2 import infer_stage1_norm_mode, prepare_stage1_input


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
    """Extracts Multi-Depth CAMs from bn7, b5_2, and b4_5 layers of WSSSTissueStage1NetCAM

    using Progressive Dropout Attention (PDA) with Prototype Cosine Gating.
    Operates in pure feedforward mode with torch.no_grad() for maximum throughput.
    """

    def __init__(
        self,
        model: WSSSTissueStage1NetCAM,
        device: torch.device,
        num_pda_steps: int = 3,
        pda_top_p: float = 0.25,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.features: Dict[str, torch.Tensor] = {}
        self.handles: List[Any] = []

        # Initialize Progressive Dropout Attention generator
        self.pda = ProgressiveDropoutAttention(
            num_steps=num_pda_steps,
            top_p=pda_top_p,
            base_dropout_rate=0.50,
            step_weight_decay=0.80,
            boundary_bleeding_control=True,
            spatial_masking=True,
            channel_masking=True,
            cumulative_mask=True,
        ).to(device)

        self._register_hooks()

    def _register_hooks(self) -> None:
        def get_activation(name: str):
            def hook(module: nn.Module, inp: Any, output: Any):
                if isinstance(output, tuple):
                    output = output[0]
                self.features[name] = output
            return hook

        # Register forward hooks on b4_5 and b5_2
        if hasattr(self.model, "b4_5"):
            self.handles.append(self.model.b4_5.register_forward_hook(get_activation("b4_5")))

        if hasattr(self.model, "b5_2"):
            self.handles.append(self.model.b5_2.register_forward_hook(get_activation("b5_2")))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def extract_cams(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-depth CAMs using Progressive Dropout Attention.

        Args:
            images: (B, 3, H, W) normalized image tensor.
            labels: (B, 4) binary ground truth indicators.

        Returns:
            Dictionary with keys 'bn7', 'b5_2', 'b4_5', each tensor shape (B, 4, H, W).
        """
        b, _, h, w = images.shape
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.features.clear()

        # Forward backbone pass
        conv6 = self.model.forward_backbone(images)  # bn7 feature: (B, 4096, H_feat, W_feat)
        feat_bn7 = conv6
        feat_b5_2 = self.features.get("b5_2", feat_bn7)
        feat_b4_5 = self.features.get("b4_5", feat_bn7)
        self.features["bn7"] = feat_bn7
        self.features["b5_2"] = feat_b5_2
        self.features["b4_5"] = feat_b4_5

        fc_weights = self.model.fc8.weight  # (4, 4096, 1, 1)

        # 1. PDA CAM for bn7 (deep semantic context, 4096 channels)
        cam_bn7 = self.pda.forward(feat_bn7, fc_weights)

        # 2. PDA CAM for b5_2 (structural tissue layout, 1024 channels)
        w_b5 = fc_weights[:, :feat_b5_2.shape[1], :, :]
        cam_b5_2 = self.pda.forward(feat_b5_2, w_b5)

        # 3. PDA CAM for b4_5 (cellular boundary fidelity, 512 channels)
        w_b4 = fc_weights[:, :feat_b4_5.shape[1], :, :]
        cam_b4_5 = self.pda.forward(feat_b4_5, w_b4)

        # Interpolate all CAMs to input image resolution (H, W)
        cam_bn7 = F.interpolate(cam_bn7, size=(h, w), mode="bilinear", align_corners=False)
        cam_b5_2 = F.interpolate(cam_b5_2, size=(h, w), mode="bilinear", align_corners=False)
        cam_b4_5 = F.interpolate(cam_b4_5, size=(h, w), mode="bilinear", align_corners=False)

        # Ground-truth label gating: mathematically extinguish inactive classes
        gate = labels.view(b, NUM_FOREGROUND, 1, 1)
        cam_bn7 = cam_bn7 * gate
        cam_b5_2 = cam_b5_2 * gate
        cam_b4_5 = cam_b4_5 * gate

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
    cams: Union[torch.Tensor, np.ndarray],
    raw_images_np: np.ndarray,
    background_threshold: float = 0.20,
) -> np.ndarray:
    """Combine 4-class CAM scores with clinical Otsu tissue gating into a 5-class mask.

    Args:
        cams: (B, 4, H, W) normalized CAM tensor or numpy array.
        raw_images_np: (B, H, W, 3) uint8 RGB image batch.
        background_threshold: Base sensitivity threshold for tissue vs background.

    Returns:
        discrete_masks: (B, H, W) int64 array with pixel values in {0, 1, 2, 3, 4}.
    """
    if isinstance(cams, torch.Tensor):
        cams_np = cams.detach().cpu().numpy()
    else:
        cams_np = cams

    b, c, h, w = cams_np.shape
    bg_channel = np.full((b, 1, h, w), fill_value=background_threshold, dtype=np.float32)
    score_volume = np.concatenate([cams_np, bg_channel], axis=1)  # (B, 5, H, W)
    initial_masks = np.argmax(score_volume, axis=1).astype(np.int64)  # (B, H, W)

    # Apply clinical Otsu gating in HSV saturation space
    gated_masks = apply_otsu_gating_to_prediction(
        pred_mask=initial_masks,
        image=raw_images_np,
        background_class_idx=4,
    )
    return gated_masks


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
    use_sg_dap: bool = False,
    stage1_norm_mode: str = "auto",
) -> Dict[str, Any]:
    """Orchestrate multi-layer pseudo-mask generation for training dataset."""
    resolved_stage1_norm = infer_stage1_norm_mode(
        None,
        checkpoint_path=checkpoint_path,
        requested_mode=stage1_norm_mode,
    )
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
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
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # nosec B614
        resolved_stage1_norm = infer_stage1_norm_mode(
            state,
            checkpoint_path=checkpoint_path,
            requested_mode=stage1_norm_mode,
        )
        state_dict = state.get("state_dict", state.get("model", state)) if isinstance(state, dict) else state
        model.load_state_dict(state_dict, strict=False)

    extractor = MultiLayerCAMExtractor(model, device=device)
    sg_dap_refiner = StainGuidedAffinityRefiner().to(device).eval() if use_sg_dap else None

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

        # Align Stage-1 input normalization with checkpoint expectations ('zero_one' vs 'imagenet')
        s1_img_batch = prepare_stage1_input(
            img_batch,
            stage1_norm_mode=resolved_stage1_norm,
            images_are_imagenet_norm=True,
        )

        # Extract multi-layer CAMs
        cams_dict = extractor.extract_cams(s1_img_batch, label_batch)

        # Pure-PyTorch GPU Affinity Matting refinement
        raw_img_tensors = torch.from_numpy(raw_imgs_np).permute(0, 3, 1, 2).float().to(device) / 255.0
        if sg_dap_refiner is not None:
            # SG-DAP mode: stain-guided affinity propagation with ignore_index=255 at uncertain tissue boundaries
            _, _, h_img, w_img = raw_img_tensors.shape

            def _apply_sg_dap(cam_tensor: torch.Tensor, deep_feats: torch.Tensor) -> np.ndarray:
                cam_up = F.interpolate(cam_tensor, size=(h_img, w_img), mode="bilinear", align_corners=False)
                cam_sum = cam_up.sum(dim=1, keepdim=True)
                cam_probs = cam_up / cam_sum.clamp(min=1e-6)
                _, sg_mask = sg_dap_refiner(
                    raw_img_tensors,
                    deep_feats,
                    cam_probs,
                    return_ignore_mask=True,
                )
                sg_np = sg_mask.cpu().numpy()
                gated_np = apply_otsu_gating_to_prediction(sg_np, image=raw_imgs_np, background_class_idx=4)
                # Preserve Otsu glass background (4); only mark uncertain tissue pixels as 255
                gated_np[(sg_np == 255) & (gated_np != 4)] = 255
                return gated_np

            masks_bn7 = _apply_sg_dap(cams_dict["bn7"], extractor.features.get("bn7", raw_img_tensors))
            masks_b5_2 = _apply_sg_dap(cams_dict["b5_2"], extractor.features.get("b5_2", raw_img_tensors))
            masks_b4_5 = _apply_sg_dap(cams_dict["b4_5"], extractor.features.get("b4_5", raw_img_tensors))
        else:
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
        "stage1_norm_mode": resolved_stage1_norm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Multi-Layer Pseudo-Masks with GPU Affinity Matting")
    parser.add_argument("--data-dir", type=str, default="data/BCSS-WSSS", help="Dataset directory")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/stage1_retrained_best.pth", help="Stage 1 checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for train_PM")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Process only 4 patches for testing")
    parser.add_argument("--use-sg-dap", action="store_true", help="Use Stain-Guided Affinity Refiner with ignore_index=255")
    parser.add_argument(
        "--stage1-norm-mode",
        type=str,
        default="auto",
        choices=["auto", "zero_one", "imagenet"],
        help="Stage-1 input normalization mode ('auto', 'zero_one', or 'imagenet')",
    )
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
        use_sg_dap=args.use_sg_dap,
        stage1_norm_mode=args.stage1_norm_mode,
    )
    print("Pseudo-Mask Generation Complete.")
    print(f"Processed: {res['processed_count']} patches")
    print(f"Saved to: {res['pm_root']}")


if __name__ == "__main__":
    main()
