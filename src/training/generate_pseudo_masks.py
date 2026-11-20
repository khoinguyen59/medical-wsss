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
import warnings
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
from src.training.train_stage2 import (
    infer_stage1_norm_mode,
    load_stage1_checkpoint_into_model,
    prepare_stage1_input,
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
    """Extracts Multi-Depth CAMs from bn7, b5_2, and b4_5 layers of WSSSTissueStage1NetCAM.

    - Deep layer bn7 (4096 channels): Progressive Dropout Attention (PDA) using fc8 weights.
    - Intermediate/shallow layers b5_2 (1024 channels) and b4_5 (512 channels): authentic
      Grad-CAM (class-score gradients global-average-pooled as channel weights + ReLU),
      matching repos/WSSS-Tissue/2_generate_PM.py and tool/gradcam.py.
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
        self.features.clear()

    def extract_cams(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-depth CAMs using PDA (bn7) and authentic Grad-CAM (b5_2, b4_5).

        Args:
            images: (B, 3, H, W) normalized image tensor.
            labels: (B, 4) binary ground truth indicators.

        Returns:
            Dictionary with keys 'bn7', 'b5_2', 'b4_5', each tensor shape (B, 4, H, W).
        """
        b, _, h, w = images.shape
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.model.eval()
        self.features.clear()

        # Authentic Grad-CAM requires autograd graph from b4_5 and b5_2 to class logits
        with torch.enable_grad():
            images_in = images.detach().requires_grad_(True)
            conv6 = self.model.forward_backbone(images_in)  # bn7 feature: (B, 4096, H_feat, W_feat)
            feat_b5_2 = self.features.get("b5_2")
            feat_b4_5 = self.features.get("b4_5")

            # Pre-sigmoid class logits (matches repos/WSSS-Tissue/tool/gradcam.py ModelOutputs)
            x_pool = self.model.pool(conv6.float())
            fc_bias = self.model.fc8.bias.float() if self.model.fc8.bias is not None else None
            logits = F.conv2d(x_pool, self.model.fc8.weight.float(), bias=fc_bias).view(b, -1)

            cam_b5_2: Optional[torch.Tensor] = (
                torch.zeros((b, NUM_FOREGROUND, feat_b5_2.shape[2], feat_b5_2.shape[3]), device=self.device, dtype=torch.float32)
                if feat_b5_2 is not None
                else None
            )
            cam_b4_5: Optional[torch.Tensor] = (
                torch.zeros((b, NUM_FOREGROUND, feat_b4_5.shape[2], feat_b4_5.shape[3]), device=self.device, dtype=torch.float32)
                if feat_b4_5 is not None
                else None
            )

            active_classes = [
                c for c in range(NUM_FOREGROUND) if bool((labels[:, c] > 0.5).any().item())
            ]
            grad_targets: List[torch.Tensor] = []
            target_names: List[str] = []
            if feat_b5_2 is not None and feat_b5_2.requires_grad:
                grad_targets.append(feat_b5_2)
                target_names.append("b5_2")
            if feat_b4_5 is not None and feat_b4_5.requires_grad:
                grad_targets.append(feat_b4_5)
                target_names.append("b4_5")

            if grad_targets and active_classes:
                for idx_c, c in enumerate(active_classes):
                    retain = idx_c < (len(active_classes) - 1)
                    active_mask_c = (labels[:, c] > 0.5).float()
                    score_c = (logits[:, c] * active_mask_c).sum()
                    grads = torch.autograd.grad(
                        score_c,
                        grad_targets,
                        retain_graph=retain,
                        create_graph=False,
                        allow_unused=True,
                    )
                    for t_name, g_val in zip(target_names, grads):
                        if g_val is None:
                            continue
                        w_c = g_val.detach().float().mean(dim=(2, 3), keepdim=True)
                        if t_name == "b5_2" and cam_b5_2 is not None and feat_b5_2 is not None:
                            cam_b5_2[:, c] = F.relu((w_c * feat_b5_2.detach().float()).sum(dim=1))
                        elif t_name == "b4_5" and cam_b4_5 is not None and feat_b4_5 is not None:
                            cam_b4_5[:, c] = F.relu((w_c * feat_b4_5.detach().float()).sum(dim=1))

            self.model.zero_grad(set_to_none=True)

        # Detach stored intermediate features cleanly to release autograd graph
        feat_bn7_det = conv6.detach()
        feat_b5_2_det = feat_b5_2.detach() if feat_b5_2 is not None else feat_bn7_det
        feat_b4_5_det = feat_b4_5.detach() if feat_b4_5 is not None else feat_bn7_det
        self.features["bn7"] = feat_bn7_det
        self.features["b5_2"] = feat_b5_2_det
        self.features["b4_5"] = feat_b4_5_det

        with torch.no_grad():
            fc_weights = self.model.fc8.weight.detach()  # (4, 4096, 1, 1)

            # 1. PDA CAM for bn7 (deep semantic context, 4096 channels matching fc8)
            cam_bn7 = self.pda.forward(feat_bn7_det, fc_weights)

            # 2 & 3. Grad-CAM for b5_2 and b4_5 (fallback to bn7 only if model lacks hooks)
            if cam_b5_2 is None:
                cam_b5_2 = cam_bn7.clone()
            if cam_b4_5 is None:
                cam_b4_5 = cam_bn7.clone()

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

            def normalize_gradcam(cam_tensor: torch.Tensor) -> torch.Tensor:
                # Global active-volume max normalization preserves relative cross-class Grad-CAM confidence
                # (matching repos/WSSS-Tissue/tool/infer_fun.py:117-118 without stretching noisy weak channels)
                gated = cam_tensor * gate
                max_val = gated.view(b, -1).max(dim=-1)[0].view(b, 1, 1, 1).clamp(min=1e-7)
                return (gated / max_val) * gate

            return {
                "bn7": normalize_cam(cam_bn7).detach(),
                "b5_2": normalize_gradcam(cam_b5_2).detach(),
                "b4_5": normalize_gradcam(cam_b4_5).detach(),
            }


def assemble_pseudo_mask(
    cams: Union[torch.Tensor, np.ndarray],
    raw_images_np: np.ndarray,
    background_threshold: float = 0.20,
    active_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> np.ndarray:
    """Combine 4-class CAM scores with clinical Otsu tissue gating into a 5-class mask.

    Args:
        cams: (B, 4, H, W) normalized CAM tensor or numpy array.
        raw_images_np: (B, H, W, 3) uint8 RGB image batch.
        background_threshold: Base sensitivity threshold for tissue vs background.
        active_labels: Optional (B, 4) binary image-level label indicator. When provided,
            absent classes (label <= 0.5) are masked to -1.0 before argmax so zero-CAM
            ties never leak Class 0 (TUM) onto non-tumor patches.

    Returns:
        discrete_masks: (B, H, W) int64 array with pixel values in {0, 1, 2, 3, 4}.
    """
    if isinstance(cams, torch.Tensor):
        cams_np = cams.detach().cpu().numpy().astype(np.float32, copy=True)
    else:
        cams_np = np.array(cams, dtype=np.float32, copy=True)

    b, c, h, w = cams_np.shape
    bg_fill = float(background_threshold)
    if active_labels is not None:
        if isinstance(active_labels, torch.Tensor):
            lbl_np = active_labels.detach().cpu().numpy()
        else:
            lbl_np = np.asarray(active_labels)
        gate_mask = (lbl_np > 0.5).reshape(b, c, 1, 1)
        has_any_active = np.any(gate_mask, axis=1, keepdims=True)
        cams_np = np.where(gate_mask | (~has_any_active), cams_np, -1.0)
        if bg_fill <= 0.0:
            bg_fill = -0.5

    bg_channel = np.full((b, 1, h, w), fill_value=bg_fill, dtype=np.float32)
    score_volume = np.concatenate([cams_np, bg_channel], axis=1)  # (B, 5, H, W)
    initial_masks = np.argmax(score_volume, axis=1).astype(np.int64)  # (B, H, W)

    # Apply clinical Otsu gating in HSV saturation space
    gated_masks = apply_otsu_gating_to_prediction(
        pred_mask=initial_masks,
        image=raw_images_np,
        background_class_idx=4,
    )
    return gated_masks


_FULL_256_BCSS_PALETTE: List[int] = list(BCSS_PALETTE) + [0] * (768 - len(BCSS_PALETTE))
_FULL_256_BCSS_PALETTE[255 * 3 : 255 * 3 + 3] = [128, 128, 128]  # 255: ignore_index - Gray


def save_palette_mask(mask: np.ndarray, save_path: Union[str, Path]) -> None:
    """Save 2D discrete segmentation mask as 8-bit paletted PNG preserving ignore_index=255.

    Uses a full 256-entry (768-byte) palette so Pillow writes an 8-bit PLTE PNG rather than
    silently quantizing to 4-bit PNG (which would corrupt ignore_index=255 into index 15).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(mask.astype(np.uint8), mode="P")
    img.putpalette(_FULL_256_BCSS_PALETTE)
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
    allow_random_init: Optional[bool] = None,
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

    # Initialize model with strict Stage-1 checkpoint verification (strips module./model., checks >=100 layers)
    model = WSSSTissueStage1NetCAM(n_class=NUM_FOREGROUND)
    effective_allow_random = dry_run if allow_random_init is None else bool(allow_random_init)
    if checkpoint_path is not None:
        resolved_stage1_norm, _ = load_stage1_checkpoint_into_model(
            model,
            checkpoint_path=checkpoint_path,
            stage1_norm_mode=stage1_norm_mode,
            min_matched_layers=100,
            allow_random_init=bool(allow_random_init is True),
        )
    elif effective_allow_random:
        warnings.warn(
            "[STAGE 1 WARN] Generating pseudo-masks without checkpoint_path because allow_random_init/dry_run is enabled.",
            UserWarning,
            stacklevel=2,
        )
    else:
        raise FileNotFoundError(
            "[STAGE 1 ERROR] checkpoint_path is required for pseudo-mask generation "
            "(pass allow_random_init=True only for synthetic tests)."
        )

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

        # Extract multi-layer CAMs (requires gradients for b5_2/b4_5 Grad-CAM)
        cams_dict = extractor.extract_cams(s1_img_batch, label_batch)

        # Pure-PyTorch GPU Affinity Matting refinement (no_grad around refinement only)
        with torch.no_grad():
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
                        active_labels=label_batch,
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

                # Assemble discrete pseudo-masks (background_threshold=0.0 matches BCSS-WSSS paper parity;
                # passing active_labels eliminates zero-tie Class-0 leaks on absent classes)
                masks_bn7 = assemble_pseudo_mask(
                    refined_bn7.cpu(), raw_imgs_np, background_threshold=0.0, active_labels=label_batch
                )
                masks_b5_2 = assemble_pseudo_mask(
                    refined_b5_2.cpu(), raw_imgs_np, background_threshold=0.0, active_labels=label_batch
                )
                masks_b4_5 = assemble_pseudo_mask(
                    refined_b4_5.cpu(), raw_imgs_np, background_threshold=0.0, active_labels=label_batch
                )

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
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/stage1_checkpoint_trained_on_bcss.pth",
        help="Stage 1 checkpoint (default: verified author baseline)",
    )
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
    parser.add_argument("--allow-random-init", action="store_true", help="Allow random init for synthetic smoke tests")
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
        allow_random_init=True if args.allow_random_init else None,
    )
    print("Pseudo-Mask Generation Complete.")
    print(f"Processed: {res['processed_count']} patches")
    print(f"Saved to: {res['pm_root']}")


if __name__ == "__main__":
    main()
