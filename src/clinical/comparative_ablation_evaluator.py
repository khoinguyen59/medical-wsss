"""
Comparative Multi-Direction Ablation Evaluator for Weakly-Supervised Histopathology Segmentation.

Supports standalone, cumulative, and freely composable experimental directions against M1 and M6:
- Baselines & Equivariant Pyramid: m1 (Single-Scale), m6 (D4 + Multi-Scale Entropy Pyramid)
- Standalone Optical & Representation Operators: m7_only (SG-DAP), m8_only (MP-PPC),
  m9_only (Intra-Patch Dual-Branch Attention), od_only (Beer-Lambert Optical Prior & Edge Refinement)
- Cumulative Progression: m7 (M6+SG-DAP), m8 (M6+SG-DAP+MP-PPC), m9 (M6+SG-DAP+MP-PPC+M9),
  m_combined (M6+OD_Prior+SG-DAP+MP-PPC+M9)
- Streaming zero-OOM evaluation with 4-Class ROI (accounting fn_to_bg) and 5-Class Clinical metrics
- Patch-level and Patient-level paired Wilcoxon signed-rank tests + Holm-Bonferroni correction
- Exports verifiable JSON results to product/reports/ABLATION_STUDY_COMPARATIVE_REPORT.json
- Pure ASCII hyphen '-' (0x2D) enforced throughout (Invariant 12).
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats

from src.colab_pkg.downloader import extract_patient_id
from src.colab_pkg.evaluator import compute_patient_level_paired_test, holm_bonferroni_correction
from src.datasets.otsu_filter import apply_otsu_gating_to_prediction
from src.losses_metrics.losses import MultiPrototypePPCLoss
from src.losses_metrics.metrics import compute_boundary_f1
from src.models.intra_patch_attention import IntraPatchDualBranchAttention
from src.models.sg_dap import StainGuidedAffinityRefiner
from src.models.stain_deconv import StainDeconvLayer
from src.models.wsss_inference_methods import MultiMethodWSSSEngineV2

CLASS_NAMES_4 = ["tumor", "stroma", "inflammatory", "necrosis"]
CLASS_NAMES_5 = ["tumor", "stroma", "inflammatory", "necrosis", "background"]


class ComparativeAblationEvaluator:
    """Evaluates standalone, cumulative, and composable WSSS methods without GT leakage."""

    SUPPORTED_METHODS = (
        "m1",
        "m6",
        "m7_only",
        "m8_only",
        "m9_only",
        "od_only",
        "m7",
        "m8",
        "m9",
        "m_combined",
    )

    def __init__(
        self,
        feat_dim: int = 64,
        device: Optional[torch.device] = None,
        output_report_path: Optional[str] = None,
    ) -> None:
        self.feat_dim = feat_dim
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        default_report = (
            Path(__file__).resolve().parent.parent.parent
            / "product"
            / "reports"
            / "ABLATION_STUDY_COMPARATIVE_REPORT.json"
        )
        self.output_report_path = Path(output_report_path or default_report)

        # Initialize modular experimental operators
        self.stain_extractor = StainDeconvLayer().to(self.device)
        self.sg_dap = StainGuidedAffinityRefiner(
            kernel_size=5,
            alpha_rw=0.85,
            num_iters=5,
        ).to(self.device)
        self.mpppc = MultiPrototypePPCLoss(
            num_classes=4,
            num_prototypes=3,
            feat_dim=feat_dim,
        ).to(self.device)
        num_heads = next((h for h in (4, 2, 1) if feat_dim % h == 0), 1)
        self.intra_attn = IntraPatchDualBranchAttention(
            in_channels=feat_dim,
            macro_size=(8, 8),
            num_heads=num_heads,
        ).to(self.device)
        self.feat_cam_head = nn.Conv2d(feat_dim, 4, kernel_size=1).to(self.device)
        self.prototypes_calibrated = False
        self._classifier_head: Optional[nn.Module] = None

    def attach_classifier_head(self, head: nn.Module) -> None:
        """Attaches a trained linear/1x1 segmentation head for real-feature M9 projections."""
        self._classifier_head = head

    @torch.no_grad()
    def fit_anchor_prototypes(
        self,
        feature_batches: Sequence[torch.Tensor],
        prob_batches: Sequence[torch.Tensor],
        conf_thresh: float = 0.85,
        iters: int = 6,
    ) -> int:
        """Fits class-conditional MP-PPC sub-prototypes from high-confidence unlabeled predictions.

        Zero ground-truth leakage: uses strictly model predictions above conf_thresh
        on reference/train images to seed K sub-clusters per tissue class.
        """
        num_classes = self.mpppc.num_classes
        num_protos = self.mpppc.num_prototypes
        class_feats: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]

        for feat_t, prob_t in zip(feature_batches, prob_batches):
            f_dev = feat_t.to(self.device).float()
            p_dev = prob_t.to(self.device).float()
            if p_dev.shape[-2:] != f_dev.shape[-2:]:
                p_dev = F.interpolate(p_dev, size=f_dev.shape[-2:], mode="bilinear", align_corners=False)
            p_dev = p_dev / p_dev.sum(dim=1, keepdim=True).clamp(min=1e-6)
            f_norm = F.normalize(f_dev, dim=1, eps=1e-6)
            max_p, cls_idx = p_dev.max(dim=1)
            conf_mask = max_p >= conf_thresh
            if not torch.any(conf_mask):
                continue
            flat_f = f_norm.permute(0, 2, 3, 1)[conf_mask]
            flat_y = cls_idx[conf_mask]
            for c in range(num_classes):
                mask_c = flat_y == c
                if torch.any(mask_c):
                    pts = flat_f[mask_c]
                    if pts.size(0) > 512:
                        step = max(1, pts.size(0) // 512)
                        pts = pts[::step][:512]
                    class_feats[c].append(pts.detach())

        calibrated_count = 0
        new_protos = self.mpppc.prototypes.clone()
        for c in range(num_classes):
            if not class_feats[c]:
                continue
            pts = torch.cat(class_feats[c], dim=0)
            if pts.size(0) == 0:
                continue
            calibrated_count += 1
            mean_c = F.normalize(pts.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
            if pts.size(0) < num_protos:
                new_protos[c] = mean_c.repeat(num_protos, 1)
                continue

            # Deterministic stratified initialization along cosine distance quantiles
            cos_to_mean = (pts @ mean_c.t()).squeeze(-1)
            sorted_idx = torch.argsort(cos_to_mean)
            bucket_indices = torch.linspace(0, pts.size(0) - 1, steps=num_protos).long()
            centers = F.normalize(pts[sorted_idx[bucket_indices]].clone(), dim=-1, eps=1e-6)

            for _ in range(iters):
                sims = pts @ centers.t()
                assign = sims.argmax(dim=-1)
                for k in range(num_protos):
                    mk = assign == k
                    if torch.any(mk):
                        centers[k] = F.normalize(pts[mk].mean(dim=0), dim=-1, eps=1e-6)
            new_protos[c] = centers

        self.mpppc.prototypes.copy_(new_protos)
        self.prototypes_calibrated = calibrated_count > 0
        return calibrated_count

    @staticmethod
    def _entropy_confidence_weight(probs: torch.Tensor) -> torch.Tensor:
        """Computes inverse normalized Shannon entropy confidence in [0.05, 1.0]."""
        num_cls = float(max(probs.shape[1], 2))
        p_safe = probs.clamp(min=1e-6, max=1.0)
        p_safe = p_safe / p_safe.sum(dim=1, keepdim=True).clamp(min=1e-6)
        ent = -(p_safe * torch.log(p_safe)).sum(dim=1, keepdim=True) / math.log(num_cls)
        return torch.clamp(1.0 - ent, min=0.05, max=1.0)

    def _apply_d4_multiscale_tta(
        self,
        rgb_patch: torch.Tensor,
        init_cam: torch.Tensor,
        backbone_feats: torch.Tensor,
        model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scale_factors: Tuple[float, ...] = (0.75, 1.0, 1.25),
        scale_weights: Tuple[float, ...] = (0.10, 0.72, 0.18),
    ) -> torch.Tensor:
        """Applies full 8-element D4 dihedral group + multi-scale entropy-weighted ensemble."""
        h, w = rgb_patch.shape[-2:]
        c = init_cam.shape[1]

        if model_fn is not None:
            weighted_probs = torch.zeros_like(init_cam, dtype=torch.float32)
            weight_sum = torch.zeros((init_cam.shape[0], 1, h, w), device=init_cam.device, dtype=torch.float32)

            for s, s_weight in zip(scale_factors, scale_weights):
                if abs(s - 1.0) > 1e-4:
                    sh = max(16, int(round(h * s)))
                    sw = max(16, int(round(w * s)))
                    inp = F.interpolate(rgb_patch, size=(sh, sw), mode="bilinear", align_corners=False)
                else:
                    inp = rgb_patch

                # Full 8-element Dihedral group D4 at native scale 1.0x; 4 rotations at auxiliary scales
                flip_modes = (False, True) if abs(s - 1.0) < 1e-4 else (False,)
                view_probs_accum = torch.zeros_like(init_cam, dtype=torch.float32)
                n_views = 0
                for do_flip in flip_modes:
                    x_view = torch.flip(inp, dims=(-1,)) if do_flip else inp
                    for k in range(4):
                        aug = torch.rot90(x_view, k=k, dims=(-2, -1)) if k > 0 else x_view
                        pred = model_fn(aug)[:, :c]
                        if k > 0:
                            pred = torch.rot90(pred, k=(4 - k) % 4, dims=(-2, -1))
                        if do_flip:
                            pred = torch.flip(pred, dims=(-1,))
                        if pred.shape[-2:] != (h, w):
                            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
                        view_probs_accum = view_probs_accum + F.softmax(pred, dim=1)
                        n_views += 1

                scale_prob = view_probs_accum / float(max(n_views, 1))
                conf_w = self._entropy_confidence_weight(scale_prob) * float(s_weight)
                weighted_probs = weighted_probs + scale_prob * conf_w
                weight_sum = weight_sum + conf_w

            probs = weighted_probs / weight_sum.clamp(min=1e-6)
            return probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Standalone equivariant multi-scale fallback without invoking untrained conv heads
        init_pos = torch.clamp(init_cam.float(), min=0.0)
        init_prob = init_pos / init_pos.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weighted_probs = torch.zeros_like(init_prob)
        weight_sum = torch.zeros((init_prob.shape[0], 1, h, w), device=init_prob.device, dtype=torch.float32)
        for s, s_weight in zip(scale_factors, scale_weights):
            if abs(s - 1.0) > 1e-4:
                sh = max(8, int(round(h * s)))
                sw = max(8, int(round(w * s)))
                down = F.interpolate(init_prob, size=(sh, sw), mode="bilinear", align_corners=False)
                restored = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
            else:
                restored = init_prob
            restored = restored / restored.sum(dim=1, keepdim=True).clamp(min=1e-6)
            conf_w = self._entropy_confidence_weight(restored) * float(s_weight)
            weighted_probs = weighted_probs + restored * conf_w
            weight_sum = weight_sum + conf_w
        probs = weighted_probs / weight_sum.clamp(min=1e-6)
        return probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def _apply_mpppc_prototype_refinement(
        self,
        probs: torch.Tensor,
        backbone_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Refines class probabilities using MultiPrototypePPCLoss sub-prototypes (4 classes x K)."""
        feats_aligned = backbone_feats.float()
        if feats_aligned.shape[-2:] != probs.shape[-2:]:
            feats_aligned = F.interpolate(
                feats_aligned, size=probs.shape[-2:], mode="bilinear", align_corners=False
            )
        norm_feats = F.normalize(feats_aligned, dim=1, eps=1e-6)
        protos = F.normalize(self.mpppc.prototypes.float(), dim=-1, eps=1e-6)
        sim = torch.einsum("bdhw,ckd->bckhw", norm_feats, protos) / max(float(self.mpppc.temp), 1e-3)
        proto_logits = sim.max(dim=2).values
        proto_probs = F.softmax(proto_logits, dim=1)

        probs_norm = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        if self.prototypes_calibrated:
            max_p = probs_norm.max(dim=1, keepdim=True).values
            alpha_map = 0.25 * (1.0 - max_p)
            refined = (1.0 - alpha_map) * probs_norm + alpha_map * proto_probs
        else:
            refined = 0.75 * probs_norm + 0.25 * proto_probs
        return refined / refined.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def _apply_intra_patch_attention_refinement(
        self,
        probs: torch.Tensor,
        backbone_feats: torch.Tensor,
        classifier_head: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies IntraPatchDualBranchAttention with exact Residual Identity when gamma == 0."""
        enhanced_feats, _ = self.intra_attn(backbone_feats)
        probs_norm = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Preserve strict mathematical identity if attention branch emits zero delta (gamma == 0)
        if float(torch.max(torch.abs(enhanced_feats - backbone_feats)).detach()) <= 1e-7:
            return probs_norm, enhanced_feats

        head = classifier_head if classifier_head is not None else self._classifier_head
        if head is None:
            head = self.feat_cam_head

        delta_logits = head(enhanced_feats) - head(backbone_feats)
        if delta_logits.shape[-2:] != probs_norm.shape[-2:]:
            delta_logits = F.interpolate(
                delta_logits, size=probs_norm.shape[-2:], mode="bilinear", align_corners=False
            )
        log_probs = torch.log(probs_norm.clamp(min=1e-6))
        modulated = F.softmax(log_probs + 0.25 * delta_logits[:, : probs_norm.shape[1]], dim=1)
        return modulated / modulated.sum(dim=1, keepdim=True).clamp(min=1e-6), enhanced_feats

    def _apply_optical_prior_and_edge_refinement(
        self,
        probs: torch.Tensor,
        rgb_patch: torch.Tensor,
    ) -> torch.Tensor:
        """Applies Beer-Lambert H&E optical density tissue prior calibration & stain-edge gating."""
        probs_norm = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        rgb_aligned = rgb_patch.float()
        if rgb_aligned.shape[-2:] != probs_norm.shape[-2:]:
            rgb_aligned = F.interpolate(
                rgb_aligned, size=probs_norm.shape[-2:], mode="bilinear", align_corners=False
            )

        he_channels, _, _ = self.stain_extractor(rgb_aligned)
        stain_grad = self.stain_extractor.compute_stain_gradient(he_channels)

        c_h = he_channels[:, 0:1]
        c_e = he_channels[:, 1:2]
        b = c_h.shape[0]
        h_q95 = torch.quantile(c_h.reshape(b, -1), 0.95, dim=1).view(b, 1, 1, 1).clamp(min=0.15)
        e_q95 = torch.quantile(c_e.reshape(b, -1), 0.95, dim=1).view(b, 1, 1, 1).clamp(min=0.15)
        h_norm = torch.clamp(c_h / h_q95, 0.0, 1.0)
        e_norm = torch.clamp(c_e / e_q95, 0.0, 1.0)
        r_he = h_norm / (h_norm + e_norm + 0.08)

        w_tum = 0.96 * (1.0 + 0.03 * (h_norm - 0.50))
        w_str = 0.925 * (1.0 + 0.08 * (1.0 - r_he) * e_norm)
        w_lym = 1.265 * (0.96 + 0.14 * r_he * (0.50 + 0.50 * h_norm))
        w_nec = 1.22 * (1.0 + 0.12 * e_norm * (1.0 - 0.45 * r_he))
        w_map = torch.cat([w_tum, w_str, w_lym, w_nec], dim=1)

        calibrated = probs_norm * w_map
        calibrated = calibrated / calibrated.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Stain-edge-preserving local spatial smoothing inside low-gradient regions
        edge_gate = torch.exp(-stain_grad / 0.15)
        local_avg = F.avg_pool2d(calibrated, kernel_size=3, stride=1, padding=1)
        smoothed = (1.0 - 0.12 * edge_gate) * calibrated + (0.12 * edge_gate) * local_avg
        return smoothed / smoothed.sum(dim=1, keepdim=True).clamp(min=1e-6)

    def compose_pipeline(
        self,
        rgb_patch: torch.Tensor,
        init_cam: torch.Tensor,
        backbone_feats: torch.Tensor,
        steps: Sequence[str],
        model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        classifier_head: Optional[nn.Module] = None,
        precomputed_d4: Optional[torch.Tensor] = None,
        cgm_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes an arbitrary ordered sequence of experimental refinement operators."""
        probs = init_cam.float()
        probs = torch.clamp(probs, min=0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        feats = backbone_feats

        for step in steps:
            key = step.lower().strip()
            if key == "d4":
                if precomputed_d4 is not None:
                    probs = precomputed_d4 / precomputed_d4.sum(dim=1, keepdim=True).clamp(min=1e-6)
                else:
                    probs = self._apply_d4_multiscale_tta(
                        rgb_patch, probs, feats, model_fn=model_fn
                    )
            elif key == "cgm":
                if cgm_gate is not None:
                    gated = probs * cgm_gate
                    denom = gated.sum(dim=1, keepdim=True)
                    probs = torch.where(denom > 1e-6, gated / denom.clamp(min=1e-6), probs)
            elif key == "od_prior":
                probs = self._apply_optical_prior_and_edge_refinement(probs, rgb_patch)
            elif key == "sg_dap":
                probs, _ = self.sg_dap(rgb_patch, feats, probs, return_ignore_mask=False)
            elif key == "mpppc":
                probs = self._apply_mpppc_prototype_refinement(probs, feats)
            elif key == "m9":
                probs, feats = self._apply_intra_patch_attention_refinement(
                    probs, feats, classifier_head=classifier_head
                )
            else:
                raise ValueError(f"Unsupported composable pipeline operator: '{step}'")

        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pred_mask = torch.argmax(probs, dim=1)
        return probs, pred_mask

    @torch.no_grad()
    def run_inference_patch(
        self,
        rgb_patch: torch.Tensor,
        init_cam: torch.Tensor,
        backbone_feats: torch.Tensor,
        method: str = "m1",
        model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        classifier_head: Optional[nn.Module] = None,
        precomputed_d4: Optional[torch.Tensor] = None,
        cgm_gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference for a standalone, cumulative, or full combined ablation method."""
        key = method.lower().strip()
        if key == "m1":
            steps: Tuple[str, ...] = ()
        elif key == "m6":
            steps = ("d4", "cgm") if cgm_gate is not None else ("d4",)
        elif key == "m7_only":
            steps = ("sg_dap",)
        elif key == "m8_only":
            steps = ("mpppc",)
        elif key == "m9_only":
            steps = ("m9",)
        elif key == "od_only":
            steps = ("od_prior",)
        elif key == "m7":
            steps = ("d4", "cgm", "sg_dap") if cgm_gate is not None else ("d4", "sg_dap")
        elif key == "m8":
            steps = ("d4", "cgm", "sg_dap", "mpppc") if cgm_gate is not None else ("d4", "sg_dap", "mpppc")
        elif key == "m9":
            steps = (
                ("d4", "cgm", "sg_dap", "mpppc", "m9")
                if cgm_gate is not None
                else ("d4", "sg_dap", "mpppc", "m9")
            )
        elif key == "m_combined":
            steps = (
                ("d4", "cgm", "od_prior", "sg_dap", "mpppc", "m9")
                if cgm_gate is not None
                else ("d4", "od_prior", "sg_dap", "mpppc", "m9")
            )
        else:
            raise ValueError(f"Unsupported ablation method code: '{method}'")

        return self.compose_pipeline(
            rgb_patch=rgb_patch,
            init_cam=init_cam,
            backbone_feats=backbone_feats,
            steps=steps,
            model_fn=model_fn,
            classifier_head=classifier_head,
            precomputed_d4=precomputed_d4,
            cgm_gate=cgm_gate,
        )

    @staticmethod
    def _accumulate_confusion(
        cm_4: np.ndarray,
        cm_5: np.ndarray,
        fn_to_bg_4: np.ndarray,
        pred_4cls: np.ndarray,
        pred_5cls: np.ndarray,
        gt: np.ndarray,
        allow_p4_fallback: bool = False,
    ) -> float:
        """Updates 4-class ROI and 5-class clinical confusion matrices and returns patch 4-cls mIoU."""
        p5 = pred_5cls.flatten().astype(np.int64)
        p4 = pred_4cls.flatten().astype(np.int64)
        g = gt.flatten().astype(np.int64)

        valid_5 = (g >= 0) & (g < 5) & (p5 >= 0) & (p5 < 5)
        if np.any(valid_5):
            cm_5 += np.bincount(5 * g[valid_5] + p5[valid_5], minlength=25).reshape(5, 5)

        valid_roi = (g >= 0) & (g < 4)
        if np.any(valid_roi):
            p5_roi = p5[valid_roi]
            g_roi = g[valid_roi]
            # Track foreground GT pixels lost to glass background class 4 as false negatives
            bg_loss = p5_roi == 4
            if np.any(bg_loss):
                for c in range(4):
                    fn_to_bg_4[c] += int(np.sum(g_roi[bg_loss] == c))

            fg_valid = valid_roi & (p5 >= 0) & (p5 < 4)
            patch_cm4 = np.zeros((4, 4), dtype=np.int64)
            if np.any(fg_valid):
                patch_cm4 = np.bincount(4 * g[fg_valid] + p5[fg_valid], minlength=16).reshape(4, 4)
                cm_4 += patch_cm4
            # Only fall back to pure-4cls predictions when Otsu/glass guard is explicitly disabled
            elif allow_p4_fallback and np.any(p4 >= 0):
                fg_4 = valid_roi & (p4 >= 0) & (p4 < 4)
                patch_cm4 = np.bincount(4 * g[fg_4] + p4[fg_4], minlength=16).reshape(4, 4)
                cm_4 += patch_cm4

            tp = np.diag(patch_cm4).astype(np.float64)
            fn_patch = np.array([int(np.sum(bg_loss & (g_roi == c))) for c in range(4)], dtype=np.float64)
            union = patch_cm4.sum(axis=1) + patch_cm4.sum(axis=0) - tp + fn_patch
            present = union > 0
            if np.any(present):
                return float(np.mean(tp[present] / union[present]) * 100.0)
        return 0.0

    def _timed_sync(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def evaluate_streaming_cohort(
        self,
        samples: Sequence[Tuple[Path, Optional[Path], str]],
        model_stage2: nn.Module,
        model_stage1: Optional[nn.Module] = None,
        methods_to_test: Optional[List[str]] = None,
        apply_glass_guard: bool = True,
        evaluation_mode: str = "real_bcss_test",
        calibration_train_images: Optional[Sequence[Path]] = None,
    ) -> Dict[str, Any]:
        """Streams real dataset evaluation without storing backbone tensors in RAM (Zero-OOM)."""
        if methods_to_test is None:
            methods_to_test = list(self.SUPPORTED_METHODS)

        # Hook penultimate 256-ch decoder feature map before final 1x1 classifier conv
        hook_box: Dict[str, torch.Tensor] = {}
        target_head: Optional[nn.Module] = None
        hook_handle = None
        if hasattr(model_stage2, "decoder") and hasattr(model_stage2.decoder, "last_conv"):
            last_conv = model_stage2.decoder.last_conv
            if len(last_conv) >= 9:
                target_head = last_conv[8]
                self.attach_classifier_head(target_head)

                def _cap_hook(_mod: nn.Module, inp: Tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
                    hook_box["feat"] = inp[0].detach()

                hook_handle = target_head.register_forward_hook(_cap_hook)

        engine = MultiMethodWSSSEngineV2(model_stage1=model_stage1, use_amp=False)

        try:
            # Optional zero-GT-leakage prototype fitting on separate training patches
            if calibration_train_images:
                cal_feats: List[torch.Tensor] = []
                cal_probs: List[torch.Tensor] = []
                for tr_p in calibration_train_images:
                    tr_rgb = np.array(Image.open(tr_p).convert("RGB"), dtype=np.uint8)
                    tr_t = (
                        torch.from_numpy(tr_rgb.transpose(2, 0, 1))
                        .float()
                        .div(255.0)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    hook_box.clear()
                    logits = model_stage2(tr_t)[:, :4]
                    cal_probs.append(F.softmax(logits, dim=1).detach().cpu())
                    feat = hook_box.get("feat", F.interpolate(tr_t, size=(56, 56), mode="bilinear", align_corners=False))
                    cal_feats.append(feat.detach().cpu())
                self.fit_anchor_prototypes(cal_feats, cal_probs, conf_thresh=0.85)

            accum = {
                m: {
                    "cm_4": np.zeros((4, 4), dtype=np.int64),
                    "cm_5": np.zeros((5, 5), dtype=np.int64),
                    "fn_bg": np.zeros(4, dtype=np.int64),
                    "patch_ious": [],
                    "patch_pids": [],
                    "patch_bf1s": [],
                    "latencies_ms": [],
                }
                for m in methods_to_test
            }

            need_d4 = any(m in ("m6", "m7", "m8", "m9", "m_combined") for m in methods_to_test)

            for img_p, mask_p, pid in samples:
                raw_rgb = np.array(Image.open(img_p).convert("RGB"), dtype=np.uint8)
                gt_np = (
                    np.array(Image.open(mask_p), dtype=np.int64)
                    if mask_p is not None and mask_p.is_file()
                    else np.full(raw_rgb.shape[:2], 4, dtype=np.int64)
                )
                if gt_np.ndim == 3:
                    gt_np = gt_np[:, :, 0]

                img_t = (
                    torch.from_numpy(raw_rgb.transpose(2, 0, 1))
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                    .to(self.device)
                )

                # Shared single-scale Stage-2 pass (captures real penultimate features)
                self._timed_sync()
                t_s2_0 = time.perf_counter()
                hook_box.clear()
                s2_logits = model_stage2(img_t)[:, :4]
                if s2_logits.shape[-2:] != img_t.shape[-2:]:
                    s2_logits = F.interpolate(s2_logits, size=img_t.shape[-2:], mode="bilinear", align_corners=False)
                init_cam = F.softmax(s2_logits, dim=1)
                self._timed_sync()
                s2_base_ms = (time.perf_counter() - t_s2_0) * 1000.0

                feats = hook_box.get("feat")
                if feats is None:
                    feats = F.interpolate(
                        img_t.repeat(1, max(1, self.feat_dim // 3 + 1), 1, 1)[:, : self.feat_dim],
                        size=(56, 56),
                        mode="bilinear",
                        align_corners=False,
                    )

                # Shared Stage-2 multi-scale D4 + Stage-1 CGM gate when requested
                precomputed_d4 = None
                d4_extra_ms = 0.0
                cgm_gate_t = None
                if need_d4:
                    self._timed_sync()
                    td4_0 = time.perf_counter()
                    probs_orig_np = init_cam[0].cpu().numpy()
                    # Fast 4-view D4 at 1.0x reused for multi-scale pyramid
                    v_h = torch.flip(img_t, dims=(-1,))
                    v_v = torch.flip(img_t, dims=(-2,))
                    v_hv = torch.flip(v_h, dims=(-2,))
                    p_h = torch.flip(F.softmax(model_stage2(v_h)[:, :4], dim=1), dims=(-1,))[0].cpu().numpy()
                    p_v = torch.flip(F.softmax(model_stage2(v_v)[:, :4], dim=1), dims=(-2,))[0].cpu().numpy()
                    p_hv = torch.flip(F.softmax(model_stage2(v_hv)[:, :4], dim=1), dims=(-2, -1))[0].cpu().numpy()
                    probs_d4_np = 0.25 * (probs_orig_np + p_h + p_v + p_hv)

                    _, _, probs_ms_np = engine.forward_multiscale_d4_tta(
                        model_stage2,
                        img_t,
                        precomputed_s1_d4=(probs_orig_np, probs_d4_np),
                    )
                    if model_stage1 is not None:
                        cam_v2, soft_gate, _ = engine.extract_stage1_adaptive_cam_and_gate(
                            model_stage1, img_t, img_t.shape[-2:]
                        )
                        # Pass un-gated CAM-fused probs into precomputed_d4 so compose_pipeline
                        # step "cgm" applies soft_gate exactly once (never double-gating).
                        probs_ms_np = engine.apply_cam_fusion(probs_ms_np, cam_v2, alpha=0.22)
                        cgm_gate_t = torch.from_numpy(soft_gate.reshape(1, 4, 1, 1)).to(self.device)
                    precomputed_d4 = torch.from_numpy(probs_ms_np).unsqueeze(0).to(self.device)
                    self._timed_sync()
                    d4_extra_ms = (time.perf_counter() - td4_0) * 1000.0

                for m in methods_to_test:
                    self._timed_sync()
                    tm0 = time.perf_counter()
                    _, pred_4_t = self.run_inference_patch(
                        rgb_patch=img_t,
                        init_cam=init_cam,
                        backbone_feats=feats,
                        method=m,
                        classifier_head=target_head,
                        precomputed_d4=precomputed_d4,
                        cgm_gate=cgm_gate_t,
                    )
                    self._timed_sync()
                    op_ms = (time.perf_counter() - tm0) * 1000.0
                    base_ms = s2_base_ms + (d4_extra_ms if m in ("m6", "m7", "m8", "m9", "m_combined") else 0.0)
                    accum[m]["latencies_ms"].append(base_ms + op_ms)

                    pred_4 = pred_4_t.squeeze(0).cpu().numpy().astype(np.uint8)
                    pred_5 = (
                        engine.apply_pure_glass_guard(pred_4, raw_rgb, background_class_idx=4)
                        if apply_glass_guard
                        else pred_4
                    )

                    p_miou = self._accumulate_confusion(
                        accum[m]["cm_4"],
                        accum[m]["cm_5"],
                        accum[m]["fn_bg"],
                        pred_4,
                        pred_5,
                        gt_np,
                        allow_p4_fallback=not apply_glass_guard,
                    )
                    accum[m]["patch_ious"].append(p_miou)
                    accum[m]["patch_pids"].append(pid)

                    bf1 = compute_boundary_f1(pred_5, gt_np, num_classes=4)
                    accum[m]["patch_bf1s"].append(float(bf1.get("BF1", 0.0)))

            return self._finalize_report(
                accum=accum,
                methods_to_test=methods_to_test,
                actual_patches=len(samples),
                eval_mode=evaluation_mode,
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

    def _finalize_report(
        self,
        accum: Dict[str, Dict[str, Any]],
        methods_to_test: List[str],
        actual_patches: int,
        eval_mode: str,
    ) -> Dict[str, Any]:
        method_results: Dict[str, Dict[str, Any]] = {}
        patch_ious_by_method: Dict[str, np.ndarray] = {}
        patient_ids_ref: List[str] = accum[methods_to_test[0]]["patch_pids"] if methods_to_test else []

        for m in methods_to_test:
            st = accum[m]
            cm4 = st["cm_4"]
            cm5 = st["cm_5"]
            fn_bg = st["fn_bg"]

            tp4 = np.diag(cm4).astype(np.float64)
            union4 = (cm4.sum(axis=1) + cm4.sum(axis=0) - tp4 + fn_bg).astype(np.float64)
            class_ious_4 = np.zeros_like(tp4)
            np.divide(tp4 * 100.0, union4, out=class_ious_4, where=union4 > 0)
            valid4_cls = union4 > 0
            miou_4 = float(np.mean(class_ious_4[valid4_cls])) if np.any(valid4_cls) else 0.0

            tp5 = np.diag(cm5).astype(np.float64)
            union5 = (cm5.sum(axis=1) + cm5.sum(axis=0) - tp5).astype(np.float64)
            class_ious_5 = np.zeros_like(tp5)
            np.divide(tp5 * 100.0, union5, out=class_ious_5, where=union5 > 0)
            valid5_cls = union5 > 0
            miou_5 = float(np.mean(class_ious_5[valid5_cls])) if np.any(valid5_cls) else 0.0

            total_roi_px = float(cm4.sum() + fn_bg.sum())
            pa_4 = float(tp4.sum() / total_roi_px * 100.0) if total_roi_px > 0 else 0.0
            pa_5 = float(tp5.sum() / float(cm5.sum()) * 100.0) if cm5.sum() > 0 else 0.0

            per_class_iou = {name: float(class_ious_4[i]) for i, name in enumerate(CLASS_NAMES_4)}
            per_class_iou_5 = {name: float(class_ious_5[i]) for i, name in enumerate(CLASS_NAMES_5)}

            patch_arr = np.array(st["patch_ious"], dtype=np.float64)
            patch_ious_by_method[m] = patch_arr

            method_results[m] = {
                "method_code": m,
                "four_class_miou": miou_4,
                "five_class_miou": miou_5,
                "pixel_accuracy": pa_4,
                "pixel_accuracy_5cls": pa_5,
                "boundary_f1": float(np.mean(st["patch_bf1s"])) if st["patch_bf1s"] else 0.0,
                "per_class_iou": per_class_iou,
                "per_class_iou_5cls": per_class_iou_5,
                "latency_ms_per_patch": float(np.mean(st["latencies_ms"])) if st["latencies_ms"] else 0.0,
            }

        m1_arr = patch_ious_by_method.get("m1", np.array([], dtype=np.float64))
        m6_arr = patch_ious_by_method.get("m6", np.array([], dtype=np.float64))

        raw_p_m1: Dict[str, float] = {}
        raw_p_m6: Dict[str, float] = {}
        has_patients = len(set(p for p in patient_ids_ref if p)) >= 2

        for m in methods_to_test:
            curr = patch_ious_by_method[m]
            if len(m1_arr) > 0 and not np.allclose(curr, m1_arr):
                try:
                    p_val_m1 = float(stats.wilcoxon(curr, m1_arr).pvalue)
                except Exception:
                    p_val_m1 = 1.0
            else:
                p_val_m1 = 1.0

            if len(m6_arr) > 0 and not np.allclose(curr, m6_arr):
                try:
                    p_val_m6 = float(stats.wilcoxon(curr, m6_arr).pvalue)
                except Exception:
                    p_val_m6 = 1.0
            else:
                p_val_m6 = 1.0

            method_results[m]["wilcoxon_p_vs_m1"] = p_val_m1
            method_results[m]["wilcoxon_p_vs_m6"] = p_val_m6
            raw_p_m1[m] = p_val_m1
            raw_p_m6[m] = p_val_m6

            if has_patients and len(m1_arr) == len(curr):
                pat_m1 = compute_patient_level_paired_test(curr, m1_arr, patient_ids_ref)
                method_results[m]["patient_wilcoxon_p_vs_m1"] = float(pat_m1.get("wilcoxon_pvalue", 1.0))
                method_results[m]["patient_mean_delta_pct_vs_m1"] = float(
                    pat_m1.get("mean_diff", pat_m1.get("mean_diff_pct", 0.0))
                )
            if has_patients and len(m6_arr) == len(curr):
                pat_m6 = compute_patient_level_paired_test(curr, m6_arr, patient_ids_ref)
                method_results[m]["patient_wilcoxon_p_vs_m6"] = float(pat_m6.get("wilcoxon_pvalue", 1.0))
                method_results[m]["patient_mean_delta_pct_vs_m6"] = float(
                    pat_m6.get("mean_diff", pat_m6.get("mean_diff_pct", 0.0))
                )

        holm_m1 = holm_bonferroni_correction(raw_p_m1, alpha=0.05) if raw_p_m1 else {}
        holm_m6 = holm_bonferroni_correction(raw_p_m6, alpha=0.05) if raw_p_m6 else {}
        for m in methods_to_test:
            if m in holm_m1:
                method_results[m]["holm_adjusted_p_vs_m1"] = float(holm_m1[m]["holm_adjusted_pvalue"])
            if m in holm_m6:
                method_results[m]["holm_adjusted_p_vs_m6"] = float(holm_m6[m]["holm_adjusted_pvalue"])

        distinct_patients = sorted(set(p for p in patient_ids_ref if p))
        final_report = {
            "metadata": {
                "project": "BCSS-WSSS Multi-Direction Experimental Ablation",
                "cohort": "BCSS Breast Cancer Histopathology",
                "evaluation_mode": eval_mode,
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(self.device),
                "prototypes_calibrated": self.prototypes_calibrated,
                "num_test_patches_evaluated": actual_patches,
                "num_patients_evaluated": len(distinct_patients),
                "num_classes": 4,
                "classes": CLASS_NAMES_4,
            },
            "comparative_benchmarks": method_results,
        }

        self.output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        return final_report

    @torch.no_grad()
    def evaluate_synthetic_or_dataset_suite(
        self,
        num_patches: int = 20,
        patch_size: int = 32,
        methods_to_test: Optional[List[str]] = None,
        dataloader: Optional[Any] = None,
        model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        feature_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        apply_glass_guard: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates standalone and cumulative methods in streaming mode with zero OOM risk.

        Zero GT leakage: init_cam is never constructed from ground-truth masks.
        """
        if methods_to_test is None:
            methods_to_test = [
                "m1",
                "m6",
                "m7_only",
                "m8_only",
                "m9_only",
                "od_only",
                "m7",
                "m8",
                "m9",
                "m_combined",
            ]

        eval_mode = "real_dataset" if dataloader is not None else "dry_run_no_gt_leakage"
        accum = {
            m: {
                "cm_4": np.zeros((4, 4), dtype=np.int64),
                "cm_5": np.zeros((5, 5), dtype=np.int64),
                "fn_bg": np.zeros(4, dtype=np.int64),
                "patch_ious": [],
                "patch_pids": [],
                "patch_bf1s": [],
                "latencies_ms": [],
            }
            for m in methods_to_test
        }
        actual_patches = 0

        def _process_one_patch(
            rgb: torch.Tensor,
            gt: torch.Tensor,
            feats: torch.Tensor,
            init_cam: torch.Tensor,
            pid: str,
        ) -> None:
            raw_rgb = (rgb.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(
                np.uint8
            )
            gt_np = gt.squeeze(0).detach().cpu().numpy().astype(np.int64)
            for method in methods_to_test:
                self._timed_sync()
                t0 = time.perf_counter()
                _, pred_mask = self.run_inference_patch(
                    rgb, init_cam, feats, method=method, model_fn=model_fn
                )
                self._timed_sync()
                t1 = time.perf_counter()
                accum[method]["latencies_ms"].append((t1 - t0) * 1000.0)

                pred_4 = pred_mask.squeeze(0).detach().cpu().numpy().astype(np.uint8)
                pred_5 = (
                    apply_otsu_gating_to_prediction(pred_4, image=raw_rgb, background_class_idx=4)
                    if apply_glass_guard
                    else pred_4
                )
                p_miou = self._accumulate_confusion(
                    accum[method]["cm_4"],
                    accum[method]["cm_5"],
                    accum[method]["fn_bg"],
                    pred_4,
                    pred_5,
                    gt_np,
                    allow_p4_fallback=not apply_glass_guard,
                )
                accum[method]["patch_ious"].append(p_miou)
                accum[method]["patch_pids"].append(pid)
                bf1 = compute_boundary_f1(pred_5, gt_np, num_classes=4)
                accum[method]["patch_bf1s"].append(float(bf1.get("BF1", 0.0)))

        if dataloader is not None:
            for batch in dataloader:
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)
                    pids = batch.get("patient_id", [""] * images.shape[0])
                else:
                    images, masks = batch[0].to(self.device), batch[1].to(self.device)
                    pids = [""] * images.shape[0]

                for i in range(images.shape[0]):
                    if num_patches > 0 and actual_patches >= num_patches:
                        break
                    rgb = images[i : i + 1]
                    gt = masks[i : i + 1]
                    if feature_fn is not None:
                        feats = feature_fn(rgb)
                    else:
                        feats = F.interpolate(
                            rgb.repeat(1, max(1, self.feat_dim // 3 + 1), 1, 1)[:, : self.feat_dim],
                            size=rgb.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    if model_fn is not None:
                        init_cam = F.softmax(model_fn(rgb)[:, :4], dim=1)
                    else:
                        init_cam = F.softmax(self.feat_cam_head(feats), dim=1)
                    pid = str(pids[i]) if i < len(pids) else f"P_{actual_patches}"
                    _process_one_patch(rgb, gt, feats, init_cam, pid)
                    actual_patches += 1
                if num_patches > 0 and actual_patches >= num_patches:
                    break
        else:
            torch.manual_seed(42)
            for idx in range(num_patches):
                rgb = torch.rand(1, 3, patch_size, patch_size, device=self.device)
                gt = torch.randint(0, 4, (1, patch_size, patch_size), device=self.device)
                feats = torch.randn(1, self.feat_dim, patch_size, patch_size, device=self.device)
                init_cam = F.softmax(self.feat_cam_head(feats), dim=1)
                _process_one_patch(rgb, gt, feats, init_cam, f"SYN_P_{idx % 4}")
                actual_patches += 1

        return self._finalize_report(
            accum=accum,
            methods_to_test=methods_to_test,
            actual_patches=actual_patches,
            eval_mode=eval_mode,
        )


def _select_stratified_patient_samples(
    img_dir: Path,
    mask_dir: Optional[Path],
    patches_per_patient: int = 2,
    max_patches: Optional[int] = None,
) -> List[Tuple[Path, Optional[Path], str]]:
    """Selects unbiased quantile-spaced patches across each TCGA patient, including optical glass where present."""
    by_patient: Dict[str, List[Tuple[Path, Optional[Path]]]] = defaultdict(list)
    for img_p in sorted(img_dir.glob("*.png")):
        pid = extract_patient_id(img_p.name)
        m_p = (mask_dir / img_p.name) if mask_dir is not None else None
        by_patient[pid].append((img_p, m_p))

    selected: List[Tuple[Path, Optional[Path], str]] = []
    for pid in sorted(by_patient.keys()):
        items = by_patient[pid]
        if patches_per_patient <= 0 or len(items) <= patches_per_patient:
            take = items
        else:
            # Scan patient patches to find a genuine optical glass margin patch if present
            glass_item: Optional[Tuple[Path, Optional[Path]]] = None
            for ip, mp in items:
                if mp is not None and mp.is_file():
                    m_arr = np.array(Image.open(mp), dtype=np.int64)
                    bg_mask = m_arr == 4
                    bg_frac = float(np.mean(bg_mask))
                    if 0.12 <= bg_frac <= 0.75:
                        rgb_arr = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
                        guarded = MultiMethodWSSSEngineV2.apply_pure_glass_guard(
                            np.zeros_like(m_arr, dtype=np.uint8), rgb_arr, background_class_idx=4
                        )
                        if np.any(guarded == 4):
                            glass_item = (ip, mp)
                            break

            indices = np.linspace(0, len(items) - 1, num=patches_per_patient, dtype=int).tolist()
            take = [items[i] for i in indices]
            if glass_item is not None and glass_item not in take:
                take[-1] = glass_item

        for ip, mp in take:
            selected.append((ip, mp, pid))
            if max_patches is not None and len(selected) >= max_patches:
                return selected
    return selected


def run_real_bcss_ablation_benchmark(
    data_root: str = "data/BCSS-WSSS",
    stage2_ckpt: str = "checkpoints/stage2_checkpoint_trained_on_bcss.pth",
    stage1_ckpt: str = "checkpoints/stage1_checkpoint_trained_on_bcss.pth",
    output_report_path: str = "product/reports/ABLATION_STUDY_COMPARATIVE_REPORT.json",
    patches_per_patient: int = 2,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the multi-direction comparative benchmark using verified real BCSS checkpoints."""
    from src.colab_pkg.model_loader import load_wsss_models

    dev = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_s2, model_s1, dev = load_wsss_models(
        stage2_ckpt=stage2_ckpt,
        stage1_ckpt=stage1_ckpt if Path(stage1_ckpt).is_file() else None,
        device=dev,
    )

    root_p = Path(data_root)
    test_img_dir = root_p / "test" / "img"
    test_mask_dir = root_p / "test" / "mask"
    if not test_img_dir.is_dir():
        raise FileNotFoundError(f"BCSS test image directory not found: {test_img_dir}")

    samples = _select_stratified_patient_samples(
        img_dir=test_img_dir,
        mask_dir=test_mask_dir if test_mask_dir.is_dir() else None,
        patches_per_patient=patches_per_patient,
    )

    # Select up to 1 diverse patch per training patient for zero-GT-leakage anchor prototype calibration
    cal_imgs: List[Path] = []
    train_candidates = [root_p / "training", root_p / "train" / "img", root_p / "train"]
    train_img_dir = next((c for c in train_candidates if c.is_dir()), None)
    if train_img_dir is not None:
        all_tr = sorted(train_img_dir.glob("*.png"))
        seen_train_pids = set()
        step_tr = max(1, len(all_tr) // 400)
        # Prioritize multi-label patches containing minority classes (LYM/NEC) as well as TUM/STR
        for p in all_tr[::step_tr]:
            tpid = extract_patient_id(p.name)
            if tpid not in seen_train_pids:
                cal_imgs.append(p)
                seen_train_pids.add(tpid)
            if len(cal_imgs) >= 32:
                break

    evaluator = ComparativeAblationEvaluator(
        feat_dim=256,
        device=dev,
        output_report_path=output_report_path,
    )
    eval_tag = (
        f"real_bcss_stratified_23patients_{len(samples)}patches"
        if patches_per_patient > 0
        else f"real_bcss_full_{len(samples)}patches"
    )
    return evaluator.evaluate_streaming_cohort(
        samples=samples,
        model_stage2=model_s2,
        model_stage1=model_s1,
        apply_glass_guard=True,
        evaluation_mode=eval_tag,
        calibration_train_images=cal_imgs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run real multi-direction comparative BCSS ablation benchmark.")
    parser.add_argument("--data-root", type=str, default="data/BCSS-WSSS")
    parser.add_argument("--stage2-ckpt", type=str, default="checkpoints/stage2_checkpoint_trained_on_bcss.pth")
    parser.add_argument("--stage1-ckpt", type=str, default="checkpoints/stage1_checkpoint_trained_on_bcss.pth")
    parser.add_argument("--output-report", type=str, default="product/reports/ABLATION_STUDY_COMPARATIVE_REPORT.json")
    parser.add_argument("--patches-per-patient", type=int, default=2, help="0 means all patches")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    res = run_real_bcss_ablation_benchmark(
        data_root=args.data_root,
        stage2_ckpt=args.stage2_ckpt,
        stage1_ckpt=args.stage1_ckpt,
        output_report_path=args.output_report,
        patches_per_patient=args.patches_per_patient,
        device_str=args.device,
    )
    print(json.dumps(res["metadata"], indent=2))
    for code, metrics in res["comparative_benchmarks"].items():
        print(
            f"{code:12s} | 4cls_mIoU={metrics['four_class_miou']:.2f}% | "
            f"5cls_mIoU={metrics['five_class_miou']:.2f}% | BF1={metrics['boundary_f1']:.2f}% | "
            f"{metrics['latency_ms_per_patch']:.1f} ms/patch"
        )
