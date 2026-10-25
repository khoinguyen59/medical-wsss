"""
MedSAM WSSS Adapter Architecture for Weakly-Supervised Histopathology Segmentation.

Course: SE374.R11 - Medical Image Processing Software Development
Grounding:
- S2C (CVPR 2024): Automated CAM-to-prompt generation
- SAM2PATH / MedSAM (Nature Communications 2024): Foundation model adaptation
- Parameter-Efficient Fine-Tuning (PEFT) with Residual Adapter Bottlenecks (256 -> 64 -> 256)
- 5-class Histological Mask Decoder (0: TUM, 1: STR, 2: LYM, 3: NEC, 4: BACK)

Strict Invariant 12 compliance: 100% ASCII 0x2D hyphens only.
"""

import os
import sys
import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

# Safeguard against OpenMP library collision on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repos/SAM2PATH to sys.path if available
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
_SAM2PATH_DIR = os.path.join(_WORKSPACE_DIR, "repos", "SAM2PATH")
if os.path.exists(_SAM2PATH_DIR) and _SAM2PATH_DIR not in sys.path:
    sys.path.insert(0, _SAM2PATH_DIR)


# ==============================================================================
# 1. Residual Adapter Bottleneck Block
# ==============================================================================

class ResidualAdapterBottleneck(nn.Module):
    """Residual Adapter Bottleneck (256 -> 64 -> 256) for Parameter-Efficient Fine-Tuning.

    Formula:
        Adapter(X) = W_up(GELU(W_down(LayerNorm(X)))) * scale
        Output = X + Adapter(X)
    """

    def __init__(
        self,
        in_dim: int = 256,
        mid_dim: int = 64,
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.scale = scale

        self.norm = nn.LayerNorm(in_dim)
        self.down_proj = nn.Linear(in_dim, mid_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(mid_dim, in_dim)

        # Zero-init up-projection so adapter starts as pure identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Input tensor of shape (B, N, C) or (B, C, H, W).
        Returns:
            Tensor with identical shape.
        """
        if x.dim() == 4:
            b, c, h, w = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
            res = self.norm(x_flat)
            res = self.down_proj(res)
            res = self.act(res)
            res = self.up_proj(res) * self.scale
            out = x_flat + res
            return out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        elif x.dim() == 3:
            res = self.norm(x)
            res = self.down_proj(res)
            res = self.act(res)
            res = self.up_proj(res) * self.scale
            return x + res
        elif x.dim() == 2:
            res = self.norm(x)
            res = self.down_proj(res)
            res = self.act(res)
            res = self.up_proj(res) * self.scale
            return x + res
        else:
            raise ValueError(f"Unsupported tensor dimension: {x.dim()}")


ResidualAdapterBlock = ResidualAdapterBottleneck


# ==============================================================================
# 2. CAM Prompt Generator
# ==============================================================================

class CAMPromptGenerator(nn.Module):
    """Automated Point Prompt Generator for Histopathology WSSS.

    Mines positive point prompts from local maxima and intensity-weighted centroids
    of high-confidence CAM activations for active classes (TUM, STR, LYM, NEC).
    Mines negative point prompts from competing tissue activations and low-affinity glass.

    Coordinate format: (x, y) = (column, row).
    Label format: +1 for positive, 0 for negative, -1 for padding/dummy.
    """

    CLASS_NAMES: List[str] = ["TUM", "STR", "LYM", "NEC"]

    def __init__(
        self,
        num_classes: int = 4,
        max_pos_points: int = 3,
        max_neg_points: int = 3,
        min_component_area: int = 16,
        pos_threshold: float = 0.60,
        neg_threshold: float = 0.50,
        cgm_threshold: float = 0.10,
        target_size: Tuple[int, int] = (1024, 1024),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_pos_points = max_pos_points
        self.max_neg_points = max_neg_points
        self.min_component_area = min_component_area
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.cgm_threshold = cgm_threshold
        self.target_size = target_size

    @torch.no_grad()
    def forward(
        self,
        cams: torch.Tensor,
        cls_scores: Optional[torch.Tensor] = None,
        cgm_gate: Optional[torch.Tensor] = None,
        hsv_saturation: Optional[torch.Tensor] = None,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Extracts per-class point prompts and labels.

        Args:
            cams: Tensor of shape (B, 4, H, W) or (4, H, W) normalized in [0, 1].
            cls_scores: Optional tensor of shape (B, 4) in [0, 1].
            cgm_gate: Optional boolean or float gate tensor of shape (B, 4).
            hsv_saturation: Optional saturation tensor of shape (B, H, W) for glass gating.

        Returns:
            Dict mapping class_idx (0..3) to:
                'point_coords': Tensor of shape (B, N_pts, 2) in target resolution (x, y).
                'point_labels': Tensor of shape (B, N_pts) int64 (+1, 0, or -1).
                'active_mask': Tensor of shape (B,) bool indicating if class is active.
        """
        if cams.dim() == 3:
            cams = cams.unsqueeze(0)
        b, c_num, h, w = cams.shape
        device = cams.device

        if cls_scores is None:
            cls_scores = torch.ones((b, c_num), device=device, dtype=torch.float32)
        elif cls_scores.dim() == 1:
            cls_scores = cls_scores.unsqueeze(0)

        # Min-max normalize CAMs per channel (ensure float32 precision for safe numpy extraction)
        cams_norm = cams.float().clone()
        for i in range(b):
            for c in range(c_num):
                c_min = cams_norm[i, c].min()
                c_max = cams_norm[i, c].max()
                denom = c_max - c_min + 1e-6
                cams_norm[i, c] = (cams_norm[i, c] - c_min) / denom

        total_pts = self.max_pos_points + self.max_neg_points
        scale_x = self.target_size[1] / float(w)
        scale_y = self.target_size[0] / float(h)

        results: Dict[int, Dict[str, torch.Tensor]] = {}

        for c in range(min(c_num, self.num_classes)):
            batch_coords: List[torch.Tensor] = []
            batch_labels: List[torch.Tensor] = []
            batch_active: List[bool] = []

            for i in range(b):
                score = cls_scores[i, c].item()
                gate = cgm_gate[i, c].item() if cgm_gate is not None else 1.0
                is_active = (score >= self.cgm_threshold) and (gate > 0.5)
                batch_active.append(bool(is_active))

                pts_list: List[Tuple[float, float]] = []
                lbl_list: List[int] = []

                cam_c = cams_norm[i, c]

                if is_active:
                    # Positive prompt mining
                    mu = cam_c.mean().item()
                    sigma = cam_c.std().item()
                    dyn_theta = max(self.pos_threshold, mu + 0.5 * sigma)
                    high_mask = (cam_c >= dyn_theta).cpu().numpy()

                    # Find candidate positive points (safely float-converted)
                    pos_pts = self._find_positive_points(
                        cam_c.float().cpu().numpy(),
                        high_mask,
                        max_points=self.max_pos_points,
                    )

                    for px, py in pos_pts:
                        pts_list.append((px * scale_x, py * scale_y))
                        lbl_list.append(1)

                    # Negative prompt mining
                    # Competing classes activation
                    other_indices = [idx for idx in range(c_num) if idx != c]
                    if other_indices:
                        competing_max = cams_norm[i, other_indices].max(dim=0)[0].float().cpu().numpy()
                    else:
                        competing_max = np.zeros((h, w), dtype=np.float32)

                    cam_c_np = cam_c.float().cpu().numpy()
                    comp_mask = (competing_max >= self.neg_threshold) & (cam_c_np <= 0.20)

                    # Background / glass check
                    bg_mask = (cams_norm[i].max(dim=0)[0].float().cpu().numpy() < 0.15)
                    if hsv_saturation is not None:
                        sat_np = hsv_saturation[i].float().cpu().numpy()
                        bg_mask = bg_mask | (sat_np < 0.08)

                    neg_mask = comp_mask | bg_mask
                    neg_pts = self._find_negative_points(
                        competing_max,
                        neg_mask,
                        max_points=self.max_neg_points,
                    )

                    for nx, ny in neg_pts:
                        pts_list.append((nx * scale_x, ny * scale_y))
                        lbl_list.append(0)

                # Ensure fixed length padding
                while len(pts_list) < total_pts:
                    pts_list.append((0.0, 0.0))
                    lbl_list.append(-1)

                pts_tensor = torch.tensor(pts_list[:total_pts], dtype=torch.float32, device=device)
                lbl_tensor = torch.tensor(lbl_list[:total_pts], dtype=torch.int64, device=device)

                batch_coords.append(pts_tensor)
                batch_labels.append(lbl_tensor)

            results[c] = {
                "point_coords": torch.stack(batch_coords, dim=0),
                "point_labels": torch.stack(batch_labels, dim=0),
                "active_mask": torch.tensor(batch_active, dtype=torch.bool, device=device),
            }

        return results

    def _find_positive_points(
        self,
        cam_map: np.ndarray,
        high_mask: np.ndarray,
        max_points: int,
    ) -> List[Tuple[float, float]]:
        """Mines positive points as intensity-weighted centroids or local peaks."""
        points: List[Tuple[float, float]] = []
        h, w = cam_map.shape

        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(high_mask)
            if num_features > 0:
                components = []
                for comp_id in range(1, num_features + 1):
                    comp_mask = (labeled == comp_id)
                    area = int(comp_mask.sum())
                    if area < self.min_component_area:
                        continue
                    weights = cam_map[comp_mask]
                    mass = float(weights.sum())

                    # Intensity-weighted centroid
                    y_indices, x_indices = np.where(comp_mask)
                    if mass > 1e-6:
                        cx = float(np.sum(x_indices * weights) / mass)
                        cy = float(np.sum(y_indices * weights) / mass)
                    else:
                        cx = float(np.mean(x_indices))
                        cy = float(np.mean(y_indices))

                    # Local peak
                    peak_sub_idx = np.argmax(weights)
                    px = float(x_indices[peak_sub_idx])
                    py = float(y_indices[peak_sub_idx])

                    # Select centroid if inside high confidence region, else local peak
                    icx, icy = int(round(np.clip(cx, 0, w - 1))), int(round(np.clip(cy, 0, h - 1)))
                    if high_mask[icy, icx]:
                        chosen = (cx, cy)
                    else:
                        chosen = (px, py)

                    components.append((mass, chosen))

                components.sort(key=lambda item: item[0], reverse=True)
                for _, pt in components[:max_points]:
                    points.append(pt)
        except Exception:
            pass

        # Fallback if no components or scipy missing
        if len(points) == 0:
            if high_mask.any():
                y_idx, x_idx = np.where(high_mask)
                peak_idx = np.argmax(cam_map[y_idx, x_idx])
                points.append((float(x_idx[peak_idx]), float(y_idx[peak_idx])))
            else:
                max_flat = np.argmax(cam_map)
                py, px = divmod(int(max_flat), w)
                points.append((float(px), float(py)))

        return points[:max_points]

    def _find_negative_points(
        self,
        priority_map: np.ndarray,
        neg_mask: np.ndarray,
        max_points: int,
    ) -> List[Tuple[float, float]]:
        """Mines negative points from competing tissue or background glass."""
        points: List[Tuple[float, float]] = []
        h, w = priority_map.shape

        if neg_mask.any():
            y_indices, x_indices = np.where(neg_mask)
            scores = priority_map[y_indices, x_indices]
            sorted_order = np.argsort(scores)[::-1]

            step = max(1, len(sorted_order) // max_points)
            for k in range(0, min(len(sorted_order), max_points * step), step):
                sel = sorted_order[k]
                points.append((float(x_indices[sel]), float(y_indices[sel])))
                if len(points) >= max_points:
                    break

        if len(points) == 0:
            # Pick corner points as safe background fallback
            corners = [(0.0, 0.0), (float(w - 1), 0.0), (0.0, float(h - 1))]
            points.extend(corners[:max_points])

        return points[:max_points]


CAM_Prompt_Generator = CAMPromptGenerator


# ==============================================================================
# 3. Two-Way Cross-Attention Transformer with Residual Adapters
# ==============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron block for hypernetworks and prediction heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class AdaptedTwoWayAttentionBlock(nn.Module):
    """Two-way cross-attention block with embedded Residual Adapter Bottlenecks."""

    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        adapter_mid_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Self-attention on tokens
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # Cross-attention: tokens attending to image
        self.cross_attn_token_to_img = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # MLP on tokens
        self.mlp_token = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embedding_dim),
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        # Cross-attention: image attending to tokens
        self.cross_attn_img_to_token = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm4 = nn.LayerNorm(embedding_dim)

        # MLP on image
        self.mlp_img = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embedding_dim),
        )
        self.norm5 = nn.LayerNorm(embedding_dim)

        # Injected Residual Adapter Bottlenecks
        self.adapter_tokens = ResidualAdapterBottleneck(embedding_dim, adapter_mid_dim)
        self.adapter_image = ResidualAdapterBottleneck(embedding_dim, adapter_mid_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: Optional[torch.Tensor] = None,
        key_pe: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            queries: Token embeddings (B, N_tokens, C).
            keys: Image embeddings (B, N_image, C).
        """
        # 1. Token Self-Attention
        q = queries if query_pe is None else queries + query_pe
        attn_out, _ = self.self_attn(q, q, queries)
        queries = self.norm1(queries + attn_out)

        # 2. Token to Image Cross-Attention
        q = queries if query_pe is None else queries + query_pe
        k = keys if key_pe is None else keys + key_pe
        attn_out, _ = self.cross_attn_token_to_img(q, k, keys)
        queries = self.norm2(queries + attn_out)

        # 3. Token MLP + Adapter
        mlp_out = self.mlp_token(queries)
        queries = self.norm3(queries + mlp_out)
        queries = self.adapter_tokens(queries)

        # 4. Image to Token Cross-Attention
        q = queries if query_pe is None else queries + query_pe
        k = keys if key_pe is None else keys + key_pe
        attn_out, _ = self.cross_attn_img_to_token(k, q, queries)
        keys = self.norm4(keys + attn_out)

        # 5. Image MLP + Adapter
        mlp_out = self.mlp_img(keys)
        keys = self.norm5(keys + mlp_out)
        keys = self.adapter_image(keys)

        return queries, keys


# ==============================================================================
# 4. Multi-Class Histological Mask Decoder
# ==============================================================================

class HistologicalMaskDecoder(nn.Module):
    """5-Class Histological Mask Decoder with Residual Adapters and Parallel Prediction Heads.

    Classes:
        0: TUM (Tumor)
        1: STR (Stroma)
        2: LYM (Lymphocyte)
        3: NEC (Necrosis)
        4: BACK (Background / Glass / Other)
    """

    def __init__(
        self,
        transformer_dim: int = 256,
        num_classes: int = 5,
        num_layers: int = 2,
        adapter_mid_dim: int = 64,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_classes = num_classes

        # 1 IoU token + 5 histological class tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(num_classes, transformer_dim)

        # Stack of adapted two-way cross-attention layers
        self.layers = nn.ModuleList([
            AdaptedTwoWayAttentionBlock(
                embedding_dim=transformer_dim,
                num_heads=8,
                mlp_dim=2048,
                adapter_mid_dim=adapter_mid_dim,
            )
            for _ in range(num_layers)
        ])

        # Upscaling module: (B, 256, H, W) -> (B, 32, 4*H, 4*W)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(8, transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # 5 Parallel Hypernetwork MLPs (mapping class tokens to 32-dim weights)
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(num_classes)
        ])

        # Mask quality / IoU prediction head (outputting 5 scores)
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, num_classes, iou_head_depth, sigmoid_output=True
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,       # (B, 256, H_feat, W_feat)
        sparse_prompt_embeddings: torch.Tensor, # (B, N_pts, 256)
        dense_prompt_embeddings: Optional[torch.Tensor] = None, # (B, 256, H_feat, W_feat)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts 5-class mask logits and IoU quality scores.

        Returns:
            logits: Tensor of shape (B, 5, 4*H_feat, 4*W_feat).
            iou_pred: Tensor of shape (B, 5).
        """
        b, c, h, w = image_embeddings.shape

        if dense_prompt_embeddings is not None:
            image_features = image_embeddings + dense_prompt_embeddings
        else:
            image_features = image_embeddings

        # Flatten image features for transformer: (B, H*W, C)
        image_tokens = image_features.flatten(2).permute(0, 2, 1)

        # Concatenate IoU token + 5 Class tokens: shape (1, 6, 256) -> (B, 6, 256)
        tokens_all = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        tokens_all = tokens_all.unsqueeze(0).expand(b, -1, -1)

        # Append sparse prompt embeddings if present: (B, 6 + N_pts, 256)
        if sparse_prompt_embeddings is not None and sparse_prompt_embeddings.shape[1] > 0:
            prompt_and_class_tokens = torch.cat([tokens_all, sparse_prompt_embeddings], dim=1)
        else:
            prompt_and_class_tokens = tokens_all

        # Cross-attention iterations
        queries = prompt_and_class_tokens
        keys = image_tokens
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys)

        iou_token_out = queries[:, 0, :]
        class_tokens_out = queries[:, 1: 1 + self.num_classes, :]

        # Upscale image features: (B, 256, H, W) -> (B, 32, 4*H, 4*W)
        spatial_features = keys.permute(0, 2, 1).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(spatial_features)
        b_up, c_up, h_up, w_up = upscaled_embedding.shape

        # Parallel hypernetworks generate mask logits
        hyper_weights = []
        for i in range(self.num_classes):
            hyper_weights.append(self.output_hypernetworks_mlps[i](class_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_weights, dim=1)  # (B, 5, 32)

        # Matrix multiplication: (B, 5, 32) x (B, 32, H_up * W_up) -> (B, 5, H_up, W_up)
        logits = (hyper_in @ upscaled_embedding.view(b_up, c_up, h_up * w_up)).view(b_up, self.num_classes, h_up, w_up)

        # Predict per-class mask IoU scores
        iou_pred = self.iou_prediction_head(iou_token_out)

        return logits, iou_pred


# ==============================================================================
# 5. Lightweight Surrogate Image & Prompt Encoder for Offline Execution
# ==============================================================================

class SurrogateImageEncoder(nn.Module):
    """Lightweight 4-stage residual convolutional encoder matching ViT-B output shape (B, 256, H/16, W/16).

    Guarantees deterministic execution and exact shape parity on CPU and GPU
    without downloading 375 MB foundation weights.
    """

    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim

        # Stem downsample 2x
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # Stage 1 downsample 2x (total 4x)
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        # Stage 2 downsample 2x (total 8x)
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        # Stage 3 downsample 2x (total 16x)
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        # Final neck matching SAM LayerNorm
        self.neck = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.neck(out)
        return out


class SurrogatePromptEncoder(nn.Module):
    """Encodes sparse 2D point coordinates and labels into prompt tokens."""

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(4)  # labels: -1, 0, 1, 4
        ])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.coord_proj = nn.Linear(2, embed_dim)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            points: Tuple of (coords: (B, N, 2), labels: (B, N)).
        Returns:
            sparse_embeddings: Tensor of shape (B, N, 256).
            dense_embeddings: Optional dense mask embeddings.
        """
        if points is None:
            return torch.empty((1, 0, self.embed_dim)), None

        coords, labels = points
        b, n, _ = coords.shape

        # Positional coordinate projection
        coord_emb = self.coord_proj(coords)

        # Label embeddings: label 1 is pos, label 0 is neg, label -1 is pad
        label_weights = []
        for i in range(b):
            sample_lbl_emb = []
            for j in range(n):
                lbl = labels[i, j].item()
                if lbl == 1:
                    sample_lbl_emb.append(self.point_embeddings[1].weight[0])
                elif lbl == 0:
                    sample_lbl_emb.append(self.point_embeddings[0].weight[0])
                else:
                    sample_lbl_emb.append(self.not_a_point_embed.weight[0])
            label_weights.append(torch.stack(sample_lbl_emb, dim=0))

        label_emb = torch.stack(label_weights, dim=0).to(coords.device)
        sparse_embeddings = coord_emb + label_emb

        return sparse_embeddings, None


# ==============================================================================
# 6. Proposed MedSAM WSSS Adapter Model
# ==============================================================================

class MedSAM_WSSS_Adapter(nn.Module):
    """Medical SAM Architecture with Frozen Image Encoder and Residual Mask Decoder Adapters.

    Key Features:
        - Image Encoder (ViT-B or Surrogate): ~86M parameters, strictly frozen.
        - Residual Adapter Bottlenecks: Injected into cross-attention blocks.
        - Multi-Class Histological Mask Decoder: 5 classes (TUM, STR, LYM, NEC, BACK).
        - Parallel prediction heads with Softmax probabilities.
        - Graceful fallback to lightweight surrogate backbone for offline/unit-test resilience.
    """

    CLASS_NAMES: List[str] = ["TUM", "STR", "LYM", "NEC", "BACK"]
    NUM_CLASSES: int = 5

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        adapter_mid_dim: int = 64,
        num_classes: int = 5,
        use_surrogate_if_missing: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.adapter_mid_dim = adapter_mid_dim
        self.checkpoint_path = checkpoint_path

        # Dual-mode initialization: Official SAM if available and file exists, else Surrogate
        self.is_surrogate = True
        self.image_encoder: nn.Module
        self.prompt_encoder: nn.Module

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            try:
                from segment_anything import sam_model_registry
                sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
                self.image_encoder = sam_model.image_encoder
                self.prompt_encoder = sam_model.prompt_encoder
                self.is_surrogate = False
            except Exception:
                if not use_surrogate_if_missing:
                    raise
                self.image_encoder = SurrogateImageEncoder(out_dim=256)
                self.prompt_encoder = SurrogatePromptEncoder(embed_dim=256)
                self.is_surrogate = True
        else:
            self.image_encoder = SurrogateImageEncoder(out_dim=256)
            self.prompt_encoder = SurrogatePromptEncoder(embed_dim=256)
            self.is_surrogate = True

        # Custom 5-class histological mask decoder with Residual Adapters
        self.mask_decoder = HistologicalMaskDecoder(
            transformer_dim=256,
            num_classes=num_classes,
            num_layers=2,
            adapter_mid_dim=adapter_mid_dim,
        )

        # Freeze Image Encoder parameters
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # Freeze Prompt Encoder parameters
        if freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        dense_prompts: Optional[torch.Tensor] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass predicting 5-class segmentation logits and probabilities.

        Args:
            images: Tensor of shape (B, 3, H, W).
            point_coords: Optional tensor of shape (B, N, 2) in target pixel frame (x, y).
            point_labels: Optional tensor of shape (B, N) int64 (+1, 0, -1).
            dense_prompts: Optional dense mask embeddings (B, 256, H_feat, W_feat).
            target_size: Desired output resolution (H, W). Defaults to input image resolution.

        Returns:
            Dict containing:
                'logits': Tensor of shape (B, 5, H, W).
                'probabilities': Tensor of shape (B, 5, H, W) via Softmax.
                'iou_predictions': Tensor of shape (B, 5).
        """
        orig_h, orig_w = images.shape[-2:]
        out_size = target_size or (orig_h, orig_w)

        # 1. Image Encoder pass (under torch.no_grad() if frozen)
        if not any(p.requires_grad for p in self.image_encoder.parameters()):
            with torch.no_grad():
                image_embeddings = self.image_encoder(images)
        else:
            image_embeddings = self.image_encoder(images)

        # Ensure image embeddings match (B, 256, H_feat, W_feat)
        if image_embeddings.shape[1] != 256:
            raise ValueError(f"Expected 256 channels in image embeddings, got {image_embeddings.shape[1]}")

        # 2. Prompt Encoder pass
        sparse_embeddings = None
        if point_coords is not None and point_labels is not None:
            points_tuple = (point_coords, point_labels)
            sparse_embeddings, _ = self.prompt_encoder(points=points_tuple, masks=None)

        # 3. Mask Decoder with Residual Adapters
        low_res_logits, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_prompts,
        )

        # 4. Bilinear interpolation to target resolution
        logits = F.interpolate(
            low_res_logits,
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )

        # 5. Spatial Softmax across 5 histological classes
        probabilities = F.softmax(logits, dim=1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "iou_predictions": iou_predictions,
        }

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns only the trainable parameters (adapters, mask tokens, hypernetwork heads)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Dict[str, int]:
        """Calculates total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }


MedSAMWSSSAdapter = MedSAM_WSSS_Adapter
