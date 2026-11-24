"""
Stage-2 v2 Training Engine for Gate-5 4-Arm Controlled Breakthrough.

Supports all 4 architectures & training recipes:
- Differential LR (lr_backbone vs lr_head), 1D/bias/norm zero weight decay
- Per-step linear warmup + Poly / Cosine LR schedule
- BF16 AMP on sm >= 80 CUDA GPUs (FP32 fallback on CPU) + GradScaler for FP16
- Gradient clipping (max_norm=5.0)
- Stage2ModelEMA evaluated alongside raw weights every epoch (saves best by Val mIoU)
- Atomic .tmp + os.replace() CPU-cloned checkpoints (best + resume latest)
- Round-2 EMA self-training hook for Arm 3 (writes separate pm_soft_r2.dat)
- Final VAL-only CGM threshold calibration + untouched TEST multi-mode eval
- Academic monochrome loss/mIoU curve PNG + STAGE2_TRAINING_METRICS.json + ZIP export

Pure ASCII hyphen '-' only (Invariant 12).
"""

import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.models.wsss_inference_methods import WSSSTissueStage1NetCAM
from src.stage2_v2.config import Stage2V2Config, validate_track_name
from src.stage2_v2.evaluator import (
    calibrate_cgm_thresholds_val_only,
    evaluate_stage2_v2_model,
)
from src.stage2_v2.losses import Stage2V2CompositeLoss
from src.stage2_v2.memmap_dataset import (
    BCSSSoftMemmapDataset,
    generate_round2_ema_memmap,
)
from src.stage2_v2.models import (
    BaseStage2DualHead,
    Stage2ModelEMA,
    load_stage2_v2_checkpoint,
)
from src.training.dataset_loader import NUM_FOREGROUND
from src.training.eval_stage1_full import compute_file_sha256


def _atomic_torch_save(payload: Dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target_path)


def _atomic_write_json(payload: Dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, target_path)


def _resolve_memmap_source(
    cfg: Stage2V2Config,
    memmap_path: Union[str, Path],
    manifest_path: Union[str, Path],
) -> tuple[Path, Path, str, str]:
    memmap_source = str(getattr(cfg, "memmap_source", "stage1")).strip().lower()
    if memmap_source not in ("stage1", "ens4_r2"):
        raise ValueError(f"Unsupported memmap_source={memmap_source!r}. Expected 'stage1' or 'ens4_r2'.")
    track_norm = validate_track_name(str(getattr(cfg, "track", "track_a_clean")))
    is_track_a = track_norm == "track_a_clean"
    mm_p = Path(memmap_path)
    man_p = Path(manifest_path)
    if memmap_source == "ens4_r2":
        if cfg.self_train_round2_start_epoch is not None:
            raise ValueError(
                "memmap_source='ens4_r2' is incompatible with self_train_round2_start_epoch != None."
            )
        target_dat = "pm_soft_ens4_r2.dat" if is_track_a else "pm_soft_ens4_r2_trackb.dat"
        target_man = (
            "pm_soft_ens4_r2_manifest.json"
            if is_track_a
            else "pm_soft_ens4_r2_trackb_manifest.json"
        )
        if is_track_a:
            if mm_p.name == "pm_soft_s1.dat":
                mm_p = mm_p.parent / target_dat
            if man_p.name == "pm_soft_manifest.json":
                man_p = man_p.parent / target_man
        else:
            if mm_p.name in ("pm_soft_s1.dat", "pm_soft_ens4_r2.dat"):
                mm_p = mm_p.parent / target_dat
            if man_p.name in ("pm_soft_manifest.json", "pm_soft_ens4_r2_manifest.json"):
                man_p = man_p.parent / target_man
        r2_sha = ""
        if man_p.is_file():
            man_obj = json.loads(man_p.read_text(encoding="utf-8"))
            if is_track_a:
                man_track = str(man_obj.get("track", "")).strip().lower()
                if man_track != "track_a_clean":
                    raise ValueError(
                        f"[TRACK A QUARANTINE] Memmap manifest {man_p} has track={man_obj.get('track')!r}; expected 'track_a_clean'."
                    )
            r2_sha = compute_file_sha256(man_p)
        return mm_p, man_p, memmap_source, r2_sha
    return mm_p, man_p, memmap_source, ""


def _build_train_loader(
    cfg: Stage2V2Config,
    img_dir: Union[str, Path],
    memmap_path: Union[str, Path],
    manifest_path: Union[str, Path],
    patient_ids: Optional[Sequence[str]],
    strict_patient_filter: bool,
    max_train_samples: Optional[int],
    device: torch.device,
) -> DataLoader:
    ds = BCSSSoftMemmapDataset(
        img_dir=img_dir,
        memmap_path=memmap_path,
        manifest_path=manifest_path,
        patient_ids=patient_ids,
        strict_patient_filter=strict_patient_filter,
        is_train=True,
        max_samples=max_train_samples,
    )
    bs = min(int(cfg.batch_size), max(1, len(ds)))
    drop_last = len(ds) >= bs and (len(ds) % bs == 1)
    if len(ds) > int(cfg.batch_size):
        bs = int(cfg.batch_size)
        drop_last = True
    nw = int(cfg.num_workers)
    dl_kwargs: Dict[str, Any] = {}
    if nw > 0:
        dl_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": 2,
            "multiprocessing_context": "spawn",
        }
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=drop_last,
        **dl_kwargs,
    )


def _compute_lr_scale(
    step_idx: int,
    total_steps: int,
    warmup_steps: int,
    scheduler_type: str,
    poly_power: float,
    min_ratio: float,
) -> float:
    if warmup_steps > 0 and step_idx < warmup_steps:
        return float(step_idx + 1) / float(warmup_steps)
    denom = max(1, total_steps - warmup_steps)
    prog = min(1.0, max(0.0, float(step_idx - warmup_steps) / float(denom)))
    if str(scheduler_type).strip().lower() == "poly":
        return max(min_ratio, (1.0 - prog) ** float(poly_power))
    return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * prog)))


def _set_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    base_lrs: Sequence[float],
    scale: float,
) -> None:
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * float(scale)


def _save_monochrome_training_plot(history: Dict[str, List[Any]], plot_path: Path) -> None:
    epochs = history.get("epoch", [])
    if not epochs:
        return
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300, facecolor="#FFFFFF")
    for ax in (ax1, ax2):
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, linestyle="--", linewidth=0.5, color="#B0B0B0", alpha=0.8)

    ax1.plot(epochs, history["train_loss"], color="#000000", linestyle="-", marker="o", markersize=3, label="Train Loss")
    ax1.set_xlabel("Epoch", color="#000000")
    ax1.set_ylabel("Loss", color="#000000")
    ax1.set_title("Stage-2 v2 Training Loss", color="#000000")
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#000000")

    ax2.plot(epochs, history["val_raw_miou"], color="#000000", linestyle="-", marker="o", markersize=3, label="Raw Val mIoU")
    if len(history.get("val_ema_miou", [])) == len(epochs):
        ax2.plot(epochs, history["val_ema_miou"], color="#555555", linestyle="--", marker="s", markersize=3, label="EMA Val mIoU")
    ax2.set_xlabel("Epoch", color="#000000")
    ax2.set_ylabel("4-Class mIoU", color="#000000")
    ax2.set_title("Validation 4-Class mIoU", color="#000000")
    ax2.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#000000")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, facecolor="#FFFFFF")
    plt.close(fig)


def train_stage2_v2(
    cfg: Stage2V2Config,
    model: BaseStage2DualHead,
    train_img_dir: Union[str, Path],
    memmap_path: Union[str, Path],
    manifest_path: Union[str, Path],
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    stage1_model: Optional[WSSSTissueStage1NetCAM] = None,
    stage1_cls_model: Optional[WSSSTissueStage1NetCAM] = None,
    stage1_norm_mode: str = "zero_one",
    stage1_sha256: str = "",
    train_patient_ids: Optional[Sequence[str]] = None,
    checkpoint_dir: Union[str, Path] = "/content/checkpoints",
    output_dir: Union[str, Path] = "/content/stage2_results",
    zip_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    amp_dtype: Optional[torch.dtype] = None,
    max_train_samples: Optional[int] = None,
    strict_patient_filter: bool = True,
    resume_from: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run complete Stage-2 v2 training, EMA tracking, VAL CGM calibration & TEST eval."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    if stage1_model is not None:
        stage1_model = stage1_model.to(dev).eval()
    if stage1_cls_model is not None:
        stage1_cls_model = stage1_cls_model.to(dev).eval()

    ckpt_dir = Path(checkpoint_dir)
    out_dir = Path(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "stage2_retrained_best.pth"
    best_raw_ckpt_path = ckpt_dir / "stage2_best_raw.pth"
    latest_ckpt_path = ckpt_dir / "stage2_latest.pth"
    save_best_raw = bool(getattr(cfg, "save_best_raw_checkpoint", False))

    use_amp = dev.type == "cuda"
    if amp_dtype is None and use_amp:
        amp_dtype = (
            torch.bfloat16
            if (torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported(including_emulation=False))
            else torch.float16
        )
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    active_memmap, active_manifest, memmap_source, r2_manifest_sha256 = _resolve_memmap_source(
        cfg, memmap_path, manifest_path
    )
    train_loader = _build_train_loader(
        cfg=cfg,
        img_dir=train_img_dir,
        memmap_path=active_memmap,
        manifest_path=active_manifest,
        patient_ids=train_patient_ids,
        strict_patient_filter=strict_patient_filter,
        max_train_samples=max_train_samples,
        device=dev,
    )

    param_groups = model.get_param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if str(cfg.optimizer).strip().lower() == "sgd":
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            param_groups, lr=cfg.lr_head, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_groups, lr=cfg.lr_head, weight_decay=cfg.weight_decay
        )
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]

    criterion = Stage2V2CompositeLoss(cfg).to(dev)
    ema = Stage2ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, int(cfg.epochs) * steps_per_epoch)
    warmup_steps = max(0, int(round(float(cfg.warmup_epochs) * steps_per_epoch)))
    min_ratio = float(cfg.eta_min) / max(float(cfg.lr_head), 1e-8)

    start_epoch = 1
    best_val_miou = -1.0
    best_raw_val_miou = -1.0
    best_raw_epoch = 0
    best_selected_weights = "raw"
    no_improve = 0
    round2_active = False
    round2_step = 0
    lr_mult = 1.0
    global_step = 0
    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "val_raw_miou": [],
        "val_ema_miou": [],
        "val_cgm_s1_miou": [],
        "val_cgm_self_miou": [],
        "val_cgm_dual_miou": [],
        "selected_weights": [],
    }

    if resume_from is not None and Path(resume_from).is_file():
        r_obj = torch.load(resume_from, map_location="cpu", weights_only=False)
        if isinstance(r_obj, dict) and "state_dict" in r_obj:
            model.load_state_dict(r_obj["state_dict"], strict=True)
            if "optimizer" in r_obj:
                optimizer.load_state_dict(r_obj["optimizer"])
            if ema is not None and "ema_state_dict" in r_obj and r_obj["ema_state_dict"]:
                ema.load_state_dict(r_obj["ema_state_dict"], model=model)
            start_epoch = int(r_obj.get("epoch", 0)) + 1
            best_val_miou = float(r_obj.get("best_val_miou", -1.0))
            best_raw_val_miou = float(r_obj.get("best_raw_val_miou", -1.0))
            best_raw_epoch = int(r_obj.get("best_raw_epoch", 0))
            best_selected_weights = str(r_obj.get("selected_weights", "raw"))
            no_improve = int(r_obj.get("epochs_without_improve", 0))
            round2_active = bool(r_obj.get("round2_active", False))
            round2_step = int(r_obj.get("round2_step", 0))
            global_step = int(r_obj.get("global_step", (start_epoch - 1) * steps_per_epoch))
            if round2_active:
                lr_mult = float(cfg.self_train_lr_scale)
            if isinstance(r_obj.get("history"), dict):
                history = r_obj["history"]

    if round2_active:
        r2_dat = active_memmap.parent / "pm_soft_r2.dat"
        r2_man = active_memmap.parent / "pm_soft_r2_manifest.json"
        s1_dat = active_memmap.parent / "pm_soft_s1.dat" if active_memmap.name == "pm_soft_r2.dat" else active_memmap
        s1_man_p = (
            active_manifest.parent / "pm_soft_manifest.json"
            if active_manifest.name == "pm_soft_r2_manifest.json"
            else active_manifest
        )
        r2_valid = False
        if r2_dat.is_file() and r2_man.is_file() and s1_man_p.is_file():
            try:
                s1_man_obj = json.loads(s1_man_p.read_text(encoding="utf-8"))
                r2_man_obj = json.loads(r2_man.read_text(encoding="utf-8"))
                n_s1 = int(s1_man_obj.get("num_patches", -1))
                n_r2 = int(r2_man_obj.get("num_patches", -2))
                expected_r2_bytes = n_s1 * NUM_FOREGROUND * 224 * 224
                r2_valid = (
                    n_s1 > 0
                    and n_r2 == n_s1
                    and r2_man_obj.get("filenames") == s1_man_obj.get("filenames")
                    and r2_dat.stat().st_size >= expected_r2_bytes
                )
            except Exception:
                r2_valid = False
        if not r2_valid:
            if ema is not None:
                with ema.swap_to_ema(model):
                    generate_round2_ema_memmap(
                        ema_model=model,
                        train_img_dir=train_img_dir,
                        s1_memmap_path=s1_dat,
                        s1_manifest_path=s1_man_p,
                        output_dat_path=r2_dat,
                        output_manifest_path=r2_man,
                        blend=cfg.self_train_blend,
                        batch_size=cfg.batch_size,
                        device=dev,
                    )
            else:
                generate_round2_ema_memmap(
                    ema_model=model,
                    train_img_dir=train_img_dir,
                    s1_memmap_path=s1_dat,
                    s1_manifest_path=s1_man_p,
                    output_dat_path=r2_dat,
                    output_manifest_path=r2_man,
                    blend=cfg.self_train_blend,
                    batch_size=cfg.batch_size,
                    device=dev,
                )
        active_memmap = r2_dat
        active_manifest = r2_man
        train_loader = _build_train_loader(
            cfg=cfg,
            img_dir=train_img_dir,
            memmap_path=active_memmap,
            manifest_path=active_manifest,
            patient_ids=train_patient_ids,
            strict_patient_filter=strict_patient_filter,
            max_train_samples=max_train_samples,
            device=dev,
        )
        print(
            f"[ROUND2 RESUME] Bound train_loader to {r2_dat.name} "
            f"(regenerated={not r2_valid}, lr_scale={lr_mult})",
            flush=True,
        )

    print(
        f"[PARITY] Model={cfg.arch} | Preset={cfg.preset_name} | Epochs={cfg.epochs} | "
        f"Loss={cfg.loss_mode} | Pretrained=True | SoftMode={cfg.soft_mode}",
        flush=True,
    )

    model_meta = model.model_metadata()
    stop_reason = "completed"

    for epoch in range(start_epoch, int(cfg.epochs) + 1):
        # Arm 3 Round-2 EMA self-training transition
        if (
            cfg.self_train_round2_start_epoch is not None
            and not round2_active
            and (
                epoch >= int(cfg.self_train_round2_start_epoch)
                or no_improve >= int(cfg.patience)
            )
        ):
            if best_ckpt_path.is_file():
                best_ckpt = torch.load(best_ckpt_path, map_location=dev, weights_only=False)
                if isinstance(best_ckpt, dict) and "state_dict" in best_ckpt:
                    model.load_state_dict(best_ckpt["state_dict"], strict=True)
                    if ema is not None:
                        ema.load_state_dict({"shadow": model.state_dict()})
                    print(
                        f"[ROUND2] Reloaded peak Round-1 checkpoint (best_val_miou={best_ckpt.get('val_miou', best_val_miou)}, epoch={best_ckpt.get('epoch')})",
                        flush=True,
                    )
            optimizer.state.clear()
            r2_dat = active_memmap.parent / "pm_soft_r2.dat"
            r2_man = active_memmap.parent / "pm_soft_r2_manifest.json"
            if ema is not None:
                with ema.swap_to_ema(model):
                    generate_round2_ema_memmap(
                        ema_model=model,
                        train_img_dir=train_img_dir,
                        s1_memmap_path=active_memmap,
                        s1_manifest_path=active_manifest,
                        output_dat_path=r2_dat,
                        output_manifest_path=r2_man,
                        blend=cfg.self_train_blend,
                        batch_size=cfg.batch_size,
                        device=dev,
                    )
            else:
                generate_round2_ema_memmap(
                    ema_model=model,
                    train_img_dir=train_img_dir,
                    s1_memmap_path=active_memmap,
                    s1_manifest_path=active_manifest,
                    output_dat_path=r2_dat,
                    output_manifest_path=r2_man,
                    blend=cfg.self_train_blend,
                    batch_size=cfg.batch_size,
                    device=dev,
                )
            active_memmap = r2_dat
            active_manifest = r2_man
            train_loader = _build_train_loader(
                cfg=cfg,
                img_dir=train_img_dir,
                memmap_path=active_memmap,
                manifest_path=active_manifest,
                patient_ids=train_patient_ids,
                strict_patient_filter=strict_patient_filter,
                max_train_samples=max_train_samples,
                device=dev,
            )
            lr_mult = float(cfg.self_train_lr_scale)
            round2_active = True
            no_improve = 0
            round2_step = 0
            print(
                f"[ROUND2] Switched to blended EMA memmap {r2_dat.name} (blend={cfg.self_train_blend}, lr_scale={lr_mult})",
                flush=True,
            )

        model.train()
        ep_loss_sum = 0.0
        ep_batches = 0

        r2_warmup_steps = min(200, max(1, steps_per_epoch // 2))

        for imgs, q_soft, y_gt, px_weight in train_loader:
            lr_scale = _compute_lr_scale(
                step_idx=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                scheduler_type=cfg.scheduler,
                poly_power=cfg.poly_power,
                min_ratio=min_ratio,
            )
            if round2_active and round2_step < r2_warmup_steps:
                step_lr_mult = lr_mult * (0.1 + 0.9 * float(round2_step + 1) / float(r2_warmup_steps))
            else:
                step_lr_mult = lr_mult
            _set_optimizer_lrs(optimizer, base_lrs, lr_scale * step_lr_mult)

            imgs = imgs.to(dev, non_blocking=True)
            q_soft = q_soft.to(dev, non_blocking=True)
            y_gt = y_gt.to(dev, non_blocking=True)
            px_weight = px_weight.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(imgs)
                loss, _ = criterion(
                    outputs,
                    q_soft,
                    y_gt,
                    valid_weight=px_weight,
                    epoch=epoch - 1,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(f"[STAGE2 ERROR] Non-finite epoch loss at epoch {epoch}, step {global_step}.")

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            if ema is not None:
                ema.update(model)

            ep_loss_sum += float(loss.detach().item())
            ep_batches += 1
            global_step += 1
            if round2_active:
                round2_step += 1

        avg_train_loss = ep_loss_sum / float(max(1, ep_batches))

        # Evaluate raw and EMA on VAL
        raw_res = evaluate_stage2_v2_model(
            model=model,
            loader=val_loader,
            device=dev,
            stage1_model=stage1_model,
            stage1_norm_mode=stage1_norm_mode,
            stage1_cls_model=stage1_cls_model,
            cgm_thresholds=0.10,
            scales=cfg.eval_scales,
            run_ms_tta=False,
            amp_dtype=amp_dtype,
        )
        raw_miou = float(raw_res["raw"]["mIoU"])

        if ema is not None:
            with ema.swap_to_ema(model):
                ema_res = evaluate_stage2_v2_model(
                    model=model,
                    loader=val_loader,
                    device=dev,
                    stage1_model=stage1_model,
                    stage1_norm_mode=stage1_norm_mode,
                    stage1_cls_model=stage1_cls_model,
                    cgm_thresholds=0.10,
                    scales=cfg.eval_scales,
                    run_ms_tta=False,
                    amp_dtype=amp_dtype,
                )
            ema_miou: Optional[float] = float(ema_res["raw"]["mIoU"])
        else:
            ema_res = raw_res
            ema_miou = None

        force_ema_only = bool(
            bool(cfg.use_ema)
            and ema is not None
            and bool(getattr(cfg, "ema_only_best_selection", False))
            and ema_miou is not None
        )
        if ema is not None and ema_miou is not None and (force_ema_only or ema_miou >= raw_miou):
            sel_tag = "ema"
            ep_res = ema_res
            ep_miou = ema_miou
            export_sd = ema.clean_model_state_dict()
        else:
            sel_tag = "raw"
            ep_res = raw_res
            ep_miou = raw_miou
            export_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        ema_miou_log = ema_miou if ema_miou is not None else raw_miou
        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 5))
        history["val_raw_miou"].append(round(raw_miou, 4))
        history["val_ema_miou"].append(round(ema_miou_log, 4))
        history["val_cgm_s1_miou"].append(
            float(ep_res["cgm_s1"]["mIoU"]) if "cgm_s1" in ep_res else None
        )
        history["val_cgm_self_miou"].append(float(ep_res["cgm_self"]["mIoU"]))
        history["val_cgm_dual_miou"].append(
            float(ep_res["cgm_dual"]["mIoU"]) if "cgm_dual" in ep_res else None
        )
        history["selected_weights"].append(sel_tag)

        improved = ep_miou > best_val_miou + 1e-5
        if improved:
            best_val_miou = ep_miou
            best_selected_weights = sel_tag
            no_improve = 0
            best_payload: Dict[str, Any] = {
                **model_meta,
                "preset": cfg.preset_name,
                "seed": int(cfg.seed),
                "epoch": int(epoch),
                "selected_weights": sel_tag,
                "val_miou": round(float(best_val_miou), 4),
                "stage1_sha256": str(stage1_sha256),
                "memmap_source": memmap_source,
                "r2_manifest_sha256": r2_manifest_sha256,
                "state_dict": export_sd,
            }
            _atomic_torch_save(best_payload, best_ckpt_path)
        else:
            no_improve += 1

        if save_best_raw and raw_miou > best_raw_val_miou + 1e-5:
            best_raw_val_miou = raw_miou
            best_raw_epoch = int(epoch)
            best_raw_payload: Dict[str, Any] = {
                **model_meta,
                "preset": cfg.preset_name,
                "seed": int(cfg.seed),
                "epoch": int(epoch),
                "selected_weights": "raw",
                "val_miou": round(float(best_raw_val_miou), 4),
                "stage1_sha256": str(stage1_sha256),
                "memmap_source": memmap_source,
                "r2_manifest_sha256": r2_manifest_sha256,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            _atomic_torch_save(best_raw_payload, best_raw_ckpt_path)

        has_frozen_params = any(not p.requires_grad for p in model.parameters())
        latest_payload: Dict[str, Any] = {
            **model_meta,
            "preset": cfg.preset_name,
            "seed": int(cfg.seed),
            "epoch": int(epoch),
            "selected_weights": sel_tag,
            "best_val_miou": round(float(best_val_miou), 4),
            "best_raw_val_miou": round(float(best_raw_val_miou), 4),
            "best_raw_epoch": int(best_raw_epoch),
            "epochs_without_improve": int(no_improve),
            "round2_active": bool(round2_active),
            "round2_step": int(round2_step),
            "global_step": int(global_step),
            "stage1_sha256": str(stage1_sha256),
            "memmap_source": memmap_source,
            "r2_manifest_sha256": r2_manifest_sha256,
            "history": history,
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "ema_state_dict": (
                ema.state_dict(trainable_only=has_frozen_params, model=model)
                if ema is not None
                else None
            ),
        }
        _atomic_torch_save(latest_payload, latest_ckpt_path)

        print(
            f"[EPOCH {epoch:02d}] Train Loss={avg_train_loss:.4f} | Val_mIoU={ep_miou:.4f} | "
            f"Raw_mIoU={raw_miou:.4f} | EMA_mIoU={ema_miou_log:.4f} | Best={best_val_miou:.4f} | "
            f"Selected={sel_tag}",
            flush=True,
        )

        if no_improve >= int(cfg.patience):
            if cfg.self_train_round2_start_epoch is None or round2_active:
                stop_reason = "early_stop_patience"
                print(
                    f"[EARLY STOP PATIENCE] No Val_mIoU improvement for {no_improve} epochs; stopping cleanly.",
                    flush=True,
                )
                break
            if epoch < int(cfg.epochs):
                print(
                    "[ROUND2 SCHEDULED] Round-1 plateau reached; transitioning to Round-2 EMA self-training on next epoch.",
                    flush=True,
                )

    # Final Step-8 Strict Reload + Fixed-0.10 & VAL-Calibrated Evaluation
    best_model, best_meta = load_stage2_v2_checkpoint(best_ckpt_path, device=dev, strict=True)
    s1_expert = stage1_cls_model if stage1_cls_model is not None else stage1_model

    val_fixed = evaluate_stage2_v2_model(
        model=best_model,
        loader=val_loader,
        device=dev,
        stage1_model=stage1_model,
        stage1_norm_mode=stage1_norm_mode,
        stage1_cls_model=stage1_cls_model,
        cgm_thresholds=0.10,
        scales=cfg.eval_scales,
        run_ms_tta=True,
        amp_dtype=amp_dtype,
    )
    gate_src = "dual" if s1_expert is not None else "self"
    cal_info = calibrate_cgm_thresholds_val_only(
        model=best_model,
        val_loader=val_loader,
        device=dev,
        stage1_model=s1_expert,
        stage1_norm_mode=stage1_norm_mode,
        gate_source=gate_src,
        amp_dtype=amp_dtype,
    )
    cal_thresholds: List[float] = list(cal_info["thresholds"])
    val_calibrated = evaluate_stage2_v2_model(
        model=best_model,
        loader=val_loader,
        device=dev,
        stage1_model=stage1_model,
        stage1_norm_mode=stage1_norm_mode,
        stage1_cls_model=stage1_cls_model,
        cgm_thresholds=cal_thresholds,
        scales=cfg.eval_scales,
        run_ms_tta=True,
        amp_dtype=amp_dtype,
    )

    test_fixed: Optional[Dict[str, Any]] = None
    test_calibrated: Optional[Dict[str, Any]] = None
    if test_loader is not None:
        test_fixed = evaluate_stage2_v2_model(
            model=best_model,
            loader=test_loader,
            device=dev,
            stage1_model=stage1_model,
            stage1_norm_mode=stage1_norm_mode,
            stage1_cls_model=stage1_cls_model,
            cgm_thresholds=0.10,
            scales=cfg.eval_scales,
            run_ms_tta=True,
            amp_dtype=amp_dtype,
        )
        test_calibrated = evaluate_stage2_v2_model(
            model=best_model,
            loader=test_loader,
            device=dev,
            stage1_model=stage1_model,
            stage1_norm_mode=stage1_norm_mode,
            stage1_cls_model=stage1_cls_model,
            cgm_thresholds=cal_thresholds,
            scales=cfg.eval_scales,
            run_ms_tta=True,
            amp_dtype=amp_dtype,
        )

    best_raw_ablation: Optional[Dict[str, Any]] = None
    if save_best_raw and best_raw_ckpt_path.is_file():
        raw_eval_model, raw_eval_meta = load_stage2_v2_checkpoint(best_raw_ckpt_path, device=dev, strict=True)
        raw_val_fixed = evaluate_stage2_v2_model(
            model=raw_eval_model,
            loader=val_loader,
            device=dev,
            stage1_model=stage1_model,
            stage1_norm_mode=stage1_norm_mode,
            stage1_cls_model=stage1_cls_model,
            cgm_thresholds=0.10,
            scales=cfg.eval_scales,
            run_ms_tta=True,
            amp_dtype=amp_dtype,
        )
        raw_cal_info = calibrate_cgm_thresholds_val_only(
            model=raw_eval_model,
            val_loader=val_loader,
            device=dev,
            stage1_model=s1_expert,
            stage1_norm_mode=stage1_norm_mode,
            gate_source=gate_src,
            amp_dtype=amp_dtype,
        )
        raw_cal_thrs: List[float] = list(raw_cal_info["thresholds"])
        raw_val_cal = evaluate_stage2_v2_model(
            model=raw_eval_model,
            loader=val_loader,
            device=dev,
            stage1_model=stage1_model,
            stage1_norm_mode=stage1_norm_mode,
            stage1_cls_model=stage1_cls_model,
            cgm_thresholds=raw_cal_thrs,
            scales=cfg.eval_scales,
            run_ms_tta=True,
            amp_dtype=amp_dtype,
        )
        raw_test_fixed: Optional[Dict[str, Any]] = None
        raw_test_cal: Optional[Dict[str, Any]] = None
        if test_loader is not None:
            raw_test_fixed = evaluate_stage2_v2_model(
                model=raw_eval_model,
                loader=test_loader,
                device=dev,
                stage1_model=stage1_model,
                stage1_norm_mode=stage1_norm_mode,
                stage1_cls_model=stage1_cls_model,
                cgm_thresholds=0.10,
                scales=cfg.eval_scales,
                run_ms_tta=True,
                amp_dtype=amp_dtype,
            )
            raw_test_cal = evaluate_stage2_v2_model(
                model=raw_eval_model,
                loader=test_loader,
                device=dev,
                stage1_model=stage1_model,
                stage1_norm_mode=stage1_norm_mode,
                stage1_cls_model=stage1_cls_model,
                cgm_thresholds=raw_cal_thrs,
                scales=cfg.eval_scales,
                run_ms_tta=True,
                amp_dtype=amp_dtype,
            )
        best_raw_ablation = {
            "best_raw_epoch": int(raw_eval_meta.get("epoch") or best_raw_epoch),
            "best_raw_val_miou": round(float(best_raw_val_miou), 4),
            "best_raw_checkpoint": str(best_raw_ckpt_path),
            "val_eval_fixed_0_10": raw_val_fixed,
            "val_cgm_calibration": raw_cal_info,
            "val_eval_calibrated": raw_val_cal,
            "test_eval_fixed_0_10": raw_test_fixed,
            "test_eval_calibrated": raw_test_cal,
        }

    metrics_payload: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "best_epoch": int(best_meta.get("epoch") or 0),
        "best_val_miou": round(float(best_val_miou), 4),
        "best_selected_weights": best_selected_weights,
        "stop_reason": stop_reason,
        "stage1_sha256": str(stage1_sha256),
        "stage1_norm_mode": stage1_norm_mode,
        "memmap_source": memmap_source,
        "r2_manifest_sha256": r2_manifest_sha256,
        "history": history,
        "val_eval_fixed_0_10": val_fixed,
        "val_cgm_calibration": cal_info,
        "val_eval_calibrated": val_calibrated,
        "test_eval_fixed_0_10": test_fixed,
        "test_eval_calibrated": test_calibrated,
        "best_checkpoint": str(best_ckpt_path),
    }
    if best_raw_ablation is not None:
        metrics_payload["best_raw_ablation"] = best_raw_ablation

    metrics_json_path = out_dir / "STAGE2_TRAINING_METRICS.json"
    _atomic_write_json(metrics_payload, metrics_json_path)

    plot_path = out_dir / "stage2_training_loss.png"
    _save_monochrome_training_plot(history, plot_path)

    if best_ckpt_path.is_file():
        shutil.copy2(best_ckpt_path, out_dir / best_ckpt_path.name)

    target_zip = (
        Path(zip_path)
        if zip_path is not None
        else (out_dir.parent / "SE374_Stage2_Training_Results.zip")
    )
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = target_zip.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(out_dir.glob("*")):
            if item.is_file() and item.name != "stage2_latest.pth":
                zf.write(item, arcname=item.name)
    os.replace(tmp_zip, target_zip)

    return metrics_payload
