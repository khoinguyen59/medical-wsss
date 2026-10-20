"""
Comparative Benchmarking Runner for SE374.R11 Medical AI.
Executes rigorous, multi-seed comparative experiments across the 4 benchmark configurations:
1. Configuration 1: Vanilla CAM Baseline
2. Configuration 2: Data-Centric (+ ClassMix)
3. Configuration 3: Energy Relocalization (ARML-inspired)
4. Configuration 4: Proposed Adaptive Pipeline (Otsu Gating + Dynamic Thresholding + Uncertainty Regularization)

Strictly adheres to:
- Zero Mockup / Academic Integrity: True computation from PyTorch modules.
- Multi-Seed Reproducibility: Evaluates across 5 seeds and calculates mean +/- std.
- Strict ASCII hyphens (-) only.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.bcss_dataset import BCSSDataset
from src.losses_metrics.metrics import MedicalSegmentationMetrics
from src.models.wsss_tissue import WSSSBenchmarkingEngine


DEFAULT_SEEDS: List[int] = [42, 100, 2024, 2025, 2026]
CONFIGURATIONS: List[str] = ["baseline", "classmix", "relocalization", "proposed"]


def set_seed(seed: int) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_single_run(
    engine: WSSSBenchmarkingEngine,
    dataset: BCSSDataset,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Runs inference on dataset using the specified engine configuration."""
    metrics = MedicalSegmentationMetrics(
        num_classes=5,
        class_names=["TUM", "STR", "LYM", "NEC", "BACK"],
    )

    num_eval = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    start_time = time.time()

    for idx in range(num_eval):
        # Unpack from BCSSDataset valid/test split
        item = dataset[idx]
        if len(item) == 3:
            img_tensor, mask_tensor, _ = item
        else:
            continue

        # Format image for inference
        img_input = img_tensor.unsqueeze(0)  # (1, 3, H, W)
        raw_img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        # Generate prediction mask
        pred_mask = engine.generate_pseudo_mask(
            image_tensor=img_input,
            current_epoch=5,
            raw_image=raw_img_np,
        )

        gt_mask = mask_tensor.numpy()
        metrics.update(pred=pred_mask, target=gt_mask)

    elapsed_time = time.time() - start_time
    summary = metrics.get_summary_dict()
    summary["fps"] = float(num_eval / max(elapsed_time, 1e-4))
    summary["elapsed_seconds"] = float(elapsed_time)
    return summary


def run_comparative_benchmark(
    data_root: str = "data/sample_dataset",
    output_dir: str = "experiments/benchmarks",
    seeds: Optional[List[int]] = None,
    configurations: Optional[List[str]] = None,
    dry_run_samples: Optional[int] = 10,
    device: str = "cpu",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Runs the 4-configuration comparative benchmark across multi-seed runs.
    
    Args:
        data_root: Root directory of BCSS-WSSS dataset.
        output_dir: Destination directory for JSON logs and markdown summaries.
        seeds: List of random seeds for multi-seed reproducibility.
        configurations: Subset of CONFIGURATIONS to test.
        dry_run_samples: Maximum number of patches to evaluate per configuration (for quick verification).
        device: 'cpu' or 'cuda'.
    """
    seeds = seeds or DEFAULT_SEEDS
    configurations = configurations or CONFIGURATIONS
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = BCSSDataset(root_dir=data_root, split="valid")
    all_results: Dict[str, List[Dict[str, float]]] = {cfg: [] for cfg in configurations}

    for cfg in configurations:
        print(f"=== Starting Benchmark Evaluation: Configuration [{cfg}] ===")
        for seed in seeds:
            set_seed(seed)
            engine = WSSSBenchmarkingEngine(mode=cfg, num_classes=5, device=device)
            run_metric = evaluate_single_run(engine, test_dataset, max_samples=dry_run_samples)
            all_results[cfg].append(run_metric)
            print(f"  Seed {seed} -> mIoU: {run_metric['mIoU']:.2f}%, mDice: {run_metric['mDice']:.2f}%")

    # Aggregate mean and standard deviation
    aggregated_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    metric_keys = [
        "mIoU",
        "mDice",
        "IoU_TUM",
        "IoU_STR",
        "IoU_LYM",
        "IoU_NEC",
        "IoU_BACK",
        "Dice_TUM",
        "Dice_STR",
        "Dice_LYM",
        "Dice_NEC",
        "Dice_BACK",
        "fps",
    ]

    for cfg in configurations:
        aggregated_results[cfg] = {}
        for key in metric_keys:
            vals = [res[key] for res in all_results[cfg]]
            aggregated_results[cfg][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    # Save detailed JSON log
    log_file = os.path.join(output_dir, "benchmark_results.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw_runs": all_results,
                "aggregated": aggregated_results,
                "seeds": seeds,
                "configurations": configurations,
                "evaluated_samples": dry_run_samples,
            },
            f,
            indent=2,
        )

    # Format Markdown Report
    md_file = os.path.join(output_dir, "BENCHMARK_REPORT.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# SE374.R11 Comparative Benchmarking Results\n\n")
        f.write(f"- Multi-Seed Runs: {seeds}\n")
        f.write(f"- Samples per Run: {dry_run_samples}\n\n")
        f.write("| Configuration | mIoU (%) | mDice (%) | TUM IoU (%) | STR IoU (%) | LYM IoU (%) | NEC IoU (%) | BACK IoU (%) | FPS |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for cfg in configurations:
            stats = aggregated_results[cfg]
            f.write(
                f"| **{cfg.upper()}** | "
                f"{stats['mIoU']['mean']:.2f} +/- {stats['mIoU']['std']:.2f} | "
                f"{stats['mDice']['mean']:.2f} +/- {stats['mDice']['std']:.2f} | "
                f"{stats['IoU_TUM']['mean']:.2f} +/- {stats['IoU_TUM']['std']:.2f} | "
                f"{stats['IoU_STR']['mean']:.2f} +/- {stats['IoU_STR']['std']:.2f} | "
                f"{stats['IoU_LYM']['mean']:.2f} +/- {stats['IoU_LYM']['std']:.2f} | "
                f"{stats['IoU_NEC']['mean']:.2f} +/- {stats['IoU_NEC']['std']:.2f} | "
                f"{stats['IoU_BACK']['mean']:.2f} +/- {stats['IoU_BACK']['std']:.2f} | "
                f"{stats['fps']['mean']:.1f} |\n"
            )

    print(f"Benchmark completed successfully! Logs saved to {log_file} and {md_file}")
    return aggregated_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SE374.R11 Comparative Benchmarking Runner")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/sample_dataset",
        help="Root directory of BCSS-WSSS dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/benchmarks",
        help="Output directory for benchmark logs",
    )
    parser.add_argument(
        "--dry-run-samples",
        type=int,
        default=None,
        help="Maximum samples per run (None for full dataset)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (cpu or cuda)",
    )
    args = parser.parse_args()

    run_comparative_benchmark(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dry_run_samples=args.dry_run_samples,
        device=args.device,
    )
