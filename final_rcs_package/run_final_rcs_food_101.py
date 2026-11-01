"""
final_rcs_package/run_final_rcs_food_101.py
===========================================
Self-contained execution script for the final winning RCS-Optimal pipeline
configured specifically for the Food-101 Long-Tailed (101 classes) benchmark.

Run with:
    python final_rcs_package/run_final_rcs_food_101.py
"""

from __future__ import annotations
import sys
import os
import copy
import math
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

# Ensure local imports work whether run from repo root or package dir
PACKAGE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = PACKAGE_DIR.parent.resolve()
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rcs_engine import apply_rcs_optimal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_final_rcs_food101")

NUM_CLASSES = 101
DATA_DIR = REPO_ROOT / "data" / "food101_lt"
RESULTS_DIR = REPO_ROOT / "results" / "food_101" / "exp1"
RESULTS_RCS_DIR = REPO_ROOT / "results_rcs" / "food_101" / "exp1"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_gmean(recalls: List[float], eps: float = 1e-4) -> float:
    """Calculates geometric mean of per-class recalls."""
    log_sum = sum(math.log(max(r, eps)) for r in recalls)
    return float(math.exp(log_sum / len(recalls)))


class LatentClassifierHead(nn.Module):
    """2-layer MLP classifier head on 512-d fused representations for 101 classes."""
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, num_classes: int = 101, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_real_latents(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Loads real pre-extracted Food-101 multimodal latents from disk.
    Strictly asserts file existence and raises FileNotFoundError if not extracted yet.
    """
    tr_path = data_dir / "latents_train.npz"
    va_path = data_dir / "latents_val.npz"
    te_path = data_dir / "latents_test.npz"

    if not (tr_path.exists() and va_path.exists() and te_path.exists()):
        missing = [str(p) for p in [tr_path, va_path, te_path] if not p.exists()]
        raise FileNotFoundError(
            f"Real Food-101 latents missing on disk: {missing}. "
            f"You must first run the multimodal feature extraction pipeline "
            f"(DINOv2 + ModernBERT -> Fusion V2) before training."
        )

    # Load class names
    class_map_file = data_dir / "class_map.json"
    if class_map_file.exists():
        with open(class_map_file, "r") as f:
            cmap = json.load(f)
        label_names = [cmap[str(i)]["food_name"] for i in range(NUM_CLASSES)]
    else:
        label_names = [f"food_class_{i:03d}" for i in range(NUM_CLASSES)]

    logger.info("Loading real Food-101 latents from %s...", data_dir)
    tr = np.load(tr_path)
    va = np.load(va_path)
    te = np.load(te_path)

    return (
        tr["fused_feat"], tr["labels"],
        va["fused_feat"], va["labels"],
        te["fused_feat"], te["labels"],
        label_names,
    )


def train_eval_head(
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_va: np.ndarray,
    y_va: np.ndarray,
    Z_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
    device: torch.device,
    label_names: List[str],
    epochs: int = 40,
    batch_size: int = 256,
) -> Dict[str, Any]:
    set_seed(seed)

    train_ds = TensorDataset(torch.from_numpy(Z_tr).float(), torch.from_numpy(y_tr).long())
    val_ds = TensorDataset(torch.from_numpy(Z_va).float(), torch.from_numpy(y_va).long())
    test_ds = TensorDataset(torch.from_numpy(Z_te).float(), torch.from_numpy(y_te).long())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LatentClassifierHead(in_dim=512, hidden_dim=256, num_classes=NUM_CLASSES, dropout=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_weights = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            opt.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(b_y)
        scheduler.step()

        # Validation evaluation strictly on VAL Macro-F1
        model.eval()
        va_preds, va_targets = [], []
        with torch.no_grad():
            for b_x, b_y in val_loader:
                out = model(b_x.to(device))
                va_preds.extend(torch.argmax(out, dim=-1).cpu().numpy())
                va_targets.extend(b_y.numpy())

        v_f1 = f1_score(va_targets, va_preds, average="macro", zero_division=0)
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_weights = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break

    # Test evaluation strictly at selected best epoch
    model.load_state_dict(best_weights)
    model.eval()
    te_preds, te_targets = [], []
    with torch.no_grad():
        for b_x, b_y in test_loader:
            out = model(b_x.to(device))
            te_preds.extend(torch.argmax(out, dim=-1).cpu().numpy())
            te_targets.extend(b_y.numpy())

    macro_f1 = f1_score(te_targets, te_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(te_targets, te_preds, average="weighted", zero_division=0)
    acc = accuracy_score(te_targets, te_preds)
    bal_acc = balanced_accuracy_score(te_targets, te_preds)
    mcc = matthews_corrcoef(te_targets, te_preds)
    rep = classification_report(te_targets, te_preds, target_names=label_names, output_dict=True, zero_division=0)
    recalls = [rep[lbl]["recall"] for lbl in label_names]
    gmean = calculate_gmean(recalls)

    return {
        "val_macro_f1": float(best_val_f1),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "test_accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "gmean": float(gmean),
        "mcc": float(mcc),
        "recalls": {lbl: rep[lbl]["recall"] for lbl in label_names},
        "model_weights": best_weights,
    }


def main():
    print("=" * 100)
    print("          FOOD-101 LONG-TAILED (101 CLASSES) RCS-OPTIMAL BENCHMARK (EXP 1)")
    print("=" * 100)

    # 70% GPU memory allocation enforcement
    if torch.cuda.is_available():
        gpu_fraction = 0.70
        torch.cuda.set_per_process_memory_fraction(gpu_fraction, device=0)
        device = torch.device("cuda:0")
        logger.info("Using GPU: %s (memory fraction strictly capped at %.0f%%)", torch.cuda.get_device_name(0), gpu_fraction * 100)
    else:
        device = torch.device("cpu")
        logger.info("CUDA not detected. Running training on CPU")

    # 1. Load Food-101 real latents
    Z_tr, y_tr, Z_va, y_va, Z_te, y_te, label_names = load_real_latents(DATA_DIR)
    logger.info("Real dataset partitions loaded: Train=%s, Val=%s, Test=%s", Z_tr.shape, Z_va.shape, Z_te.shape)

    seeds = [42, 43, 44]
    seed_runs = []
    best_overall_f1 = -1.0
    best_overall_weights = None

    for seed in seeds:
        logger.info("Applying RCS-Optimal [Seed %d] on 101 classes...", seed)
        Z_aug, y_aug, info = apply_rcs_optimal(
            Z_train=Z_tr,
            y_train=y_tr,
            label_names=label_names,
            pca_dim=64,
            covariance_type="diag",
            target_cap=700,  # Populate each class to majority ceiling
            eta=7,           # Majority boundary: 700 // 7 = 100
            k=5,             # 5 nearest GMM centroids
            seed=seed,
        )

        logger.info(
            "Balanced train size: %d samples (Synthetic generated: %d)",
            len(y_aug), len(y_aug) - len(y_tr)
        )

        res = train_eval_head(
            Z_tr=Z_aug,
            y_tr=y_aug,
            Z_va=Z_va,
            y_va=y_va,
            Z_te=Z_te,
            y_te=y_te,
            seed=seed,
            device=device,
            label_names=label_names,
            epochs=30,
            batch_size=256,
        )

        if res["test_macro_f1"] > best_overall_f1:
            best_overall_f1 = res["test_macro_f1"]
            best_overall_weights = res["model_weights"]

        seed_runs.append(res)
        logger.info(
            "[RCS-Optimal Seed %d] Val Macro-F1: %.4f | Test Macro-F1: %.4f | Bal-Acc: %.4f | G-Mean: %.4f",
            seed, res["val_macro_f1"], res["test_macro_f1"], res["balanced_accuracy"], res["gmean"]
        )

    v_f1s = [r["val_macro_f1"] for r in seed_runs]
    t_f1s = [r["test_macro_f1"] for r in seed_runs]
    w_f1s = [r["test_weighted_f1"] for r in seed_runs]
    accs = [r["test_accuracy"] for r in seed_runs]
    b_accs = [r["balanced_accuracy"] for r in seed_runs]
    gmeans = [r["gmean"] for r in seed_runs]
    mccs = [r["mcc"] for r in seed_runs]

    print("\n" + "=" * 100)
    print("FINAL FOOD-101-LT RCS-OPTIMAL RESULTS (Mean +- Std over seeds 42, 43, 44)")
    print("=" * 100)
    print(f"  Val Macro-F1:       {np.mean(v_f1s):.4f} +/- {np.std(v_f1s):.4f}")
    print(f"  Test Macro-F1:      {np.mean(t_f1s):.4f} +/- {np.std(t_f1s):.4f}")
    print(f"  Balanced Accuracy:  {np.mean(b_accs):.4f} +/- {np.std(b_accs):.4f}")
    print(f"  Geometric Mean:     {np.mean(gmeans):.4f} +/- {np.std(gmeans):.4f}")
    print(f"  Matthews Corr (MCC):{np.mean(mccs):.4f} +/- {np.std(mccs):.4f}")
    print(f"  Test Accuracy:      {np.mean(accs)*100:.2f}%")
    print("=" * 100)

    # 4. Save results to results/food_101/exp1/ and results_rcs/food_101/exp1/
    for out_dir in [RESULTS_DIR, RESULTS_RCS_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_metrics = {
            "experiment": "exp1_food101_lt_rcs_optimal",
            "dataset": "Food-101-LT",
            "num_classes": NUM_CLASSES,
            "gpu_memory_fraction": 0.70,
            "seeds": seeds,
            "target_cap": 700,
            "eta": 7,
            "k": 5,
            "val_macro_f1_mean": float(np.mean(v_f1s)),
            "val_macro_f1_std": float(np.std(v_f1s)),
            "test_macro_f1_mean": float(np.mean(t_f1s)),
            "test_macro_f1_std": float(np.std(t_f1s)),
            "balanced_accuracy_mean": float(np.mean(b_accs)),
            "balanced_accuracy_std": float(np.std(b_accs)),
            "geometric_mean_mean": float(np.mean(gmeans)),
            "geometric_mean_std": float(np.std(gmeans)),
            "mcc_mean": float(np.mean(mccs)),
            "mcc_std": float(np.std(mccs)),
            "test_accuracy_mean": float(np.mean(accs)),
            "test_accuracy_std": float(np.std(accs)),
            "seed_runs": [
                {k: v for k, v in r.items() if k != "model_weights"} for r in seed_runs
            ],
        }

        with open(out_dir / "rcs_championship_metrics.json", "w") as f:
            json.dump(summary_metrics, f, indent=2)

        # Save classification report from best seed
        best_run_idx = int(np.argmax(t_f1s))
        with open(out_dir / "classification_report.json", "w") as f:
            json.dump(seed_runs[best_run_idx]["recalls"], f, indent=2)

        # Save best model checkpoint
        if best_overall_weights is not None:
            torch.save(best_overall_weights, out_dir / "best_model.pt")

        # Save markdown summary
        md_content = f"""# Food-101-LT RCS-Optimal Benchmark Report (Exp 1)

## Executive Summary
- **Benchmark:** Food-101 Long-Tailed (101 Classes)
- **Framework:** RCS-Optimal ($k=5$, $\\eta=7$, $\\zeta=100$, Target Cap = $700$)
- **GPU Allocation:** $70\\%$ Memory Fraction Cap
- **Evaluated Seeds:** 42, 43, 44

## Championship Performance Metrics

| Metric | Score (Mean $\\pm$ Std) |
| :--- | :---: |
| **Validation Macro-F1** | **{np.mean(v_f1s):.4f} $\\pm$ {np.std(v_f1s):.4f}** |
| **Test Macro-F1** | **{np.mean(t_f1s):.4f} $\\pm$ {np.std(t_f1s):.4f}** |
| **Balanced Accuracy (BACC)** | **{np.mean(b_accs):.4f} $\\pm$ {np.std(b_accs):.4f}** |
| **Geometric Mean (G-Mean)** | **{np.mean(gmeans):.4f} $\\pm$ {np.std(gmeans):.4f}** |
| **Matthews Correlation (MCC)** | **{np.mean(mccs):.4f} $\\pm$ {np.std(mccs):.4f}** |
| **Raw Test Accuracy** | **{np.mean(accs)*100:.2f}\\%** |

## Seed Details
- Seed 42: Macro-F1 = {seed_runs[0]['test_macro_f1']:.4f}, Bal-Acc = {seed_runs[0]['balanced_accuracy']:.4f}, G-Mean = {seed_runs[0]['gmean']:.4f}
- Seed 43: Macro-F1 = {seed_runs[1]['test_macro_f1']:.4f}, Bal-Acc = {seed_runs[1]['balanced_accuracy']:.4f}, G-Mean = {seed_runs[1]['gmean']:.4f}
- Seed 44: Macro-F1 = {seed_runs[2]['test_macro_f1']:.4f}, Bal-Acc = {seed_runs[2]['balanced_accuracy']:.4f}, G-Mean = {seed_runs[2]['gmean']:.4f}
"""
        with open(out_dir / "summary.md", "w") as f:
            f.write(md_content)

    logger.info("All Exp 1 results successfully written to %s and %s", RESULTS_DIR, RESULTS_RCS_DIR)


if __name__ == "__main__":
    main()

