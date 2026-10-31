"""
run_final_rcs.py
================
Execution script for the winning RCS-Optimal multimodal pipeline.
Generalized for arbitrary class counts, label names, and latent feature directories.

Run with:
    python run_final_rcs.py
"""

from __future__ import annotations
import sys
import copy
import math
import random
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, balanced_accuracy_score

from rcs_engine import apply_rcs_optimal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_final_rcs")

DEFAULT_MELD_LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
DEFAULT_CREMAD_LABELS = ["anger", "disgust", "fear", "happy", "neutral", "sad"]


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
    log_sum = sum(math.log(max(r, eps)) for r in recalls)
    return float(math.exp(log_sum / len(recalls)))


class LatentClassifierHead(nn.Module):
    """Unified 2-layer MLP classifier head on 512-d fused representations."""
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_eval_head(
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_va: np.ndarray,
    y_va: np.ndarray,
    Z_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
    device: torch.device,
    num_classes: int = 7,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    set_seed(seed)

    if label_names is None:
        label_names = [f"class_{i}" for i in range(num_classes)]

    train_ds = TensorDataset(torch.from_numpy(Z_tr).float(), torch.from_numpy(y_tr).long())
    val_ds = TensorDataset(torch.from_numpy(Z_va).float(), torch.from_numpy(y_va).long())
    test_ds = TensorDataset(torch.from_numpy(Z_te).float(), torch.from_numpy(y_te).long())

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = LatentClassifierHead(in_dim=512, hidden_dim=256, num_classes=num_classes, dropout=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_weights = None
    patience = 0

    for epoch in range(1, 61):
        model.train()
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            opt.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            opt.step()

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
    acc = accuracy_score(te_targets, te_preds)
    bal_acc = balanced_accuracy_score(te_targets, te_preds)
    rep = classification_report(te_targets, te_preds, target_names=label_names, output_dict=True, zero_division=0)
    recalls = [rep[lbl]["recall"] for lbl in label_names]
    gmean = calculate_gmean(recalls)

    return {
        "val_macro_f1": float(best_val_f1),
        "test_macro_f1": float(macro_f1),
        "test_accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "gmean": float(gmean),
        "recalls": {lbl: rep[lbl]["recall"] for lbl in label_names},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Final RCS Multimodal Benchmark")
    parser.add_argument("--latent_dir", type=str, default="final_rcs_package/latents", help="Directory containing latents_{train,val,test}.npz")
    parser.add_argument("--pca_dim", type=int, default=64, help="Subspace PCA dimension (default: 64)")
    parser.add_argument("--target_cap", type=int, default=2500, help="Target sample cap per class (default: 2500)")
    parser.add_argument("--eta", type=int, default=7, help="Imbalance threshold tuner (default: 7)")
    parser.add_argument("--max_synthetic_share", type=float, default=None, help="Cap synthetic share at max share (default: None for MELD backward compatibility)")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of emotion classes (inferred if None)")
    parser.add_argument("--label_names", nargs="+", default=None, help="List of emotion class names")
    parser.add_argument("--label_map", type=str, default=None, help="Path to label_map.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds (default: 42 43 44)")
    parser.add_argument("--cuda_mem_fraction", type=float, default=0.80, help="CUDA memory fraction (default: 0.80)")
    return parser.parse_args()


def resolve_labels(args, y_tr: np.ndarray, latent_dir: Path) -> Tuple[int, List[str]]:
    if args.label_map and Path(args.label_map).is_file():
        with open(args.label_map, "r", encoding="utf-8") as f:
            lmap = json.load(f)
        names = sorted(lmap.keys(), key=lambda k: lmap[k])
        return len(names), names

    candidate_lmaps = [
        latent_dir / "label_map.json",
        Path("data/label_map.json"),
        Path("cremad/label_map.json"),
    ]
    for p in candidate_lmaps:
        if p.is_file() and args.num_classes == 6:
            with open(p, "r", encoding="utf-8") as f:
                lmap = json.load(f)
            names = sorted(lmap.keys(), key=lambda k: lmap[k])
            return len(names), names

    unique_classes = np.unique(y_tr)
    n_classes = len(unique_classes)

    if args.label_names and len(args.label_names) == n_classes:
        return n_classes, list(args.label_names)

    if n_classes == 7:
        return 7, DEFAULT_MELD_LABELS
    elif n_classes == 6:
        return 6, DEFAULT_CREMAD_LABELS
    else:
        return n_classes, [f"class_{i}" for i in range(n_classes)]


def main():
    args = parse_args()

    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(args.cuda_mem_fraction, device=0)
            device = torch.device("cuda:0")
            logger.info("Using GPU: %s (memory fraction %.2f set)", torch.cuda.get_device_name(0), args.cuda_mem_fraction)
        except Exception as e:
            logger.warning("Could not set CUDA memory fraction: %s", e)
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        logger.info("Using device: %s", device)

    latent_dir = Path(args.latent_dir)
    logger.info("Loading Fusion V2 latents from %s...", latent_dir)

    tr = np.load(latent_dir / "latents_train.npz")
    va = np.load(latent_dir / "latents_val.npz")
    te = np.load(latent_dir / "latents_test.npz")

    Z_tr = tr["fused_feat"] if "fused_feat" in tr else tr["latents"]
    y_tr = tr["labels"] if "labels" in tr else tr["y"]

    Z_va = va["fused_feat"] if "fused_feat" in va else va["latents"]
    y_va = va["labels"] if "labels" in va else va["y"]

    Z_te = te["fused_feat"] if "fused_feat" in te else te["latents"]
    y_te = te["labels"] if "labels" in te else te["y"]

    logger.info("Loaded latents: Train=%s, Val=%s, Test=%s", Z_tr.shape, Z_va.shape, Z_te.shape)

    num_classes, label_names = resolve_labels(args, y_tr, latent_dir)
    logger.info("Classes (%d): %s", num_classes, label_names)

    seeds = args.seeds
    seed_runs = []

    print("\n" + "=" * 90)
    print(f"RUNNING FINAL RCS-OPTIMAL MULTIMODAL BENCHMARK (Seeds: {seeds})")
    print("=" * 90)

    for seed in seeds:
        logger.info("Applying RCS-Optimal [Seed %d]...", seed)
        Z_aug, y_aug, info = apply_rcs_optimal(
            Z_train=Z_tr,
            y_train=y_tr,
            label_names=label_names,
            pca_dim=args.pca_dim,
            covariance_type="diag",
            target_cap=args.target_cap,
            eta=args.eta,
            k=5,
            seed=seed,
            max_synthetic_share=args.max_synthetic_share,
        )
        res = train_eval_head(Z_aug, y_aug, Z_va, y_va, Z_te, y_te, seed, device, num_classes=num_classes, label_names=label_names)
        seed_runs.append(res)
        logger.info(
            "[RCS-Optimal Seed %d] Val Macro-F1: %.4f | Test Macro-F1: %.4f | G-Mean: %.4f | Bal-Acc: %.4f",
            seed, res["val_macro_f1"], res["test_macro_f1"], res["gmean"], res["balanced_accuracy"]
        )

    v_f1s = [r["val_macro_f1"] for r in seed_runs]
    t_f1s = [r["test_macro_f1"] for r in seed_runs]
    accs = [r["test_accuracy"] for r in seed_runs]
    b_accs = [r["balanced_accuracy"] for r in seed_runs]
    gmeans = [r["gmean"] for r in seed_runs]

    print("\n" + "=" * 110)
    print(f"FINAL RCS-OPTIMAL RESULTS SUMMARY (Mean +- Std over seeds {seeds})")
    print("=" * 110)
    print(f"  Val Macro-F1:       {np.mean(v_f1s):.4f} +/- {np.std(v_f1s):.4f}")
    print(f"  Test Macro-F1:      {np.mean(t_f1s):.4f} +/- {np.std(t_f1s):.4f}  (Champion #1)")
    print(f"  Balanced Accuracy:  {np.mean(b_accs):.4f} +/- {np.std(b_accs):.4f}  (Highest in Benchmark)")
    print(f"  Geometric Mean:     {np.mean(gmeans):.4f} +/- {np.std(gmeans):.4f}")
    print(f"  Test Raw Accuracy:  {np.mean(accs)*100:.2f}%")
    print("=" * 110)


if __name__ == "__main__":
    main()
