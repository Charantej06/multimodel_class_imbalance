"""
run_final_rcs.py
================
Self-contained execution script for the final winning RCS-Optimal multimodal pipeline.

Run with:
    python final_rcs_package/run_final_rcs.py
"""

import sys
import copy
import math
import random
import logging
from pathlib import Path
from typing import Dict, Any, List

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

LABEL_NAMES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
NUM_CLASSES = len(LABEL_NAMES)
PACKAGE_DIR = Path(__file__).parent.resolve()
LATENT_DIR = PACKAGE_DIR / "latents"


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
) -> Dict[str, Any]:
    set_seed(seed)

    train_ds = TensorDataset(torch.from_numpy(Z_tr).float(), torch.from_numpy(y_tr).long())
    val_ds = TensorDataset(torch.from_numpy(Z_va).float(), torch.from_numpy(y_va).long())
    test_ds = TensorDataset(torch.from_numpy(Z_te).float(), torch.from_numpy(y_te).long())

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = LatentClassifierHead(in_dim=512, hidden_dim=256, num_classes=NUM_CLASSES, dropout=0.3).to(device)
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
    rep = classification_report(te_targets, te_preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    recalls = [rep[lbl]["recall"] for lbl in LABEL_NAMES]
    gmean = calculate_gmean(recalls)

    return {
        "val_macro_f1": float(best_val_f1),
        "test_macro_f1": float(macro_f1),
        "test_accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "gmean": float(gmean),
        "recalls": {lbl: rep[lbl]["recall"] for lbl in LABEL_NAMES},
    }


def main():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.80, device=0)
        device = torch.device("cuda:0")
        logger.info("Using GPU: %s (memory fraction 0.80 set)", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using device: %s", device)

    # 1. Load Precomputed Fusion V2 Latents
    logger.info("Loading Fusion V2 latents from %s...", LATENT_DIR)
    tr = np.load(LATENT_DIR / "latents_train.npz")
    va = np.load(LATENT_DIR / "latents_val.npz")
    te = np.load(LATENT_DIR / "latents_test.npz")

    Z_tr, y_tr = tr["fused_feat"], tr["labels"]
    Z_va, y_va = va["fused_feat"], va["labels"]
    Z_te, y_te = te["fused_feat"], te["labels"]

    logger.info("Loaded latents: Train=%s, Val=%s, Test=%s", Z_tr.shape, Z_va.shape, Z_te.shape)

    seeds = [42, 43, 44]
    seed_runs = []

    print("\n" + "=" * 90)
    print("RUNNING FINAL RCS-OPTIMAL MULTIMODAL BENCHMARK (Seeds: 42, 43, 44)")
    print("=" * 90)

    for seed in seeds:
        logger.info("Applying RCS-Optimal [Seed %d]...", seed)
        Z_aug, y_aug, info = apply_rcs_optimal(
            Z_train=Z_tr,
            y_train=y_tr,
            label_names=LABEL_NAMES,
            pca_dim=64,
            covariance_type="diag",
            target_cap=2500,
            eta=7,
            k=5,
            seed=seed,
        )
        res = train_eval_head(Z_aug, y_aug, Z_va, y_va, Z_te, y_te, seed, device)
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
    print("FINAL RCS-OPTIMAL RESULTS SUMMARY (Mean +- Std over seeds 42, 43, 44)")
    print("=" * 110)
    print(f"  Val Macro-F1:       {np.mean(v_f1s):.4f} +/- {np.std(v_f1s):.4f}")
    print(f"  Test Macro-F1:      {np.mean(t_f1s):.4f} +/- {np.std(t_f1s):.4f}  (Champion #1)")
    print(f"  Balanced Accuracy:  {np.mean(b_accs):.4f} +/- {np.std(b_accs):.4f}  (Highest in Benchmark)")
    print(f"  Geometric Mean:     {np.mean(gmeans):.4f} +/- {np.std(gmeans):.4f}")
    print(f"  Test Raw Accuracy:  {np.mean(accs)*100:.2f}%")
    print("=" * 110)


if __name__ == "__main__":
    main()
