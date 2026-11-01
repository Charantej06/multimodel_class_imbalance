"""
final_rcs_package/benchmark_all_methods_food_101.py
===================================================
Comprehensive benchmark on Food-101 Long-Tail (101 classes) comparing 6 methods:
  1. Real Only / No RCS (Baseline control)
  2. SMOTE (imblearn.over_sampling.SMOTE)
  3. BorderlineSMOTE (imblearn.over_sampling.BorderlineSMOTE)
  4. DeepSMOTE (Autoencoder + Latent SMOTE adapted from DeepSMOTE-main)
  5. GAMO (Convex-combination Conditional GAN adapted from GAMO-master)
  6. RCS-Optimal (Subspace PCA-64 + Diag GMM + Calibrated k-NN)

Hyperparameters & Training:
  - Classification Head Epochs: 60 (increased from 30, with CosineAnnealingLR & EarlyStopping patience=10)
  - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  - Target Cap: 500 (balances each class up to majority count)
  - Seeds: [42, 43, 44]
  - Results saved to results/food_101/exp1/
"""

from __future__ import annotations
import os
import sys
import copy
import math
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "final_rcs_package"
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rcs_engine import apply_rcs_optimal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark_food101")

DATA_DIR = REPO_ROOT / "data" / "food101_lt"
RESULTS_DIR = REPO_ROOT / "results" / "food_101" / "exp1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NUM_CLASSES = 101


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
    """2-layer MLP classifier head on 512-d representations for 101 classes."""
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


# --- DeepSMOTE Implementation for 512-d Latents ---
class LatentAutoEncoder(nn.Module):
    def __init__(self, in_dim: int = 512, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, in_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


def run_deepsmote(Z_train: np.ndarray, y_train: np.ndarray, seed: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    ae = LatentAutoEncoder(in_dim=512, latent_dim=128).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    mse = nn.MSELoss()

    tensor_Z = torch.tensor(Z_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_Z), batch_size=128, shuffle=True)

    ae.train()
    for _ in range(15):
        for (bx,) in loader:
            bx = bx.to(device)
            opt.zero_grad()
            rec, _ = ae(bx)
            loss = mse(rec, bx)
            loss.backward()
            opt.step()

    ae.eval()
    with torch.no_grad():
        _, z_all = ae(tensor_Z.to(device))
        z_np = z_all.cpu().numpy()

    sm = SMOTE(k_neighbors=5, random_state=seed)
    z_res, y_res = sm.fit_resample(z_np, y_train)

    n_orig = len(Z_train)
    z_syn = z_res[n_orig:]
    y_syn = y_res[n_orig:]

    if len(z_syn) > 0:
        with torch.no_grad():
            x_syn = ae.decoder(torch.tensor(z_syn, dtype=torch.float32, device=device)).cpu().numpy()
        Z_res = np.concatenate([Z_train, x_syn], axis=0)
        y_final = np.concatenate([y_train, y_syn], axis=0)
    else:
        Z_res = Z_train.copy()
        y_final = y_train.copy()

    return Z_res, y_final


# --- GAMO Implementation for 512-d Latents ---
class DenseGamoGenerator(nn.Module):
    def __init__(self, num_minority: int, noise_dim: int = 64, num_classes: int = 101):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_minority),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor, data_minority: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, labels], dim=1)
        weights = F.softmax(self.net(x), dim=1)
        return torch.matmul(weights, data_minority)


class DenseGamoDiscriminator(nn.Module):
    def __init__(self, in_dim: int = 512, num_classes: int = 101):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + num_classes, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, labels], dim=1)
        return self.net(inp)


def run_gamo(Z_train: np.ndarray, y_train: np.ndarray, seed: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    target = int(counts.max())

    # Identify minority samples (classes with counts < target)
    minority_mask = np.isin(y_train, np.where(counts < target)[0])
    minority_indices = np.where(minority_mask)[0]

    if len(minority_indices) == 0:
        return Z_train.copy(), y_train.copy()

    # Subsample minority pool for memory efficiency if large
    max_pool = min(len(minority_indices), 2000)
    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(minority_indices, size=max_pool, replace=False)
    data_minority = torch.tensor(Z_train[sampled_indices], dtype=torch.float32, device=device)

    generator = DenseGamoGenerator(num_minority=max_pool, noise_dim=64, num_classes=NUM_CLASSES).to(device)
    discriminator = DenseGamoDiscriminator(in_dim=512, num_classes=NUM_CLASSES).to(device)

    opt_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    # Pre-calculate needed counts per class
    needed = {c: target - counts[c] for c in range(NUM_CLASSES) if counts[c] < target}

    # Lightweight adversarial training
    tensor_Z = torch.tensor(Z_train, dtype=torch.float32)
    tensor_y = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(tensor_Z, tensor_y), batch_size=128, shuffle=True)

    generator.train()
    discriminator.train()
    for _ in range(8):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            bs = bx.size(0)
            by_oh = F.one_hot(by, num_classes=NUM_CLASSES).float()

            # Train Discriminator
            noise = torch.randn(bs, 64, device=device)
            syn_feat = generator(noise, by_oh, data_minority)

            d_real = discriminator(bx, by_oh)
            d_fake = discriminator(syn_feat.detach(), by_oh)

            loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_D.zero_grad()
            loss_d.backward()
            opt_D.step()

            # Train Generator
            d_fake_g = discriminator(syn_feat, by_oh)
            loss_g = bce(d_fake_g, torch.ones_like(d_fake_g))
            opt_G.zero_grad()
            loss_g.backward()
            opt_G.step()

    # Generate synthetic samples
    generator.eval()
    syn_chunks = []
    syn_labels = []

    with torch.no_grad():
        for c, n_c in needed.items():
            if n_c <= 0:
                continue
            noise_c = torch.randn(n_c, 64, device=device)
            lbl_c = torch.full((n_c,), c, dtype=torch.long, device=device)
            lbl_oh = F.one_hot(lbl_c, num_classes=NUM_CLASSES).float()
            syn = generator(noise_c, lbl_oh, data_minority)
            syn_chunks.append(syn.cpu().numpy())
            syn_labels.append(lbl_c.cpu().numpy())

    if syn_chunks:
        Z_res = np.concatenate([Z_train, np.concatenate(syn_chunks, axis=0)], axis=0)
        y_final = np.concatenate([y_train, np.concatenate(syn_labels, axis=0)], axis=0)
    else:
        Z_res = Z_train.copy()
        y_final = y_train.copy()

    return Z_res, y_final


def train_and_eval_head(
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_va: np.ndarray,
    y_va: np.ndarray,
    Z_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
    epochs: int = 60,
    patience: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Any]:
    set_seed(seed)
    ds_tr = TensorDataset(torch.from_numpy(Z_tr).float(), torch.from_numpy(y_tr).long())
    ds_va = TensorDataset(torch.from_numpy(Z_va).float(), torch.from_numpy(y_va).long())
    ds_te = TensorDataset(torch.from_numpy(Z_te).float(), torch.from_numpy(y_te).long())

    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
    loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    head = LatentClassifierHead(in_dim=512, hidden_dim=256, num_classes=NUM_CLASSES, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_weights = None
    no_improve = 0

    for ep in range(epochs):
        head.train()
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = head(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation Macro-F1 check
        head.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for bx, by in loader_va:
                bx = bx.to(device)
                preds = torch.argmax(head(bx), dim=1).cpu().numpy()
                v_preds.extend(preds)
                v_targets.extend(by.numpy())
        vf1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_weights = copy.deepcopy(head.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best weights and test
    head.load_state_dict(best_weights)
    head.eval()
    t_preds, t_targets = [], []
    with torch.no_grad():
        for bx, by in loader_te:
            bx = bx.to(device)
            preds = torch.argmax(head(bx), dim=1).cpu().numpy()
            t_preds.extend(preds)
            t_targets.extend(by.numpy())

    macro_f1 = f1_score(t_targets, t_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(t_targets, t_preds, average="weighted", zero_division=0)
    raw_acc = accuracy_score(t_targets, t_preds)
    bal_acc = balanced_accuracy_score(t_targets, t_preds)

    rep = classification_report(t_targets, t_preds, output_dict=True, zero_division=0)
    per_class_recalls = [rep[str(c)]["recall"] for c in range(NUM_CLASSES) if str(c) in rep]
    gmean = calculate_gmean(per_class_recalls)

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "raw_accuracy": float(raw_acc),
        "balanced_accuracy": float(bal_acc),
        "gmean": float(gmean),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.70, device=0)
    logger.info("Running Food-101 Comparative Benchmark across 6 methods on %s...", device)

    tr_data = np.load(DATA_DIR / "latents_train.npz")
    va_data = np.load(DATA_DIR / "latents_val.npz")
    te_data = np.load(DATA_DIR / "latents_test.npz")

    Z_tr_real = tr_data["latents"]
    y_tr_real = tr_data["labels"]
    Z_va = va_data["latents"]
    y_va = va_data["labels"]
    Z_te = te_data["latents"]
    y_te = te_data["labels"]

    logger.info("Loaded real latents: Train=%s, Val=%s, Test=%s", Z_tr_real.shape, Z_va.shape, Z_te.shape)

    methods = [
        ("Real Only (Baseline)", "real_only"),
        ("SMOTE", "smote"),
        ("Borderline-SMOTE", "bsmote"),
        ("DeepSMOTE", "deepsmote"),
        ("GAMO", "gamo"),
        ("RCS-Optimal (Ours)", "rcs_optimal"),
    ]

    seeds = [42, 43, 44]
    final_results = {}

    for name, key in methods:
        logger.info("==========================================")
        logger.info("Evaluating Condition: %s (%s)", name, key)
        logger.info("==========================================")
        metrics_by_seed = []

        for s in seeds:
            set_seed(s)
            logger.info("  [Seed %d] Generating rebalanced latents for %s...", s, name)

            if key == "real_only":
                Z_aug, y_aug = Z_tr_real.copy(), y_tr_real.copy()
            elif key == "smote":
                sm = SMOTE(k_neighbors=5, random_state=s)
                Z_aug, y_aug = sm.fit_resample(Z_tr_real, y_tr_real)
            elif key == "bsmote":
                bsm = BorderlineSMOTE(k_neighbors=5, random_state=s)
                Z_aug, y_aug = bsm.fit_resample(Z_tr_real, y_tr_real)
            elif key == "deepsmote":
                Z_aug, y_aug = run_deepsmote(Z_tr_real, y_tr_real, s, device)
            elif key == "gamo":
                Z_aug, y_aug = run_gamo(Z_tr_real, y_tr_real, s, device)
            elif key == "rcs_optimal":
                Z_aug, y_aug, _ = apply_rcs_optimal(
                    Z_train=Z_tr_real,
                    y_train=y_tr_real,
                    pca_dim=64,
                    covariance_type="diag",
                    target_cap=500,
                    eta=7,
                    k=5,
                    seed=s,
                )
            else:
                raise ValueError(f"Unknown key: {key}")

            res = train_and_eval_head(
                Z_tr=Z_aug,
                y_tr=y_aug,
                Z_va=Z_va,
                y_va=y_va,
                Z_te=Z_te,
                y_te=y_te,
                seed=s,
                epochs=60,
                patience=10,
                device=device,
            )
            metrics_by_seed.append(res)
            logger.info("    Seed %d -> Macro-F1: %.4f | Bal-Acc: %.2f%% | G-Mean: %.4f",
                        s, res["macro_f1"], res["balanced_accuracy"] * 100, res["gmean"])

        f1s = [m["macro_f1"] for m in metrics_by_seed]
        bal_accs = [m["balanced_accuracy"] * 100 for m in metrics_by_seed]
        gmeans = [m["gmean"] for m in metrics_by_seed]
        raw_accs = [m["raw_accuracy"] * 100 for m in metrics_by_seed]

        final_results[name] = {
            "key": key,
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "bal_acc_mean": float(np.mean(bal_accs)),
            "bal_acc_std": float(np.std(bal_accs)),
            "gmean_mean": float(np.mean(gmeans)),
            "gmean_std": float(np.std(gmeans)),
            "raw_acc_mean": float(np.mean(raw_accs)),
            "raw_acc_std": float(np.std(raw_accs)),
            "per_seed": metrics_by_seed,
        }

    # Save metrics JSON
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Generate Markdown Summary table
    md_lines = [
        "# Comparative Benchmark on Food-101 Long-Tailed Dataset (101 Classes)",
        "",
        "Evaluated strictly across random seeds `[42, 43, 44]` on the test set:",
        "",
        "| Method | Rebalancing Approach | Test Macro-F1 | Balanced Accuracy | G-Mean | Raw Accuracy |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for name, data in final_results.items():
        rebal_map = {
            "real_only": "None (Control)",
            "smote": "Linear Interpolation (k=5)",
            "bsmote": "Borderline Frontier Interpolation",
            "deepsmote": "Autoencoder Bottleneck SMOTE",
            "gamo": "Convex Conditional GAN",
            "rcs_optimal": "**Subspace PCA-64 + Diag GMM (Ours)**",
        }
        approach = rebal_map.get(data["key"], "-")
        md_lines.append(
            f"| **{name}** | {approach} | **{data['macro_f1_mean']:.4f} ± {data['macro_f1_std']:.4f}** | "
            f"**{data['bal_acc_mean']:.2f}% ± {data['bal_acc_std']:.2f}%** | "
            f"{data['gmean_mean']:.4f} ± {data['gmean_std']:.4f} | {data['raw_acc_mean']:.2f}% |"
        )

    RESULTS_RCS_DIR = REPO_ROOT / "results_rcs" / "food_101" / "exp1"
    RESULTS_RCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_RCS_DIR / "metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)
    with open(RESULTS_RCS_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    summary_file = RESULTS_DIR / "summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Benchmark complete! Results written to %s and %s", summary_file, RESULTS_DIR / "metrics.json")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()

