#!/usr/bin/env python3
"""
scripts/run_cremad_experiments.py
=================================
Executes the comprehensive Step 9 evaluation benchmark for CREMA-D:
Evaluates Fusion V2, Fusion V2 (no gate), and Concat-MLP raw features across:
  - 4 imbalance settings: [balanced, lt_rho10, lt_rho20, lt_rho50]
  - 3 random seeds: [42, 43, 44]
  - 6 rebalancing strategies on latents:
      1. no_rebalance
      2. class_weighted_CE
      3. random_oversample
      4. latent_SMOTE
      5. balanced_softmax
      6. RCS-Optimal (tuned on validation set)

Protocol & Safeguards:
----------------------
1. Validation-Only Selection:
   - Checkpoint selection for all models and heads is strictly by validation Macro-F1.
   - RCS hyperparameters (pca_dim, eta, target_cap, k, max_synthetic_share) are tuned on the validation set.
   - Test split is evaluated strictly ONCE per (setting, seed, representation, method) at the val-selected checkpoint.
2. Train-Only Fitting:
   - All rebalancing methods (ROS, SMOTE, RCS) are fit strictly on the train split.
   - Feature scalers were fit on balanced train only (Step 7).
3. Modality and Class Count Agnostic:
   - Dynamically resolves modalities ("audio", "video") and emotion classes (6 classes for CREMA-D).
4. Master Output:
   - results_cremad/rcs_grid_{setting}.csv
   - results_cremad/master_table.csv (mean +/- std over seeds for Macro-F1, Bal-Acc, G-Mean, Accuracy, per-class F1)
   - results_cremad/all_runs_detail.csv
   - results_cremad/confusion_matrices.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler, SMOTE

from models.regularized_gated_fusion_v2 import RegularizedGatedMultimodalClassifierV2
from train_fusion_v2 import decoupled_supervised_contrastive_loss
from rcs_engine import apply_rcs_optimal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_cremad_experiments")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_gmean(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, eps: float = 1e-4) -> float:
    """Computes geometric mean of per-class recalls: G-Mean = (prod max(rec_c, eps))^(1/C)."""
    recalls: List[float] = []
    for c in range(num_classes):
        mask = (y_true == c)
        if np.sum(mask) == 0:
            continue
        rec = float(np.sum((y_pred == c) & mask) / np.sum(mask))
        recalls.append(max(rec, eps))
    if not recalls:
        return 0.0
    log_sum = sum(math.log(r) for r in recalls)
    return float(math.exp(log_sum / len(recalls)))


class LatentClassifierHead(nn.Module):
    """Unified 2-layer MLP classifier head on latent or concatenated representations."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 6, dropout: float = 0.3):
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


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax Loss (Ren et al., NeurIPS 2020) with log prior class frequency shift."""

    def __init__(self, class_counts: np.ndarray, device: torch.device):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
        self.log_counts = torch.log(counts.clamp(min=1.0)).unsqueeze(0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.log_counts
        return F.cross_entropy(adjusted_logits, targets)


def evaluate_head(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs inference on a DataLoader of latents and returns (predictions, targets)."""
    model.eval()
    preds_list: List[int] = []
    targets_list: List[int] = []
    with torch.no_grad():
        for b_x, b_y in loader:
            b_x = b_x.to(device)
            out = model(b_x)
            preds = torch.argmax(out, dim=-1).cpu().numpy()
            preds_list.extend(preds.tolist())
            targets_list.extend(b_y.numpy().tolist())
    return np.array(preds_list, dtype=np.int64), np.array(targets_list, dtype=np.int64)


def train_eval_head(
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_va: np.ndarray,
    y_va: np.ndarray,
    Z_te: Optional[np.ndarray],
    y_te: Optional[np.ndarray],
    num_classes: int,
    label_names: List[str],
    method: str,
    real_y_tr: np.ndarray,
    seed: int,
    device: torch.device,
    max_epochs: int = 60,
    patience: int = 10,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Trains LatentClassifierHead on training latents, checkpoints strictly on val Macro-F1,
    and evaluates test set ONCE at the selected checkpoint.
    """
    set_seed(seed)
    in_dim = Z_tr.shape[1]

    train_ds = TensorDataset(torch.from_numpy(Z_tr).float(), torch.from_numpy(y_tr).long())
    val_ds = TensorDataset(torch.from_numpy(Z_va).float(), torch.from_numpy(y_va).long())

    batch_size = min(batch_size, max(1, len(Z_tr)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LatentClassifierHead(in_dim=in_dim, hidden_dim=256, num_classes=num_classes, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Loss function configuration
    if method == "class_weighted_CE":
        counts = np.bincount(real_y_tr, minlength=num_classes).astype(np.float32)
        total_samples = len(real_y_tr)
        weights = total_samples / (num_classes * np.maximum(counts, 1.0))
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    elif method in ["balanced_softmax", "RCS_balanced_softmax"]:
        counts = np.bincount(real_y_tr, minlength=num_classes).astype(np.float32)
        criterion = BalancedSoftmaxLoss(counts, device=device)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()

        # Validation evaluation strictly on VAL Macro-F1
        va_preds, va_targets = evaluate_head(model, val_loader, device)
        v_f1 = float(f1_score(va_targets, va_preds, average="macro", zero_division=0))
        v_acc = float(accuracy_score(va_targets, va_preds))

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_val_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_weights)

    # Evaluate test split strictly ONCE at best validation epoch if provided
    result: Dict[str, Any] = {
        "val_macro_f1": float(best_val_f1),
        "val_accuracy": float(best_val_acc),
    }

    if Z_te is not None and y_te is not None:
        test_ds = TensorDataset(torch.from_numpy(Z_te).float(), torch.from_numpy(y_te).long())
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        te_preds, te_targets = evaluate_head(model, test_loader, device)

        te_macro_f1 = float(f1_score(te_targets, te_preds, average="macro", zero_division=0))
        te_acc = float(accuracy_score(te_targets, te_preds))
        te_bal_acc = float(balanced_accuracy_score(te_targets, te_preds))
        te_gmean = float(calculate_gmean(te_targets, te_preds, num_classes))
        per_class_f1 = f1_score(te_targets, te_preds, average=None, zero_division=0)
        cm = confusion_matrix(te_targets, te_preds, labels=list(range(num_classes))).tolist()

        result.update({
            "test_macro_f1": te_macro_f1,
            "test_accuracy": te_acc,
            "balanced_accuracy": te_bal_acc,
            "gmean": te_gmean,
            "per_class_f1": {label_names[i]: float(per_class_f1[i]) for i in range(num_classes)},
            "confusion_matrix": cm,
            "test_predictions": te_preds.tolist(),
        })

    return result


def train_fusion_model(
    X_tr: Dict[str, torch.Tensor],
    y_tr: torch.Tensor,
    X_va: Dict[str, torch.Tensor],
    y_va: torch.Tensor,
    modalities: List[str],
    modality_dims: Dict[str, int],
    num_classes: int,
    cfg: Dict[str, Any],
    use_gate: bool,
    seed: int,
    device: torch.device,
    max_epochs: int = 25,
    patience: int = 6,
) -> nn.Module:
    """Trains Fusion V2 on train features, checkpoints on highest val Macro-F1."""
    set_seed(seed)

    model = RegularizedGatedMultimodalClassifierV2(
        modalities=modalities,
        modality_dims=modality_dims,
        hidden_dim=cfg["hidden_dim"],
        num_classes=num_classes,
        modality_dropout=cfg.get("modality_dropout", 0.1),
        classifier_dropout=cfg.get("dropout", 0.3),
        use_gate=use_gate,
    ).to(device)

    # Class-balanced WeightedRandomSampler for training stability
    class_counts = torch.bincount(y_tr, minlength=num_classes).float()
    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_tr), replacement=True)

    tr_dataset = TensorDataset(*[X_tr[m] for m in modalities], y_tr)
    va_dataset = TensorDataset(*[X_va[m] for m in modalities], y_va)

    batch_size = cfg.get("batch_size", 256)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, sampler=sampler)
    va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    aux_weight = cfg.get("aux_weight", 0.1)
    supcon_weight = cfg.get("supcon_weight", 0.2)
    supcon_tau = cfg.get("supcon_temperature", 0.07)

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in tr_loader:
            b_embs = {m: batch[idx].to(device) for idx, m in enumerate(modalities)}
            b_y = batch[-1].to(device)

            optimizer.zero_grad()
            out = model(**b_embs)
            loss = ce_loss_fn(out["logits"], b_y)

            # Aux losses
            if aux_weight > 0.0 and len(modalities) > 1:
                aux_term = sum(ce_loss_fn(out[f"aux_{m}_logits"], b_y) for m in modalities)
                loss = loss + (aux_weight / len(modalities)) * aux_term

            # SupCon loss
            if supcon_weight > 0.0 and "z_con" in out:
                l_con = decoupled_supervised_contrastive_loss(out["z_con"], b_y, tau=supcon_tau)
                loss = loss + supcon_weight * l_con

            loss.backward()
            optimizer.step()

        # Validation evaluation strictly on Macro-F1
        model.eval()
        va_preds, va_targets = [], []
        with torch.no_grad():
            for batch in va_loader:
                b_embs = {m: batch[idx].to(device) for idx, m in enumerate(modalities)}
                b_y = batch[-1]
                out = model(**b_embs)
                preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
                va_preds.extend(preds.tolist())
                va_targets.extend(b_y.numpy().tolist())

        val_f1 = float(f1_score(va_targets, va_preds, average="macro", zero_division=0))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_weights)
    logger.info("  Trained %s (seed %d): Best Val Macro-F1 = %.4f",
                "Fusion V2" if use_gate else "Fusion V2 (No Gate)", seed, best_val_f1)
    return model


def extract_latents(
    model: nn.Module,
    X_dict: Dict[str, torch.Tensor],
    modalities: List[str],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extracts fused latent representations (fused_feat) from trained model."""
    model.eval()
    num_samples = len(X_dict[modalities[0]])
    latents: List[np.ndarray] = []
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            b_embs = {m: X_dict[m][start_idx:end_idx].to(device) for m in modalities}
            out = model(**b_embs)
            latents.append(out["fused_feat"].cpu().numpy())
    return np.vstack(latents)


def tune_rcs_grid(
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_va: np.ndarray,
    y_va: np.ndarray,
    num_classes: int,
    label_names: List[str],
    setting: str,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Performs validation-only grid tuning of RCS-Optimal parameters.
    Saves rcs_grid_{setting}.csv and returns the winning configuration.
    """
    grid_csv_path = output_dir / f"rcs_grid_{setting}.csv"
    counts = np.bincount(y_tr)
    min_count = int(counts.min())
    max_count = int(counts.max())
    max_allowed_pca = max(1, min_count // 4)

    logger.info("--- Tuning RCS Grid for '%s' (N_min=%d, N_max=%d, max_pca=%d) ---",
                setting, min_count, max_count, max_allowed_pca)

    # Grid definition: pca_dim in {4, 8, 16, 32} (plus 64 for balanced if allowed), max_synthetic_share in {0.85, None}
    if setting == "balanced":
        raw_pca_dims = [4, 8, 16, 32, 64]
        etas = [3, 5]
    else:
        raw_pca_dims = [4, 8, 16, 32]
        etas = [5, 7]

    pca_dims = sorted(list(set([min(p, max_allowed_pca) for p in raw_pca_dims])))
    target_caps = [max_count]
    ks = [3, 5]
    max_syn_shares = [0.85, None]

    trials: List[Dict[str, Any]] = []
    best_config: Optional[Dict[str, Any]] = None
    best_val_f1 = -1.0

    trial_idx = 0
    for p_dim in pca_dims:
        for eta in etas:
            for t_cap in target_caps:
                for k in ks:
                    for mss in max_syn_shares:
                        trial_idx += 1
                        try:
                            Z_aug, y_aug, info = apply_rcs_optimal(
                                Z_train=Z_tr,
                                y_train=y_tr,
                                label_names=label_names,
                                pca_dim=p_dim,
                                covariance_type="diag",
                                target_cap=t_cap,
                                eta=eta,
                                k=k,
                                seed=42,
                                max_synthetic_share=mss,
                            )
                            # Evaluate on validation latents ONLY
                            res = train_eval_head(
                                Z_tr=Z_aug,
                                y_tr=y_aug,
                                Z_va=Z_va,
                                y_va=y_va,
                                Z_te=None,
                                y_te=None,
                                num_classes=num_classes,
                                label_names=label_names,
                                method="RCS-Optimal",
                                real_y_tr=y_tr,
                                seed=42,
                                device=device,
                                max_epochs=40,
                                patience=8,
                            )
                            v_f1 = res["val_macro_f1"]
                            v_acc = res["val_accuracy"]
                        except Exception as ex:
                            logger.warning("RCS Grid trial failed: %s", ex)
                            v_f1, v_acc = 0.0, 0.0
                            info = {"pca_dim": p_dim, "n_synthetic_total": 0}

                        trial_record = {
                            "trial": trial_idx,
                            "setting": setting,
                            "pca_dim": p_dim,
                            "clamped_pca_dim": info.get("pca_dim", p_dim),
                            "eta": eta,
                            "target_cap": t_cap,
                            "k": k,
                            "max_synthetic_share": str(mss),
                            "n_synthetic": info.get("n_synthetic_total", 0),
                            "val_macro_f1": v_f1,
                            "val_accuracy": v_acc,
                        }
                        trials.append(trial_record)

                        if v_f1 > best_val_f1:
                            best_val_f1 = v_f1
                            best_config = {
                                "pca_dim": p_dim,
                                "eta": eta,
                                "target_cap": t_cap,
                                "k": k,
                                "max_synthetic_share": mss,
                            }

    if best_config is None:
        best_config = {
            "pca_dim": pca_dims[-1],
            "eta": 7 if "lt" in setting else 3,
            "target_cap": max_count,
            "k": 5,
            "max_synthetic_share": 0.85 if "lt" in setting else None,
        }

    df_grid = pd.DataFrame(trials)
    df_grid.to_csv(grid_csv_path, index=False)
    logger.info("Saved RCS grid tuning results (%d trials) to %s", len(trials), grid_csv_path)
    logger.info("Winning RCS config for '%s': %s (Val Macro-F1 = %.4f)", setting, best_config, best_val_f1)
    return best_config


def main():
    parser = argparse.ArgumentParser(description="Run complete CREMA-D experiments across balanced and LT settings.")
    parser.add_argument("--features_root", type=str, default="data", help="Root data directory containing feature sets.")
    parser.add_argument("--output_dir", type=str, default="results_cremad", help="Output directory for results.")
    parser.add_argument("--threads", type=int, default=8, help="PyTorch CPU threads.")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["balanced", "lt_rho10", "lt_rho20", "lt_rho50"],
        help="List of settings to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="List of random seeds to evaluate.",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_cache_dir = output_dir / "latents"
    latents_cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    logger.info("Using device: %s (CPU threads: %d)", device, torch.get_num_threads())

    # Load tuned best configuration for Fusion V2
    best_config_path = output_dir / "best_config.json"
    if not best_config_path.is_file():
        raise FileNotFoundError(f"FATAL: {best_config_path} not found. Run Step 8 tuning first.")
    with open(best_config_path, "r", encoding="utf-8") as f:
        best_cfg = json.load(f)
    logger.info("Loaded best Fusion V2 configuration: %s", best_cfg)

    # Label map and classes
    label_map_path = Path("data/features_cremad/label_map.json")
    if not label_map_path.is_file():
        raise FileNotFoundError(f"FATAL: {label_map_path} not found.")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    # Inverse map to get list of label names ordered by integer index
    inv_map = {v: k for k, v in label_map.items()}
    label_names = [inv_map[i] for i in range(num_classes)]
    logger.info("Emotion classes (%d): %s", num_classes, label_names)

    settings = args.settings
    seeds = args.seeds
    representations = ["fusion_v2", "fusion_v2_nogate", "concat_MLP_raw_features"]
    methods = [
        "no_rebalance",
        "class_weighted_CE",
        "random_oversample",
        "latent_SMOTE",
        "balanced_softmax",
        "RCS-Optimal",
        "RCS_balanced_softmax",
    ]

    all_detailed_runs: List[Dict[str, Any]] = []
    confusion_matrices_dict: Dict[str, Any] = {}
    test_predictions_dict: Dict[str, Any] = {}
    gate_weights_dict: Dict[str, Any] = {}

    for setting in settings:
        print("\n" + "=" * 90)
        print(f"PROCESSING SETTING: {setting.upper()}")
        print("=" * 90)

        # Resolve setting features directory
        if setting == "balanced":
            feat_dir = Path("data/features_cremad")
        else:
            feat_dir = Path(f"data/features_cremad_{setting}")

        tr_data = np.load(feat_dir / "train_features.npz")
        va_data = np.load(feat_dir / "val_features.npz")
        te_data = np.load(feat_dir / "test_features.npz")

        modalities = ["audio", "video"]
        modality_dims = {
            "audio": int(tr_data["X_audio"].shape[1]),
            "video": int(tr_data["X_video"].shape[1]),
        }
        logger.info("Modality dimensions: %s", modality_dims)

        # Convert to PyTorch tensors
        X_tr = {"audio": torch.from_numpy(tr_data["X_audio"]).float(), "video": torch.from_numpy(tr_data["X_video"]).float()}
        y_tr = torch.from_numpy(tr_data["y"]).long()
        X_va = {"audio": torch.from_numpy(va_data["X_audio"]).float(), "video": torch.from_numpy(va_data["X_video"]).float()}
        y_va = torch.from_numpy(va_data["y"]).long()
        X_te = {"audio": torch.from_numpy(te_data["X_audio"]).float(), "video": torch.from_numpy(te_data["X_video"]).float()}
        y_te = torch.from_numpy(te_data["y"]).long()

        y_tr_np = tr_data["y"]
        y_va_np = va_data["y"]
        y_te_np = te_data["y"]

        # Cache/extract representations for each seed
        latents_store: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {rep: {} for rep in representations}

        for seed in seeds:
            logger.info("--- Preparing latents for setting '%s' [Seed %d] ---", setting, seed)

            # 1. Fusion V2
            cache_file_v2 = latents_cache_dir / f"{setting}_fusion_v2_seed{seed}.npz"
            gate_file = latents_cache_dir / f"{setting}_fusion_v2_seed{seed}_gates.npy"
            if cache_file_v2.is_file():
                logger.info("  Loading cached Fusion V2 latents: %s", cache_file_v2)
                f_data = np.load(cache_file_v2)
                Z_tr_v2, Z_va_v2, Z_te_v2 = f_data["Z_tr"], f_data["Z_va"], f_data["Z_te"]
            else:
                logger.info("  Training Fusion V2 (seed %d)...", seed)
                model_v2 = train_fusion_model(
                    X_tr, y_tr, X_va, y_va, modalities, modality_dims, num_classes,
                    best_cfg, use_gate=True, seed=seed, device=device
                )
                Z_tr_v2 = extract_latents(model_v2, X_tr, modalities, device)
                Z_va_v2 = extract_latents(model_v2, X_va, modalities, device)
                Z_te_v2 = extract_latents(model_v2, X_te, modalities, device)
                np.savez_compressed(cache_file_v2, Z_tr=Z_tr_v2, Z_va=Z_va_v2, Z_te=Z_te_v2)

            if not gate_file.is_file():
                logger.info("  Extracting gate scores on test set (seed %d)...", seed)
                model_v2 = train_fusion_model(
                    X_tr, y_tr, X_va, y_va, modalities, modality_dims, num_classes,
                    best_cfg, use_gate=True, seed=seed, device=device
                )
                model_v2.eval()
                with torch.no_grad():
                    out = model_v2(audio=X_te["audio"].to(device), video=X_te["video"].to(device))
                    scores = out["gate_scores"].cpu().numpy()
                np.save(gate_file, scores)

            latents_store["fusion_v2"][seed] = {"tr": Z_tr_v2, "va": Z_va_v2, "te": Z_te_v2}

            # 2. Fusion V2 No Gate
            cache_file_nogate = latents_cache_dir / f"{setting}_fusion_v2_nogate_seed{seed}.npz"
            if cache_file_nogate.is_file():
                logger.info("  Loading cached Fusion V2 (No Gate) latents: %s", cache_file_nogate)
                f_data = np.load(cache_file_nogate)
                Z_tr_ng, Z_va_ng, Z_te_ng = f_data["Z_tr"], f_data["Z_va"], f_data["Z_te"]
            else:
                logger.info("  Training Fusion V2 No Gate (seed %d)...", seed)
                model_ng = train_fusion_model(
                    X_tr, y_tr, X_va, y_va, modalities, modality_dims, num_classes,
                    best_cfg, use_gate=False, seed=seed, device=device
                )
                Z_tr_ng = extract_latents(model_ng, X_tr, modalities, device)
                Z_va_ng = extract_latents(model_ng, X_va, modalities, device)
                Z_te_ng = extract_latents(model_ng, X_te, modalities, device)
                np.savez_compressed(cache_file_nogate, Z_tr=Z_tr_ng, Z_va=Z_va_ng, Z_te=Z_te_ng)
            latents_store["fusion_v2_nogate"][seed] = {"tr": Z_tr_ng, "va": Z_va_ng, "te": Z_te_ng}

            # 3. Concat MLP Raw Features (2304-d)
            Z_tr_raw = np.concatenate([tr_data["X_audio"], tr_data["X_video"]], axis=-1)
            Z_va_raw = np.concatenate([va_data["X_audio"], va_data["X_video"]], axis=-1)
            Z_te_raw = np.concatenate([te_data["X_audio"], te_data["X_video"]], axis=-1)
            latents_store["concat_MLP_raw_features"][seed] = {"tr": Z_tr_raw, "va": Z_va_raw, "te": Z_te_raw}

        # Step: Aggregate gate weights per emotion across seeds for setting
        setting_gates = []
        for s in seeds:
            scores = np.load(latents_cache_dir / f"{setting}_fusion_v2_seed{s}_gates.npy")
            per_emo = {}
            for c, lbl in enumerate(label_names):
                mask = (y_te_np == c)
                per_emo[lbl] = {
                    "audio": float(np.mean(scores[mask, 0])),
                    "video": float(np.mean(scores[mask, 1])),
                }
            setting_gates.append(per_emo)

        gate_weights_dict[setting] = {
            lbl: {
                "audio_mean": float(np.mean([sg[lbl]["audio"] for sg in setting_gates])),
                "audio_std": float(np.std([sg[lbl]["audio"] for sg in setting_gates])),
                "video_mean": float(np.mean([sg[lbl]["video"] for sg in setting_gates])),
                "video_std": float(np.std([sg[lbl]["video"] for sg in setting_gates])),
            }
            for lbl in label_names
        }

        # Step: RCS Parameter Tuning (Validation Only on Seed 42 Fusion V2 latents)
        best_rcs_cfg = tune_rcs_grid(
            Z_tr=latents_store["fusion_v2"][42]["tr"],
            y_tr=y_tr_np,
            Z_va=latents_store["fusion_v2"][42]["va"],
            y_va=y_va_np,
            num_classes=num_classes,
            label_names=label_names,
            setting=setting,
            output_dir=output_dir,
            device=device,
        )

        # Step: Benchmark evaluation across representations, methods, and seeds
        print("\n" + "-" * 80)
        print(f"BENCHMARKING REBALANCING METHODS FOR SETTING: {setting}")
        print("-" * 80)

        for rep in representations:
            logger.info("Evaluating Representation: %s [%s]", rep, setting)

            for seed in seeds:
                Z_tr_base = latents_store[rep][seed]["tr"]
                Z_va_base = latents_store[rep][seed]["va"]
                Z_te_base = latents_store[rep][seed]["te"]

                for method in methods:
                    # Prepare training data based on rebalancing strategy
                    if method == "no_rebalance":
                        Z_tr_run, y_tr_run = Z_tr_base, y_tr_np

                    elif method == "class_weighted_CE":
                        Z_tr_run, y_tr_run = Z_tr_base, y_tr_np

                    elif method == "balanced_softmax":
                        Z_tr_run, y_tr_run = Z_tr_base, y_tr_np

                    elif method == "random_oversample":
                        ros = RandomOverSampler(random_state=seed)
                        Z_tr_run, y_tr_run = ros.fit_resample(Z_tr_base, y_tr_np)

                    elif method == "latent_SMOTE":
                        min_c = int(np.min(np.bincount(y_tr_np)))
                        k_smote = min(5, max(1, min_c - 1))
                        smote = SMOTE(k_neighbors=k_smote, random_state=seed)
                        Z_tr_run, y_tr_run = smote.fit_resample(Z_tr_base, y_tr_np)

                    elif method in ["RCS-Optimal", "RCS_balanced_softmax"]:
                        # Clamp pca_dim to min(pca_dim, smallest_class // 4)
                        min_c = int(np.min(np.bincount(y_tr_np)))
                        max_allowed_pca = max(1, min_c // 4)
                        p_dim = min(best_rcs_cfg["pca_dim"], max_allowed_pca)

                        Z_tr_run, y_tr_run, _ = apply_rcs_optimal(
                            Z_train=Z_tr_base,
                            y_train=y_tr_np,
                            label_names=label_names,
                            pca_dim=p_dim,
                            covariance_type="diag",
                            target_cap=best_rcs_cfg["target_cap"],
                            eta=best_rcs_cfg["eta"],
                            k=best_rcs_cfg["k"],
                            seed=seed,
                            max_synthetic_share=best_rcs_cfg["max_synthetic_share"],
                        )
                    else:
                        raise ValueError(f"Unknown rebalancing method: {method}")

                    # Train and evaluate LatentClassifierHead
                    res = train_eval_head(
                        Z_tr=Z_tr_run,
                        y_tr=y_tr_run,
                        Z_va=Z_va_base,
                        y_va=y_va_np,
                        Z_te=Z_te_base,
                        y_te=y_te_np,
                        num_classes=num_classes,
                        label_names=label_names,
                        method=method,
                        real_y_tr=y_tr_np,
                        seed=seed,
                        device=device,
                    )

                    run_key = f"{setting}/{rep}/{method}/seed_{seed}"
                    confusion_matrices_dict[run_key] = res["confusion_matrix"]
                    test_predictions_dict[run_key] = res["test_predictions"]

                    run_record = {
                        "setting": setting,
                        "representation": rep,
                        "method": method,
                        "seed": seed,
                        "val_macro_f1": res["val_macro_f1"],
                        "val_accuracy": res["val_accuracy"],
                        "test_macro_f1": res["test_macro_f1"],
                        "balanced_accuracy": res["balanced_accuracy"],
                        "gmean": res["gmean"],
                        "accuracy": res["test_accuracy"],
                    }
                    for lbl, f1_val in res["per_class_f1"].items():
                        run_record[f"f1_{lbl}"] = f1_val

                    all_detailed_runs.append(run_record)

                    logger.info("  [%s | %s | %s | seed %d] Val F1: %.4f | Test F1: %.4f | Bal-Acc: %.4f | G-Mean: %.4f",
                                setting, rep, method, seed,
                                res["val_macro_f1"], res["test_macro_f1"], res["balanced_accuracy"], res["gmean"])

    # Save all detailed runs
    df_runs = pd.DataFrame(all_detailed_runs)
    df_runs.to_csv(output_dir / "all_runs_detail.csv", index=False)
    logger.info("Saved all detailed runs (%d records) to %s", len(df_runs), output_dir / "all_runs_detail.csv")

    # Save confusion matrices dictionary
    with open(output_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(confusion_matrices_dict, f, indent=2)
    logger.info("Saved confusion matrices to %s", output_dir / "confusion_matrices.json")

    # Save test predictions dictionary
    with open(output_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_predictions_dict, f)
    logger.info("Saved test predictions to %s", output_dir / "test_predictions.json")

    # Save gate weights dictionary
    with open(output_dir / "gate_weights.json", "w", encoding="utf-8") as f:
        json.dump(gate_weights_dict, f, indent=2)
    logger.info("Saved gate weights to %s", output_dir / "gate_weights.json")

    # Build master summary table (aggregating mean +/- std over seeds 42, 43, 44)
    summary_rows: List[Dict[str, Any]] = []
    group_cols = ["setting", "representation", "method"]

    metric_cols = ["val_macro_f1", "test_macro_f1", "balanced_accuracy", "gmean", "accuracy"]
    per_class_cols = [f"f1_{lbl}" for lbl in label_names]
    all_metrics = metric_cols + per_class_cols

    for (st, rep, mth), group in df_runs.groupby(group_cols, sort=False):
        row: Dict[str, Any] = {
            "setting": st,
            "representation": rep,
            "method": mth,
            "num_seeds": len(group),
        }
        for col in all_metrics:
            vals = group[col].values
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals))

        summary_rows.append(row)

    master_df = pd.DataFrame(summary_rows)
    master_table_path = output_dir / "master_table.csv"
    master_df.to_csv(master_table_path, index=False)
    logger.info("=" * 90)
    logger.info("SUCCESS: Master results table written to %s (%d rows)", master_table_path, len(master_df))
    logger.info("=" * 90)

    # Print summary highlights
    print("\n" + "=" * 115)
    print("MASTER SUMMARY HIGHLIGHTS ACROSS IMBALANCE SETTINGS (Mean Test Macro-F1 +/- Std)")
    print("=" * 115)
    print(f"{'Setting':<12} {'Representation':<26} {'Method':<20} {'Test Macro-F1':<20} {'Bal-Acc':<18} {'G-Mean':<18}")
    print("-" * 115)
    for _, r in master_df.iterrows():
        f1_str = f"{r['test_macro_f1_mean']:.4f} +/- {r['test_macro_f1_std']:.4f}"
        bacc_str = f"{r['balanced_accuracy_mean']:.4f} +/- {r['balanced_accuracy_std']:.4f}"
        gm_str = f"{r['gmean_mean']:.4f} +/- {r['gmean_std']:.4f}"
        print(f"{r['setting']:<12} {r['representation']:<26} {r['method']:<20} {f1_str:<20} {bacc_str:<18} {gm_str:<18}")
    print("=" * 115)


if __name__ == "__main__":
    main()
