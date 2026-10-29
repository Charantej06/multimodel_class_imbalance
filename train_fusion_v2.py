"""
train_fusion_v2.py
==================
Training and evaluation script for Fusion V2 with Decoupled Supervised Contrastive Loss
across configurations V0..V6 and seeds {42, 43, 44}.

Hardware:
  - torch.cuda.set_per_process_memory_fraction(0.80, device=0) if CUDA available

Training setup:
  - Batch size 256
  - Class-balanced WeightedRandomSampler on y_train with weights 1 / class_count
  - 1 epoch = len(train) draws (9989 draws)
  - AdamW (lr=1e-4, weight_decay=1e-4)
  - Max 50 epochs, early stopping patience 10 on VAL macro-F1
  - Total Loss: L_CE(fused) + 0.3*L_CE(text) + 0.3*L_CE(video) + 0.5*L_con

Contrastive Loss:
  - Decoupled Supervised Contrastive Loss on z_con:
    L_i = -alpha_pos * (1 / |P(i)|) * sum_{p in P(i)} (s_ip / tau)
          + alpha_neg * log( sum_{n in N(i)} exp(s_in / tau) )
    where s_ij = z_i^T z_j / tau.
  - Anchor losses averaged per-class first, then across classes present in batch.

Model selection:
  - Strictly on VAL macro-F1. Test set evaluated once at the selected epoch.
"""

from __future__ import annotations
import os
import sys
import copy
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, accuracy_score

from models.regularized_gated_fusion_v2 import RegularizedGatedMultimodalClassifierV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_fusion_v2")

LABEL_NAMES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
NUM_CLASSES = len(LABEL_NAMES)
OUT_DIR = Path("results_v3")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def decoupled_supervised_contrastive_loss(
    z_con: torch.Tensor,
    labels: torch.Tensor,
    alpha_pos: float = 1.0,
    alpha_neg: float = 1.0,
    tau: float = 0.1,
) -> torch.Tensor:
    """
    Decoupled Supervised Contrastive Loss with class-balanced anchor averaging.

    z_con : (B, D) L2-normalized projection embeddings
    labels : (B,) integer emotion class labels
    """
    device = z_con.device
    B = z_con.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Cosine similarity matrix scaled by tau
    sim_matrix = torch.matmul(z_con, z_con.T) / tau  # (B, B)

    # Masks
    labels = labels.view(-1, 1)
    mask_same_class = torch.eq(labels, labels.T).to(device)  # (B, B)
    mask_self = torch.eye(B, dtype=torch.bool, device=device)
    mask_pos = mask_same_class & (~mask_self)                 # (B, B)
    mask_neg = ~mask_same_class                              # (B, B)

    pos_counts = mask_pos.sum(dim=1)  # (B,)
    neg_counts = mask_neg.sum(dim=1)  # (B,)

    # Valid anchors must have at least one positive and one negative in the batch
    valid_anchor_mask = (pos_counts > 0) & (neg_counts > 0)
    if not valid_anchor_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Positive attraction term: -alpha_pos * (1 / |P(i)|) * sum_{p} s_ip
    sim_pos = sim_matrix * mask_pos.float()
    pos_term = -alpha_pos * (sim_pos.sum(dim=1) / torch.clamp(pos_counts.float(), min=1.0))

    # Negative repulsion term: alpha_neg * logsumexp_{n in N(i)} (s_in)
    # Mask out non-negatives with -1e9 for stable logsumexp
    sim_neg = sim_matrix.masked_fill(~mask_neg, -1e9)
    neg_term = alpha_neg * torch.logsumexp(sim_neg, dim=1)

    anchor_loss = pos_term + neg_term  # (B,)
    valid_anchor_loss = anchor_loss[valid_anchor_mask]
    valid_labels = labels.view(-1)[valid_anchor_mask]

    # Class-balanced aggregation: average within each class first, then across classes present
    unique_classes = torch.unique(valid_labels)
    class_losses = []
    for c in unique_classes:
        c_mask = (valid_labels == c)
        class_losses.append(valid_anchor_loss[c_mask].mean())

    if len(class_losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    total_con_loss = torch.stack(class_losses).mean()
    return total_con_loss


def train_single_run(
    cfg_name: str,
    alpha_pos: float,
    alpha_neg: float,
    tau: float,
    use_contrastive: bool,
    seed: int,
    X_tr_t: torch.Tensor,
    X_tr_v: torch.Tensor,
    y_tr: torch.Tensor,
    X_va_t: torch.Tensor,
    X_va_v: torch.Tensor,
    y_va: torch.Tensor,
    X_te_t: torch.Tensor,
    X_te_v: torch.Tensor,
    y_te: torch.Tensor,
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    set_seed(seed)

    # Create class-balanced WeightedRandomSampler for training
    class_counts = torch.bincount(y_tr, minlength=NUM_CLASSES).float()
    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_tr),
        replacement=True,
    )

    train_dataset = TensorDataset(X_tr_t, X_tr_v, y_tr)
    val_dataset = TensorDataset(X_va_t, X_va_v, y_va)
    test_dataset = TensorDataset(X_te_t, X_te_v, y_te)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = RegularizedGatedMultimodalClassifierV2(
        text_dim=768,
        video_dim=768,
        hidden_dim=512,
        num_classes=NUM_CLASSES,
        modality_dropout=0.15,
        classifier_dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_macro_f1 = -1.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        for b_text, b_video, b_y in train_loader:
            b_text = b_text.to(device)
            b_video = b_video.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            out = model(b_text, b_video)

            l_fused = ce_loss_fn(out["logits"], b_y)
            l_aux_t = ce_loss_fn(out["aux_text_logits"], b_y)
            l_aux_v = ce_loss_fn(out["aux_video_logits"], b_y)

            loss = l_fused + 0.3 * l_aux_t + 0.3 * l_aux_v

            if use_contrastive:
                l_con = decoupled_supervised_contrastive_loss(
                    out["z_con"],
                    b_y,
                    alpha_pos=alpha_pos,
                    alpha_neg=alpha_neg,
                    tau=tau,
                )
                loss = loss + 0.5 * l_con

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * b_y.size(0)

        # Validation evaluation strictly on VAL macro-F1
        val_metrics = evaluate_model(model, val_loader, device)
        val_macro_f1 = val_metrics["macro_f1"]

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model checkpoint and evaluate once on val and test
    model.load_state_dict(best_model_state)
    final_val = evaluate_model(model, val_loader, device)
    final_test = evaluate_model(model, test_loader, device)

    return {
        "config": cfg_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "val": final_val,
        "test": final_test,
        "best_model_state": best_model_state,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    all_preds, all_targets = [], []
    all_gate_t, all_gate_v = [], []

    with torch.no_grad():
        for b_text, b_video, b_y in loader:
            b_text = b_text.to(device)
            b_video = b_video.to(device)
            out = model(b_text, b_video)

            preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
            gates = out["gate_scores"].cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(b_y.numpy())
            all_gate_t.extend(gates[:, 0])
            all_gate_v.extend(gates[:, 1])

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    recalls = {lbl: report[lbl]["recall"] for lbl in LABEL_NAMES}

    mean_gate_t = float(np.mean(all_gate_t))
    mean_gate_v = float(np.mean(all_gate_v))

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(acc),
        "recalls": recalls,
        "gate_weights": {
            "text": mean_gate_t,
            "video": mean_gate_v,
        },
    }


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Fusion V2 Multimodal Classifier with Decoupled SupCon")
    parser.add_argument("--features_dir", type=str, default="embeddings_v3", help="Directory with train_features.npz, val_features.npz, test_features.npz")
    parser.add_argument("--output_dir", type=str, default="results_v3", help="Directory where models and results will be saved")
    parser.add_argument("--configs", nargs="+", default=["V2"], help="Configurations to train (choose from: V0, V1, V2, V3, V4, V5, V6; default: V2)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds to evaluate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max_epochs", type=int, default=50, help="Max training epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # GPU Memory Setup
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.80, device=0)
        device = torch.device("cuda:0")
        logger.info("Using GPU: %s (memory fraction 0.80 set)", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")

    # Load precomputed features from npz cache
    cache_root = Path(args.features_dir)
    tr_data = np.load(cache_root / "train_features.npz")
    va_data = np.load(cache_root / "val_features.npz")
    te_data = np.load(cache_root / "test_features.npz")

    X_tr_t = torch.tensor(tr_data["X_text"], dtype=torch.float32)
    X_tr_v = torch.tensor(tr_data["X_video"], dtype=torch.float32)
    y_tr = torch.tensor(tr_data["y"], dtype=torch.int64)

    X_va_t = torch.tensor(va_data["X_text"], dtype=torch.float32)
    X_va_v = torch.tensor(va_data["X_video"], dtype=torch.float32)
    y_va = torch.tensor(va_data["y"], dtype=torch.int64)

    X_te_t = torch.tensor(te_data["X_text"], dtype=torch.float32)
    X_te_v = torch.tensor(te_data["X_video"], dtype=torch.float32)
    y_te = torch.tensor(te_data["y"], dtype=torch.int64)

    logger.info("Loaded datasets: Train=%d, Val=%d, Test=%d", len(y_tr), len(y_va), len(y_te))

    # Define Configurations V0..V6
    all_configs = {
        "V0": {"use_con": False, "alpha_pos": 0.0, "alpha_neg": 0.0, "tau": 0.0, "desc": "No contrastive (cross-entropy only)"},
        "V1": {"use_con": True, "alpha_pos": 1.0, "alpha_neg": 1.0, "tau": 0.1, "desc": "pos=1.0, neg=1.0, tau=0.1"},
        "V2": {"use_con": True, "alpha_pos": 1.0, "alpha_neg": 1.0, "tau": 0.2, "desc": "pos=1.0, neg=1.0, tau=0.2 (Best)"},
        "V3": {"use_con": True, "alpha_pos": 1.0, "alpha_neg": 0.5, "tau": 0.1, "desc": "pos=1.0, neg=0.5, tau=0.1"},
        "V4": {"use_con": True, "alpha_pos": 1.0, "alpha_neg": 0.5, "tau": 0.2, "desc": "pos=1.0, neg=0.5, tau=0.2"},
        "V5": {"use_con": True, "alpha_pos": 1.0, "alpha_neg": 0.25, "tau": 0.1, "desc": "pos=1.0, neg=0.25, tau=0.1"},
        "V6": {"use_con": True, "alpha_pos": 1.0, "alpha_neg": 0.25, "tau": 0.2, "desc": "pos=1.0, neg=0.25, tau=0.2"},
    }

    configs = {k: all_configs[k] for k in args.configs if k in all_configs}
    seeds = args.seeds
    all_runs = {}
    aggregated_results = {}
    best_overall_val_f1 = -1.0
    best_overall_state = None

    for cfg_name, cfg in configs.items():
        logger.info("\n" + "=" * 80)
        logger.info("STARTING CONFIGURATION %s: %s", cfg_name, cfg["desc"])
        logger.info("=" * 80)

        cfg_runs = []
        for seed in seeds:
            logger.info("Running %s with seed %d...", cfg_name, seed)
            run_res = train_single_run(
                cfg_name=cfg_name,
                alpha_pos=cfg["alpha_pos"],
                alpha_neg=cfg["alpha_neg"],
                tau=cfg["tau"],
                use_contrastive=cfg["use_con"],
                seed=seed,
                X_tr_t=X_tr_t,
                X_tr_v=X_tr_v,
                y_tr=y_tr,
                X_va_t=X_va_t,
                X_va_v=X_va_v,
                y_va=y_va,
                X_te_t=X_te_t,
                X_te_v=X_te_v,
                y_te=y_te,
                device=device,
            )
            cfg_runs.append(run_res)
            logger.info(
                "[%s seed %d] Best Epoch: %d | Val Macro-F1: %.4f | Test Macro-F1: %.4f (Gate Text: %.3f, Video: %.3f)",
                cfg_name, seed, run_res["best_epoch"],
                run_res["val"]["macro_f1"], run_res["test"]["macro_f1"],
                run_res["test"]["gate_weights"]["text"], run_res["test"]["gate_weights"]["video"],
            )

        all_runs[cfg_name] = cfg_runs

        # Aggregate across seeds
        val_f1s = [r["val"]["macro_f1"] for r in cfg_runs]
        test_f1s = [r["test"]["macro_f1"] for r in cfg_runs]
        test_wf1s = [r["test"]["weighted_f1"] for r in cfg_runs]
        test_accs = [r["test"]["accuracy"] for r in cfg_runs]
        gate_videos = [r["test"]["gate_weights"]["video"] for r in cfg_runs]

        # Aggregate per-class recalls
        class_recalls_agg = {}
        for lbl in LABEL_NAMES:
            lbl_vals = [r["test"]["recalls"][lbl] for r in cfg_runs]
            class_recalls_agg[lbl] = {
                "mean": float(np.mean(lbl_vals)),
                "std": float(np.std(lbl_vals)),
            }

        aggregated_results[cfg_name] = {
            "description": cfg["desc"],
            "val_macro_f1": {"mean": float(np.mean(val_f1s)), "std": float(np.std(val_f1s)), "values": val_f1s},
            "test_macro_f1": {"mean": float(np.mean(test_f1s)), "std": float(np.std(test_f1s)), "values": test_f1s},
            "test_weighted_f1": {"mean": float(np.mean(test_wf1s)), "std": float(np.std(test_wf1s)), "values": test_wf1s},
            "test_accuracy": {"mean": float(np.mean(test_accs)), "std": float(np.std(test_accs)), "values": test_accs},
            "test_gate_video": {"mean": float(np.mean(gate_videos)), "std": float(np.std(gate_videos)), "values": gate_videos},
            "per_class_recalls": class_recalls_agg,
        }

    # Save detailed runs and aggregated results
    summary_data = {
        "aggregated": aggregated_results,
        "runs": all_runs,
    }

    out_file = out_dir / "fusion_v2_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved full Fusion V2 results to %s", out_file)

    # Save best model checkpoint
    for cfg_runs in all_runs.values():
        for run in cfg_runs:
            if run["val"]["macro_f1"] > best_overall_val_f1:
                best_overall_val_f1 = run["val"]["macro_f1"]
                best_overall_state = run.get("best_model_state", None)

    if best_overall_state is not None:
        torch.save({"model_state_dict": best_overall_state}, out_dir / "best_fusion_v2_model.pt")
        logger.info("Saved best overall checkpoint to %s (Val Macro-F1: %.4f)", out_dir / "best_fusion_v2_model.pt", best_overall_val_f1)

    # Print summary table
    print("\n" + "=" * 105)
    print("STEP 4 & 5 - FUSION V2 EVALUATION (Mean +- Std over seeds 42, 43, 44)")
    print("=" * 105)
    print(f"{'Config':<8} | {'Description':<32} | {'Val Macro-F1':<16} | {'Test Macro-F1':<16} | {'Test Weighted-F1':<16} | {'Video Gate %':<10}")
    print("-" * 105)
    for cfg_name, agg in aggregated_results.items():
        v_f1 = f"{agg['val_macro_f1']['mean']:.4f} +/- {agg['val_macro_f1']['std']:.4f}"
        t_f1 = f"{agg['test_macro_f1']['mean']:.4f} +/- {agg['test_macro_f1']['std']:.4f}"
        t_wf1 = f"{agg['test_weighted_f1']['mean']:.4f} +/- {agg['test_weighted_f1']['std']:.4f}"
        v_gate = f"{agg['test_gate_video']['mean']*100:.2f}%"
        print(f"{cfg_name:<8} | {agg['description']:<32} | {v_f1:<16} | {t_f1:<16} | {t_wf1:<16} | {v_gate:<10}")
    print("=" * 105)


if __name__ == "__main__":
    main()
