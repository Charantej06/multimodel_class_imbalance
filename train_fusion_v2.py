"""
train_fusion_v2.py
==================
Modality- and class-count-agnostic training and evaluation script for Fusion V2
with Decoupled Supervised Contrastive Loss across configurations V0..V6 and random seeds.

Hardware:
  - Configurable GPU memory fraction (default 0.80) if CUDA available, with graceful CPU fallback.

Training setup:
  - Class-balanced WeightedRandomSampler on y_train with weights 1 / class_count
  - Dynamic modality inputs (supports arbitrary modalities, e.g. text+video, audio+video)
  - Total Loss: L_CE(fused) + sum_m (0.6/M * L_CE(aux_m)) + 0.5 * L_con
  - Model selection strictly on VAL macro-F1. Test set evaluated once at selected epoch.
  - Per-run config JSON saved in output_dir.
"""

from __future__ import annotations
import argparse
import copy
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score, accuracy_score

from models.regularized_gated_fusion_v2 import RegularizedGatedMultimodalClassifierV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_fusion_v2")

MELD_LABEL_NAMES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
CREMAD_LABEL_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]


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
    """Decoupled Supervised Contrastive Loss with class-balanced anchor averaging."""
    device = z_con.device
    B = z_con.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    sim_matrix = torch.matmul(z_con, z_con.T) / tau
    labels = labels.view(-1, 1)
    mask_same_class = torch.eq(labels, labels.T).to(device)
    mask_self = torch.eye(B, dtype=torch.bool, device=device)
    mask_pos = mask_same_class & (~mask_self)
    mask_neg = ~mask_same_class

    pos_counts = mask_pos.sum(dim=1)
    neg_counts = mask_neg.sum(dim=1)

    valid_anchor_mask = (pos_counts > 0) & (neg_counts > 0)
    if not valid_anchor_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    sim_pos = sim_matrix * mask_pos.float()
    pos_term = -alpha_pos * (sim_pos.sum(dim=1) / torch.clamp(pos_counts.float(), min=1.0))
    sim_neg = sim_matrix.masked_fill(~mask_neg, -1e9)
    neg_term = alpha_neg * torch.logsumexp(sim_neg, dim=1)

    anchor_loss = pos_term + neg_term
    valid_anchor_loss = anchor_loss[valid_anchor_mask]
    valid_labels = labels.view(-1)[valid_anchor_mask]

    unique_classes = torch.unique(valid_labels)
    class_losses = []
    for c in unique_classes:
        c_mask = (valid_labels == c)
        class_losses.append(valid_anchor_loss[c_mask].mean())

    if len(class_losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return torch.stack(class_losses).mean()


def train_single_run(
    cfg_name: str,
    alpha_pos: float,
    alpha_neg: float,
    tau: float,
    use_contrastive: bool,
    seed: int,
    X_tr: Dict[str, torch.Tensor],
    y_tr: torch.Tensor,
    X_va: Dict[str, torch.Tensor],
    y_va: torch.Tensor,
    X_te: Dict[str, torch.Tensor],
    y_te: torch.Tensor,
    modalities: List[str],
    modality_dims: Dict[str, int],
    num_classes: int,
    label_names: List[str],
    device: torch.device,
    max_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    set_seed(seed)

    # Class-balanced WeightedRandomSampler
    class_counts = torch.bincount(y_tr, minlength=num_classes).float()
    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_tr),
        replacement=True,
    )

    tr_tensors = [X_tr[m] for m in modalities] + [y_tr]
    va_tensors = [X_va[m] for m in modalities] + [y_va]
    te_tensors = [X_te[m] for m in modalities] + [y_te]

    train_dataset = TensorDataset(*tr_tensors)
    val_dataset = TensorDataset(*va_tensors)
    test_dataset = TensorDataset(*te_tensors)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = RegularizedGatedMultimodalClassifierV2(
        modalities=modalities,
        modality_dims=modality_dims,
        hidden_dim=512,
        num_classes=num_classes,
        modality_dropout=0.15,
        classifier_dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_macro_f1 = -1.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    aux_weight = 0.6 / len(modalities)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_data in train_loader:
            b_embs = {m: batch_data[idx].to(device) for idx, m in enumerate(modalities)}
            b_y = batch_data[-1].to(device)

            optimizer.zero_grad()
            out = model(**b_embs)

            l_fused = ce_loss_fn(out["logits"], b_y)
            l_aux = sum(aux_weight * ce_loss_fn(out[f"aux_{m}_logits"], b_y) for m in modalities)
            loss = l_fused + l_aux

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

        val_metrics = evaluate_model(model, val_loader, modalities, label_names, device)
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

    model.load_state_dict(best_model_state)
    final_val = evaluate_model(model, val_loader, modalities, label_names, device)
    final_test = evaluate_model(model, test_loader, modalities, label_names, device)

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
    modalities: List[str],
    label_names: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    all_preds, all_targets = [], []
    all_gates = {m: [] for m in modalities}

    with torch.no_grad():
        for batch_data in loader:
            b_embs = {m: batch_data[idx].to(device) for idx, m in enumerate(modalities)}
            b_y = batch_data[-1]
            out = model(**b_embs)

            preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
            gates = out["gate_scores"].cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(b_y.numpy())
            for idx, m in enumerate(modalities):
                all_gates[m].extend(gates[:, idx])

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    recalls = {lbl: report[lbl]["recall"] for lbl in label_names}

    mean_gates = {m: float(np.mean(all_gates[m])) for m in modalities}

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(acc),
        "recalls": recalls,
        "gate_weights": mean_gates,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fusion V2 Multimodal Classifier with Decoupled SupCon")
    parser.add_argument("--features_dir", type=str, default="embeddings_v3", help="Features directory containing train_features.npz, val_features.npz, test_features.npz")
    parser.add_argument("--output_dir", type=str, default="results_v3", help="Directory where models and results will be saved")
    parser.add_argument("--modalities", nargs="+", default=["text", "video"], help="Modalities to use (e.g. text video, audio video; default: text video)")
    parser.add_argument("--label_map", type=str, default=None, help="Path to label_map.json")
    parser.add_argument("--configs", nargs="+", default=["V2"], help="Configurations to train (choose from: V0..V6; default: V2)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds to evaluate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max_epochs", type=int, default=50, help="Max training epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--cuda_mem_fraction", type=float, default=0.80, help="CUDA per-process memory fraction (default: 0.80)")
    return parser.parse_args()


def load_split_features(npz_path: Path, modalities: List[str]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Loads modality tensors and labels from an NPZ file."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature file not found: {npz_path}")
    data = np.load(npz_path)

    embs: Dict[str, torch.Tensor] = {}
    for m in modalities:
        key = None
        for cand in [f"X_{m}", m, f"emb_{m}", f"{m}_feat"]:
            if cand in data:
                key = cand
                break
        if key is None:
            raise KeyError(f"Could not find tensor for modality '{m}' in {npz_path}. Available keys: {list(data.keys())}")
        embs[m] = torch.tensor(data[key], dtype=torch.float32)

    y = torch.tensor(data["y"], dtype=torch.int64)
    return embs, y


def resolve_label_names(args_label_map: Optional[str], features_dir: Path, max_y: int, modalities: List[str]) -> Tuple[int, List[str]]:
    """Resolves num_classes and ordered label names from label_map.json or dataset conventions."""
    candidate_paths = []
    if args_label_map:
        candidate_paths.append(Path(args_label_map))
    candidate_paths.extend([
        features_dir / "label_map.json",
        Path("data/label_map.json"),
        Path("cremad/label_map.json"),
    ])

    for p in candidate_paths:
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                lmap = json.load(f)
            sorted_labels = sorted(lmap.keys(), key=lambda k: lmap[k])
            logger.info("Loaded label map from %s (%d classes): %s", p, len(sorted_labels), sorted_labels)
            return len(sorted_labels), sorted_labels

    if "audio" in modalities and max_y == 5:
        logger.info("Using CREMA-D 6-class labels (audio modality detected)")
        return 6, CREMAD_LABEL_NAMES

    logger.info("Defaulting to MELD 7-class labels")
    return 7, MELD_LABEL_NAMES


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-run config JSON
    config_dict = vars(args)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Saved run configuration to: %s", out_dir / "config.json")

    # GPU Memory Setup
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(args.cuda_mem_fraction, device=0)
            logger.info("Using GPU: %s (memory fraction %.2f set)", torch.cuda.get_device_name(0), args.cuda_mem_fraction)
        except Exception as e:
            logger.warning("Could not set CUDA memory fraction: %s", e)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, using CPU")

    features_dir = Path(args.features_dir)
    modalities = list(args.modalities)

    X_tr, y_tr = load_split_features(features_dir / "train_features.npz", modalities)
    X_va, y_va = load_split_features(features_dir / "val_features.npz", modalities)
    X_te, y_te = load_split_features(features_dir / "test_features.npz", modalities)

    modality_dims = {m: X_tr[m].shape[1] for m in modalities}
    logger.info("Modalities: %s | Dimensions: %s", modalities, modality_dims)
    logger.info("Loaded dataset splits: Train=%d, Val=%d, Test=%d", len(y_tr), len(y_va), len(y_te))

    max_y = int(torch.max(torch.cat([y_tr, y_va, y_te])).item())
    num_classes, label_names = resolve_label_names(args.label_map, features_dir, max_y, modalities)

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
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                X_te=X_te,
                y_te=y_te,
                modalities=modalities,
                modality_dims=modality_dims,
                num_classes=num_classes,
                label_names=label_names,
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            cfg_runs.append(run_res)
            gates_str = ", ".join(f"{m.capitalize()}: {run_res['test']['gate_weights'][m]:.3f}" for m in modalities)
            logger.info(
                "[%s seed %d] Best Epoch: %d | Val Macro-F1: %.4f | Test Macro-F1: %.4f (Gates %s)",
                cfg_name, seed, run_res["best_epoch"],
                run_res["val"]["macro_f1"], run_res["test"]["macro_f1"],
                gates_str,
            )

        all_runs[cfg_name] = cfg_runs

        val_f1s = [r["val"]["macro_f1"] for r in cfg_runs]
        test_f1s = [r["test"]["macro_f1"] for r in cfg_runs]
        test_wf1s = [r["test"]["weighted_f1"] for r in cfg_runs]
        test_accs = [r["test"]["accuracy"] for r in cfg_runs]

        class_recalls_agg = {}
        for lbl in label_names:
            lbl_vals = [r["test"]["recalls"][lbl] for r in cfg_runs]
            class_recalls_agg[lbl] = {
                "mean": float(np.mean(lbl_vals)),
                "std": float(np.std(lbl_vals)),
            }

        gates_agg = {}
        for m in modalities:
            g_vals = [r["test"]["gate_weights"][m] for r in cfg_runs]
            gates_agg[m] = {"mean": float(np.mean(g_vals)), "std": float(np.std(g_vals))}

        aggregated_results[cfg_name] = {
            "description": cfg["desc"],
            "val_macro_f1": {"mean": float(np.mean(val_f1s)), "std": float(np.std(val_f1s)), "values": val_f1s},
            "test_macro_f1": {"mean": float(np.mean(test_f1s)), "std": float(np.std(test_f1s)), "values": test_f1s},
            "test_weighted_f1": {"mean": float(np.mean(test_wf1s)), "std": float(np.std(test_wf1s)), "values": test_wf1s},
            "test_accuracy": {"mean": float(np.mean(test_accs)), "std": float(np.std(test_accs)), "values": test_accs},
            "gate_weights": gates_agg,
            "per_class_recalls": class_recalls_agg,
        }

    summary_data = {
        "aggregated": aggregated_results,
        "runs": all_runs,
        "modalities": modalities,
        "num_classes": num_classes,
        "label_names": label_names,
    }

    out_file = out_dir / "fusion_v2_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved full Fusion V2 results to %s", out_file)

    for cfg_runs in all_runs.values():
        for run in cfg_runs:
            if run["val"]["macro_f1"] > best_overall_val_f1:
                best_overall_val_f1 = run["val"]["macro_f1"]
                best_overall_state = run.get("best_model_state", None)

    if best_overall_state is not None:
        torch.save({"model_state_dict": best_overall_state}, out_dir / "best_fusion_v2_model.pt")
        logger.info("Saved best overall checkpoint to %s (Val Macro-F1: %.4f)", out_dir / "best_fusion_v2_model.pt", best_overall_val_f1)

    print("\n" + "=" * 105)
    print("FUSION V2 EVALUATION (Mean +- Std over seeds)")
    print("=" * 105)
    print(f"{'Config':<8} | {'Description':<32} | {'Val Macro-F1':<16} | {'Test Macro-F1':<16} | {'Test Weighted-F1':<16}")
    print("-" * 105)
    for cfg_name, agg in aggregated_results.items():
        v_f1 = f"{agg['val_macro_f1']['mean']:.4f} +/- {agg['val_macro_f1']['std']:.4f}"
        t_f1 = f"{agg['test_macro_f1']['mean']:.4f} +/- {agg['test_macro_f1']['std']:.4f}"
        t_wf1 = f"{agg['test_weighted_f1']['mean']:.4f} +/- {agg['test_weighted_f1']['std']:.4f}"
        print(f"{cfg_name:<8} | {agg['description']:<32} | {v_f1:<16} | {t_f1:<16} | {t_wf1:<16}")
    print("=" * 105)


if __name__ == "__main__":
    main()
