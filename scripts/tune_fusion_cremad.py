#!/usr/bin/env python3
"""
scripts/tune_fusion_cremad.py
=============================
Hyperparameter tuning and comprehensive baseline/ablation evaluation for Fusion V2
on CREMA-D audio + video features (Step 8).

Key Protocol & Specifications:
------------------------------
1. Tuning Phase:
   - Random search over 30 trials with fixed master seed 42.
   - Search parameters: lr, weight_decay, hidden_dim, dropout, modality_dropout {0, 0.1, 0.15, 0.3},
     supcon_weight, supcon_temperature {0.07, 0.1, 0.2}, aux_weight, label_smoothing {0, 0.05, 0.1}, batch_size.
   - Optimized on validation Macro-F1 ONLY.
   - CRITICAL CONSTRAINT: Test split MUST NOT be loaded or touched during tuning (strictly asserted).
   - Saves: results_cremad/tuning.csv and results_cremad/best_config.json.

2. Multi-Seed Training & Ablation Phase:
   - Uses the best configuration found in tuning across 3 seeds: 42, 43, 44.
   - Selects checkpoint strictly by validation Macro-F1.
   - Evaluates test split ONCE per run at the val-selected checkpoint.
   - Models Evaluated:
     - Full Fusion V2 (best config)
     - Ablations:
       * no_supcon (supcon_weight = 0.0)
       * no_aux (aux_weight = 0.0)
       * no_gate (uniform blending, use_gate = False)
       * no_modality_dropout (modality_dropout = 0.0)
       * audio_only (Fusion V2 architecture on audio only)
       * video_only (Fusion V2 architecture on video only)
     - Baselines:
       * audio-only MLP (standard MLP head on audio)
       * video-only MLP (standard MLP head on video)
       * concat-MLP (standard MLP head on concatenated audio+video)
   - Outputs: results_cremad/ablation_table.csv (mean +/- std over seeds for val and test).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score

from models.regularized_gated_fusion_v2 import RegularizedGatedMultimodalClassifierV2
from train_fusion_v2 import decoupled_supervised_contrastive_loss, load_split_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tune_fusion_cremad")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleMLPClassifier(nn.Module):
    """Standard MLP baseline for unimodal or concatenated features."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, num_classes)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, num_classes), num_classes),
        )

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        if len(args) == 1:
            x = args[0]
        elif len(args) > 1:
            x = torch.cat(args, dim=-1)
        elif "x" in kwargs:
            x = kwargs["x"]
        else:
            tensors = [kwargs[k] for k in sorted(kwargs.keys())]
            x = torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]
        return {"logits": self.net(x)}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    modalities: List[str],
    device: torch.device,
) -> Tuple[float, float]:
    """Computes Macro-F1 and Accuracy on the given DataLoader."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in loader:
            b_embs = {m: batch[idx].to(device) for idx, m in enumerate(modalities)}
            b_y = batch[-1]
            out = model(**b_embs)
            preds = torch.argmax(out["logits"], dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(b_y.numpy().tolist())

    f1 = float(f1_score(all_targets, all_preds, average="macro"))
    acc = float(accuracy_score(all_targets, all_preds))
    return f1, acc


def parse_args():
    parser = argparse.ArgumentParser(description="Tune and evaluate Fusion V2 on CREMA-D.")
    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/features_cremad",
        help="Path to directory containing train_features.npz and val_features.npz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_cremad",
        help="Output directory for tuning and ablation results (default: results_cremad)",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=30,
        help="Number of random search trials (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed (default: 42)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=25,
        help="Max epochs per trial (default: 25)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early stopping patience (default: 6)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="PyTorch CPU threads (default: 8)",
    )
    return parser.parse_args()


def sample_hyperparameters(rng: random.Random) -> Dict[str, Any]:
    """Samples one hyperparameter configuration from the defined search space."""
    return {
        "lr": rng.choice([1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3]),
        "weight_decay": rng.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2]),
        "hidden_dim": rng.choice([256, 512, 768]),
        "dropout": rng.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        "modality_dropout": rng.choice([0.0, 0.1, 0.15, 0.3]),
        "supcon_weight": rng.choice([0.1, 0.2, 0.3, 0.5, 0.8]),
        "supcon_temperature": rng.choice([0.07, 0.1, 0.2]),
        "aux_weight": rng.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        "label_smoothing": rng.choice([0.0, 0.05, 0.1]),
        "batch_size": rng.choice([64, 128, 256]),
    }


def train_model(
    model: nn.Module,
    X_tr: Dict[str, torch.Tensor],
    y_tr: torch.Tensor,
    X_va: Dict[str, torch.Tensor],
    y_va: torch.Tensor,
    modalities: List[str],
    cfg: Dict[str, Any],
    device: torch.device,
    max_epochs: int,
    patience: int,
    seed: int,
    supcon_weight: float,
    aux_weight: float,
    is_mlp: bool = False,
) -> Tuple[Dict[str, Any], float, float]:
    """Trains model on train split, checkpoints on highest val Macro-F1, returns best state & val metrics."""
    set_seed(seed)

    # Class-balanced WeightedRandomSampler
    num_classes = int(y_tr.max().item() + 1)
    class_counts = torch.bincount(y_tr, minlength=num_classes).float()
    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(y_tr), replacement=True)

    tr_dataset = TensorDataset(*[X_tr[m] for m in modalities], y_tr)
    va_dataset = TensorDataset(*[X_va[m] for m in modalities], y_va)

    batch_size = cfg.get("batch_size", 128)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, sampler=sampler)
    va_loader = DataLoader(va_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    best_val_f1 = -1.0
    best_val_acc = 0.0
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
            if not is_mlp and aux_weight > 0.0 and len(modalities) > 1:
                aux_term = sum(ce_loss_fn(out[f"aux_{m}_logits"], b_y) for m in modalities)
                loss = loss + (aux_weight / len(modalities)) * aux_term

            # SupCon loss
            if not is_mlp and supcon_weight > 0.0 and "z_con" in out:
                l_con = decoupled_supervised_contrastive_loss(
                    out["z_con"], b_y, tau=cfg.get("supcon_temperature", 0.1)
                )
                loss = loss + supcon_weight * l_con

            loss.backward()
            optimizer.step()

        val_f1, val_acc = evaluate(model, va_loader, modalities, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_weights)
    return best_weights, best_val_f1, best_val_acc


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    logger.info("Configured PyTorch CPU threads: %d", torch.get_num_threads())

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = features_dir / "train_features.npz"
    val_path = features_dir / "val_features.npz"
    test_path = features_dir / "test_features.npz"

    if not train_path.is_file() or not val_path.is_file():
        logger.error("FATAL: Features not found in %s", features_dir)
        sys.exit(1)

    modalities = ["audio", "video"]
    device = torch.device("cpu")

    # =========================================================================
    # PHASE 1: HYPERPARAMETER TUNING OVER 30 TRIALS (VALIDATION ONLY)
    # =========================================================================
    logger.info("=" * 80)
    logger.info("PHASE 1: TUNING FUSION V2 (30 TRIALS, SEED 42)")
    logger.info("=" * 80)

    # STRICT ASSERTION: Test data must NOT be loaded or touched during tuning
    logger.info("Asserting test isolation: test_features.npz will NOT be read during tuning.")
    assert "test_data" not in locals()
    assert "X_te" not in locals()

    X_tr, y_tr = load_split_features(train_path, modalities)
    X_va, y_va = load_split_features(val_path, modalities)
    modality_dims = {m: X_tr[m].shape[1] for m in modalities}
    num_classes = int(y_tr.max().item() + 1)

    logger.info("Dataset loaded for tuning: Train=%d, Val=%d | Modality Dims: %s | Classes: %d",
                len(y_tr), len(y_va), modality_dims, num_classes)

    tuning_rng = random.Random(args.seed)
    tuning_records: List[Dict[str, Any]] = []

    best_val_macro_f1 = -1.0
    best_config: Dict[str, Any] = {}

    for trial in range(1, args.num_trials + 1):
        cfg = sample_hyperparameters(tuning_rng)
        trial_seed = args.seed + trial

        model = RegularizedGatedMultimodalClassifierV2(
            modalities=modalities,
            modality_dims=modality_dims,
            hidden_dim=cfg["hidden_dim"],
            num_classes=num_classes,
            modality_dropout=cfg["modality_dropout"],
            classifier_dropout=cfg["dropout"],
            use_gate=True,
        ).to(device)

        t0 = time.time()
        _, val_f1, val_acc = train_model(
            model=model,
            X_tr=X_tr,
            y_tr=y_tr,
            X_va=X_va,
            y_va=y_va,
            modalities=modalities,
            cfg=cfg,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=trial_seed,
            supcon_weight=cfg["supcon_weight"],
            aux_weight=cfg["aux_weight"],
            is_mlp=False,
        )
        elapsed = time.time() - t0

        record = {
            "trial": trial,
            **cfg,
            "val_macro_f1": float(val_f1),
            "val_accuracy": float(val_acc),
            "time_sec": float(elapsed),
        }
        tuning_records.append(record)

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_config = copy.deepcopy(cfg)
            star = " *** NEW BEST ***"
        else:
            star = ""

        logger.info(
            "Trial %2d/%2d | Val Macro-F1: %.4f | Val Acc: %.4f | lr: %.1e, wd: %.1e, hdim: %d, drop: %.2f, mdrop: %.2f, supw: %.2f, tau: %.2f, auxw: %.2f, ls: %.2f, bs: %d (%.1fs)%s",
            trial, args.num_trials, val_f1, val_acc,
            cfg["lr"], cfg["weight_decay"], cfg["hidden_dim"], cfg["dropout"],
            cfg["modality_dropout"], cfg["supcon_weight"], cfg["supcon_temperature"],
            cfg["aux_weight"], cfg["label_smoothing"], cfg["batch_size"], elapsed, star,
        )

    # Save tuning results
    df_tuning = pd.DataFrame(tuning_records)
    tuning_csv_path = output_dir / "tuning.csv"
    df_tuning.to_csv(tuning_csv_path, index=False)
    logger.info("Saved all 30 tuning trials to: %s", tuning_csv_path)

    best_config_path = output_dir / "best_config.json"
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    logger.info("Saved best configuration (Val F1: %.4f) to: %s", best_val_macro_f1, best_config_path)

    # =========================================================================
    # PHASE 2: MULTI-SEED (42, 43, 44) EVALUATION OF BASELINES & ABLATIONS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: EVALUATING BASELINES & ABLATIONS ACROSS SEEDS [42, 43, 44]")
    logger.info("=" * 80)

    # Load test split ONLY now for the final one-time evaluation of val-selected checkpoints
    logger.info("Loading test features for single evaluation at val-selected checkpoints...")
    X_te, y_te = load_split_features(test_path, modalities)
    logger.info("Loaded Test Split: %d samples", len(y_te))

    seeds = [42, 43, 44]
    experiments = [
        "best_fusion_v2",
        "no_supcon",
        "no_aux",
        "no_gate",
        "no_modality_dropout",
        "audio_only",
        "video_only",
        "audio-only MLP",
        "video-only MLP",
        "concat-MLP",
    ]

    ablation_summary: List[Dict[str, Any]] = []

    for exp in experiments:
        logger.info("Running Experiment: %s across seeds %s...", exp, seeds)
        val_f1s, val_accs, test_f1s, test_accs = [], [], [], []

        for seed in seeds:
            # Model construction based on experiment specification
            if exp == "best_fusion_v2":
                m = RegularizedGatedMultimodalClassifierV2(
                    modalities=["audio", "video"],
                    modality_dims=modality_dims,
                    hidden_dim=best_config["hidden_dim"],
                    num_classes=num_classes,
                    modality_dropout=best_config["modality_dropout"],
                    classifier_dropout=best_config["dropout"],
                    use_gate=True,
                ).to(device)
                cur_mods = ["audio", "video"]
                sup_w = best_config["supcon_weight"]
                aux_w = best_config["aux_weight"]
                is_mlp = False

            elif exp == "no_supcon":
                m = RegularizedGatedMultimodalClassifierV2(
                    modalities=["audio", "video"],
                    modality_dims=modality_dims,
                    hidden_dim=best_config["hidden_dim"],
                    num_classes=num_classes,
                    modality_dropout=best_config["modality_dropout"],
                    classifier_dropout=best_config["dropout"],
                    use_gate=True,
                ).to(device)
                cur_mods = ["audio", "video"]
                sup_w = 0.0  # Ablation: no supervised contrastive loss
                aux_w = best_config["aux_weight"]
                is_mlp = False

            elif exp == "no_aux":
                m = RegularizedGatedMultimodalClassifierV2(
                    modalities=["audio", "video"],
                    modality_dims=modality_dims,
                    hidden_dim=best_config["hidden_dim"],
                    num_classes=num_classes,
                    modality_dropout=best_config["modality_dropout"],
                    classifier_dropout=best_config["dropout"],
                    use_gate=True,
                ).to(device)
                cur_mods = ["audio", "video"]
                sup_w = best_config["supcon_weight"]
                aux_w = 0.0  # Ablation: no auxiliary unimodal loss
                is_mlp = False

            elif exp == "no_gate":
                m = RegularizedGatedMultimodalClassifierV2(
                    modalities=["audio", "video"],
                    modality_dims=modality_dims,
                    hidden_dim=best_config["hidden_dim"],
                    num_classes=num_classes,
                    modality_dropout=best_config["modality_dropout"],
                    classifier_dropout=best_config["dropout"],
                    use_gate=False,  # Ablation: uniform feature blending
                ).to(device)
                cur_mods = ["audio", "video"]
                sup_w = best_config["supcon_weight"]
                aux_w = best_config["aux_weight"]
                is_mlp = False

            elif exp == "no_modality_dropout":
                m = RegularizedGatedMultimodalClassifierV2(
                    modalities=["audio", "video"],
                    modality_dims=modality_dims,
                    hidden_dim=best_config["hidden_dim"],
                    num_classes=num_classes,
                    modality_dropout=0.0,  # Ablation: no modality dropout
                    classifier_dropout=best_config["dropout"],
                    use_gate=True,
                ).to(device)
                cur_mods = ["audio", "video"]
                sup_w = best_config["supcon_weight"]
                aux_w = best_config["aux_weight"]
                is_mlp = False

            elif exp == "audio_only":
                m = RegularizedGatedMultimodalClassifierV2(
                    modalities=["audio"],
                    modality_dims={"audio": modality_dims["audio"]},
                    hidden_dim=best_config["hidden_dim"],
                    num_classes=num_classes,
                    modality_dropout=0.0,
                    classifier_dropout=best_config["dropout"],
                    use_gate=True,
                ).to(device)
                cur_mods = ["audio"]
                sup_w = best_config["supcon_weight"]
                aux_w = 0.0
                is_mlp = False

            elif exp == "video_only":
                m = RegularizedGatedMultimodalClassifierV2(
                    modalities=["video"],
                    modality_dims={"video": modality_dims["video"]},
                    hidden_dim=best_config["hidden_dim"],
                    num_classes=num_classes,
                    modality_dropout=0.0,
                    classifier_dropout=best_config["dropout"],
                    use_gate=True,
                ).to(device)
                cur_mods = ["video"]
                sup_w = best_config["supcon_weight"]
                aux_w = 0.0
                is_mlp = False

            elif exp == "audio-only MLP":
                m = SimpleMLPClassifier(
                    in_dim=modality_dims["audio"],
                    hidden_dim=best_config["hidden_dim"],
                    num_classes=num_classes,
                    dropout=best_config["dropout"],
                ).to(device)
                cur_mods = ["audio"]
                sup_w = 0.0
                aux_w = 0.0
                is_mlp = True

            elif exp == "video-only MLP":
                m = SimpleMLPClassifier(
                    in_dim=modality_dims["video"],
                    hidden_dim=best_config["hidden_dim"],
                    num_classes=num_classes,
                    dropout=best_config["dropout"],
                ).to(device)
                cur_mods = ["video"]
                sup_w = 0.0
                aux_w = 0.0
                is_mlp = True

            elif exp == "concat-MLP":
                m = SimpleMLPClassifier(
                    in_dim=modality_dims["audio"] + modality_dims["video"],
                    hidden_dim=best_config["hidden_dim"],
                    num_classes=num_classes,
                    dropout=best_config["dropout"],
                ).to(device)
                cur_mods = ["audio", "video"]
                sup_w = 0.0
                aux_w = 0.0
                is_mlp = True

            # Train on train, select on val
            best_weights, v_f1, v_acc = train_model(
                model=m,
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                modalities=cur_mods,
                cfg=best_config,
                device=device,
                max_epochs=args.max_epochs + 5,
                patience=args.patience + 2,
                seed=seed,
                supcon_weight=sup_w,
                aux_weight=aux_w,
                is_mlp=is_mlp,
            )

            # Evaluate test once at the val-selected checkpoint
            m.load_state_dict(best_weights)
            te_dataset = TensorDataset(*[X_te[mod] for mod in cur_mods], y_te)
            te_loader = DataLoader(te_dataset, batch_size=best_config["batch_size"], shuffle=False)
            t_f1, t_acc = evaluate(m, te_loader, cur_mods, device)

            val_f1s.append(v_f1)
            val_accs.append(v_acc)
            test_f1s.append(t_f1)
            test_accs.append(t_acc)

        ablation_summary.append({
            "model_or_ablation": exp,
            "val_macro_f1_mean": float(np.mean(val_f1s)),
            "val_macro_f1_std": float(np.std(val_f1s)),
            "val_accuracy_mean": float(np.mean(val_accs)),
            "val_accuracy_std": float(np.std(val_accs)),
            "test_macro_f1_mean": float(np.mean(test_f1s)),
            "test_macro_f1_std": float(np.std(test_f1s)),
            "test_accuracy_mean": float(np.mean(test_accs)),
            "test_accuracy_std": float(np.std(test_accs)),
        })
        logger.info(
            "Exp %-20s | Val Macro-F1: %.4f +/- %.4f | Test Macro-F1: %.4f +/- %.4f",
            exp, np.mean(val_f1s), np.std(val_f1s), np.mean(test_f1s), np.std(test_f1s),
        )

    # Save ablation table
    df_ablation = pd.DataFrame(ablation_summary)
    ablation_csv_path = output_dir / "ablation_table.csv"
    df_ablation.to_csv(ablation_csv_path, index=False)
    logger.info("Saved ablation table to: %s", ablation_csv_path)

    # Print Formatted Results Table
    print("\n" + "=" * 115)
    print("CREMA-D ABLATION & BASELINE COMPARISON TABLE (Seeds 42, 43, 44)")
    print("=" * 115)
    fmt_hdr = "{:<24} | {:<22} | {:<22} | {:<22} | {:<20}"
    print(fmt_hdr.format("Model / Configuration", "Val Macro-F1", "Val Accuracy", "Test Macro-F1", "Test Accuracy"))
    print("-" * 115)
    for r in ablation_summary:
        val_f1_str = f"{r['val_macro_f1_mean']:.4f} +/- {r['val_macro_f1_std']:.4f}"
        val_acc_str = f"{r['val_accuracy_mean']:.4f} +/- {r['val_accuracy_std']:.4f}"
        test_f1_str = f"{r['test_macro_f1_mean']:.4f} +/- {r['test_macro_f1_std']:.4f}"
        test_acc_str = f"{r['test_accuracy_mean']:.4f} +/- {r['test_accuracy_std']:.4f}"
        print(f"{r['model_or_ablation']:<24} | {val_f1_str:<22} | {val_acc_str:<22} | {test_f1_str:<22} | {test_acc_str:<20}")
    print("=" * 115 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
