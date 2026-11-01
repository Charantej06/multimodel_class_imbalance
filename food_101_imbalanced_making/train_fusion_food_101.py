"""
food_101_imbalanced_making/train_fusion_food_101.py
==================================================
Trains Fusion V2 on real extracted unimodal features (ModernBERT 768-d + DINOv2 768-d)
with increased epochs (40 epochs) and Decoupled Supervised Contrastive Loss.

Once trained, extracts 512-d joint multimodal latents to:
  data/food101_lt/latents_train.npz
  data/food101_lt/latents_val.npz
  data/food101_lt/latents_test.npz
"""

from __future__ import annotations
import os
import sys
import copy
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "final_rcs_package"
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fusion_model_food_101 import RegularizedGatedMultimodalClassifierFood101

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_fusion_food101")

DATA_DIR = REPO_ROOT / "data" / "food101_lt"
CHECKPOINT_PATH = DATA_DIR / "best_fusion_v2_food101.pt"


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def decoupled_supervised_contrastive_loss(
    proj_feats: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.2,
    alpha_pos: float = 1.0,
    alpha_neg: float = 1.0,
) -> torch.Tensor:
    """
    Decoupled Supervised Contrastive Loss with anchor averaging.
    proj_feats: (B, 128) L2-normalized
    labels: (B,)
    """
    device = proj_feats.device
    B = proj_feats.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=device)

    # Cosine similarity matrix scaled by tau
    sim_matrix = torch.matmul(proj_feats, proj_feats.T) / tau  # (B, B)

    # Numerically stable exp
    sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_exp = torch.exp(sim_matrix - sim_max.detach())

    # Mask self-contrast
    diag_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim_exp_no_diag = sim_exp.masked_fill(diag_mask, 0.0)

    # Positive mask
    labels = labels.view(-1, 1)
    pos_mask = torch.eq(labels, labels.T) & (~diag_mask)
    neg_mask = (~torch.eq(labels, labels.T)) & (~diag_mask)

    # Decoupled denominator
    denom = (alpha_neg * (sim_exp_no_diag * neg_mask.float()).sum(dim=1)) + 1e-8

    num_pos = pos_mask.sum(dim=1)
    valid_anchors = num_pos > 0

    if not valid_anchors.any():
        return torch.tensor(0.0, device=device)

    # Anchor loss
    sim_exp_pos = sim_exp_no_diag * pos_mask.float()
    log_terms = torch.log(1e-8 + (sim_exp_pos / denom.unsqueeze(1)))
    loss_per_anchor = - (log_terms * pos_mask.float()).sum(dim=1) / torch.clamp(num_pos.float(), min=1.0)

    return loss_per_anchor[valid_anchors].mean() * alpha_pos


def load_dataset_tensors(data_dir: Path) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    f_tr = np.load(data_dir / "features_train.npz")
    f_va = np.load(data_dir / "features_val.npz")
    f_te = np.load(data_dir / "features_test.npz")

    ds_tr = TensorDataset(
        torch.from_numpy(f_tr["text_embeds"]).float(),
        torch.from_numpy(f_tr["image_embeds"]).float(),
        torch.from_numpy(f_tr["labels"]).long(),
    )
    ds_va = TensorDataset(
        torch.from_numpy(f_va["text_embeds"]).float(),
        torch.from_numpy(f_va["image_embeds"]).float(),
        torch.from_numpy(f_va["labels"]).long(),
    )
    ds_te = TensorDataset(
        torch.from_numpy(f_te["text_embeds"]).float(),
        torch.from_numpy(f_te["image_embeds"]).float(),
        torch.from_numpy(f_te["labels"]).long(),
    )
    return ds_tr, ds_va, ds_te


def train_fusion_model(
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_supcon: float = 0.5,
    patience: int = 10,
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.70, device=0)
    logger.info("Training Fusion V2 on %s for %d epochs...", device, epochs)

    ds_tr, ds_va, ds_te = load_dataset_tensors(DATA_DIR)
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)
    loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    model = RegularizedGatedMultimodalClassifierFood101(
        text_dim=768,
        image_dim=768,
        hidden_dim=512,
        num_classes=101,
        modality_dropout=0.15,
        classifier_dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_macro_f1 = 0.0
    best_state = None
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_con = 0.0

        for txt, img, y in loader_tr:
            txt, img, y = txt.to(device), img.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(txt, img)
            logits = out["logits"]
            proj_feat = out["z_con"]
            aux_txt_logits = out["aux_text_logits"]
            aux_img_logits = out["aux_image_logits"]

            ce_fused = ce_loss_fn(logits, y)
            ce_aux = 0.2 * ce_loss_fn(aux_txt_logits, y) + 0.2 * ce_loss_fn(aux_img_logits, y)
            supcon = decoupled_supervised_contrastive_loss(proj_feat, y)

            loss = ce_fused + ce_aux + lambda_supcon * supcon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_fused.item()
            total_con += supcon.item()

        scheduler.step()

        # Validation
        model.eval()
        va_preds, va_targets = [], []
        with torch.no_grad():
            for txt, img, y in loader_va:
                txt, img = txt.to(device), img.to(device)
                logits = model(txt, img)["logits"]
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                va_preds.extend(preds)
                va_targets.extend(y.numpy())

        val_f1 = f1_score(va_targets, va_preds, average="macro", zero_division=0)
        logger.info("Epoch %02d/%d | Loss: %.4f (CE: %.4f, SupCon: %.4f) | Val Macro-F1: %.4f",
                    ep, epochs, total_loss / len(loader_tr), total_ce / len(loader_tr), total_con / len(loader_tr), val_f1)

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_state, CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered at epoch %d. Best Val Macro-F1: %.4f", ep, best_val_macro_f1)
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    logger.info("Loaded best model checkpoint. Saving extracted 512-d latents...")

    # Extract 512-d latents
    model.eval()
    for name, loader in [("train", loader_tr), ("val", loader_va), ("test", loader_te)]:
        latents_list, labels_list = [], []
        # Non-shuffled loader for extraction
        ds = loader.dataset
        full_loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for txt, img, y in full_loader:
                txt, img = txt.to(device), img.to(device)
                fused = model.extract_fused_features(txt, img)
                latents_list.append(fused.cpu().numpy())
                labels_list.append(y.numpy())

        z_all = np.concatenate(latents_list, axis=0)
        y_all = np.concatenate(labels_list, axis=0)
        out_npz = DATA_DIR / f"latents_{name}.npz"
        np.savez_compressed(out_npz, latents=z_all, labels=y_all)
        logger.info("Saved %s latents: %s to %s", name, z_all.shape, out_npz)

    logger.info("Fusion V2 training and latent extraction complete!")


if __name__ == "__main__":
    train_fusion_model(epochs=40)

