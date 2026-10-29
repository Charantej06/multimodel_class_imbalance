"""
extract_latents.py
==================
Extracts 512-d fused latent representations (fused_feat \\in R^{512}) from a trained
Fusion V2 checkpoint across train, val, and test splits.

Usage:
------
    python extract_latents.py --features_dir embeddings_v3 --checkpoint results_v3/best_fusion_v2_model.pt --output_dir results_rcs/latents
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.regularized_gated_fusion_v2 import RegularizedGatedMultimodalClassifierV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("extract_latents")

LABEL_NAMES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]


def extract_split(
    model: torch.nn.Module,
    X_text: torch.Tensor,
    X_video: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Passes unimodal embeddings through the frozen fusion model to extract fused representations."""
    ds = TensorDataset(X_text, X_video, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    feats, targets = [], []
    with torch.no_grad():
        for t_emb, v_emb, label_batch in loader:
            t_emb = t_emb.to(device)
            v_emb = v_emb.to(device)
            out = model(t_emb, v_emb)
            fused = out["fused_feat"].cpu().numpy()
            feats.append(fused)
            targets.extend(label_batch.numpy())

    return np.vstack(feats), np.array(targets, dtype=np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract 512-d fused latents from trained Fusion V2 model.")
    parser.add_argument(
        "--features_dir",
        type=str,
        default="embeddings_v3",
        help="Directory containing train_features.npz, val_features.npz, test_features.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final_rcs_package/latents/best_fusion_v2_model.pt",
        help="Path to trained Fusion V2 model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_rcs/latents",
        help="Directory where extracted latents will be saved",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument("--text_dim", type=int, default=768, help="Dimension of text embeddings")
    parser.add_argument("--video_dim", type=int, default=768, help="Dimension of video embeddings")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Dimension of fused representations")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of emotion classes")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.80, device=0)
    logger.info("Using device: %s", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    logger.info("Loading Fusion V2 checkpoint from %s...", ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = RegularizedGatedMultimodalClassifierV2(
        text_dim=args.text_dim,
        video_dim=args.video_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        modality_dropout=0.0,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    features_dir = Path(args.features_dir)
    splits = ["train", "val", "test"]

    for split in splits:
        npz_file = features_dir / f"{split}_features.npz"
        if not npz_file.exists():
            logger.warning("Feature file %s not found. Skipping split '%s'.", npz_file, split)
            continue

        data = np.load(npz_file)
        X_t = torch.tensor(data["X_text"], dtype=torch.float32)
        X_v = torch.tensor(data["X_video"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.int64)

        logger.info("Extracting latents for split '%s' (%d samples)...", split, len(y))
        Z_fused, y_np = extract_split(model, X_t, X_v, y, device, batch_size=args.batch_size)
        logger.info("Split '%s': Z_fused shape=%s, y shape=%s", split, Z_fused.shape, y_np.shape)

        out_npz = out_dir / f"latents_{split}.npz"
        np.savez_compressed(out_npz, fused_feat=Z_fused, labels=y_np)
        logger.info("Saved extracted latents to %s", out_npz)

    logger.info("Latent extraction pipeline finished successfully!")


if __name__ == "__main__":
    main()
