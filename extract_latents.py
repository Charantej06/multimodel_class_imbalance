"""
extract_latents.py
==================
Modality- and class-count-agnostic script to extract 512-d fused latent representations
(fused_feat in R^{512}) from a trained Fusion V2 checkpoint across train, val, and test splits.

Usage:
------
    python extract_latents.py --features_dir embeddings_v3 --checkpoint results_v3/best_fusion_v2_model.pt --output_dir results_rcs/latents
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

MELD_LABEL_NAMES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
CREMAD_LABEL_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]


def extract_split(
    model: torch.nn.Module,
    X_dict: Dict[str, torch.Tensor],
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Passes multimodal embeddings through the frozen fusion model to extract fused representations."""
    modalities = model.modalities
    tensors = [X_dict[m] for m in modalities] + [y]
    ds = TensorDataset(*tensors)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    feats, targets = [], []
    with torch.no_grad():
        for batch in loader:
            b_embs = {m: batch[idx].to(device) for idx, m in enumerate(modalities)}
            label_batch = batch[-1]
            out = model(**b_embs)
            fused = out["fused_feat"].cpu().numpy()
            feats.append(fused)
            targets.extend(label_batch.numpy())

    return np.vstack(feats), np.array(targets, dtype=np.int64)


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
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["text", "video"],
        help="Modalities to extract (default: text video)",
    )
    parser.add_argument("--label_map", type=str, default=None, help="Path to label_map.json")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Dimension of fused representations")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of emotion classes (inferred if None)")
    parser.add_argument("--cuda_mem_fraction", type=float, default=0.80, help="CUDA per-process memory fraction")
    return parser.parse_args()


def resolve_num_classes(args_label_map: Optional[str], features_dir: Path, max_y: int, modalities: List[str]) -> Tuple[int, List[str]]:
    """Resolves num_classes and label names."""
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
            return len(sorted_labels), sorted_labels

    if "audio" in modalities and max_y == 5:
        return 6, CREMAD_LABEL_NAMES

    return 7, MELD_LABEL_NAMES


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(args.cuda_mem_fraction, device=0)
        except Exception as e:
            logger.warning("Could not set CUDA memory fraction: %s", e)
    logger.info("Using device: %s", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-run config JSON
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    features_dir = Path(args.features_dir)
    splits = ["train", "val", "test"]
    modalities = list(args.modalities)

    # Peek at train set to get dimensions
    train_npz = features_dir / "train_features.npz"
    X_tr_peek, y_tr_peek = load_split_features(train_npz, modalities)
    modality_dims = {m: X_tr_peek[m].shape[1] for m in modalities}
    max_y = int(torch.max(y_tr_peek).item())

    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes, _ = resolve_num_classes(args.label_map, features_dir, max_y, modalities)

    ckpt_path = Path(args.checkpoint)
    logger.info("Loading Fusion V2 checkpoint from %s...", ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = RegularizedGatedMultimodalClassifierV2(
        modalities=modalities,
        modality_dims=modality_dims,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        modality_dropout=0.0,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    for split in splits:
        npz_file = features_dir / f"{split}_features.npz"
        if not npz_file.exists():
            logger.warning("Feature file %s not found. Skipping split '%s'.", npz_file, split)
            continue

        X_dict, y = load_split_features(npz_file, modalities)
        logger.info("Extracting latents for split '%s' (%d samples)...", split, len(y))
        Z_fused, y_np = extract_split(model, X_dict, y, device, batch_size=args.batch_size)
        logger.info("Split '%s': Z_fused shape=%s, y shape=%s", split, Z_fused.shape, y_np.shape)

        out_npz = out_dir / f"latents_{split}.npz"
        np.savez_compressed(out_npz, fused_feat=Z_fused, labels=y_np)
        logger.info("Saved extracted latents to %s", out_npz)

    logger.info("Latent extraction pipeline finished successfully!")


if __name__ == "__main__":
    main()
