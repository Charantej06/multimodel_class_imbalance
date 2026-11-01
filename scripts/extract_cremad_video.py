#!/usr/bin/env python3
"""
scripts/extract_cremad_video.py
===============================
Extracts frame- or clip-level representations from frozen pretrained video backbones
for CREMA-D face-cropped video clips (Step 7a).

Supported Backbones:
--------------------
- dinov2: facebook/dinov2-base (86.6M params, per-frame CLS embedding, temporal mean & std, 768-d)
- videomae_k400: MCG-NJU/videomae-base-finetuned-kinetics (86.2M params, 16 frames, token mean pooling, 768-d)
- videomae: MCG-NJU/videomae-base (86.2M params, 16 frames, token mean pooling, 768-d)

Key Specifications:
-------------------
- Input: Preprocessed face-cropped video frames from cache/cremad_video/{id}.npy
  (16 frames, 224x224, RGB uint8).
- Normalization: ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
- Output: cache/cremad_video_feats/{backbone}.npz containing:
  - 'ids': strictly aligned to data/cremad_manifest.csv row order.
  - 'features': float32 array of shape (N, 768).
  - For dinov2: also includes 'temporal_std' of shape (N, 768).
- CPU Execution: Single process, torch.set_num_threads(8), batch size of 4 clips,
  torch.inference_mode(), float32.
- Resumable: Checkpoints every 500 clips, skips already extracted clips upon resumption.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from encoders.registry import VIDEO_BACKBONES, load_video_backbone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("extract_cremad_video")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract video representations for CREMA-D.")
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=list(VIDEO_BACKBONES.keys()),
        help=f"Video backbone to use: {list(VIDEO_BACKBONES.keys())}",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/cremad_manifest.csv",
        help="Path to cremad_manifest.csv (default: data/cremad_manifest.csv)",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="cache/cremad_video",
        help="Directory with face-cropped video npy files (default: cache/cremad_video)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cache/cremad_video_feats",
        help="Directory to save extracted features (default: cache/cremad_video_feats)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="Number of frames to sample for dinov2 (default: 8, uniformly from 16)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size of clips (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of clips to extract for benchmarking (default: None)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="PyTorch intra-op CPU threads (default: 8)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=500,
        help="Save partial checkpoint every N clips (default: 500)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip extraction if final output file already exists",
    )
    return parser.parse_args()


def load_clip_frames(clip_path: Path) -> np.ndarray:
    """
    Loads preprocessed clip frames from .npy file.
    Validates shape (16, 224, 224, 3) and uint8 dtype.
    """
    if not clip_path.is_file():
        raise FileNotFoundError(f"Video clip npy not found: {clip_path}")
    frames = np.load(clip_path)
    if frames.ndim != 4 or frames.shape[1:] != (224, 224, 3):
        raise ValueError(f"Unexpected shape {frames.shape} for {clip_path}; expected (16, 224, 224, 3)")
    if frames.shape[0] != 16:
        raise ValueError(f"Expected 16 frames in {clip_path}, found {frames.shape[0]}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return frames


def save_npz(
    path: Path,
    ids: List[str],
    features: np.ndarray,
    temporal_std: Optional[np.ndarray] = None,
):
    """Saves features and IDs to a compressed npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "ids": np.array(ids),
        "features": features.astype(np.float32),
    }
    if temporal_std is not None:
        save_dict["temporal_std"] = temporal_std.astype(np.float32)
    np.savez_compressed(path, **save_dict)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    logger.info("Configured PyTorch CPU threads: %d", torch.get_num_threads())

    manifest_path = Path(args.manifest)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate manifest and paths
    if not manifest_path.is_file():
        alt_manifests = [
            Path("data/cremad_manifest.csv"),
            Path("CREMAD/cremad_manifest.csv"),
            Path("../data/cremad_manifest.csv"),
        ]
        for alt in alt_manifests:
            if alt.is_file():
                manifest_path = alt
                break
    if not manifest_path.is_file():
        logger.error("FATAL: Manifest not found at %s", manifest_path)
        sys.exit(1)

    if not video_dir.is_dir():
        alt_dirs = [
            Path("cache/cremad_video"),
            Path("../cache/cremad_video"),
        ]
        for alt in alt_dirs:
            if alt.is_dir():
                video_dir = alt
                break
    if not video_dir.is_dir():
        logger.error("FATAL: Video cache directory not found at %s", video_dir)
        sys.exit(1)

    df_manifest = pd.read_csv(manifest_path)
    if "id" not in df_manifest.columns:
        logger.error("FATAL: 'id' column missing in manifest %s", manifest_path)
        sys.exit(1)

    manifest_ids = df_manifest["id"].tolist()
    if args.limit is not None and args.limit > 0:
        logger.info("Applying --limit %d (from %d total manifest clips)", args.limit, len(manifest_ids))
        manifest_ids = manifest_ids[: args.limit]

    total_target_clips = len(manifest_ids)
    target_id_to_idx = {cid: idx for idx, cid in enumerate(manifest_ids)}

    final_npz_path = output_dir / f"{args.backbone}.npz"
    chk_npz_path = output_dir / f"{args.backbone}_checkpoint.npz"

    if args.skip_existing and final_npz_path.is_file():
        logger.info("Output file %s already exists and --skip_existing is set. Exiting.", final_npz_path)
        return 0

    # In-memory storage for extracted features
    extracted_features: Dict[str, np.ndarray] = {}
    extracted_std: Dict[str, np.ndarray] = {}

    # Check for existing checkpoint or final output to resume
    resume_source = None
    if final_npz_path.is_file():
        resume_source = final_npz_path
    elif chk_npz_path.is_file():
        resume_source = chk_npz_path

    if resume_source is not None:
        try:
            logger.info("Found existing cache at %s. Loading completed clips...", resume_source)
            cached_data = np.load(resume_source)
            c_ids = list(cached_data["ids"])
            c_feats = cached_data["features"]
            has_std = "temporal_std" in cached_data
            c_std = cached_data["temporal_std"] if has_std else None

            for i, cid in enumerate(c_ids):
                if cid in target_id_to_idx:
                    extracted_features[cid] = c_feats[i].astype(np.float32)
                    if has_std and c_std is not None:
                        extracted_std[cid] = c_std[i].astype(np.float32)
            logger.info("Successfully resumed %d clips from cache.", len(extracted_features))
        except Exception as e:
            logger.warning("Could not read resume cache %s: %s. Starting fresh.", resume_source, e)
            extracted_features = {}
            extracted_std = {}

    remaining_ids = [cid for cid in manifest_ids if cid not in extracted_features]
    logger.info("Total clips: %d | Already cached: %d | Remaining to extract: %d",
                total_target_clips, len(extracted_features), len(remaining_ids))

    if len(remaining_ids) == 0:
        logger.info("All %d clips are already extracted! Saving/verifying final file...", total_target_clips)
        ordered_feats = np.stack([extracted_features[cid] for cid in manifest_ids], axis=0)
        ordered_std = (
            np.stack([extracted_std[cid] for cid in manifest_ids], axis=0)
            if args.backbone == "dinov2" and extracted_std
            else None
        )
        save_npz(final_npz_path, manifest_ids, ordered_feats, ordered_std)
        logger.info("Verified and saved %s with shape %s", final_npz_path, ordered_feats.shape)
        return 0

    # 2. Load Pretrained Backbone
    logger.info("Loading pretrained video backbone: %s (%s)...",
                args.backbone, VIDEO_BACKBONES[args.backbone])
    encoder_dict = load_video_backbone(args.backbone, device=torch.device("cpu"))
    model = encoder_dict["model"]
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded successfully: %s", encoder_dict["model_id"])
    logger.info("Parameter count: %d (%.2fM)", num_params, num_params / 1e6)
    if num_params < 80_000_000:
        logger.error("FATAL: Parameter count (%d) is unexpectedly low! Stopping.", num_params)
        sys.exit(1)

    # 3. Setup normalization tensors
    mean_tensor = IMAGENET_MEAN.view(1, 1, 3, 1, 1)
    std_tensor = IMAGENET_STD.view(1, 1, 3, 1, 1)

    # DINOv2 frame sampling index
    if args.backbone == "dinov2":
        frame_indices = np.linspace(0, 15, args.num_frames, dtype=int)
        logger.info("DINOv2 frame sampling indices (num_frames=%d): %s",
                    args.num_frames, frame_indices.tolist())

    # 4. Batch Inference Loop
    batch_size = args.batch_size
    num_batches = (len(remaining_ids) + batch_size - 1) // batch_size
    logger.info("Beginning video feature extraction: %d clips, %d batches (batch size: %d, CPU threads: %d)...",
                len(remaining_ids), num_batches, batch_size, torch.get_num_threads())

    start_time = time.time()
    processed_count = 0
    last_checkpoint_count = len(extracted_features)

    for b_idx in range(num_batches):
        b_cids = remaining_ids[b_idx * batch_size : (b_idx + 1) * batch_size]
        B = len(b_cids)

        # Load raw frames
        clip_frames_list = []
        for cid in b_cids:
            clip_file = video_dir / f"{cid}.npy"
            try:
                frames = load_clip_frames(clip_file)
            except Exception as e:
                logger.error("FATAL: Missing or corrupt video clip %s: %s", clip_file, e)
                sys.exit(1)
            clip_frames_list.append(frames)

        raw_batch = np.stack(clip_frames_list, axis=0)  # (B, 16, 224, 224, 3) uint8

        if args.backbone == "dinov2":
            # Sample num_frames frames uniformly
            sampled_batch = raw_batch[:, frame_indices]  # (B, num_frames, 224, 224, 3)
            # Permute to (B, num_frames, 3, 224, 224) and normalize
            t_batch = torch.from_numpy(sampled_batch).permute(0, 1, 4, 2, 3).float().div(255.0)
            t_batch = (t_batch - mean_tensor) / std_tensor
            t_flat = t_batch.reshape(B * args.num_frames, 3, 224, 224)

            with torch.inference_mode():
                out = model(t_flat)
                # CLS token is at index 0 of last_hidden_state
                cls_tokens = out.last_hidden_state[:, 0, :].reshape(B, args.num_frames, 768)
                t_mean = cls_tokens.mean(dim=1).cpu().numpy()  # (B, 768)
                t_std = cls_tokens.std(dim=1).cpu().numpy()    # (B, 768)

            for i, cid in enumerate(b_cids):
                extracted_features[cid] = t_mean[i].astype(np.float32)
                extracted_std[cid] = t_std[i].astype(np.float32)

        else:
            # VideoMAE / VideoMAE K400: expects (B, 16, 3, 224, 224)
            t_batch = torch.from_numpy(raw_batch).permute(0, 1, 4, 2, 3).float().div(255.0)
            t_batch = ((t_batch - mean_tensor) / std_tensor).contiguous()

            with torch.inference_mode():
                out = model(pixel_values=t_batch)
                # Mean over token dimension (1568 patch tokens) -> (B, 768)
                feat = out.last_hidden_state.mean(dim=1).cpu().numpy()

            for i, cid in enumerate(b_cids):
                extracted_features[cid] = feat[i].astype(np.float32)

        processed_count += B
        elapsed = time.time() - start_time
        s_per_clip = elapsed / processed_count

        # Progress reporting: every batch if small run (<= 40 clips), else every 100 clips or last batch
        if len(remaining_ids) <= 40 or processed_count % 100 < B or b_idx == num_batches - 1:
            remaining_clips = len(remaining_ids) - processed_count
            eta_s = remaining_clips * s_per_clip
            logger.info(
                "Progress: %d/%d clips processed (%.1f%%) | %.3f s/clip | Elapsed: %.1fs | ETA: %.1fs",
                processed_count, len(remaining_ids), 100.0 * processed_count / len(remaining_ids),
                s_per_clip, elapsed, eta_s,
            )

        # Periodic checkpointing
        if (len(extracted_features) - last_checkpoint_count >= args.checkpoint_interval) or (b_idx == num_batches - 1):
            curr_cids = [cid for cid in manifest_ids if cid in extracted_features]
            curr_feats = np.stack([extracted_features[cid] for cid in curr_cids], axis=0)
            curr_std = (
                np.stack([extracted_std[cid] for cid in curr_cids], axis=0)
                if args.backbone == "dinov2" and extracted_std
                else None
            )
            save_npz(chk_npz_path, curr_cids, curr_feats, curr_std)
            last_checkpoint_count = len(extracted_features)

    # 5. Save final result in exact manifest row order
    logger.info("Saving final features strictly in manifest order...")
    ordered_feats = np.stack([extracted_features[cid] for cid in manifest_ids], axis=0)
    ordered_std = (
        np.stack([extracted_std[cid] for cid in manifest_ids], axis=0)
        if args.backbone == "dinov2" and extracted_std
        else None
    )
    save_npz(final_npz_path, manifest_ids, ordered_feats, ordered_std)

    # Remove temporary checkpoint file upon full completion
    if len(extracted_features) == total_target_clips and chk_npz_path.is_file():
        try:
            chk_npz_path.unlink()
        except OSError:
            pass

    total_time = time.time() - start_time
    avg_s_per_clip = total_time / max(processed_count, 1)
    logger.info(
        "Extraction completed successfully! Backbone: %s | Total: %d clips | Time: %.2fs | Avg: %.3f s/clip",
        args.backbone, processed_count, total_time, avg_s_per_clip,
    )
    logger.info("Final output saved: %s (shape: %s)", final_npz_path, ordered_feats.shape)
    if ordered_std is not None:
        logger.info("Temporal std saved: shape: %s", ordered_std.shape)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
