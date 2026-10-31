#!/usr/bin/env python3
"""
scripts/extract_cremad_audio.py
===============================
Extracts multi-layer representations from frozen pretrained WavLM (microsoft/wavlm-base-plus)
for all CREMA-D audio clips (Step 6).

Key Specifications:
- Model: microsoft/wavlm-base-plus (94M parameters, frozen, eval mode).
- Input: Mono 16 kHz WAV audio from {media_root}/AudioWAV/{id}.wav.
- Preprocessing: Per-utterance zero-mean and unit-variance normalization.
- Forward: output_hidden_states=True, masked mean pooling for EVERY layer (13 layers: 0..12).
- Batching: Clips sorted by duration into batches of 8 with proper attention mask.
- Output: cache/cremad_audio/wavlm-base-plus/all_layers.npz
  containing 'ids' and 'features' array [N, 13, 768], strictly aligned to data/cremad_manifest.csv order.
- Resumable: Saves checkpoints every 500 clips; skips already processed clips upon resumption.
- CPU Optimized: Single process with torch.set_num_threads(8), torch.inference_mode(), float32.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import WavLMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("extract_cremad_audio")


def load_audio_waveform(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Loads audio from WAV file and converts to mono 16 kHz float32."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sr = sf.read(str(audio_path), dtype="float32")
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    if sr != target_sr:
        # Resample if sample rate doesn't match
        try:
            import scipy.signal as sps
            num_samples = int(round(len(waveform) * float(target_sr) / sr))
            waveform = sps.resample(waveform, num_samples).astype(np.float32)
        except ImportError:
            raise RuntimeError(f"Audio {audio_path} has sample rate {sr} != {target_sr} and scipy is not installed.")

    # Per-utterance zero-mean and unit-variance normalization
    mean = np.mean(waveform)
    std = np.std(waveform)
    norm_waveform = (waveform - mean) / (std + 1e-7)
    return norm_waveform.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract multi-layer WavLM representations for CREMA-D.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/cremad_manifest.csv",
        help="Path to cremad_manifest.csv (default: data/cremad_manifest.csv)",
    )
    parser.add_argument(
        "--media_root",
        type=str,
        default="CREMAD",
        help="Media root directory containing AudioWAV/ (default: CREMAD)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cache/cremad_audio/wavlm-base-plus",
        help="Output directory (default: cache/cremad_audio/wavlm-base-plus)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/wavlm-base-plus",
        help="HuggingFace model identifier (default: microsoft/wavlm-base-plus)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of clips to process for testing/smoke check (default: None)",
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
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    logger.info("Configured PyTorch CPU threads: %d", torch.get_num_threads())

    media_root = Path(args.media_root)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve manifest path
    if not manifest_path.is_file():
        alt_manifests = [
            Path("data/cremad_manifest.csv"),
            Path("CREMAD/cremad_manifest.csv"),
            Path("../data/cremad_manifest.csv"),
        ]
        for ap in alt_manifests:
            if ap.is_file():
                manifest_path = ap
                break

    if not manifest_path.is_file():
        logger.error("FATAL: Manifest not found at %s", manifest_path)
        sys.exit(1)

    # Resolve media root
    if not (media_root / "AudioWAV").is_dir():
        alt_roots = [
            Path("CREMAD"),
            Path("multimodel_class_imbalance/CREMAD"),
            Path("../CREMAD"),
        ]
        for ar in alt_roots:
            if (ar / "AudioWAV").is_dir():
                media_root = ar
                break

    if not (media_root / "AudioWAV").is_dir():
        logger.error("FATAL: AudioWAV directory not found under %s", media_root)
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
    target_id_to_manifest_idx = {cid: idx for idx, cid in enumerate(manifest_ids)}

    final_npz_path = output_dir / "all_layers.npz"
    chk_npz_path = output_dir / "all_layers_checkpoint.npz"

    # In-memory storage for extracted features: id -> (13, 768) float32
    extracted_features: Dict[str, np.ndarray] = {}

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
            for cid, feat in zip(c_ids, c_feats):
                if cid in target_id_to_manifest_idx:
                    extracted_features[cid] = feat.astype(np.float32)
            logger.info("Successfully resumed %d clips from cache.", len(extracted_features))
        except Exception as e:
            logger.warning("Could not read resume cache %s: %s. Starting fresh.", resume_source, e)
            extracted_features = {}

    # Determine clips remaining to process
    remaining_ids = [cid for cid in manifest_ids if cid not in extracted_features]
    logger.info("Total clips: %d | Already cached: %d | Remaining to extract: %d",
                total_target_clips, len(extracted_features), len(remaining_ids))

    if len(remaining_ids) == 0:
        logger.info("All %d clips are already extracted! Saving/verifying final file...", total_target_clips)
        ordered_feats = np.stack([extracted_features[cid] for cid in manifest_ids], axis=0)
        np.savez_compressed(
            final_npz_path,
            ids=np.array(manifest_ids),
            features=ordered_feats,
            all_layers=ordered_feats,
        )
        logger.info("Verified and saved %s with shape %s", final_npz_path, ordered_feats.shape)
        return 0

    # 1. Load Pretrained Model (Strict verification: NO random initialization)
    logger.info("Loading pretrained model: %s...", args.model_name)
    try:
        model = WavLMModel.from_pretrained(args.model_name)
    except Exception as e:
        logger.error("FATAL: Failed to download/load pretrained model '%s': %s", args.model_name, e)
        logger.error("Pretrained weights are required. Do not fall back to random initialization. Stopping.")
        sys.exit(1)

    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded successfully: %s", args.model_name)
    logger.info("Parameter count: %d (%.2fM)", num_params, num_params / 1e6)
    if num_params < 80_000_000:
        logger.error("FATAL: Parameter count (%d) is unexpectedly low for wavlm-base-plus! Stopping.", num_params)
        sys.exit(1)

    # 2. Pre-load audio waveforms and determine duration for length-based bucketing
    logger.info("Pre-loading audio waveforms and calculating durations for %d clips...", len(remaining_ids))
    clip_audio_data: List[Tuple[str, np.ndarray, int]] = []
    load_start = time.time()
    for idx, cid in enumerate(remaining_ids, 1):
        wav_path = media_root / "AudioWAV" / f"{cid}.wav"
        try:
            wf = load_audio_waveform(wav_path, target_sr=16000)
            clip_audio_data.append((cid, wf, len(wf)))
        except Exception as e:
            logger.error("FATAL: Failed reading audio %s: %s", wav_path, e)
            sys.exit(1)
        if idx % 1000 == 0 or idx == len(remaining_ids):
            logger.info("Loaded %d/%d audio clips (%.1fs)", idx, len(remaining_ids), time.time() - load_start)

    # Sort remaining clips by duration for optimal batch padding efficiency
    clip_audio_data.sort(key=lambda x: x[2])
    logger.info("Clips sorted by duration. Min length: %d (%.2fs), Max length: %d (%.2fs)",
                clip_audio_data[0][2], clip_audio_data[0][2] / 16000.0,
                clip_audio_data[-1][2], clip_audio_data[-1][2] / 16000.0)

    # 3. Batch Inference Loop
    batch_size = args.batch_size
    num_batches = (len(clip_audio_data) + batch_size - 1) // batch_size
    logger.info("Beginning feature extraction: %d batches (batch size: %d, float32, CPU threads: %d)...",
                num_batches, batch_size, torch.get_num_threads())

    start_time = time.time()
    processed_since_start = 0
    last_checkpoint_count = len(extracted_features)

    for b_idx in range(num_batches):
        batch_slice = clip_audio_data[b_idx * batch_size : (b_idx + 1) * batch_size]
        b_cids = [item[0] for item in batch_slice]
        b_wfs = [item[1] for item in batch_slice]
        b_lens = [item[2] for item in batch_slice]

        b_max_len = max(b_lens)
        B = len(b_cids)

        # Pad and build attention mask
        padded_inputs = np.zeros((B, b_max_len), dtype=np.float32)
        attn_mask = np.zeros((B, b_max_len), dtype=np.int64)
        for i, (wf, length) in enumerate(zip(b_wfs, b_lens)):
            padded_inputs[i, :length] = wf
            attn_mask[i, :length] = 1

        t_inputs = torch.from_numpy(padded_inputs)
        t_mask = torch.from_numpy(attn_mask)

        with torch.inference_mode():
            outputs = model(
                input_values=t_inputs,
                attention_mask=t_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states  # tuple of 13 tensors: (B, T_feat, 768)

            # Calculate exact feature-level sequence lengths after conv downsampling
            feat_lengths = model._get_feat_extract_output_lengths(torch.tensor(b_lens))
            T_feat = hidden_states[0].shape[1]

            # Construct feature mask: (B, T_feat)
            feat_mask = torch.arange(T_feat).expand(B, T_feat) < feat_lengths.unsqueeze(1)
            feat_mask_exp = feat_mask.unsqueeze(-1).float()  # (B, T_feat, 1)
            feat_mask_sum = feat_mask_exp.sum(dim=1).clamp(min=1.0)  # (B, 1)

            # Masked mean pooling across all 13 layers
            pooled_layers = []
            for h in hidden_states:
                pooled = (h * feat_mask_exp).sum(dim=1) / feat_mask_sum  # (B, 768)
                pooled_layers.append(pooled)

            b_stacked = torch.stack(pooled_layers, dim=1).cpu().numpy().astype(np.float32)  # (B, 13, 768)

        # Store in dict
        for cid, feat in zip(b_cids, b_stacked):
            extracted_features[cid] = feat

        processed_since_start += B
        total_extracted_so_far = len(extracted_features)

        # Progress reporting every 100 clips (or on completion)
        if processed_since_start % 100 < B or processed_since_start == len(remaining_ids):
            elapsed = time.time() - start_time
            sec_per_clip = elapsed / max(1, processed_since_start)
            clips_left = len(remaining_ids) - processed_since_start
            eta_sec = clips_left * sec_per_clip
            logger.info(
                "[%d/%d extracted] %.1f%% complete | %.3f s/clip | Elapsed: %.1fs | ETA: %.1fs (%.1f min)",
                total_extracted_so_far, total_target_clips,
                (total_extracted_so_far / total_target_clips) * 100.0,
                sec_per_clip, elapsed, eta_sec, eta_sec / 60.0,
            )

        # Save partial checkpoint every args.checkpoint_interval clips
        if total_extracted_so_far - last_checkpoint_count >= args.checkpoint_interval:
            logger.info("Saving checkpoint at %d clips to %s...", total_extracted_so_far, chk_npz_path)
            cur_ids = list(extracted_features.keys())
            cur_feats = np.stack([extracted_features[cid] for cid in cur_ids], axis=0)
            np.savez_compressed(chk_npz_path, ids=np.array(cur_ids), features=cur_feats)
            last_checkpoint_count = total_extracted_so_far

    # 4. Final Assembly: strictly reorder to match manifest row order
    logger.info("All extractions complete! Assembling array aligned strictly with manifest row order...")
    ordered_feats = np.stack([extracted_features[cid] for cid in manifest_ids], axis=0)
    ordered_ids = np.array(manifest_ids)

    logger.info("Final feature array shape: %s (dtype: %s)", ordered_feats.shape, ordered_feats.dtype)
    np.savez_compressed(
        final_npz_path,
        ids=ordered_ids,
        features=ordered_feats,
        all_layers=ordered_feats,
    )
    logger.info("Saved final multi-layer features to: %s", final_npz_path)

    # Remove temporary checkpoint if final save succeeded
    if chk_npz_path.is_file():
        try:
            chk_npz_path.unlink()
        except OSError:
            pass

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("WAVLM AUDIO EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Model Name:              {args.model_name}")
    print(f"Model Parameters:        {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Total Clips Processed:   {total_target_clips:,}")
    print(f"Output File:             {final_npz_path}")
    print(f"Output Array Shape:      {ordered_feats.shape} (N, 13, 768)")
    print(f"Average Seconds / Clip:  {total_time / max(1, processed_since_start):.3f} s")
    print(f"Total Extraction Time:   {total_time:.2f} s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    main()

