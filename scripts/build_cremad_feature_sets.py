#!/usr/bin/env python3
"""
scripts/build_cremad_feature_sets.py
====================================
Builds standardized multimodal feature sets for CREMA-D (Step 7b).

Outputs:
--------
1. Balanced Feature Set:
   data/features_cremad/{train,val,test}_features.npz
   Containing keys:
     - X_audio : (N, 768) float32 (standardized WavLM avg_layers_4_9)
     - X_video : (N, 1536) float32 (standardized DINOv2 mean + temporal_std)
     - y       : (N,) int64 (0..5 emotion class labels)
     - ids     : (N,) string (clip IDs)
     - actor   : (N,) int64 (actor IDs)
   Also saves:
     - data/features_cremad/scalers.joblib (audio and video StandardScaler fitted on train)
     - data/features_cremad/label_map.json

2. Long-Tailed Feature Sets:
   data/features_cremad_lt_rho{10,20,50}/
     - val_features.npz and test_features.npz (identical to the balanced set)
     - train_features.npz (subset of balanced train rows per data/cremad_lt_rho*.csv)
     - label_map.json and scalers.joblib

Assertions Enforced:
--------------------
- No NaN or Inf values in any array.
- Row counts match manifests exactly.
- No actor overlap between train, val, and test splits.
- Integer labels match data/label_map.json.
- Feature standardization uses train statistics only.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("build_cremad_feature_sets")


def parse_args():
    parser = argparse.ArgumentParser(description="Build standardized feature sets for CREMA-D.")
    parser.add_argument(
        "--audio_npz",
        type=str,
        default="cache/cremad_audio/wavlm-base-plus/all_layers.npz",
        help="Path to audio all_layers.npz (default: cache/cremad_audio/wavlm-base-plus/all_layers.npz)",
    )
    parser.add_argument(
        "--video_npz",
        type=str,
        default="cache/cremad_video_feats/dinov2.npz",
        help="Path to video dinov2.npz (default: cache/cremad_video_feats/dinov2.npz)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/cremad_manifest.csv",
        help="Path to cremad_manifest.csv (default: data/cremad_manifest.csv)",
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default="data/label_map.json",
        help="Path to label_map.json (default: data/label_map.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/features_cremad",
        help="Output base directory for balanced features (default: data/features_cremad)",
    )
    parser.add_argument(
        "--lt_dir_prefix",
        type=str,
        default="data/features_cremad_lt_rho",
        help="Prefix for long-tail output directories (default: data/features_cremad_lt_rho)",
    )
    return parser.parse_args()


def verify_feature_dict(
    split_name: str,
    data: Dict[str, np.ndarray],
    expected_rows: int,
    expected_audio_dim: int = 768,
    expected_video_dim: int = 1536,
) -> None:
    """Rigorous assertion checks on an exported split dictionary."""
    for key in ["X_audio", "X_video", "y", "ids", "actor"]:
        assert key in data, f"Missing key '{key}' in split '{split_name}'!"

    X_a = data["X_audio"]
    X_v = data["X_video"]
    y = data["y"]
    cids = data["ids"]
    actors = data["actor"]

    # Row count verification
    assert len(X_a) == expected_rows, f"X_audio rows ({len(X_a)}) != expected ({expected_rows}) in {split_name}"
    assert len(X_v) == expected_rows, f"X_video rows ({len(X_v)}) != expected ({expected_rows}) in {split_name}"
    assert len(y) == expected_rows, f"y rows ({len(y)}) != expected ({expected_rows}) in {split_name}"
    assert len(cids) == expected_rows, f"ids rows ({len(cids)}) != expected ({expected_rows}) in {split_name}"
    assert len(actors) == expected_rows, f"actor rows ({len(actors)}) != expected ({expected_rows}) in {split_name}"

    # Dimensionality verification
    assert X_a.ndim == 2 and X_a.shape[1] == expected_audio_dim, (
        f"X_audio shape {X_a.shape} != expected (N, {expected_audio_dim})"
    )
    assert X_v.ndim == 2 and X_v.shape[1] == expected_video_dim, (
        f"X_video shape {X_v.shape} != expected (N, {expected_video_dim})"
    )
    assert y.ndim == 1, f"y ndim {y.ndim} != 1"

    # NaN / Inf assertion
    assert not np.isnan(X_a).any(), f"NaN in X_audio for {split_name}!"
    assert not np.isinf(X_a).any(), f"Inf in X_audio for {split_name}!"
    assert not np.isnan(X_v).any(), f"NaN in X_video for {split_name}!"
    assert not np.isinf(X_v).any(), f"Inf in X_video for {split_name}!"

    # Dtype verification
    assert X_a.dtype == np.float32, f"X_audio dtype {X_a.dtype} != float32"
    assert X_v.dtype == np.float32, f"X_video dtype {X_v.dtype} != float32"
    assert np.issubdtype(y.dtype, np.integer), f"y dtype {y.dtype} is not integer"
    assert cids.dtype.kind in ('U', 'S'), f"ids dtype {cids.dtype} is not string/unicode in {split_name}"
    assert np.issubdtype(actors.dtype, np.integer), f"actor dtype {actors.dtype} is not integer in {split_name}"

    # Label range check
    assert y.min() >= 0 and y.max() <= 5, f"y out of bounds [0, 5]: min={y.min()}, max={y.max()}"


def main():
    args = parse_args()
    audio_path = Path(args.audio_npz)
    video_path = Path(args.video_npz)
    manifest_path = Path(args.manifest)
    label_map_path = Path(args.label_map)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Path validations
    for p, name in [
        (audio_path, "Audio features"),
        (video_path, "Video features"),
        (manifest_path, "Manifest CSV"),
        (label_map_path, "Label map JSON"),
    ]:
        if not p.is_file():
            logger.error("FATAL: %s file not found at: %s", name, p)
            sys.exit(1)

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map: Dict[str, int] = json.load(f)

    df_manifest = pd.read_csv(manifest_path)
    total_manifest_clips = len(df_manifest)
    manifest_ids = df_manifest["id"].tolist()
    logger.info("Loaded manifest: %d rows from %s", total_manifest_clips, manifest_path)

    # 2. Load audio and video features with strict ID alignment
    logger.info("Loading audio features from %s...", audio_path)
    aud_data = np.load(audio_path)
    assert list(aud_data["ids"]) == manifest_ids, "Audio IDs do not strictly match manifest IDs!"
    raw_audio = aud_data["features"]  # (7442, 13, 768)
    X_audio_raw = raw_audio[:, 4:10, :].mean(axis=1).astype(np.float32)  # (7442, 768)

    logger.info("Loading video features from %s...", video_path)
    vid_data = np.load(video_path)
    assert list(vid_data["ids"]) == manifest_ids, "Video IDs do not strictly match manifest IDs!"
    assert "features" in vid_data and "temporal_std" in vid_data, "Missing features or temporal_std in video npz!"

    # Use winning video variant by Val Macro-F1 (DINOv2 mean + temporal_std, 1536-d)
    X_video_raw = np.concatenate([vid_data["features"], vid_data["temporal_std"]], axis=1).astype(np.float32)
    logger.info("Selected video variant: mean + temporal_std (dim: %d)", X_video_raw.shape[1])

    # 3. Actor Isolation Check
    train_mask = (df_manifest["split"] == "train").values
    val_mask = (df_manifest["split"] == "val").values
    test_mask = (df_manifest["split"] == "test").values

    train_actors = set(df_manifest.loc[train_mask, "actor"])
    val_actors = set(df_manifest.loc[val_mask, "actor"])
    test_actors = set(df_manifest.loc[test_mask, "actor"])

    assert len(train_actors & val_actors) == 0, f"Actor leakage between Train and Val: {train_actors & val_actors}"
    assert len(train_actors & test_actors) == 0, f"Actor leakage between Train and Test: {train_actors & test_actors}"
    assert len(val_actors & test_actors) == 0, f"Actor leakage between Val and Test: {val_actors & test_actors}"
    logger.info("Actor isolation verified across all splits: Train=%d, Val=%d, Test=%d actors.",
                len(train_actors), len(val_actors), len(test_actors))

    # 4. Fit StandardScaler on TRAIN SPLIT ONLY
    logger.info("Fitting StandardScaler on TRAIN split only...")
    scaler_audio = StandardScaler().fit(X_audio_raw[train_mask])
    scaler_video = StandardScaler().fit(X_video_raw[train_mask])

    # Transform full dataset using train statistics
    X_audio_norm = scaler_audio.transform(X_audio_raw).astype(np.float32)
    X_video_norm = scaler_video.transform(X_video_raw).astype(np.float32)

    # Verify train standardization
    tr_aud_norm = X_audio_norm[train_mask]
    tr_vid_norm = X_video_norm[train_mask]
    assert np.allclose(tr_aud_norm.mean(axis=0), 0.0, atol=1e-5), "Train audio mean is not 0!"
    assert np.allclose(tr_aud_norm.std(axis=0), 1.0, atol=1e-4), "Train audio std is not 1!"
    assert np.allclose(tr_vid_norm.mean(axis=0), 0.0, atol=1e-5), "Train video mean is not 0!"
    assert np.allclose(tr_vid_norm.std(axis=0), 1.0, atol=1e-4), "Train video std is not 1!"
    logger.info("Standardization verified: Train audio/video have mean 0.0 and std 1.0.")

    # Save scalers bundle
    scaler_save_path = out_dir / "scalers.joblib"
    joblib.dump({"audio_scaler": scaler_audio, "video_scaler": scaler_video}, scaler_save_path)
    logger.info("Saved fitted train scalers to: %s", scaler_save_path)

    # Copy label map to output directory
    shutil.copy2(label_map_path, out_dir / "label_map.json")

    # Map labels and prepare full data arrays
    y_full = np.array([label_map[l] for l in df_manifest["label"].values], dtype=np.int64)
    ids_full = np.array(df_manifest["id"].values, dtype=np.str_)
    actors_full = np.array(df_manifest["actor"].values, dtype=np.int64)

    # 5. Build Balanced Feature Set
    split_counts = {"train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": int(test_mask.sum())}
    balanced_split_data: Dict[str, Dict[str, np.ndarray]] = {}

    for split in ["train", "val", "test"]:
        s_mask = (df_manifest["split"] == split).values
        s_dict = {
            "X_audio": X_audio_norm[s_mask],
            "X_video": X_video_norm[s_mask],
            "y": y_full[s_mask],
            "ids": ids_full[s_mask],
            "actor": actors_full[s_mask],
        }
        verify_feature_dict(f"balanced_{split}", s_dict, expected_rows=split_counts[split])
        out_file = out_dir / f"{split}_features.npz"
        np.savez_compressed(out_file, **s_dict)
        logger.info("Saved %s with %d samples (X_audio: %s, X_video: %s, y: %s)",
                    out_file, len(s_dict["y"]), s_dict["X_audio"].shape, s_dict["X_video"].shape, s_dict["y"].shape)
        balanced_split_data[split] = s_dict

    # Build lookup map: clip_id -> row index in balanced train set
    train_ids_list = list(balanced_split_data["train"]["ids"])
    train_id_to_idx = {cid: idx for idx, cid in enumerate(train_ids_list)}

    # 6. Build Long-Tailed Feature Sets: rho in {10, 20, 50}
    for rho in [10, 20, 50]:
        lt_manifest_path = Path(f"data/cremad_lt_rho{rho}.csv")
        lt_out_dir = Path(f"{args.lt_dir_prefix}{rho}")
        lt_out_dir.mkdir(parents=True, exist_ok=True)

        if not lt_manifest_path.is_file():
            logger.error("FATAL: Long-tail manifest not found: %s", lt_manifest_path)
            sys.exit(1)

        df_lt = pd.read_csv(lt_manifest_path)
        lt_train_df = df_lt[df_lt["split"] == "train"]
        lt_val_df = df_lt[df_lt["split"] == "val"]
        lt_test_df = df_lt[df_lt["split"] == "test"]

        logger.info("Building long-tail feature set rho=%d (train=%d, val=%d, test=%d)...",
                    rho, len(lt_train_df), len(lt_val_df), len(lt_test_df))

        # Check val and test match balanced set identically
        assert list(lt_val_df["id"]) == list(balanced_split_data["val"]["ids"]), (
            f"LT rho={rho} val IDs do not match balanced val IDs!"
        )
        assert list(lt_test_df["id"]) == list(balanced_split_data["test"]["ids"]), (
            f"LT rho={rho} test IDs do not match balanced test IDs!"
        )

        # Val and test files are identical to the balanced set
        np.savez_compressed(lt_out_dir / "val_features.npz", **balanced_split_data["val"])
        np.savez_compressed(lt_out_dir / "test_features.npz", **balanced_split_data["test"])

        # Train rows are the exact subset of balanced train rows
        lt_train_ids = lt_train_df["id"].tolist()
        lt_indices = [train_id_to_idx[cid] for cid in lt_train_ids]

        lt_train_dict = {
            "X_audio": balanced_split_data["train"]["X_audio"][lt_indices],
            "X_video": balanced_split_data["train"]["X_video"][lt_indices],
            "y": balanced_split_data["train"]["y"][lt_indices],
            "ids": balanced_split_data["train"]["ids"][lt_indices],
            "actor": balanced_split_data["train"]["actor"][lt_indices],
        }
        verify_feature_dict(f"lt_rho{rho}_train", lt_train_dict, expected_rows=len(lt_train_df))
        np.savez_compressed(lt_out_dir / "train_features.npz", **lt_train_dict)

        # Copy metadata
        shutil.copy2(label_map_path, lt_out_dir / "label_map.json")
        shutil.copy2(scaler_save_path, lt_out_dir / "scalers.joblib")
        logger.info("Completed long-tailed feature set rho=%d in %s", rho, lt_out_dir)

    print("\n" + "=" * 90)
    print("CREMA-D FEATURE SETS GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 90)
    print(f"Balanced set:      {out_dir}  (Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']})")
    for rho in [10, 20, 50]:
        lt_p = Path(f"{args.lt_dir_prefix}{rho}")
        df_l = pd.read_csv(f"data/cremad_lt_rho{rho}.csv")
        n_tr = (df_l["split"] == "train").sum()
        print(f"Long-tail rho={rho:<2}:   {lt_p}  (Train={n_tr}, Val={split_counts['val']}, Test={split_counts['test']})")
    print("=" * 90 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
