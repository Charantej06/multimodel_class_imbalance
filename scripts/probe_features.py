#!/usr/bin/env python3
"""
scripts/probe_features.py
=========================
Linear diagnostic probing of unimodal and multimodal features on CREMA-D (Step 7b).

Probing Protocol:
-----------------
- Classifiers: LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0, random_state=42).
- Scaler: StandardScaler fitted strictly on the TRAIN split only.
- Audio: Mean of WavLM layers 4-9 ("avg_layers_4_9", 768-d).
- Video: Evaluates two variants from DINOv2:
    (a) video-only (mean): temporal mean of 8 frame CLS embeddings (768-d).
    (b) video-only (mean+std): temporal mean concatenated with temporal std (1536-d).
- Multimodal Concat: Concatenation of standardized audio (768-d) and the better
  video variant (selected by Validation Macro-F1).
- Metrics: Validation Macro-F1, Validation Accuracy, Test Macro-F1, Test Accuracy,
  and per-class F1 scores on the validation split.

Outputs:
--------
- Writes cremad/feature_probe.csv and prints a formatted results table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("probe_features")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate linear probes on CREMA-D audio/video features.")
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
        "--out_csv",
        type=str,
        default="cremad/feature_probe.csv",
        help="Output CSV path (default: cremad/feature_probe.csv)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum iterations for LogisticRegression (default: 1000)",
    )
    parser.add_argument(
        "--c_val",
        type=float,
        default=1.0,
        help="Inverse regularization parameter C (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    audio_path = Path(args.audio_npz)
    video_path = Path(args.video_npz)
    manifest_path = Path(args.manifest)
    label_map_path = Path(args.label_map)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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

    # 2. Load manifest and verify splits
    df_manifest = pd.read_csv(manifest_path)
    manifest_ids = df_manifest["id"].tolist()
    manifest_splits = df_manifest["split"].values

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map: Dict[str, int] = json.load(f)

    # Ordered class names matching integer label indices
    inv_label_map = {idx: name for name, idx in label_map.items()}
    class_names = [inv_label_map[i] for i in range(len(label_map))]
    logger.info("Loaded label map (%d classes): %s", len(class_names), label_map)

    y_all = np.array([label_map[lbl] for lbl in df_manifest["label"].values], dtype=np.int64)

    train_mask = manifest_splits == "train"
    val_mask = manifest_splits == "val"
    test_mask = manifest_splits == "test"

    logger.info(
        "Manifest split counts: train=%d, val=%d, test=%d (total=%d)",
        train_mask.sum(), val_mask.sum(), test_mask.sum(), len(df_manifest),
    )

    # 3. Load feature arrays and strictly assert manifest ID row-order alignment
    logger.info("Loading audio features from %s...", audio_path)
    audio_data = np.load(audio_path)
    audio_ids = list(audio_data["ids"])
    assert audio_ids == manifest_ids, "CRITICAL ERROR: Audio IDs do not match manifest IDs exactly in row order!"

    # Mean of WavLM layers 4-9: index 4 to 9 inclusive -> slice 4:10
    raw_audio_all_layers = audio_data["features"]  # (N, 13, 768)
    assert raw_audio_all_layers.ndim == 3 and raw_audio_all_layers.shape[1] == 13, (
        f"Unexpected audio feature shape: {raw_audio_all_layers.shape}"
    )
    X_audio_raw = raw_audio_all_layers[:, 4:10, :].mean(axis=1)  # (N, 768)
    logger.info("Audio features (avg_layers_4_9) prepared: shape %s", X_audio_raw.shape)

    logger.info("Loading video features from %s...", video_path)
    video_data = np.load(video_path)
    video_ids = list(video_data["ids"])
    assert video_ids == manifest_ids, "CRITICAL ERROR: Video IDs do not match manifest IDs exactly in row order!"

    X_video_mean = video_data["features"]  # (N, 768)
    assert "temporal_std" in video_data, "FATAL: 'temporal_std' key missing from video features npz!"
    X_video_std = video_data["temporal_std"]  # (N, 768)
    X_video_mean_std = np.concatenate([X_video_mean, X_video_std], axis=1)  # (N, 1536)
    logger.info("Video features prepared: mean=%s, mean+std=%s", X_video_mean.shape, X_video_mean_std.shape)

    # Check for NaNs/Infs
    for arr, name in [
        (X_audio_raw, "Audio raw"),
        (X_video_mean, "Video mean"),
        (X_video_mean_std, "Video mean+std"),
    ]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            logger.error("FATAL: NaN or Inf detected in %s array!", name)
            sys.exit(1)

    # 4. Probing helper function
    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    def fit_and_eval_probe(
        X_tr: np.ndarray,
        X_va: np.ndarray,
        X_te: np.ndarray,
        feature_name: str,
    ) -> Dict[str, Any]:
        """Trains LogisticRegression probe and returns evaluation metrics."""
        clf = LogisticRegression(
            max_iter=args.max_iter,
            C=args.c_val,
            random_state=args.seed,
            solver="lbfgs",
        )
        clf.fit(X_tr, y_train)

        val_pred = clf.predict(X_va)
        test_pred = clf.predict(X_te)

        val_macro_f1 = f1_score(y_val, val_pred, average="macro")
        val_acc = accuracy_score(y_val, val_pred)
        test_macro_f1 = f1_score(y_test, test_pred, average="macro")
        test_acc = accuracy_score(y_test, test_pred)

        # Per-class validation F1
        val_per_class_f1 = f1_score(y_val, val_pred, average=None)
        class_f1_dict = {f"f1_{class_names[i]}": float(val_per_class_f1[i]) for i in range(len(class_names))}

        res = {
            "feature_set": feature_name,
            "dim": int(X_tr.shape[1]),
            "val_macro_f1": float(val_macro_f1),
            "val_accuracy": float(val_acc),
            "test_macro_f1": float(test_macro_f1),
            "test_accuracy": float(test_acc),
            **class_f1_dict,
        }
        return res

    results: List[Dict[str, Any]] = []

    # Experiment 1: Audio-only (avg_layers_4_9)
    logger.info("Evaluating Audio-Only (avg_layers_4_9, 768-d)...")
    scaler_audio = StandardScaler().fit(X_audio_raw[train_mask])
    X_aud_tr = scaler_audio.transform(X_audio_raw[train_mask])
    X_aud_va = scaler_audio.transform(X_audio_raw[val_mask])
    X_aud_te = scaler_audio.transform(X_audio_raw[test_mask])
    res_audio = fit_and_eval_probe(X_aud_tr, X_aud_va, X_aud_te, "audio_only")
    results.append(res_audio)

    # Experiment 2: Video-only (mean)
    logger.info("Evaluating Video-Only (mean, 768-d)...")
    scaler_vid_mean = StandardScaler().fit(X_video_mean[train_mask])
    X_vmean_tr = scaler_vid_mean.transform(X_video_mean[train_mask])
    X_vmean_va = scaler_vid_mean.transform(X_video_mean[val_mask])
    X_vmean_te = scaler_vid_mean.transform(X_video_mean[test_mask])
    res_vmean = fit_and_eval_probe(X_vmean_tr, X_vmean_va, X_vmean_te, "video_only_mean")
    results.append(res_vmean)

    # Experiment 3: Video-only (mean+std)
    logger.info("Evaluating Video-Only (mean+std, 1536-d)...")
    scaler_vid_mean_std = StandardScaler().fit(X_video_mean_std[train_mask])
    X_vstd_tr = scaler_vid_mean_std.transform(X_video_mean_std[train_mask])
    X_vstd_va = scaler_vid_mean_std.transform(X_video_mean_std[val_mask])
    X_vstd_te = scaler_vid_mean_std.transform(X_video_mean_std[test_mask])
    res_vstd = fit_and_eval_probe(X_vstd_tr, X_vstd_va, X_vstd_te, "video_only_mean_std")
    results.append(res_vstd)

    # Determine better video variant by Validation Macro-F1
    if res_vstd["val_macro_f1"] >= res_vmean["val_macro_f1"]:
        chosen_video_variant = "video_only_mean_std"
        chosen_video_name = "mean+std (1536-d)"
        X_vid_chosen_tr = X_vstd_tr
        X_vid_chosen_va = X_vstd_va
        X_vid_chosen_te = X_vstd_te
        better_dim = 1536
    else:
        chosen_video_variant = "video_only_mean"
        chosen_video_name = "mean (768-d)"
        X_vid_chosen_tr = X_vmean_tr
        X_vid_chosen_va = X_vmean_va
        X_vid_chosen_te = X_vmean_te
        better_dim = 768

    logger.info(
        "Better video variant by Validation Macro-F1: %s (Val F1: %.4f vs %.4f)",
        chosen_video_name,
        res_vstd["val_macro_f1"] if chosen_video_variant == "video_only_mean_std" else res_vmean["val_macro_f1"],
        res_vmean["val_macro_f1"] if chosen_video_variant == "video_only_mean_std" else res_vstd["val_macro_f1"],
    )

    # Experiment 4: Multimodal Concat (Audio + Better Video Variant)
    logger.info("Evaluating Multimodal Concat (Audio + %s)...", chosen_video_name)
    X_cat_tr = np.concatenate([X_aud_tr, X_vid_chosen_tr], axis=1)
    X_cat_va = np.concatenate([X_aud_va, X_vid_chosen_va], axis=1)
    X_cat_te = np.concatenate([X_aud_te, X_vid_chosen_te], axis=1)
    res_cat = fit_and_eval_probe(X_cat_tr, X_cat_va, X_cat_te, f"concat_audio_{chosen_video_variant}")
    results.append(res_cat)

    # Annotate chosen video variant flag
    for r in results:
        r["chosen_video_variant"] = (r["feature_set"] == chosen_video_variant)

    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_csv, index=False)
    logger.info("Diagnostic probe results saved to: %s", out_csv)

    # Print Summary Table
    print("\n" + "=" * 115)
    print("CREMA-D FEATURE PROBE DIAGNOSTIC SUMMARY (Step 7b)")
    print("=" * 115)
    header_fmt = "{:<28} | {:<5} | {:<12} | {:<10} | {:<12} | {:<10} | {:<7}"
    row_fmt = "{:<28} | {:<5} | {:<12.4f} | {:<10.4f} | {:<12.4f} | {:<10.4f} | {:<7}"
    print(header_fmt.format("Feature Representation", "Dim", "Val Macro-F1", "Val Acc", "Test Macro-F1", "Test Acc", "Chosen"))
    print("-" * 115)
    for r in results:
        chosen_str = "YES (*)" if r.get("chosen_video_variant") or "concat" in r["feature_set"] else ""
        print(row_fmt.format(
            r["feature_set"],
            r["dim"],
            r["val_macro_f1"],
            r["val_accuracy"],
            r["test_macro_f1"],
            r["test_accuracy"],
            chosen_str,
        ))
    print("=" * 115)

    # Print Per-Class F1 on Validation Split Table
    print("\n" + "=" * 95)
    print("PER-CLASS VALIDATION F1 SCORES")
    print("=" * 95)
    cls_header = "{:<28} | " + " | ".join([f"{c:<8}" for c in class_names])
    print(cls_header.format("Feature Representation"))
    print("-" * 95)
    for r in results:
        cls_vals = " | ".join([f"{r[f'f1_{c}']:<8.4f}" for c in class_names])
        print(f"{r['feature_set']:<28} | {cls_vals}")
    print("=" * 95 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

