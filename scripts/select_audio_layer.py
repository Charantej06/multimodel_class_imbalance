#!/usr/bin/env python3
"""
scripts/select_audio_layer.py
=============================
Evaluates multi-layer representations from WavLM (microsoft/wavlm-base-plus) on CREMA-D (Step 6).

Trains a linear probe (LogisticRegression) on the train split only, standardized using
train statistics only, and evaluates validation Macro-F1 and Accuracy across:
- All 13 individual layers (layer 0 through layer 12)
- Average of layers 4-9
- Average of layers 6-9
- Average of all layers (0-12)

Outputs the diagnostic probe table to cremad/audio_layer_probe.csv with the winning recommendation.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
logger = logging.getLogger("select_audio_layer")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WavLM layer representations on CREMA-D.")
    parser.add_argument(
        "--features_npz",
        type=str,
        default="cache/cremad_audio/wavlm-base-plus/all_layers.npz",
        help="Path to all_layers.npz extracted by extract_cremad_audio.py",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/cremad_manifest.csv",
        help="Path to cremad_manifest.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="cremad/audio_layer_probe.csv",
        help="Output CSV path (default: cremad/audio_layer_probe.csv)",
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
        help="Random state seed (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    features_path = Path(args.features_npz)
    manifest_path = Path(args.manifest)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not features_path.is_file():
        logger.error("FATAL: Features file not found at %s", features_path)
        sys.exit(1)

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
        logger.error("FATAL: Manifest file not found at %s", manifest_path)
        sys.exit(1)

    logger.info("Loading extracted features from %s...", features_path)
    data = np.load(features_path)
    clip_ids = list(data["ids"])
    feats = data["features"] if "features" in data else data["all_layers"]  # (N, 13, 768)
    logger.info("Loaded feature tensor: %s for %d clips", feats.shape, len(clip_ids))

    df_manifest = pd.read_csv(manifest_path)
    df_manifest = df_manifest.set_index("id")

    # Align manifest with loaded IDs
    valid_mask = [cid in df_manifest.index for cid in clip_ids]
    if not all(valid_mask):
        logger.warning("Filtering out %d clips not in manifest", len(valid_mask) - sum(valid_mask))
        clip_ids = [cid for cid, v in zip(clip_ids, valid_mask) if v]
        feats = feats[valid_mask]

    matched_df = df_manifest.loc[clip_ids].copy()
    splits = matched_df["split"].values

    # Encode labels to integers
    unique_labels = sorted(matched_df["label"].unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    y_all = np.array([label_to_idx[l] for l in matched_df["label"].values], dtype=np.int64)

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    logger.info("Dataset split distribution: Train=%d, Val=%d, Test=%d (Total=%d)",
                train_mask.sum(), val_mask.sum(), test_mask.sum(), len(clip_ids))

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        logger.error("FATAL: Train or Validation split has 0 samples in manifest! Train=%d, Val=%d",
                     train_mask.sum(), val_mask.sum())
        sys.exit(1)

    y_train = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask] if test_mask.sum() > 0 else None

    # Define candidate feature configurations
    configs: Dict[str, np.ndarray] = {}
    for layer_idx in range(13):
        configs[f"layer_{layer_idx}"] = feats[:, layer_idx, :]

    # Averaged representations
    configs["avg_layers_4_9"] = feats[:, 4:10, :].mean(axis=1)
    configs["avg_layers_6_9"] = feats[:, 6:10, :].mean(axis=1)
    configs["avg_all_layers"] = feats[:, :, :].mean(axis=1)

    results: List[Dict[str, Any]] = []

    logger.info("=" * 80)
    logger.info("EVALUATING WAVLM LAYER PROBES (Train-only standardization & Logistic Regression)")
    logger.info("=" * 80)
    logger.info(f"{'Feature Set':<20} | {'Val Macro-F1':<14} | {'Val Accuracy':<14} | {'Test Macro-F1':<14} | {'Test Accuracy':<14}")
    logger.info("-" * 80)

    for name, X_full in configs.items():
        X_tr = X_full[train_mask]
        X_va = X_full[val_mask]
        X_te = X_full[test_mask] if test_mask.sum() > 0 else None

        # GLOBAL RULE: Fit scaler on TRAIN ONLY!
        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr)
        X_va_std = scaler.transform(X_va)
        X_te_std = scaler.transform(X_te) if X_te is not None else None

        clf = LogisticRegression(
            max_iter=args.max_iter,
            C=args.c_val,
            random_state=args.seed,
            solver="lbfgs",
        )
        clf.fit(X_tr_std, y_train)

        val_preds = clf.predict(X_va_std)
        val_f1 = f1_score(y_val, val_preds, average="macro")
        val_acc = accuracy_score(y_val, val_preds)

        test_f1 = None
        test_acc = None
        if X_te_std is not None and y_test is not None:
            test_preds = clf.predict(X_te_std)
            test_f1 = f1_score(y_test, test_preds, average="macro")
            test_acc = accuracy_score(y_test, test_preds)

        t_f1_str = f"{test_f1:.4f}" if test_f1 is not None else "N/A"
        t_acc_str = f"{test_acc:.4f}" if test_acc is not None else "N/A"

        logger.info(f"{name:<20} | {val_f1:<14.4f} | {val_acc:<14.4f} | {t_f1_str:<14} | {t_acc_str:<14}")

        results.append({
            "feature_set": name,
            "val_macro_f1": float(val_f1),
            "val_accuracy": float(val_acc),
            "test_macro_f1": float(test_f1) if test_f1 is not None else None,
            "test_accuracy": float(test_acc) if test_acc is not None else None,
        })

    # Find winning layer by validation Macro-F1
    best_config = max(results, key=lambda r: r["val_macro_f1"])
    for r in results:
        r["recommended"] = (r["feature_set"] == best_config["feature_set"])

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_csv, index=False)
    logger.info("Saved layer probe evaluation report to: %s", out_csv)

    print("\n" + "=" * 80)
    print("AUDIO LAYER SELECTION RECOMMENDATION")
    print("=" * 80)
    print(f"Top Recommended Feature Set:  {best_config['feature_set']}")
    print(f"Validation Macro-F1:          {best_config['val_macro_f1']:.4f}")
    print(f"Validation Accuracy:          {best_config['val_accuracy']:.4f}")
    if best_config["test_macro_f1"] is not None:
        print(f"Test Macro-F1:                {best_config['test_macro_f1']:.4f}")
        print(f"Test Accuracy:                {best_config['test_accuracy']:.4f}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    main()

