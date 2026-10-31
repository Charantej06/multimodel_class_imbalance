"""
prepare_cremad.py
=================
Prepares CREMA-D dataset manifests and actor splits for the multimodal emotion pipeline.

Responsibilities (Step 3):
--------------------------
1. Discovers and parses all 7,442 clips from {media_root}/VideoFlash and {media_root}/AudioWAV.
   NO absolute paths stored in manifests.
2. Creates data/label_map.json with fixed emotion class mapping:
   {"anger": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5}
3. Performs actor-level split (70% train, 15% val, 15% test) with seed 42.
   Asserts zero actor overlap across splits.
4. Produces:
   - data/cremad_manifest.csv (id, actor, sentence, intensity, label, split)
   - data/label_map.json
   - data/cremad_folds.csv (5-fold GroupKFold by actor over train+val, test marked fold -1)
   - data/cremad_lt_rho10.csv (long-tailed train-only manifest, rho=10, val/test unchanged)
   - data/cremad_lt_rho20.csv (long-tailed train-only manifest, rho=20, val/test unchanged)
   - data/cremad_lt_rho50.csv (long-tailed train-only manifest, rho=50, val/test unchanged)
5. Prints per-split class and actor counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("prepare_cremad")

LABEL_MAP: Dict[str, int] = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
}

EMOTION_CODE_TO_LABEL: Dict[str, str] = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_cremad_dataset(media_root: Path) -> pd.DataFrame:
    """
    Parses clips from media_root/VideoFlash or media_root/AudioWAV.
    Returns DataFrame with columns: id, actor, sentence, intensity, label.
    """
    video_dir = media_root / "VideoFlash"
    audio_dir = media_root / "AudioWAV"

    if not video_dir.exists() and not audio_dir.exists():
        raise FileNotFoundError(
            f"Neither VideoFlash nor AudioWAV found under media_root: {media_root}"
        )

    # Use whichever exists or union
    stems = set()
    if video_dir.exists():
        for entry in os.scandir(video_dir):
            if entry.is_file() and entry.name.endswith(".flv"):
                stems.add(entry.name[:-4])
    if audio_dir.exists():
        for entry in os.scandir(audio_dir):
            if entry.is_file() and entry.name.endswith(".wav"):
                stems.add(entry.name[:-4])

    sorted_stems = sorted(stems)
    records = []
    for s in sorted_stems:
        parts = s.split("_")
        if len(parts) != 4:
            logger.warning("Skipping unexpected filename format: %s", s)
            continue
        actor, sentence, emo_code, intensity = parts
        label = EMOTION_CODE_TO_LABEL.get(emo_code)
        if label is None:
            logger.warning("Skipping unrecognized emotion code '%s' in clip: %s", emo_code, s)
            continue
        records.append({
            "id": s,
            "actor": actor,
            "sentence": sentence,
            "intensity": intensity,
            "label": label,
        })

    df = pd.DataFrame(records)
    logger.info("Parsed %d clips across %d unique actors.", len(df), df["actor"].nunique())
    return df


def split_by_actor(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Splits DataFrame by actor ensuring zero actor overlap.
    Assigns 'train', 'val', 'test' to df['split'].
    """
    unique_actors = sorted(df["actor"].unique())
    n_actors = len(unique_actors)

    rng = np.random.RandomState(seed)
    shuffled_actors = rng.permutation(unique_actors)

    n_train = int(round(n_actors * train_ratio))
    n_val = int(round(n_actors * val_ratio))
    n_test = n_actors - n_train - n_val

    train_actors = set(shuffled_actors[:n_train])
    val_actors = set(shuffled_actors[n_train:n_train + n_val])
    test_actors = set(shuffled_actors[n_train + n_val:])

    # Strict assertion: zero actor overlap
    assert len(train_actors & val_actors) == 0, "Actor overlap detected between train and val!"
    assert len(train_actors & test_actors) == 0, "Actor overlap detected between train and test!"
    assert len(val_actors & test_actors) == 0, "Actor overlap detected between val and test!"

    df_out = df.copy()
    df_out["split"] = "unknown"
    df_out.loc[df_out["actor"].isin(train_actors), "split"] = "train"
    df_out.loc[df_out["actor"].isin(val_actors), "split"] = "val"
    df_out.loc[df_out["actor"].isin(test_actors), "split"] = "test"

    logger.info(
        "Actor split completed: Train=%d actors (%d clips), Val=%d actors (%d clips), Test=%d actors (%d clips).",
        len(train_actors), (df_out["split"] == "train").sum(),
        len(val_actors), (df_out["split"] == "val").sum(),
        len(test_actors), (df_out["split"] == "test").sum(),
    )
    return df_out


def create_group_kfold(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    Creates 5-fold GroupKFold by actor over train+val.
    For test samples, fold is set to -1.
    """
    df_folds = df.copy()
    df_folds["fold"] = -1

    train_val_mask = df_folds["split"].isin(["train", "val"])
    train_val_df = df_folds[train_val_mask].copy()

    gkf = GroupKFold(n_splits=n_splits)
    groups = train_val_df["actor"]

    for fold_idx, (_, val_idx) in enumerate(gkf.split(train_val_df, groups=groups)):
        orig_indices = train_val_df.iloc[val_idx].index
        df_folds.loc[orig_indices, "fold"] = fold_idx

    # Assert zero actor overlap across folds in train+val
    for f1 in range(n_splits):
        for f2 in range(f1 + 1, n_splits):
            a1 = set(df_folds[df_folds["fold"] == f1]["actor"])
            a2 = set(df_folds[df_folds["fold"] == f2]["actor"])
            assert len(a1 & a2) == 0, f"Actor leakage detected between fold {f1} and fold {f2}!"

    return df_folds


def create_long_tailed_manifest(
    df: pd.DataFrame,
    rho: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Creates long-tailed TRAIN-only manifest using:
        n_c = n_max * (1 / rho) ** (c / (C - 1))
    where C = 6 classes in fixed order (anger=0, disgust=1, fear=2, happy=3, neutral=4, sad=5).
    Subsamples from train split only. Val and test splits remain completely unchanged.
    """
    train_df = df[df["split"] == "train"].copy()
    val_test_df = df[df["split"] != "train"].copy()

    ordered_classes = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
    C = len(ordered_classes)

    # Class 0 count in train is n_max
    class_0_samples = train_df[train_df["label"] == ordered_classes[0]]
    n_max = len(class_0_samples)

    rng = np.random.RandomState(seed)
    sampled_train_dfs = []

    for c, cls_name in enumerate(ordered_classes):
        cls_df = train_df[train_df["label"] == cls_name]
        n_c = int(round(n_max * ((1.0 / rho) ** (c / (C - 1)))))
        if n_c > len(cls_df):
            raise ValueError(
                f"Requested {n_c} samples for class '{cls_name}', but only {len(cls_df)} available in train!"
            )
        indices = rng.choice(cls_df.index, size=n_c, replace=False)
        sampled_train_dfs.append(cls_df.loc[indices])

    new_train_df = pd.concat(sampled_train_dfs).sort_index()
    combined_df = pd.concat([new_train_df, val_test_df]).sort_values(by="id").reset_index(drop=True)

    n_first = len(sampled_train_dfs[0])
    n_last = len(sampled_train_dfs[-1])
    actual_rho = n_first / n_last
    rel_diff = abs(actual_rho - rho) / rho

    logger.info(
        "Created long-tailed manifest rho=%g: Train=%d samples (class 0: %d, class %d: %d, actual rho=%.3f, error=%.2f%%). Val/test=%d samples.",
        rho, len(new_train_df), n_first, C - 1, n_last, actual_rho, rel_diff * 100, len(val_test_df),
    )
    return combined_df


def print_split_statistics(df: pd.DataFrame, title: str = "MANIFEST STATISTICS"):
    """Prints clean table of per-split class and actor counts."""
    ordered_classes = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
    print("\n" + "=" * 90)
    print(f"{title:^90}")
    print("=" * 90)
    print(f"{'Split':<8} | {'Total':<6} | {'Actors':<6} | " + " | ".join(f"{c.capitalize():<7}" for c in ordered_classes))
    print("-" * 90)

    for sp in ["train", "val", "test"]:
        sub = df[df["split"] == sp]
        counts = sub["label"].value_counts().to_dict()
        class_str = " | ".join(f"{counts.get(c, 0):<7}" for c in ordered_classes)
        print(f"{sp:<8} | {len(sub):<6} | {sub['actor'].nunique():<6} | {class_str}")

    print("-" * 90)
    total_counts = df["label"].value_counts().to_dict()
    total_class_str = " | ".join(f"{total_counts.get(c, 0):<7}" for c in ordered_classes)
    print(f"{'Total':<8} | {len(df):<6} | {df['actor'].nunique():<6} | {total_class_str}")
    print("=" * 90 + "\n")


def prepare_cremad(
    media_root: Union[str, Path] = "CREMAD",
    output_dir: Union[str, Path] = "data",
    seed: int = 42,
) -> Dict[str, Path]:
    """
    Main orchestration function for Step 3.
    Writes all required manifests, label map, and folds.
    """
    media_path = Path(media_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse dataset
    df = parse_cremad_dataset(media_path)
    if len(df) == 0:
        raise ValueError(f"No valid CREMA-D clips parsed from {media_path}!")

    # 2. Write label_map.json
    label_map_path = out_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)
    logger.info("Saved label map to: %s", label_map_path)

    # 3. Actor-level 70/15/15 split
    df_manifest = split_by_actor(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed)
    
    # Required columns in exact order
    manifest_cols = ["id", "actor", "sentence", "intensity", "label", "split"]
    df_manifest = df_manifest[manifest_cols]

    manifest_path = out_dir / "cremad_manifest.csv"
    df_manifest.to_csv(manifest_path, index=False)
    logger.info("Saved main manifest to: %s", manifest_path)
    print_split_statistics(df_manifest, title="CREMA-D BALANCED MANIFEST (data/cremad_manifest.csv)")

    # 4. GroupKFold (5-fold) over train+val
    df_folds = create_group_kfold(df_manifest, n_splits=5)
    folds_path = out_dir / "cremad_folds.csv"
    df_folds.to_csv(folds_path, index=False)
    logger.info("Saved 5-fold cross-validation manifest to: %s", folds_path)

    # 5. Long-tailed train-only manifests (rho=10, 20, 50)
    created_files = {
        "manifest": manifest_path,
        "label_map": label_map_path,
        "folds": folds_path,
    }

    for rho in [10, 20, 50]:
        df_lt = create_long_tailed_manifest(df_manifest, rho=rho, seed=seed)
        lt_path = out_dir / f"cremad_lt_rho{rho}.csv"
        df_lt.to_csv(lt_path, index=False)
        logger.info("Saved long-tailed manifest (rho=%d) to: %s", rho, lt_path)
        print_split_statistics(df_lt, title=f"CREMA-D LONG-TAILED MANIFEST rho={rho} (data/cremad_lt_rho{rho}.csv)")
        created_files[f"lt_rho{rho}"] = lt_path

    # Also mirror into cremad/ directory for global output rule
    cremad_dir = Path("cremad")
    cremad_dir.mkdir(parents=True, exist_ok=True)
    for src_path in created_files.values():
        dst_path = cremad_dir / src_path.name
        shutil.copy2(src_path, dst_path)

    return created_files


def main():
    parser = argparse.ArgumentParser(description="Prepare CREMA-D Manifests and Actor Splits")
    parser.add_argument("--media_root", type=str, default="CREMAD", help="Directory containing VideoFlash and AudioWAV")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where manifests and label_map will be saved")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for actor split and long-tail sampling")
    args = parser.parse_args()

    prepare_cremad(
        media_root=args.media_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

