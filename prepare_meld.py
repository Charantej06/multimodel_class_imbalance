"""
prepare_meld.py
===============
Prepares the MELD (Multimodal EmotionLines Dataset) raw data into a single unified
manifest suitable for DatasetAdapter and FeatureExtractor.

Responsibilities:
-----------------
1. Locates train_sent_emo, dev_sent_emo, and test_sent_emo CSV files.
2. Preserves the original splits ('train', 'val' for dev, 'test').
3. Maps 'Utterance' -> 'text' and 'Emotion' -> 'label'.
4. Generates each utterance video clip path from Dialogue_ID and Utterance_ID
   (e.g., dia{Dialogue_ID}_utt{Utterance_ID}.mp4).
5. Validates video paths on disk, with an option to filter out known missing clips.
6. Writes a clean CSV manifest (e.g. data/meld_manifest.csv).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _find_csv(search_dirs: List[Path], filenames: List[str]) -> Optional[Path]:
    """Search for the first matching CSV filename across given directories."""
    for d in search_dirs:
        for fname in filenames:
            candidate = d / fname
            if candidate.is_file():
                return candidate
    return None


def _find_video_dir(search_dirs: List[Path], subdirs: List[str]) -> Optional[Path]:
    """Search for the first matching video directory."""
    for d in search_dirs:
        for s in subdirs:
            candidate = d / s
            if candidate.is_dir():
                return candidate
    return None


def prepare_meld(
    meld_dir: Union[str, Path] = "MELD.Raw",
    output_path: Union[str, Path] = "data/meld_manifest.csv",
    filter_missing: bool = True,
    prefix_split_id: bool = False,
) -> pd.DataFrame:
    """
    Combine MELD train, dev, and test CSVs into one unified manifest.

    Parameters
    ----------
    meld_dir : str or Path
        Root directory containing raw MELD files and split subdirectories.
    output_path : str or Path
        Path where the combined CSV manifest will be saved.
    filter_missing : bool
        If True, samples whose video files do not exist on disk are filtered out.
    prefix_split_id : bool
        If True, sample IDs will be '{split}_dia{d}_utt{u}'. Otherwise 'dia{d}_utt{u}'.

    Returns
    -------
    pd.DataFrame
        The combined manifest DataFrame.
    """
    meld_path = Path(meld_dir)
    if not meld_path.exists():
        raise FileNotFoundError(f"MELD root directory not found: {meld_path}")

    # Search candidates for each split
    split_configs = {
        "train": {
            "split_name": "train",
            "csv_search": [meld_path / "train", meld_path],
            "csv_names": ["train_sent_emo.csv"],
            "video_search": [meld_path / "train", meld_path],
            "video_dirs": ["train_splits", "train"],
        },
        "dev": {
            "split_name": "val",  # Normalized split name
            "csv_search": [meld_path, meld_path / "dev"],
            "csv_names": ["dev_sent_emo.csv"],
            "video_search": [meld_path / "dev", meld_path],
            "video_dirs": ["dev_splits_complete", "dev_splits", "dev"],
        },
        "test": {
            "split_name": "test",
            "csv_search": [meld_path, meld_path / "test"],
            "csv_names": ["test_sent_emo.csv"],
            "video_search": [meld_path / "test", meld_path],
            "video_dirs": ["output_repeated_splits_test", "test_splits", "test"],
        },
    }

    dfs: List[pd.DataFrame] = []

    for key, cfg in split_configs.items():
        csv_path = _find_csv(cfg["csv_search"], cfg["csv_names"])
        if csv_path is None:
            raise FileNotFoundError(
                f"Could not find CSV for split '{key}' in {[str(p) for p in cfg['csv_search']]}"
            )

        video_dir = _find_video_dir(cfg["video_search"], cfg["video_dirs"])
        if video_dir is None:
            logger.warning(
                "Could not find video directory for split '%s'. Video paths will be constructed using default relative path.",
                key,
            )
            # Default fallback directory
            video_dir = meld_path / key / f"{key}_splits"

        logger.info(
            "Split '%s': loading CSV from '%s' | video directory: '%s'",
            key, csv_path, video_dir,
        )

        try:
            df_split = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df_split = pd.read_csv(csv_path, encoding="latin-1")
        logger.info("  Loaded %d records for '%s'.", len(df_split), key)

        # Set normalized split
        split_val = cfg["split_name"]
        df_split["split"] = split_val

        # Construct video file paths
        video_paths: List[str] = []
        for d, u in zip(df_split["Dialogue_ID"], df_split["Utterance_ID"]):
            fname = f"dia{d}_utt{u}.mp4"
            vid_p = video_dir / fname
            # Store path formatted with forward slashes for cross-platform portability
            video_paths.append(str(vid_p).replace("\\", "/"))

        df_split["video"] = video_paths
        df_split["video_path"] = video_paths

        dfs.append(df_split)

    manifest = pd.concat(dfs, ignore_index=True)

    # Standardize column mappings for text and label
    manifest["text"] = manifest["Utterance"].astype(str)
    manifest["label"] = manifest["Emotion"].astype(str)

    # Generate sample IDs
    if prefix_split_id:
        manifest["id"] = [
            f"{s}_dia{d}_utt{u}"
            for s, d, u in zip(manifest["split"], manifest["Dialogue_ID"], manifest["Utterance_ID"])
        ]
    else:
        manifest["id"] = [
            f"dia{d}_utt{u}"
            for d, u in zip(manifest["Dialogue_ID"], manifest["Utterance_ID"])
        ]

    # Check for missing video files
    initial_count = len(manifest)
    exists_mask = manifest["video"].apply(lambda p: Path(p).exists())
    missing_count = int((~exists_mask).sum())

    if missing_count > 0:
        logger.warning(
            "Found %d / %d samples where video file does not exist on disk.",
            missing_count, initial_count,
        )
        if filter_missing:
            manifest = manifest[exists_mask].reset_index(drop=True)
            logger.info(
                "Filtered missing video files: %d -> %d samples retained.",
                initial_count, len(manifest),
            )
        else:
            logger.info("Retaining all samples (filter_missing=False).")

    # Save to disk
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_p, index=False)
    logger.info("Saved combined MELD manifest (%d rows) to '%s'", len(manifest), out_p)

    # Print summary
    logger.info("Split distribution:\n%s", manifest["split"].value_counts().to_string())
    logger.info("Emotion class distribution:\n%s", manifest["label"].value_counts().to_string())

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare MELD multimodal manifest.")
    parser.add_argument(
        "--meld_dir",
        type=str,
        default="MELD.Raw",
        help="Path to MELD.Raw directory containing raw data (default: MELD.Raw)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/meld_manifest.csv",
        help="Destination path for manifest CSV (default: data/meld_manifest.csv)",
    )
    parser.add_argument(
        "--keep_missing",
        action="store_true",
        help="Do not filter out samples whose video files are missing on disk.",
    )
    parser.add_argument(
        "--prefix_split_id",
        action="store_true",
        help="Prefix sample ID with split name, e.g. 'train_dia0_utt0'.",
    )
    args = parser.parse_args()

    prepare_meld(
        meld_dir=args.meld_dir,
        output_path=args.output_path,
        filter_missing=not args.keep_missing,
        prefix_split_id=args.prefix_split_id,
    )


if __name__ == "__main__":
    main()
