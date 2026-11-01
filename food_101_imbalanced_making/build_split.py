"""
food101_lt/build_split.py
=========================
Builds the Long-Tailed Food-101 dataset split for the RCS paper pipeline:
  Autoencoder + SupCon -> GMM / Calibrated Gaussian Sampling -> ResNet32.

Specification:
  - Input: Food-101 root containing images/ and meta/train.txt, meta/test.txt.
  - For each class (750 official train files):
      1. Shuffle with seeded RNG.
      2. Take 50 as VAL (balanced across all 101 classes -> 5,050 total).
      3. From remaining 700 files, randomly pick counts[rank] as TRAIN.
  - TEST: Full official test split, 250 per class untouched (25,250 total).
  - Randomly permute class ranks so label 0 = largest class (n_max), label 100 = smallest (n_min).
  - Strict assertions:
      * Train, Val, Test are strictly disjoint by file path.
      * Val has exactly 50 samples per class.
      * Test has exactly 250 samples per class.
      * Train matches counts[rank] exactly.
  - Outputs saved to ./data/food101_lt/:
      * manifest.csv (id, label, split, image_path)
      * class_map.json (label -> food name, rank, counts)
      * config.json (seed, n_max, n_min, eta, k, etc.)
  - Prints a summary table at the end.
"""

from __future__ import annotations
import os
import sys
import json
import csv
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Ensure local imports work whether run from repo root or package dir
sys.path.insert(0, str(Path(__file__).parent.resolve()))
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

try:
    from food101_profile import make_counts, check_profile, PRESETS
except ImportError:
    from food_101_imbalanced_making.food101_profile import make_counts, check_profile, PRESETS


def parse_meta_split_file(meta_file: Path, images_root: Path) -> Dict[str, List[str]]:
    """
    Parses official Food-101 meta split file (train.txt or test.txt).
    Each line has the form: class_name/image_id (e.g. 'apple_pie/1005649').
    Returns a dict mapping class_name -> list of relative image paths.
    """
    assert meta_file.exists(), f"Assert failed: metadata split file not found: {meta_file}"
    class_to_files: Dict[str, List[str]] = {}

    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("/")
            assert len(parts) == 2, f"Assert failed: invalid line in {meta_file}: '{line}'"
            cls_name, img_id = parts[0], parts[1]
            if not img_id.endswith(".jpg"):
                img_rel = f"{cls_name}/{img_id}.jpg"
            else:
                img_rel = f"{cls_name}/{img_id}"

            if cls_name not in class_to_files:
                class_to_files[cls_name] = []
            class_to_files[cls_name].append(img_rel)

    return class_to_files


def create_mock_food101_meta(mock_root: Path) -> Tuple[Path, Path, Path]:
    """
    Creates a valid synthetic Food-101 directory with meta/train.txt (750/class)
    and meta/test.txt (250/class) across 101 standard classes for verification.
    """
    meta_dir = mock_root / "meta"
    images_dir = mock_root / "images"
    meta_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Standard 101 Food-101 class names
    classes = [f"food_class_{i:03d}" for i in range(101)]

    train_lines = []
    test_lines = []

    for c in classes:
        cls_img_dir = images_dir / c
        cls_img_dir.mkdir(parents=True, exist_ok=True)

        # 750 train items
        for i in range(750):
            item_id = f"{c}_{i:04d}"
            train_lines.append(f"{c}/{item_id}")

        # 250 test items
        for i in range(750, 1000):
            item_id = f"{c}_{i:04d}"
            test_lines.append(f"{c}/{item_id}")

    train_txt = meta_dir / "train.txt"
    test_txt = meta_dir / "test.txt"

    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(test_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + "\n")

    return mock_root, train_txt, test_txt


def build_food101_lt_split(
    food101_root: Path,
    output_dir: Path,
    seed: int = 42,
    n_max: int = 700,
    n_min: int = 40,
    C: int = 101,
    eta: int = 7,
    k: int = 5,
    allow_below_paper_min: bool = False,
    preset_name: str = "main",
) -> Dict[str, Any]:
    """
    Builds and writes the long-tailed Food-101 split according to the RCS paper pipeline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    food101_root = food101_root.resolve()

    # 1. Compute and verify exponential count profile
    counts = make_counts(n_max=n_max, n_min=n_min, C=C)
    profile_info = check_profile(
        counts=counts,
        eta=eta,
        k=k,
        allow_below_paper_min=allow_below_paper_min,
        verbose=False,
    )

    # 2. Locate meta files and images directory
    meta_dir = food101_root / "meta"
    train_meta_file = meta_dir / "train.txt"
    test_meta_file = meta_dir / "test.txt"
    images_dir = food101_root / "images"

    assert train_meta_file.exists(), (
        f"Assert failed: missing official Food-101 train metadata at '{train_meta_file}'. "
        f"Ensure --food101_root contains meta/train.txt."
    )
    assert test_meta_file.exists(), (
        f"Assert failed: missing official Food-101 test metadata at '{test_meta_file}'. "
        f"Ensure --food101_root contains meta/test.txt."
    )
    assert images_dir.exists(), (
        f"Assert failed: missing images directory at '{images_dir}'."
    )

    # Parse official split files
    train_files_by_class = parse_meta_split_file(train_meta_file, images_dir)
    test_files_by_class = parse_meta_split_file(test_meta_file, images_dir)

    all_classes = sorted(list(train_files_by_class.keys()))
    assert len(all_classes) == C, (
        f"Assert failed: expected exactly {C} classes in train.txt, found {len(all_classes)}."
    )
    assert set(all_classes) == set(test_files_by_class.keys()), (
        f"Assert failed: class sets between train.txt and test.txt do not match."
    )

    # Verify per-class official split counts: exactly 750 train and 250 test
    for cls in all_classes:
        n_tr = len(train_files_by_class[cls])
        n_te = len(test_files_by_class[cls])
        assert n_tr == 750, (
            f"Assert failed: class '{cls}' has {n_tr} official train files, expected exactly 750."
        )
        assert n_te == 250, (
            f"Assert failed: class '{cls}' has {n_te} official test files, expected exactly 250."
        )

    # 3. Randomly permute class rank assignment (seeded)
    # Permute which Food-101 class gets which rank, then relabel so label 0 = largest, label 100 = smallest
    rng_perm = random.Random(seed)
    permuted_classes = all_classes.copy()
    rng_perm.shuffle(permuted_classes)

    rank_to_class: Dict[int, str] = {rank: permuted_classes[rank] for rank in range(C)}
    class_to_label: Dict[str, int] = {permuted_classes[rank]: rank for rank in range(C)}

    # 4. Partition samples per class using seeded RNG
    manifest_rows = []
    class_map: Dict[str, Any] = {}

    train_path_set = set()
    val_path_set = set()
    test_path_set = set()

    for rank in range(C):
        cls_name = rank_to_class[rank]
        label = rank  # label 0 = largest class, label 100 = smallest class
        target_train_count = counts[rank]

        # Seeded RNG per class ensures absolute determinism independent of execution order
        rng_class = random.Random(seed + rank * 1000 + 13)

        # Shuffle the 750 official train files
        official_train_pool = train_files_by_class[cls_name].copy()
        rng_class.shuffle(official_train_pool)

        # Take exactly 50 as balanced VAL
        val_selected = official_train_pool[:50]
        remaining_700 = official_train_pool[50:]

        assert len(val_selected) == 50, (
            f"Assert failed: expected 50 validation samples for '{cls_name}', got {len(val_selected)}."
        )
        assert len(remaining_700) == 700, (
            f"Assert failed: expected 700 remaining candidate training files for '{cls_name}', got {len(remaining_700)}."
        )

        # From remaining 700, randomly pick counts[rank] as TRAIN (never just the first k files)
        # rng_class.sample performs uniform random subset sampling across all 700 candidates
        train_selected = rng_class.sample(remaining_700, target_train_count)
        assert len(train_selected) == target_train_count, (
            f"Assert failed: expected {target_train_count} train samples for label {label}, got {len(train_selected)}."
        )

        # TEST = full official test split, 250 per class, untouched
        test_selected = test_files_by_class[cls_name].copy()
        assert len(test_selected) == 250, (
            f"Assert failed: expected 250 test samples for '{cls_name}', got {len(test_selected)}."
        )

        # Record paths and verify disjointness within class
        cls_val_paths = [str((images_dir / f).resolve()) for f in val_selected]
        cls_train_paths = [str((images_dir / f).resolve()) for f in train_selected]
        cls_test_paths = [str((images_dir / f).resolve()) for f in test_selected]

        # Verify within-class disjointness
        assert set(cls_train_paths).isdisjoint(set(cls_val_paths)), (
            f"Assert failed: train and val overlap for class '{cls_name}'!"
        )
        assert set(cls_train_paths).isdisjoint(set(cls_test_paths)), (
            f"Assert failed: train and test overlap for class '{cls_name}'!"
        )
        assert set(cls_val_paths).isdisjoint(set(cls_test_paths)), (
            f"Assert failed: val and test overlap for class '{cls_name}'!"
        )

        train_path_set.update(cls_train_paths)
        val_path_set.update(cls_val_paths)
        test_path_set.update(cls_test_paths)

        # Append to manifest
        for p in cls_train_paths:
            sample_id = Path(p).stem
            manifest_rows.append({"id": sample_id, "label": label, "split": "train", "image_path": p})
        for p in cls_val_paths:
            sample_id = Path(p).stem
            manifest_rows.append({"id": sample_id, "label": label, "split": "val", "image_path": p})
        for p in cls_test_paths:
            sample_id = Path(p).stem
            manifest_rows.append({"id": sample_id, "label": label, "split": "test", "image_path": p})

        class_map[str(label)] = {
            "label": label,
            "food_name": cls_name,
            "rank": rank,
            "train_count": target_train_count,
            "val_count": 50,
            "test_count": 250,
            "is_gmm_majority": bool(target_train_count > profile_info["zeta"]),
        }

    # 5. Global disjointness and structural assertions
    assert train_path_set.isdisjoint(val_path_set), (
        "Assert failed: Global train and val splits share overlapping file paths!"
    )
    assert train_path_set.isdisjoint(test_path_set), (
        "Assert failed: Global train and test splits share overlapping file paths!"
    )
    assert val_path_set.isdisjoint(test_path_set), (
        "Assert failed: Global val and test splits share overlapping file paths!"
    )

    # Assert exactly 50 val per class and 250 test per class
    val_counts_by_label = [sum(1 for r in manifest_rows if r["split"] == "val" and r["label"] == lbl) for lbl in range(C)]
    test_counts_by_label = [sum(1 for r in manifest_rows if r["split"] == "test" and r["label"] == lbl) for lbl in range(C)]
    train_counts_by_label = [sum(1 for r in manifest_rows if r["split"] == "train" and r["label"] == lbl) for lbl in range(C)]

    assert all(c == 50 for c in val_counts_by_label), (
        f"Assert failed: validation split is not exactly 50 per class for all classes! Counts: {val_counts_by_label}"
    )
    assert all(c == 250 for c in test_counts_by_label), (
        f"Assert failed: test split is not exactly 250 per class for all classes! Counts: {test_counts_by_label}"
    )
    assert train_counts_by_label == counts, (
        f"Assert failed: train split counts do not match expected exponential counts profile!"
    )

    # 6. Save manifest.csv, class_map.json, config.json
    manifest_csv_path = output_dir / "manifest.csv"
    with open(manifest_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "split", "image_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    class_map_path = output_dir / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    config_data = {
        "dataset_name": "Food-101-LT",
        "seed": seed,
        "n_max": n_max,
        "n_min": n_min,
        "C": C,
        "eta": eta,
        "k": k,
        "imbalance_ratio": profile_info["imbalance_ratio"],
        "zeta": profile_info["zeta"],
        "allow_below_paper_min": allow_below_paper_min,
        "preset": preset_name,
        "n_train_total": len(train_path_set),
        "n_val_total": len(val_path_set),
        "n_test_total": len(test_path_set),
        "n_total": len(manifest_rows),
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # Print summary table
    print_split_summary_table(config_data, class_map, output_dir)

    return {
        "manifest_csv": manifest_csv_path,
        "class_map_json": class_map_path,
        "config_json": config_path,
        "config": config_data,
        "class_map": class_map,
    }


def print_split_summary_table(config: Dict[str, Any], class_map: Dict[str, Any], output_dir: Path):
    """Prints a clear summary table of the generated Long-Tailed Food-101 split."""
    print("\n" + "=" * 105)
    print(f"LONG-TAILED FOOD-101 (FOOD101-LT) SPLIT GENERATION SUMMARY")
    print("=" * 105)
    print(f"  Preset                 : {config['preset']}")
    print(f"  Random Seed            : {config['seed']} (fixed)")
    print(f"  Classes (C)            : {config['C']} (Label 0 = largest, Label {config['C']-1} = smallest)")
    print(f"  Train Profile (n_max)  : {config['n_max']} samples (Label 0: {class_map['0']['food_name']})")
    print(f"  Train Profile (n_min)  : {config['n_min']} samples (Label {config['C']-1}: {class_map[str(config['C']-1)]['food_name']})")
    print(f"  Imbalance Ratio (IR)   : {config['imbalance_ratio']:.2f}:1")
    print(f"  Majority Threshold zeta: {config['zeta']} (n_max // {config['eta']})")
    print("-" * 105)
    print(f"  TRAIN Split Total      : {config['n_train_total']:,} samples (Long-Tailed: {config['n_max']} -> {config['n_min']})")
    print(f"  VAL Split Total        : {config['n_val_total']:,} samples (Exactly 50 per class balanced)")
    print(f"  TEST Split Total       : {config['n_test_total']:,} samples (Exactly 250 per class untouched)")
    print(f"  Grand Total Samples    : {config['n_total']:,} samples across all 101 classes")
    print("-" * 105)
    print(f"  Saved Artifacts in     : {output_dir.resolve()}")
    print(f"    - manifest.csv       : {output_dir / 'manifest.csv'}")
    print(f"    - class_map.json     : {output_dir / 'class_map.json'}")
    print(f"    - config.json        : {output_dir / 'config.json'}")
    print("=" * 105 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build long-tailed Food-101 splits for the RCS paper pipeline."
    )
    parser.add_argument(
        "--food101_root",
        type=str,
        default="./data/food-101",
        help="Path to official Food-101 root directory containing images/ and meta/ (default: ./data/food-101)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/food101_lt",
        help="Directory where manifest.csv, class_map.json, and config.json are saved (default: ./data/food101_lt)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="main",
        choices=["mild", "main", "stress"],
        help="Preset profile: 'mild' (700->70), 'main' (700->40), 'stress' (700->20). Default: main",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (default: 42)")
    parser.add_argument("--n_max", type=int, default=None, help="Override n_max (default: from preset)")
    parser.add_argument("--n_min", type=int, default=None, help="Override n_min (default: from preset)")
    parser.add_argument("--C", type=int, default=101, help="Number of classes (default: 101)")
    parser.add_argument("--eta", type=int, default=7, help="Imbalance tuner parameter (default: 7)")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest components (default: 5)")
    parser.add_argument(
        "--allow_below_paper_min",
        action="store_true",
        help="Allow n_min < 40 for stress testing (required for stress preset).",
    )
    parser.add_argument(
        "--mock_test",
        action="store_true",
        help="Create a temporary mock official Food-101 directory to verify pipeline end-to-end.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preset_cfg = PRESETS[args.preset]
    n_max = args.n_max if args.n_max is not None else preset_cfg["n_max"]
    n_min = args.n_min if args.n_min is not None else preset_cfg["n_min"]
    allow_flag = args.allow_below_paper_min or (args.preset == "stress" and args.allow_below_paper_min)

    food101_root = Path(args.food101_root)
    output_dir = Path(args.output_dir)

    # Optional mock mode for self-contained testing / verification
    if args.mock_test:
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="mock_food101_"))
        print(f"[MOCK MODE] Generating synthetic official Food-101 structure in {temp_dir}...")
        food101_root, _, _ = create_mock_food101_meta(temp_dir)

    build_food101_lt_split(
        food101_root=food101_root,
        output_dir=output_dir,
        seed=args.seed,
        n_max=n_max,
        n_min=n_min,
        C=args.C,
        eta=args.eta,
        k=args.k,
        allow_below_paper_min=allow_flag,
        preset_name=args.preset,
    )


if __name__ == "__main__":
    main()
