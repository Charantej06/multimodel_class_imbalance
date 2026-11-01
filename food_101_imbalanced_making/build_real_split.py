"""
food_101_imbalanced_making/build_real_split.py
=============================================
Builds the real Long-Tailed Food-101 multimodal dataset split using actual
images and text titles on disk:
  - Input images: food_101/images/train, food_101/images/test
  - Input texts:  food_101/texts/train_titles.csv, food_101/texts/test_titles.csv
  - Outputs to:   data/food101_lt/manifest.csv, class_map.json, config.json

Imbalance Profile:
  - C = 101 classes
  - n_max = 500 (majority class)
  - n_min = 40  (minority class)
  - Imbalance Ratio (IR) = 12.50:1
  - Train: 16,736 samples (exponential decay)
  - Val:   5,050 samples (exactly 50 / class, balanced)
  - Test:  15,150 samples (exactly 150 / class, balanced)
"""

from __future__ import annotations
import os
import sys
import json
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FOOD101_ROOT = REPO_ROOT / "food_101"
OUTPUT_DIR = REPO_ROOT / "data" / "food101_lt"


def make_exponential_counts(n_max: int = 500, n_min: int = 40, C: int = 101) -> List[int]:
    return [round(n_max * (n_min / n_max) ** (i / (C - 1))) for i in range(C)]


def build_real_split(
    food101_root: Path = FOOD101_ROOT,
    output_dir: Path = OUTPUT_DIR,
    seed: int = 42,
    n_max: int = 500,
    n_min: int = 40,
    val_per_class: int = 50,
    test_per_class: int = 150,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load text titles
    train_text_file = food101_root / "texts" / "train_titles.csv"
    test_text_file = food101_root / "texts" / "test_titles.csv"
    
    assert train_text_file.exists(), f"Missing {train_text_file}"
    assert test_text_file.exists(), f"Missing {test_text_file}"
    
    df_tr = pd.read_csv(train_text_file, header=None, names=["filename", "title", "class_name"])
    df_te = pd.read_csv(test_text_file, header=None, names=["filename", "title", "class_name"])
    
    # Clean titles
    df_tr["title"] = df_tr["title"].fillna("").astype(str).str.strip()
    df_te["title"] = df_te["title"].fillna("").astype(str).str.strip()
    
    train_img_root = food101_root / "images" / "train"
    test_img_root = food101_root / "images" / "test"
    
    classes = sorted(df_tr["class_name"].unique())
    C = len(classes)
    assert C == 101, f"Expected 101 classes, got {C}"
    
    counts = make_exponential_counts(n_max=n_max, n_min=n_min, C=C)
    
    # Permute class ranks seeded
    rng_perm = random.Random(seed)
    permuted_classes = classes.copy()
    rng_perm.shuffle(permuted_classes)
    
    class_to_label = {permuted_classes[r]: r for r in range(C)}
    label_to_class = {r: permuted_classes[r] for r in range(C)}
    
    # Group available pairs by class
    tr_pairs_by_class: Dict[str, List[Dict[str, str]]] = {c: [] for c in classes}
    for _, row in df_tr.iterrows():
        img_path = (train_img_root / row["class_name"] / row["filename"]).resolve()
        if img_path.exists():
            tr_pairs_by_class[row["class_name"]].append({
                "filename": row["filename"],
                "class_name": row["class_name"],
                "image_path": str(img_path),
                "title": row["title"],
            })
            
    te_pairs_by_class: Dict[str, List[Dict[str, str]]] = {c: [] for c in classes}
    for _, row in df_te.iterrows():
        img_path = (test_img_root / row["class_name"] / row["filename"]).resolve()
        if img_path.exists():
            te_pairs_by_class[row["class_name"]].append({
                "filename": row["filename"],
                "class_name": row["class_name"],
                "image_path": str(img_path),
                "title": row["title"],
            })
            
    manifest_rows = []
    class_map = {}
    
    train_paths = set()
    val_paths = set()
    test_paths = set()
    
    for rank in range(C):
        cname = label_to_class[rank]
        target_train_count = counts[rank]
        label = rank
        
        pool = tr_pairs_by_class[cname].copy()
        assert len(pool) >= val_per_class + target_train_count, (
            f"Class {cname} only has {len(pool)} train samples, need {val_per_class + target_train_count}"
        )
        
        # Seeded per-class RNG
        rng_c = random.Random(seed + rank * 1000 + 7)
        rng_c.shuffle(pool)
        
        val_items = pool[:val_per_class]
        rem_pool = pool[val_per_class:]
        
        train_items = rng_c.sample(rem_pool, target_train_count)
        
        # Test items
        te_pool = te_pairs_by_class[cname].copy()
        assert len(te_pool) >= test_per_class, (
            f"Class {cname} only has {len(te_pool)} test samples, need {test_per_class}"
        )
        rng_c.shuffle(te_pool)
        test_items = te_pool[:test_per_class]
        
        # Verify disjointness
        v_paths = {it["image_path"] for it in val_items}
        tr_paths = {it["image_path"] for it in train_items}
        te_paths = {it["image_path"] for it in test_items}
        
        assert v_paths.isdisjoint(tr_paths), f"Overlap train/val in {cname}"
        assert tr_paths.isdisjoint(te_paths), f"Overlap train/test in {cname}"
        assert v_paths.isdisjoint(te_paths), f"Overlap val/test in {cname}"
        
        train_paths.update(tr_paths)
        val_paths.update(v_paths)
        test_paths.update(te_paths)
        
        for it in train_items:
            manifest_rows.append({
                "id": f"{cname}_{it['filename']}",
                "label": label,
                "class_name": cname,
                "split": "train",
                "image_path": it["image_path"],
                "title": it["title"],
            })
            
        for it in val_items:
            manifest_rows.append({
                "id": f"{cname}_{it['filename']}",
                "label": label,
                "class_name": cname,
                "split": "val",
                "image_path": it["image_path"],
                "title": it["title"],
            })
            
        for it in test_items:
            manifest_rows.append({
                "id": f"{cname}_{it['filename']}",
                "label": label,
                "class_name": cname,
                "split": "test",
                "image_path": it["image_path"],
                "title": it["title"],
            })
            
        class_map[str(label)] = {
            "rank": rank,
            "class_name": cname,
            "train_count": target_train_count,
            "val_count": val_per_class,
            "test_count": test_per_class,
        }

    # Strict global disjointness assertions
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
    
    # Save manifest.csv
    manifest_file = output_dir / "manifest.csv"
    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "label", "class_name", "split", "image_path", "title"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    # Save class_map.json
    class_map_file = output_dir / "class_map.json"
    with open(class_map_file, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)
        
    config = {
        "dataset": "food_101",
        "num_classes": C,
        "n_max": n_max,
        "n_min": n_min,
        "imbalance_ratio": float(n_max / n_min),
        "seed": seed,
        "total_train": sum(counts),
        "total_val": C * val_per_class,
        "total_test": C * test_per_class,
        "val_per_class": val_per_class,
        "test_per_class": test_per_class,
    }
    config_file = output_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    print(f"Successfully generated Food-101 Long-Tailed real split:")
    print(f"  - Total Train: {config['total_train']} (n_max={n_max}, n_min={n_min}, IR={config['imbalance_ratio']:.2f}:1)")
    print(f"  - Total Val:   {config['total_val']} ({val_per_class} per class balanced)")
    print(f"  - Total Test:  {config['total_test']} ({test_per_class} per class balanced)")
    print(f"  - Manifest:    {manifest_file}")
    return config


if __name__ == "__main__":
    build_real_split()

