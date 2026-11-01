"""
food_101_imbalanced_making/extract_food101_features.py
======================================================
Extracts real unimodal feature representations from raw Food-101 images and text titles:
  - Text:  answerdotai/ModernBERT-base -> CLS token (768-d)
  - Image: facebook/dinov2-base        -> CLS token (768-d)

Features are saved to:
  data/food101_lt/features_train.npz
  data/food101_lt/features_val.npz
  data/food101_lt/features_test.npz
"""

from __future__ import annotations
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("extract_food101")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "food101_lt"


class Food101Dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "id": row["id"],
            "label": int(row["label"]),
            "image_path": row["image_path"],
            "title": str(row["title"]) if pd.notna(row["title"]) else "",
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ids": [item["id"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "image_paths": [item["image_path"] for item in batch],
        "titles": [item["title"] for item in batch],
    }


def extract_split_features(
    df_split: pd.DataFrame,
    split_name: str,
    img_model: Any,
    img_processor: Any,
    txt_model: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    dataset = Food101Dataset(df_split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    all_ids = []
    all_labels = []
    img_embed_list = []
    txt_embed_list = []

    logger.info("Extracting %s split: %d samples in %d batches (batch_size=%d)...", split_name, len(dataset), len(dataloader), batch_size)
    t_start = time.time()
    last_log_time = t_start

    for b_idx, batch in enumerate(dataloader):
        ids = batch["ids"]
        labels = batch["labels"].numpy()
        image_paths = batch["image_paths"]
        titles = batch["titles"]

        # 1. Process images
        imgs = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                logger.warning("Failed to open %s: %s. Using black image.", p, e)
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            imgs.append(img)

        img_inputs = img_processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            img_out = img_model(**img_inputs)
            # CLS token is at index 0
            img_cls = img_out.last_hidden_state[:, 0, :].cpu().numpy()

        # 2. Process text
        # If title is empty, fallback to clean text
        cleaned_titles = [t if len(t.strip()) > 0 else "food dish recipe" for t in titles]
        txt_inputs = tokenizer(
            cleaned_titles,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            txt_out = txt_model(**txt_inputs)
            txt_cls = txt_out.last_hidden_state[:, 0, :].cpu().numpy()

        img_embed_list.append(img_cls)
        txt_embed_list.append(txt_cls)
        all_labels.append(labels)
        all_ids.extend(ids)

        if time.time() - last_log_time > 15.0 or (b_idx + 1) == len(dataloader):
            elapsed = time.time() - t_start
            processed = (b_idx + 1) * batch_size
            processed = min(processed, len(dataset))
            speed = processed / elapsed if elapsed > 0 else 0
            logger.info("  [%s] %d/%d (%.1f%%) | %.1f samples/sec | Elapsed: %.1fs",
                        split_name, processed, len(dataset), (processed / len(dataset)) * 100, speed, elapsed)
            last_log_time = time.time()

    all_img = np.concatenate(img_embed_list, axis=0)
    all_txt = np.concatenate(txt_embed_list, axis=0)
    all_labels_arr = np.concatenate(all_labels, axis=0)

    logger.info("Finished %s in %.1fs: img=%s, txt=%s, labels=%s",
                split_name, time.time() - t_start, all_img.shape, all_txt.shape, all_labels_arr.shape)

    return {
        "ids": np.array(all_ids),
        "labels": all_labels_arr,
        "image_embeds": all_img,
        "text_embeds": all_txt,
    }


def main():
    manifest_path = DATA_DIR / "manifest.csv"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"
    df = pd.read_csv(manifest_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.70, device=0)
        logger.info("GPU Memory fraction 70%% allocated on %s", torch.cuda.get_device_name(0))

    logger.info("Loading DINOv2 (facebook/dinov2-base)...")
    img_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    img_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

    logger.info("Loading ModernBERT (answerdotai/ModernBERT-base)...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    txt_model = AutoModel.from_pretrained("answerdotai/ModernBERT-base").to(device).eval()

    # Freeze all weights
    for p in img_model.parameters():
        p.requires_grad = False
    for p in txt_model.parameters():
        p.requires_grad = False

    for split in ["val", "train", "test"]:
        out_file = DATA_DIR / f"features_{split}.npz"
        if out_file.exists():
            logger.info("Features for %s already exist at %s. Skipping.", split, out_file)
            continue

        df_split = df[df["split"] == split].reset_index(drop=True)
        split_dict = extract_split_features(
            df_split=df_split,
            split_name=split,
            img_model=img_model,
            img_processor=img_processor,
            txt_model=txt_model,
            tokenizer=tokenizer,
            device=device,
            batch_size=32,
        )

        np.savez_compressed(
            out_file,
            ids=split_dict["ids"],
            labels=split_dict["labels"],
            image_embeds=split_dict["image_embeds"],
            text_embeds=split_dict["text_embeds"],
        )
        logger.info("Saved %s features to %s (%.2f MB)", split, out_file, out_file.stat().st_size / (1024 * 1024))

    logger.info("Feature extraction complete for all splits!")


if __name__ == "__main__":
    main()

