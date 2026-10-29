









"""
dataset_adapter.py
==================
Dataset-independent adapter for multimodal projects.

Responsibilities
----------------
- Read CSV or JSON metadata files.
- Auto-detect or accept explicit column mappings for each modality.
- Map sample IDs and string labels to integer class indices.
- Expose per-split (train / val / test) PyTorch Datasets and DataLoaders.
- Stay format-agnostic: raw file paths are returned, NOT loaded tensors.

Expected metadata schema (minimum required columns):
    id          : unique sample identifier (string or int)
    label       : class label (string or int)
    split       : "train" / "val" / "test"   (optional; derived from split_ratios)
    <modality>  : path to the raw file, e.g. text_path, image_path, audio_path, video_path

Example config dict
-------------------
config = {
    "metadata_path": "data/metadata.csv",   # or .json / .jsonl
    "id_col":        "id",                  # None -> auto-detect
    "label_col":     "label",               # None -> auto-detect
    "split_col":     "split",               # None -> derive from split_ratios
    "split_ratios":  [0.7, 0.15, 0.15],     # only used when split_col is absent
    "split_seed":    42,
    "modality_cols": {                       # None -> auto-detect from column names
        "text":  "text_path",
        "image": "image_path",
        "audio": "audio_path",
        "video": "video_path",
    },
    "label_map": None,   # None -> built automatically; or pass {"cat": 0, "dog": 1}
}
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_MODALITIES = ("text", "image", "audio", "video")

# Heuristic column-name candidates for auto-detection (order = priority)
_MODALITY_HINTS: Dict[str, List[str]] = {
    "text":  ["text_path", "text_file", "text", "caption", "transcript", "utterance"],
    "image": ["image_path", "image_file", "img_path", "image", "img", "photo"],
    "audio": ["audio_path", "audio_file", "wav_path", "audio", "sound"],
    "video": ["video_path", "video_file", "mp4_path", "video", "clip"],
}
_ID_HINTS    = ["id", "sample_id", "uid", "filename", "index"]
_LABEL_HINTS = ["label", "class", "category", "target", "y", "emotion", "sentiment"]
_SPLIT_HINTS = ["split", "subset", "partition", "fold"]

# Explicit configuration preset for MELD dataset
MELD_CONFIG: Dict[str, Any] = {
    "id_col": "id",
    "label_col": "Emotion",
    "split_col": "split",
    "modality_cols": {
        "text": "Utterance",
        "video": "video",
    },
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_column(columns: List[str], hints: List[str], name: str) -> Optional[str]:
    """Return the first hint column that exists in *columns* (case-insensitive)."""
    lower_map = {c.lower(): c for c in columns}
    for hint in hints:
        if hint.lower() in lower_map:
            matched = lower_map[hint.lower()]
            logger.debug("Auto-detected %s column: '%s'", name, matched)
            return matched
    return None


def _load_metadata(path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV or JSON (records list / nested-by-split dict) into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    elif suffix in {".json", ".jsonl"}:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            # Support {"train": [...], "val": [...], "test": [...]} layout
            records: List[Dict] = []
            for split_name, items in raw.items():
                for item in items:
                    item = dict(item)
                    item.setdefault("split", split_name)
                    records.append(item)
            df = pd.DataFrame(records)
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    else:
        raise ValueError(f"Unsupported metadata format '{suffix}'. Use .csv or .json")

    df = df.reset_index(drop=True)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def _assign_splits(
    df: pd.DataFrame,
    split_col: str,
    ratios: Sequence[float],
    seed: int,
) -> pd.DataFrame:
    """
    Use an existing split column when valid; otherwise derive splits from *ratios*.
    Always writes to df['split'].
    """
    if split_col in df.columns and df[split_col].notna().all():
        raw_splits = df[split_col].astype(str).str.strip().str.lower()
        # Normalize split naming variations like dev -> val
        df["split"] = raw_splits.replace({
            "dev": "val",
            "development": "val",
            "validation": "val",
            "valid": "val",
            "eval": "val",
            "training": "train",
            "testing": "test",
        })
        logger.info("Using existing split column '%s'.", split_col)
        return df

    if split_col in df.columns:
        logger.warning(
            "Column '%s' contains nulls; ignoring it and re-assigning splits.", split_col
        )

    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("split_ratios must be a 3-tuple [train, val, test] summing to 1.0")

    n = len(df)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    t_end = int(n * ratios[0])
    v_end = t_end + int(n * ratios[1])

    splits = ["train"] * n
    for i in idxs[t_end:v_end]:
        splits[i] = "val"
    for i in idxs[v_end:]:
        splits[i] = "test"
    df["split"] = splits

    logger.info(
        "Assigned splits — train: %d | val: %d | test: %d",
        splits.count("train"), splits.count("val"), splits.count("test"),
    )
    return df


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

class DatasetAdapter:
    """
    Adapts a multimodal metadata file into per-split PyTorch Datasets.

    Usage::

        adapter = DatasetAdapter(config).load()
        print(adapter.summary())
        train_ds  = adapter.get_dataset("train")
        val_loader = adapter.get_dataloader("val", batch_size=32)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._df: Optional[pd.DataFrame] = None
        self._label_map: Optional[Dict[str, int]] = None
        self._modality_cols: Optional[Dict[str, str]] = None
        self._id_col: Optional[str] = None
        self._label_col: Optional[str] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "DatasetAdapter":
        """Parse metadata, resolve columns, assign splits, build label map."""
        cfg = self.config

        # 1. Raw DataFrame
        df = _load_metadata(cfg["metadata_path"])
        cols = df.columns.tolist()

        # 2. ID column
        id_col = cfg.get("id_col") or _find_column(cols, _ID_HINTS, "id")
        if id_col is None:
            df.insert(0, "__id__", df.index.astype(str))
            id_col = "__id__"
            logger.warning("No ID column found; using row index as sample ID.")
        self._id_col = id_col

        # 3. Label column
        label_col = cfg.get("label_col") or _find_column(cols, _LABEL_HINTS, "label")
        if label_col is None:
            raise ValueError(
                "Could not find a label column. "
                f"Set 'label_col' in config or name a column from: {_LABEL_HINTS}"
            )
        self._label_col = label_col

        # 4. Split assignment
        split_col = (
            cfg.get("split_col")
            or _find_column(cols, _SPLIT_HINTS, "split")
            or "split"
        )
        df = _assign_splits(
            df,
            split_col=split_col,
            ratios=cfg.get("split_ratios", [0.7, 0.15, 0.15]),
            seed=cfg.get("split_seed", 42),
        )

        # 5. Modality columns
        # If modality_cols is explicitly provided, treat it as the *complete*
        # specification — do not auto-detect additional modalities beyond it.
        # Auto-detection only runs when modality_cols is absent from config.
        mod_cfg: Optional[Dict[str, str]] = cfg.get("modality_cols")
        detected: Dict[str, str] = {}

        if mod_cfg is not None:
            # Explicit config: honour only the listed modalities
            for mod, col in mod_cfg.items():
                if col not in df.columns:
                    raise ValueError(
                        f"Configured modality column '{col}' not found in metadata."
                    )
                detected[mod] = col
                logger.info("Modality '%s' -> column '%s' (config).", mod, col)
        else:
            # No config given: auto-detect from column names
            for mod in SUPPORTED_MODALITIES:
                found = _find_column(cols, _MODALITY_HINTS[mod], mod)
                if found:
                    detected[mod] = found
                    logger.info("Modality '%s' -> column '%s' (auto-detected).", mod, found)

        if not detected:
            raise ValueError(
                "No modality columns detected. "
                f"Set 'modality_cols' in config or name columns from: {_MODALITY_HINTS}"
            )
        self._modality_cols = detected

        # 6. Label map
        label_map = cfg.get("label_map")
        if label_map is None:
            unique_labels = sorted(df[label_col].dropna().unique().tolist())
            label_map = {str(lbl): idx for idx, lbl in enumerate(unique_labels)}
            logger.info("Built label map (%d classes): %s", len(label_map), label_map)
        self._label_map = label_map

        # 7. Warn about missing files (non-blocking)
        self._validate_paths(df)

        self._df = df
        self._loaded = True
        return self

    def get_dataset(self, split: str) -> "MultimodalDataset":
        """Return a :class:`MultimodalDataset` for *split* (train/val/test)."""
        self._ensure_loaded()
        if split == "dev":
            split = "val"
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        subset = self._df[self._df["split"] == split].reset_index(drop=True)
        logger.info("Split '%s': %d samples.", split, len(subset))
        return MultimodalDataset(
            df=subset,
            id_col=self._id_col,
            label_col=self._label_col,
            modality_cols=self._modality_cols,
            label_map=self._label_map,
        )

    def get_dataloader(
        self,
        split: str,
        batch_size: int = 16,
        shuffle: Optional[bool] = None,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """Convenience wrapper: dataset + DataLoader for a given split."""
        if shuffle is None:
            shuffle = split == "train"
        ds = self.get_dataset(split)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=multimodal_collate_fn,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Properties / introspection
    # ------------------------------------------------------------------

    @property
    def label_map(self) -> Dict[str, int]:
        self._ensure_loaded()
        return self._label_map

    @property
    def modalities(self) -> List[str]:
        self._ensure_loaded()
        return list(self._modality_cols.keys())

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    def class_counts(self, split: str = "train") -> Dict[str, int]:
        """Return {label_string: count} for *split*."""
        self._ensure_loaded()
        if split == "dev":
            split = "val"
        subset = self._df[self._df["split"] == split]
        return dict(subset[self._label_col].value_counts())

    def summary(self) -> str:
        """Human-readable overview of the dataset."""
        self._ensure_loaded()
        lines = [
            "DatasetAdapter Summary",
            "======================",
            f"  Metadata path  : {self.config['metadata_path']}",
            f"  Total samples  : {len(self._df)}",
            f"  Modalities     : {self.modalities}",
            f"  Num classes    : {self.num_classes}",
            f"  Label map      : {self._label_map}",
        ]
        for sp in ("train", "val", "test"):
            n = int((self._df["split"] == sp).sum())
            lines.append(f"  {sp:5s} samples  : {n}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call adapter.load() before accessing data.")

    def _validate_paths(self, df: pd.DataFrame):
        missing: Dict[str, int] = defaultdict(int)
        for mod, col in self._modality_cols.items():
            for val in df[col].dropna():
                s = str(val)
                # If text column contains inline text rather than file paths, don't flag as missing files
                if mod == "text" and (Path(s).suffix.lower() not in {".txt", ".json", ".csv", ".md", ".text", ".log", ".tsv"} or " " in s or "\n" in s):
                    continue
                if not Path(s).exists():
                    missing[mod] += 1
        for mod, count in missing.items():
            if count > 0:
                logger.warning(
                    "Modality '%s': %d / %d paths not found on disk "
                    "(may be relative paths not yet resolved).",
                    mod, count, len(df),
                )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MultimodalDataset(Dataset):
    """
    A ``torch.utils.data.Dataset`` over a metadata DataFrame.

    Each ``__getitem__`` returns a plain dict::

        {
            "id":        str,        # sample identifier
            "label":     int,        # integer class index  (-1 if unknown)
            "label_str": str,        # original string label
            "text":      str | None, # raw file path (or text string)
            "image":     str | None, # raw file path
            "audio":     str | None, # raw file path
            "video":     str | None, # raw file path
        }

    Tensors are NOT loaded here; modality-specific preprocessing
    is the responsibility of ``feature_extraction.py``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str,
        label_col: str,
        modality_cols: Dict[str, str],
        label_map: Dict[str, int],
    ):
        self._df = df
        self._id_col = id_col
        self._label_col = label_col
        self._modality_cols = modality_cols
        self._label_map = label_map

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._df.iloc[idx]
        label_str = str(row[self._label_col])
        item: Dict[str, Any] = {
            "id":        str(row[self._id_col]),
            "label":     self._label_map.get(label_str, -1),
            "label_str": label_str,
        }
        for mod in SUPPORTED_MODALITIES:
            col = self._modality_cols.get(mod)
            val = row.get(col) if col else None
            item[mod] = str(val) if (col and pd.notna(val)) else None
        return item

    @property
    def label_map(self) -> Dict[str, int]:
        return self._label_map

    @property
    def modalities(self) -> List[str]:
        return list(self._modality_cols.keys())


# ---------------------------------------------------------------------------
# Custom collate function
# ---------------------------------------------------------------------------

def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates a list of sample dicts into a batched dict.
    - ``label``     → torch.LongTensor  [B]
    - everything else → list of strings (or None)
    """
    import torch

    keys = batch[0].keys()
    result: Dict[str, Any] = {}
    for k in keys:
        values = [item[k] for item in batch]
        result[k] = torch.tensor(values, dtype=torch.long) if k == "label" else values
    return result


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Quick-test DatasetAdapter on a metadata file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example:
              python dataset_adapter.py --meta data/metadata.csv \\
                  --id_col id --label_col label \\
                  --modality_cols text=text_path image=image_path
            """
        ),
    )
    parser.add_argument("--meta", required=True, help="Path to CSV or JSON metadata")
    parser.add_argument("--id_col",    default=None)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--split_col", default=None)
    parser.add_argument(
        "--modality_cols", nargs="*",
        help="Modality=column pairs, e.g. text=text_path image=img_path",
    )
    args = parser.parse_args()

    mod_cols = None
    if args.modality_cols:
        mod_cols = dict(pair.split("=", 1) for pair in args.modality_cols)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    adapter = DatasetAdapter({
        "metadata_path": args.meta,
        "id_col":        args.id_col,
        "label_col":     args.label_col,
        "split_col":     args.split_col,
        "modality_cols": mod_cols,
    }).load()

    print(adapter.summary())
    train_ds = adapter.get_dataset("train")
    print(f"\nFirst training sample:\n{train_ds[0]}")
