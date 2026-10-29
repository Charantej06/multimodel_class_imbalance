"""
tests/test_dataset_adapter.py
=============================
Unit tests for DatasetAdapter and MultimodalDataset.
No network access or GPU required.

Run:
    python -m pytest tests/test_dataset_adapter.py -v
"""

import json
import os
import textwrap
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from dataset_adapter import (
    DatasetAdapter,
    MultimodalDataset,
    multimodal_collate_fn,
    SUPPORTED_MODALITIES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_csv(tmp_path: Path, rows) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "metadata.csv"
    df.to_csv(p, index=False)
    return p


def _make_json(tmp_path: Path, data) -> Path:
    p = tmp_path / "metadata.json"
    with open(p, "w") as f:
        json.dump(data, f)
    return p


SAMPLE_ROWS = [
    {"id": str(i), "label": "cat" if i % 2 == 0 else "dog",
     "split": "train" if i < 7 else ("val" if i < 9 else "test"),
     "image_path": f"/data/img_{i}.jpg",
     "audio_path": f"/data/aud_{i}.wav"}
    for i in range(10)
]


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------

class TestCSVIngestion:
    def test_load_csv(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        assert len(adapter._df) == 10

    def test_auto_detect_id_label(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        assert adapter._id_col == "id"
        assert adapter._label_col == "label"

    def test_auto_detect_modalities(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        assert "image" in adapter.modalities
        assert "audio" in adapter.modalities
        assert "text" not in adapter.modalities

    def test_label_map_built(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        assert set(adapter.label_map.keys()) == {"cat", "dog"}
        assert set(adapter.label_map.values()) == {0, 1}

    def test_num_classes(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        assert adapter.num_classes == 2


# ---------------------------------------------------------------------------
# JSON ingestion
# ---------------------------------------------------------------------------

class TestJSONIngestion:
    def test_load_json_list(self, tmp_path):
        meta = _make_json(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        assert len(adapter._df) == 10

    def test_load_json_nested_by_split(self, tmp_path):
        nested = {
            "train": SAMPLE_ROWS[:7],
            "val":   SAMPLE_ROWS[7:9],
            "test":  SAMPLE_ROWS[9:],
        }
        meta = _make_json(tmp_path, nested)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        assert len(adapter._df) == 10

    def test_load_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("nothing")
        with pytest.raises(ValueError, match="Unsupported metadata format"):
            DatasetAdapter({"metadata_path": str(p)}).load()


# ---------------------------------------------------------------------------
# Split handling
# ---------------------------------------------------------------------------

class TestSplits:
    def test_explicit_split_column(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        train_ds = adapter.get_dataset("train")
        val_ds   = adapter.get_dataset("val")
        test_ds  = adapter.get_dataset("test")
        assert len(train_ds) == 7
        assert len(val_ds)   == 2
        assert len(test_ds)  == 1

    def test_derived_splits(self, tmp_path):
        rows = [{"id": str(i), "label": "a", "image_path": f"/img_{i}.jpg"}
                for i in range(100)]
        meta = _make_csv(tmp_path, rows)
        adapter = DatasetAdapter({
            "metadata_path": str(meta),
            "split_ratios": [0.7, 0.15, 0.15],
            "split_seed": 0,
        }).load()
        n_train = len(adapter.get_dataset("train"))
        n_val   = len(adapter.get_dataset("val"))
        n_test  = len(adapter.get_dataset("test"))
        assert n_train + n_val + n_test == 100
        assert abs(n_train - 70) <= 2   # allow rounding

    def test_invalid_split_name_raises(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        with pytest.raises(ValueError):
            adapter.get_dataset("holdout")


# ---------------------------------------------------------------------------
# MultimodalDataset
# ---------------------------------------------------------------------------

class TestMultimodalDataset:
    def _make_adapter(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        return DatasetAdapter({"metadata_path": str(meta)}).load()

    def test_len(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        ds = adapter.get_dataset("train")
        assert len(ds) == 7

    def test_item_keys(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        ds = adapter.get_dataset("train")
        item = ds[0]
        assert "id" in item
        assert "label" in item
        assert "label_str" in item
        for mod in SUPPORTED_MODALITIES:
            assert mod in item

    def test_label_is_int(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        ds = adapter.get_dataset("train")
        assert isinstance(ds[0]["label"], int)

    def test_missing_modality_is_none(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        ds = adapter.get_dataset("train")
        # text and video columns not in CSV → None
        assert ds[0]["text"] is None
        assert ds[0]["video"] is None


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

class TestCollateFn:
    def test_label_tensor(self, tmp_path):
        import torch
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        ds = adapter.get_dataset("train")
        batch = [ds[i] for i in range(3)]
        out = multimodal_collate_fn(batch)
        assert isinstance(out["label"], torch.Tensor)
        assert out["label"].dtype == torch.long
        assert out["label"].shape == (3,)

    def test_ids_are_list(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        ds = adapter.get_dataset("train")
        batch = [ds[i] for i in range(3)]
        out = multimodal_collate_fn(batch)
        assert isinstance(out["id"], list)
        assert len(out["id"]) == 3


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------

class TestConfigOverrides:
    def test_custom_label_map(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({
            "metadata_path": str(meta),
            "label_map": {"cat": 10, "dog": 20},
        }).load()
        ds = adapter.get_dataset("train")
        assert ds[0]["label"] in {10, 20}

    def test_explicit_modality_cols(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({
            "metadata_path": str(meta),
            "modality_cols": {"image": "image_path"},
        }).load()
        assert adapter.modalities == ["image"]

    def test_invalid_modality_col_raises(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        with pytest.raises(ValueError, match="not found in metadata"):
            DatasetAdapter({
                "metadata_path": str(meta),
                "modality_cols": {"image": "nonexistent_col"},
            }).load()

    def test_missing_metadata_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DatasetAdapter({"metadata_path": "/nonexistent/path.csv"}).load()

    def test_summary_string(self, tmp_path):
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        s = adapter.summary()
        assert "DatasetAdapter Summary" in s
        assert "train" in s


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_dataloader_yields_batches(self, tmp_path):
        import torch
        meta = _make_csv(tmp_path, SAMPLE_ROWS)
        adapter = DatasetAdapter({"metadata_path": str(meta)}).load()
        loader = adapter.get_dataloader("train", batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert isinstance(batch["label"], torch.Tensor)
        assert batch["label"].shape[0] <= 4
