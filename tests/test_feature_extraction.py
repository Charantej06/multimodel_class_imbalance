"""
tests/test_feature_extraction.py
=================================
Unit tests for feature_extraction.py.
Uses mock encoders — no network, no GPU required.

Run:
    python -m pytest tests/test_feature_extraction.py -v
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from dataset_adapter import DatasetAdapter, MultimodalDataset
from feature_extraction import (
    EmbeddingCache,
    FeatureExtractor,
    EXAMPLE_CONFIG,
    _safe_id,
    _resolve_device,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(tmp_path: Path, modality: str = "image") -> DatasetAdapter:
    col = f"{modality}_path"
    rows = [
        {"id": str(i), "label": "a" if i % 2 == 0 else "b",
         "split": "train" if i < 6 else "val",
         col: f"/data/{modality}_{i}.ext"}
        for i in range(8)
    ]
    df = pd.DataFrame(rows)
    meta = tmp_path / "meta.csv"
    df.to_csv(meta, index=False)
    return DatasetAdapter({"metadata_path": str(meta)}).load()


def _fake_encoder_dict(modality: str, embed_dim: int = 4) -> Dict[str, Any]:
    """Minimal encoder dict with a mocked model."""
    model = MagicMock()
    model.parameters.return_value = iter([torch.tensor([1.0])])
    return {
        "model":     model,
        "processor": MagicMock(),
        "embed_dim": embed_dim,
        "modality":  modality,
        "model_id":  f"fake/{modality}",
        "device":    torch.device("cpu"),
    }


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------

class TestEmbeddingCache:
    def test_save_and_load(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path), "image", "fake/model", 4)
        emb = torch.randn(4)
        cache.save("sample_0", "train", emb)
        loaded = cache.load("sample_0", "train")
        assert torch.allclose(emb, loaded)

    def test_exists_true_after_save(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path), "image", "fake/model", 4)
        cache.save("s1", "val", torch.zeros(4))
        assert cache.exists("s1", "val")

    def test_exists_false_before_save(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path), "image", "fake/model", 4)
        assert not cache.exists("never_saved", "train")

    def test_valid_cache_meta(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path), "text", "fake/text", 8)
        cache.save("x", "train", torch.zeros(8))
        cache2 = EmbeddingCache(str(tmp_path), "text", "fake/text", 8)
        assert cache2.is_valid()

    def test_stale_cache_model_id(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path), "text", "model-v1", 8)
        cache.save("x", "train", torch.zeros(8))
        cache2 = EmbeddingCache(str(tmp_path), "text", "model-v2", 8)
        assert not cache2.is_valid()

    def test_stale_cache_embed_dim(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path), "text", "m", 8)
        cache.save("x", "train", torch.zeros(8))
        cache2 = EmbeddingCache(str(tmp_path), "text", "m", 16)
        assert not cache2.is_valid()

    def test_load_split(self, tmp_path):
        adapter = _make_adapter(tmp_path)
        ds = adapter.get_dataset("train")
        cache = EmbeddingCache(str(tmp_path / "cache"), "image", "fake/model", 4)
        for i in range(len(ds)):
            cache.save(ds[i]["id"], "train", torch.ones(4) * i)
        ids, embs, labels = cache.load_split(ds, "train")
        assert len(ids) == len(ds)
        assert embs.shape == (len(ds), 4)
        assert labels.shape == (len(ds),)

    def test_safe_id_replaces_slashes(self):
        assert "/" not in _safe_id("path/to/file")
        assert "\\" not in _safe_id("path\\to\\file")


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------

class TestResolveDevice:
    def test_cpu_string(self):
        assert _resolve_device("cpu") == torch.device("cpu")

    def test_none_returns_device(self):
        dev = _resolve_device(None)
        assert isinstance(dev, torch.device)


# ---------------------------------------------------------------------------
# FeatureExtractor (mocked encoders)
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def _make_extractor(self, tmp_path, modality="image"):
        cfg = dict(EXAMPLE_CONFIG)
        cfg["metadata_path"] = str(tmp_path / "meta.csv")
        cfg["cache_root"]    = str(tmp_path / "cache")
        cfg["modalities"]    = [modality]
        cfg["batch_size"]    = 4
        cfg["device"]        = "cpu"
        return cfg

    def test_extract_split_image(self, tmp_path):
        """Test image extraction with a mocked DINOv2-style encoder."""
        adapter = _make_adapter(tmp_path, modality="image")
        cfg = self._make_extractor(tmp_path, "image")

        extractor = FeatureExtractor(cfg)
        # Inject a fake encoder
        fake_enc = _fake_encoder_dict("image", embed_dim=4)
        extractor._encoders["image"] = fake_enc

        # Mock _forward to return fixed embeddings
        dummy = torch.randn(4, 4)  # batch_size=4, embed_dim=4
        with patch.object(extractor, "_forward", return_value=dummy):
            ds_train = adapter.get_dataset("train")
            extractor.extract_split("image", "train", ds_train)

        cache = EmbeddingCache(cfg["cache_root"], "image", "fake/image", 4)
        ds_train = adapter.get_dataset("train")
        # At least some samples should be cached
        saved = sum(
            cache.exists(ds_train[i]["id"], "train") for i in range(len(ds_train))
        )
        assert saved > 0

    def test_extract_all_skips_cached(self, tmp_path):
        """Re-running extract_all on a fully-cached split should not call _forward."""
        adapter = _make_adapter(tmp_path, modality="image")
        cfg = self._make_extractor(tmp_path, "image")

        extractor = FeatureExtractor(cfg)
        fake_enc = _fake_encoder_dict("image", embed_dim=4)
        extractor._encoders["image"] = fake_enc

        # Pre-populate cache
        ds_train = adapter.get_dataset("train")
        cache_obj = EmbeddingCache(cfg["cache_root"], "image", "fake/image", 4)
        for i in range(len(ds_train)):
            cache_obj.save(ds_train[i]["id"], "train", torch.zeros(4))

        call_count = {"n": 0}
        original_forward = extractor._forward

        def counting_forward(*args, **kwargs):
            call_count["n"] += 1
            return original_forward(*args, **kwargs)

        with patch.object(extractor, "_forward", side_effect=counting_forward):
            extractor.extract_split("image", "train", ds_train)

        assert call_count["n"] == 0  # nothing should be re-extracted

    def test_load_embeddings(self, tmp_path):
        """load_embeddings should return correct shapes."""
        adapter = _make_adapter(tmp_path, modality="image")
        cfg = self._make_extractor(tmp_path, "image")

        extractor = FeatureExtractor(cfg)
        fake_enc = _fake_encoder_dict("image", embed_dim=4)
        extractor._encoders["image"] = fake_enc

        ds_train = adapter.get_dataset("train")
        dummy = torch.randn(4, 4)
        with patch.object(extractor, "_forward", return_value=dummy):
            extractor.extract_split("image", "train", ds_train)

        ids, embs, labels = extractor.load_embeddings("image", "train", ds_train)
        assert isinstance(ids, list)
        assert embs.ndim == 2
        assert embs.shape[1] == 4
        assert labels.shape[0] == embs.shape[0]

    def test_unknown_modality_raises(self, tmp_path):
        adapter = _make_adapter(tmp_path, modality="image")
        cfg = self._make_extractor(tmp_path, "image")
        extractor = FeatureExtractor(cfg)
        extractor._encoders["image"] = _fake_encoder_dict("image")
        ds = adapter.get_dataset("train")
        with pytest.raises(ValueError, match="not loaded"):
            extractor.extract_split("audio", "train", ds)


# ---------------------------------------------------------------------------
# EXAMPLE_CONFIG
# ---------------------------------------------------------------------------

class TestExampleConfig:
    def test_all_keys_present(self):
        for key in ("metadata_path", "cache_root", "modalities", "checkpoints",
                    "batch_size", "text_max_length", "video_num_frames"):
            assert key in EXAMPLE_CONFIG, f"Missing key: {key}"

    def test_checkpoints_match_registry(self):
        from encoders.registry import DEFAULT_CHECKPOINTS
        for mod, ckpt in EXAMPLE_CONFIG["checkpoints"].items():
            assert DEFAULT_CHECKPOINTS.get(mod) == ckpt
