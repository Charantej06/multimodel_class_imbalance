"""
tests/test_extract_cremad_video.py
==================================
Unit tests for CREMA-D video feature extraction (Step 7a).
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.extract_cremad_video import (
    load_clip_frames,
    save_npz,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from encoders.registry import VIDEO_BACKBONES

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestLoadClipFrames:
    def test_load_valid_npy(self):
        """Loads valid video frame npy and checks shape and dtype."""
        sample_path = REPO_ROOT / "cache/cremad_video/1001_DFA_ANG_XX.npy"
        assert sample_path.exists(), f"Sample npy not found at {sample_path}"

        frames = load_clip_frames(sample_path)
        assert frames.shape == (16, 224, 224, 3)
        assert frames.dtype == np.uint8

    def test_missing_file_raises(self, tmp_path):
        """Missing npy file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_clip_frames(tmp_path / "nonexistent.npy")

    def test_invalid_shape_raises(self, tmp_path):
        """Array with wrong shape must raise ValueError."""
        bad_file = tmp_path / "bad.npy"
        np.save(bad_file, np.zeros((8, 224, 224, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="Expected 16 frames"):
            load_clip_frames(bad_file)


class TestNormalization:
    def test_imagenet_normalization(self):
        """Verifies ImageNet mean and std tensors."""
        assert torch.allclose(IMAGENET_MEAN, torch.tensor([0.485, 0.456, 0.406]))
        assert torch.allclose(IMAGENET_STD, torch.tensor([0.229, 0.224, 0.225]))


class TestSaveNpz:
    def test_save_and_load_npz(self, tmp_path):
        """Verifies save_npz produces valid npz with expected keys and manifest ordering."""
        test_file = tmp_path / "test_out.npz"
        ids = ["id_0", "id_1", "id_2"]
        feats = np.random.randn(3, 768).astype(np.float32)
        stds = np.random.randn(3, 768).astype(np.float32)

        save_npz(test_file, ids, feats, temporal_std=stds)
        assert test_file.is_file()

        loaded = np.load(test_file)
        assert list(loaded["ids"]) == ids
        assert np.allclose(loaded["features"], feats)
        assert np.allclose(loaded["temporal_std"], stds)


class TestVideoBackbonesRegistry:
    def test_backbones_exist(self):
        expected = ["dinov2", "videomae_k400", "videomae"]
        for b in expected:
            assert b in VIDEO_BACKBONES

