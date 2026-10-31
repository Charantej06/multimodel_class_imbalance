"""
tests/test_preprocess_cremad_video.py
=====================================
Unit tests for CREMA-D video preprocessing (Step 5).
"""

from pathlib import Path
import numpy as np
import pytest

from scripts.preprocess_cremad_video import (
    calculate_crop_window,
    decode_video_frames,
    process_single_clip,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCropCalculations:
    def test_center_crop_fallback(self):
        """When detection fails (None box), calculate_crop_window returns center crop."""
        y1, y2, x1, x2 = calculate_crop_window(None, img_w=480, img_h=360, margin=0.2, square=True)
        # In a 480x360 frame, square center crop should have size min(480, 360) = 360
        assert (y2 - y1) == 360
        assert (x2 - x1) == 360
        assert y1 == 0 and y2 == 360
        assert x1 == 60 and x2 == 420

    def test_square_crop_with_margin(self):
        """Median face box is expanded with 20% margin and kept square."""
        # Face box: w = 100, h = 100, centered at (240, 180)
        box = (190.0, 130.0, 290.0, 230.0)
        y1, y2, x1, x2 = calculate_crop_window(box, img_w=480, img_h=360, margin=0.2, square=True)
        # side = 100 * (1 + 2 * 0.2) = 140
        assert (x2 - x1) == 140
        assert (y2 - y1) == 140
        assert x1 == 240 - 70 and x2 == 240 + 70
        assert y1 == 180 - 70 and y2 == 180 + 70

    def test_boundary_shift(self):
        """Square crop near boundary is shifted inside frame dimensions."""
        box = (10.0, 10.0, 110.0, 110.0)  # Near top-left
        y1, y2, x1, x2 = calculate_crop_window(box, img_w=480, img_h=360, margin=0.2, square=True)
        assert x1 >= 0 and y1 >= 0
        assert x2 <= 480 and y2 <= 360


class TestVideoPreprocessingPipeline:
    def test_cached_output_structure(self):
        """Verifies preprocessed .npy files have exact shape (16, 224, 224, 3) and uint8 dtype."""
        cache_dir = REPO_ROOT / "cache/cremad_video"
        npys = list(cache_dir.glob("*.npy"))
        assert len(npys) > 0, "No preprocessed .npy files found in cache/cremad_video"

        sample_npy = npys[0]
        data = np.load(sample_npy)
        assert data.shape == (16, 224, 224, 3)
        assert data.dtype == np.uint8
        # Ensure values are valid pixel values in [0, 255]
        assert data.min() >= 0 and data.max() <= 255

    def test_resumability(self):
        """If clip is already processed, returns cached status without reprocessing."""
        cache_dir = REPO_ROOT / "cache/cremad_video"
        npys = list(cache_dir.glob("*.npy"))
        assert len(npys) > 0
        clip_id = npys[0].stem

        res = process_single_clip(
            clip_id=clip_id,
            media_root=REPO_ROOT / "CREMAD",
            output_dir=cache_dir,
            num_frames=16,
            image_size=224,
            skip_existing=True,
        )
        assert res["status"] == "cached"

