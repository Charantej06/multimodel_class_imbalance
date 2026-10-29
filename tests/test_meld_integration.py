"""
tests/test_meld_integration.py
==============================
Tests for MELD preparation script, MELD-specific column mappings in DatasetAdapter,
and video-derived image (DINOv2) and audio (AudioMAE) extraction pipelines.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from dataset_adapter import DatasetAdapter, MELD_CONFIG
from feature_extraction import (
    _preprocess_image,
    _preprocess_audio,
    _sample_representative_frame,
    _is_video_path,
    _extract_audio_from_video,
    FeatureExtractor,
)
from prepare_meld import prepare_meld


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_meld_dir(tmp_path):
    """Create a minimal mock MELD directory tree with CSVs and dummy video files."""
    meld_root = tmp_path / "MELD.Raw"
    train_dir = meld_root / "train" / "train_splits"
    dev_dir = meld_root / "dev" / "dev_splits_complete"
    test_dir = meld_root / "test" / "output_repeated_splits_test"

    for d in (train_dir, dev_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Train CSV
    train_csv = meld_root / "train" / "train_sent_emo.csv"
    pd.DataFrame({
        "Sr No.": [1, 2],
        "Utterance": ["Hello there", "General Kenobi"],
        "Speaker": ["Obi-Wan", "Grievous"],
        "Emotion": ["joy", "anger"],
        "Sentiment": ["positive", "negative"],
        "Dialogue_ID": [0, 0],
        "Utterance_ID": [0, 1],
    }).to_csv(train_csv, index=False)

    # Dev CSV
    dev_csv = meld_root / "dev_sent_emo.csv"
    pd.DataFrame({
        "Sr No.": [3],
        "Utterance": ["I have a bad feeling about this"],
        "Speaker": ["Luke"],
        "Emotion": ["fear"],
        "Sentiment": ["negative"],
        "Dialogue_ID": [1],
        "Utterance_ID": [0],
    }).to_csv(dev_csv, index=False)

    # Test CSV
    test_csv = meld_root / "test_sent_emo.csv"
    pd.DataFrame({
        "Sr No.": [4],
        "Utterance": ["May the force be with you"],
        "Speaker": ["Yoda"],
        "Emotion": ["neutral"],
        "Sentiment": ["neutral"],
        "Dialogue_ID": [2],
        "Utterance_ID": [0],
    }).to_csv(test_csv, index=False)

    # Create dummy video files
    (train_dir / "dia0_utt0.mp4").write_bytes(b"dummy video content 0")
    (train_dir / "dia0_utt1.mp4").write_bytes(b"dummy video content 1")
    (dev_dir / "dia1_utt0.mp4").write_bytes(b"dummy video content 2")
    (test_dir / "dia2_utt0.mp4").write_bytes(b"dummy video content 3")

    return meld_root


# ---------------------------------------------------------------------------
# Test prepare_meld
# ---------------------------------------------------------------------------

class TestPrepareMELD:
    def test_prepare_meld_output(self, mock_meld_dir, tmp_path):
        out_csv = tmp_path / "manifest.csv"
        df = prepare_meld(meld_dir=mock_meld_dir, output_path=out_csv, filter_missing=True)

        assert out_csv.exists()
        assert len(df) == 4
        # Check mapped columns
        assert "text" in df.columns
        assert "label" in df.columns
        assert "video" in df.columns
        assert "split" in df.columns
        assert "id" in df.columns

        # Check split preservation and normalization
        assert set(df["split"].unique()) == {"train", "val", "test"}
        assert (df["split"] == "train").sum() == 2
        assert (df["split"] == "val").sum() == 1
        assert (df["split"] == "test").sum() == 1

        # Check text and label mappings
        assert list(df["text"]) == [
            "Hello there",
            "General Kenobi",
            "I have a bad feeling about this",
            "May the force be with you",
        ]
        assert list(df["label"]) == ["joy", "anger", "fear", "neutral"]

        # Check generated video paths
        for p in df["video"]:
            assert Path(p).exists()
            assert p.endswith(".mp4")

    def test_filter_missing_video(self, mock_meld_dir, tmp_path):
        # Delete one video file
        (mock_meld_dir / "train" / "train_splits" / "dia0_utt1.mp4").unlink()

        out_csv = tmp_path / "filtered_manifest.csv"
        df = prepare_meld(meld_dir=mock_meld_dir, output_path=out_csv, filter_missing=True)
        assert len(df) == 3
        assert "dia0_utt1" not in list(df["id"])


# ---------------------------------------------------------------------------
# Test DatasetAdapter MELD integration
# ---------------------------------------------------------------------------

class TestDatasetAdapterMELD:
    def test_auto_detection_of_utterance_and_emotion(self, tmp_path):
        """Adapter should auto-detect Utterance as text and Emotion as label."""
        csv_file = tmp_path / "meld_sample.csv"
        pd.DataFrame({
            "id": ["1", "2", "3"],
            "Utterance": ["Hello", "World", "Test"],
            "Emotion": ["joy", "sadness", "joy"],
            "split": ["train", "dev", "test"],
            "video_path": ["/path/1.mp4", "/path/2.mp4", "/path/3.mp4"],
        }).to_csv(csv_file, index=False)

        adapter = DatasetAdapter({"metadata_path": str(csv_file)}).load()

        assert adapter.modalities == ["text", "video"]
        assert adapter.num_classes == 2
        assert "joy" in adapter.label_map
        assert "sadness" in adapter.label_map

        # dev split should normalize to val
        train_ds = adapter.get_dataset("train")
        val_ds = adapter.get_dataset("val")
        dev_ds = adapter.get_dataset("dev")  # dev alias should work
        test_ds = adapter.get_dataset("test")

        assert len(train_ds) == 1
        assert len(val_ds) == 1
        assert len(dev_ds) == 1
        assert len(test_ds) == 1

    def test_meld_config_preset(self, tmp_path):
        """Verify MELD_CONFIG preset maps columns properly."""
        csv_file = tmp_path / "meld_sample.csv"
        pd.DataFrame({
            "id": ["1", "2"],
            "Utterance": ["Hi", "Bye"],
            "Emotion": ["neutral", "surprise"],
            "split": ["train", "val"],
            "video": ["v1.mp4", "v2.mp4"],
        }).to_csv(csv_file, index=False)

        cfg = dict(MELD_CONFIG)
        cfg["metadata_path"] = str(csv_file)
        adapter = DatasetAdapter(cfg).load()

        assert adapter.modalities == ["text", "video"]
        assert adapter.num_classes == 2


# ---------------------------------------------------------------------------
# Test FeatureExtraction video-to-image and video-to-audio
# ---------------------------------------------------------------------------

class TestFeatureExtractionVideoHandling:
    def test_is_video_path(self):
        assert _is_video_path("video.mp4") is True
        assert _is_video_path("video.AVI") is True
        assert _is_video_path("image.jpg") is False
        assert _is_video_path("audio.wav") is False
        assert _is_video_path(None) is False

    def test_preprocess_image_with_video_sampling(self):
        """DINOv2 should sample representative frame in memory from video."""
        from PIL import Image
        dummy_frame = Image.new("RGB", (224, 224), color=(255, 0, 0))

        with patch("feature_extraction._sample_representative_frame", return_value=dummy_frame) as mock_sample:
            batch = {"id": ["sample_1"], "video": ["dummy.mp4"]}
            mock_processor = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 224, 224)})

            out = _preprocess_image(batch, mock_processor, device=torch.device("cpu"))
            assert out is not None
            mock_sample.assert_called_once_with("dummy.mp4")
            mock_processor.assert_called_once()
            images_arg = mock_processor.call_args[1]["images"]
            assert len(images_arg) == 1
            assert isinstance(images_arg[0], Image.Image)

    def test_preprocess_audio_with_video_extraction(self, tmp_path):
        """AudioMAE should extract audio track from video to cached WAV."""
        vid_path = tmp_path / "test_video.mp4"
        vid_path.write_bytes(b"dummy video")

        cached_wav = tmp_path / "extracted" / "sample_1.wav"

        with patch("feature_extraction._extract_audio_from_video", return_value=str(cached_wav)) as mock_extract:
            batch = {"id": ["sample_1"], "video": [str(vid_path)]}
            out_paths = _preprocess_audio(
                batch,
                device=torch.device("cpu"),
                audio_cache_dir=tmp_path / "extracted",
            )
            assert out_paths == [str(cached_wav)]
            mock_extract.assert_called_once_with(str(vid_path), cached_wav)
