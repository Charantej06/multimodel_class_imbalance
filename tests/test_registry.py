"""
tests/test_registry.py
=======================
Unit tests for encoders/registry.py that do NOT require downloading
pretrained weights or a GPU.

Tests verify:
- Registry structure and default checkpoints
- freeze / unfreeze helpers on a tiny dummy model
- load_encoders validates unknown modality

Run:
    python -m pytest tests/test_registry.py -v
"""

import pytest
import torch
import torch.nn as nn

from encoders.registry import (
    DEFAULT_CHECKPOINTS,
    EMBED_DIMS,
    SUPPORTED_MODALITIES,
    _freeze,
    _param_count,
    freeze_encoder,
    unfreeze_encoder,
    encoder_info,
    load_encoder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model():
    """A minimal nn.Linear to stand in for a real encoder."""
    m = nn.Linear(4, 4)
    return m


def _fake_encoder(modality="text"):
    m = _tiny_model()
    return {
        "model":     m,
        "processor": None,
        "embed_dim": 768,
        "modality":  modality,
        "model_id":  "fake/model",
        "device":    torch.device("cpu"),
    }


# ---------------------------------------------------------------------------
# Registry constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_all_modalities_have_checkpoints(self):
        for mod in ("text", "image", "audio", "video"):
            assert mod in DEFAULT_CHECKPOINTS, f"Missing checkpoint for '{mod}'"

    def test_all_modalities_have_embed_dims(self):
        for mod in ("text", "image", "audio", "video"):
            assert mod in EMBED_DIMS
            assert isinstance(EMBED_DIMS[mod], int)
            assert EMBED_DIMS[mod] > 0

    def test_modernbert_checkpoint(self):
        assert DEFAULT_CHECKPOINTS["text"] == "answerdotai/ModernBERT-base"

    def test_dinov2_checkpoint(self):
        assert DEFAULT_CHECKPOINTS["image"] == "facebook/dinov2-base"

    def test_audiomae_checkpoint(self):
        assert DEFAULT_CHECKPOINTS["audio"] == "hance-ai/audiomae"

    def test_videomae_checkpoint(self):
        assert DEFAULT_CHECKPOINTS["video"] == "MCG-NJU/videomae-base"


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

class TestFreezeHelpers:
    def test_freeze_sets_requires_grad_false(self):
        m = _tiny_model()
        for p in m.parameters():
            assert p.requires_grad  # unfrozen by default
        _freeze(m)
        for p in m.parameters():
            assert not p.requires_grad

    def test_freeze_idempotent(self):
        m = _tiny_model()
        _freeze(m)
        _freeze(m)  # second call must not raise
        for p in m.parameters():
            assert not p.requires_grad

    def test_freeze_encoder_helper(self):
        enc = _fake_encoder()
        freeze_encoder(enc)
        for p in enc["model"].parameters():
            assert not p.requires_grad

    def test_unfreeze_encoder_helper(self):
        enc = _fake_encoder()
        freeze_encoder(enc)
        unfreeze_encoder(enc)
        for p in enc["model"].parameters():
            assert p.requires_grad


# ---------------------------------------------------------------------------
# encoder_info
# ---------------------------------------------------------------------------

class TestEncoderInfo:
    def test_returns_string(self):
        enc = _fake_encoder("image")
        info = encoder_info(enc)
        assert isinstance(info, str)

    def test_contains_modality(self):
        enc = _fake_encoder("audio")
        info = encoder_info(enc)
        assert "AUDIO" in info

    def test_reports_frozen_true(self):
        enc = _fake_encoder()
        freeze_encoder(enc)
        assert "frozen=True" in encoder_info(enc)

    def test_reports_frozen_false(self):
        enc = _fake_encoder()
        unfreeze_encoder(enc)
        assert "frozen=False" in encoder_info(enc)


# ---------------------------------------------------------------------------
# param_count
# ---------------------------------------------------------------------------

class TestParamCount:
    def test_param_count_format(self):
        m = _tiny_model()   # Linear(4,4): 4*4 + 4 = 20 params
        s = _param_count(m)
        assert s.endswith("M")
        assert float(s[:-1]) >= 0

    def test_none_returns_question_mark(self):
        assert _param_count(None) == "?"


# ---------------------------------------------------------------------------
# load_encoder error handling (no network)
# ---------------------------------------------------------------------------

class TestLoadEncoderErrors:
    def test_unknown_modality_raises(self):
        with pytest.raises(ValueError, match="Unknown modality"):
            load_encoder("lidar")
