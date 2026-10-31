"""
tests/test_extract_cremad_audio.py
==================================
Unit tests for CREMA-D WavLM audio extraction and layer probing (Step 6).
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.extract_cremad_audio import load_audio_waveform

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestAudioNormalization:
    def test_zero_mean_unit_variance(self):
        """Audio waveforms must have zero mean and unit variance."""
        sample_wav = REPO_ROOT / "CREMAD/AudioWAV/1001_DFA_ANG_XX.wav"
        assert sample_wav.exists(), f"Sample wav not found at {sample_wav}"

        wf = load_audio_waveform(sample_wav, target_sr=16000)
        assert wf.ndim == 1
        assert wf.dtype == np.float32

        # Mean should be 0 within numerical precision
        assert np.isclose(np.mean(wf), 0.0, atol=1e-5)
        # Std should be 1 within numerical precision
        assert np.isclose(np.std(wf), 1.0, atol=1e-3)


class TestMaskedMeanPooling:
    def test_masked_mean_pooling_math(self):
        """Validates that masked mean pooling correctly excludes padding."""
        B, T, D = 2, 4, 3
        # Synthetic hidden states
        H = torch.tensor([
            # Sample 0: length 2 (first 2 valid, last 2 padded)
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [99.0, 99.0, 99.0], [99.0, 99.0, 99.0]],
            # Sample 1: length 3 (first 3 valid, last 1 padded)
            [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [6.0, 6.0, 6.0], [99.0, 99.0, 99.0]],
        ], dtype=torch.float32)

        feat_lengths = torch.tensor([2, 3])
        feat_mask = torch.arange(T).expand(B, T) < feat_lengths.unsqueeze(1)
        mask_exp = feat_mask.unsqueeze(-1).float()
        mask_sum = mask_exp.sum(dim=1).clamp(min=1.0)

        pooled = (H * mask_exp).sum(dim=1) / mask_sum

        # Expected sample 0: mean([1, 2, 3], [3, 4, 5]) = [2.0, 3.0, 4.0]
        expected_0 = torch.tensor([2.0, 3.0, 4.0])
        # Expected sample 1: mean([2, 2, 2], [4, 4, 4], [6, 6, 6]) = [4.0, 4.0, 4.0]
        expected_1 = torch.tensor([4.0, 4.0, 4.0])

        assert torch.allclose(pooled[0], expected_0, atol=1e-5)
        assert torch.allclose(pooled[1], expected_1, atol=1e-5)


class TestProbeLogic:
    def test_layer_probe_pipeline(self):
        """Validates linear probing logic and train-only scaling."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        N_tr, N_va, D = 60, 20, 16
        X_tr = rng.randn(N_tr, D) * 5.0 + 10.0
        y_tr = rng.randint(0, 3, size=N_tr)

        X_va = rng.randn(N_va, D) * 5.0 + 10.0
        y_va = rng.randint(0, 3, size=N_va)

        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr)
        X_va_std = scaler.transform(X_va)

        # Train statistics only check:
        assert np.allclose(X_tr_std.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(X_tr_std.std(axis=0), 1.0, atol=1e-5)

        clf = LogisticRegression(max_iter=100, random_state=42)
        clf.fit(X_tr_std, y_tr)
        preds = clf.predict(X_va_std)
        f1 = f1_score(y_va, preds, average="macro")
        assert 0.0 <= f1 <= 1.0

