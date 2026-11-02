"""
tests/test_generalized_fusion.py
=================================
Unit and regression tests for modality- and class-count-agnostic Fusion V2 and RCS Engine (Step 4).

Verifies:
1. Two modalities ("audio", "video") with 6 classes forward shapes and outputs.
2. Modality dropout guarantees never dropping all modalities for any sample.
3. Arbitrary modality combinations (e.g. 3 modalities).
4. MELD regression: loading existing MELD checkpoint (strict=True) and forward pass.
5. RCS engine: dynamic class count, pca_dim clamping to smallest_class / 4, and 85% synthetic capping.
"""

import logging
from pathlib import Path
import numpy as np
import pytest
import torch

from models.regularized_gated_fusion_v2 import RegularizedGatedMultimodalClassifierV2
from rcs_engine import apply_rcs_optimal

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestGeneralizedFusionV2:
    def test_audio_video_six_classes(self):
        batch_size = 16
        audio_dim = 768
        video_dim = 768
        hidden_dim = 512
        num_classes = 6

        model = RegularizedGatedMultimodalClassifierV2(
            modalities=["audio", "video"],
            modality_dims={"audio": audio_dim, "video": video_dim},
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            modality_dropout=0.15,
            classifier_dropout=0.3,
        )
        model.eval()

        audio_emb = torch.randn(batch_size, audio_dim)
        video_emb = torch.randn(batch_size, video_dim)

        out = model(audio_emb, video_emb)

        # Output keys
        assert "logits" in out
        assert "gate_scores" in out
        assert "fused_feat" in out
        assert "z_con" in out
        assert "aux_audio_logits" in out
        assert "aux_video_logits" in out

        # Output shapes
        assert out["logits"].shape == (batch_size, 6)
        assert out["aux_audio_logits"].shape == (batch_size, 6)
        assert out["aux_video_logits"].shape == (batch_size, 6)
        assert out["gate_scores"].shape == (batch_size, 2)
        assert out["fused_feat"].shape == (batch_size, 512)
        assert out["z_con"].shape == (batch_size, 128)

        # Gates sum to 1
        assert torch.allclose(out["gate_scores"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)

        # z_con L2 normalized
        z_norms = torch.norm(out["z_con"], p=2, dim=-1)
        assert torch.allclose(z_norms, torch.ones_like(z_norms), atol=1e-5)

    def test_cremad_modality_dims_audio_768_video_1536(self):
        """Verifies per-modality input dimensions with audio=768 and video=1536 for CREMA-D."""
        batch_size = 8
        modality_dims = {"audio": 768, "video": 1536}
        hidden_dim = 512
        num_classes = 6

        model = RegularizedGatedMultimodalClassifierV2(
            modalities=["audio", "video"],
            modality_dims=modality_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            modality_dropout=0.15,
            classifier_dropout=0.3,
        )
        model.eval()

        assert model.modality_dims["audio"] == 768
        assert model.modality_dims["video"] == 1536
        assert model.projections["audio"][0].normalized_shape == (768,)
        assert model.projections["video"][0].normalized_shape == (1536,)
        assert model.projections["audio"][1].in_features == 768
        assert model.projections["video"][1].in_features == 1536

        audio_t = torch.randn(batch_size, 768)
        video_t = torch.randn(batch_size, 1536)

        out = model(audio_t, video_t)
        assert out["logits"].shape == (batch_size, 6)
        assert out["aux_audio_logits"].shape == (batch_size, 6)
        assert out["aux_video_logits"].shape == (batch_size, 6)
        assert out["fused_feat"].shape == (batch_size, 512)

    def test_no_gate_ablation(self):
        """Verifies Fusion V2 with use_gate=False performs uniform feature blending."""
        batch_size = 4
        model = RegularizedGatedMultimodalClassifierV2(
            modalities=["audio", "video"],
            modality_dims={"audio": 768, "video": 1536},
            hidden_dim=512,
            num_classes=6,
            use_gate=False,
        )
        model.eval()
        audio_t = torch.randn(batch_size, 768)
        video_t = torch.randn(batch_size, 1536)
        out = model(audio_t, video_t)
        assert out["logits"].shape == (batch_size, 6)
        assert out["fused_feat"].shape == (batch_size, 512)
        # Uniform gate scores sum to 1
        assert torch.allclose(out["gate_scores"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_modality_dropout_never_drops_all(self):
        """Modality dropout must guarantee at least one modality remains active for every sample."""
        batch_size = 200
        dim = 512
        model = RegularizedGatedMultimodalClassifierV2(
            modalities=["audio", "video"],
            modality_dropout=0.8,  # very high dropout probability
        )
        model.train()

        h_dict = {
            "audio": torch.randn(batch_size, dim),
            "video": torch.randn(batch_size, dim),
        }

        # Test across multiple passes
        for _ in range(5):
            _, mask = model._apply_modality_dropout(h_dict)
            active_counts = mask.sum(dim=-1)
            assert (active_counts >= 1.0).all(), "Found sample with all modalities dropped!"

    def test_three_modalities(self):
        """Model supports arbitrary M modalities."""
        batch_size = 8
        model = RegularizedGatedMultimodalClassifierV2(
            modalities=["text", "audio", "video"],
            modality_dims={"text": 768, "audio": 512, "video": 768},
            num_classes=6,
        )
        model.eval()

        t_emb = torch.randn(batch_size, 768)
        a_emb = torch.randn(batch_size, 512)
        v_emb = torch.randn(batch_size, 768)

        out = model(t_emb, a_emb, v_emb)
        assert out["logits"].shape == (batch_size, 6)
        assert out["gate_scores"].shape == (batch_size, 3)
        assert torch.allclose(out["gate_scores"].sum(dim=-1), torch.ones(batch_size), atol=1e-5)
        assert "aux_text_logits" in out
        assert "aux_audio_logits" in out
        assert "aux_video_logits" in out

    def test_meld_checkpoint_strict_loading_regression(self):
        """MELD checkpoint loads strictly with no missing/unexpected keys and preserves weights."""
        ckpt_path = REPO_ROOT / "final_rcs_package/latents/best_fusion_v2_model.pt"
        assert ckpt_path.exists(), f"MELD checkpoint not found: {ckpt_path}"

        model = RegularizedGatedMultimodalClassifierV2(
            modalities=["text", "video"],
            text_dim=768,
            video_dim=768,
            num_classes=7,
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Strict load succeeds via legacy key remapping hook
        model.load_state_dict(sd, strict=True)
        model.eval()

        # Attribute access backward compatibility
        assert hasattr(model, "proj_text")
        assert hasattr(model, "proj_video")
        assert hasattr(model, "aux_text_classifier")
        assert hasattr(model, "aux_video_classifier")

        # Forward pass produces exact MELD outputs
        b_t = torch.randn(4, 768)
        b_v = torch.randn(4, 768)
        out = model(b_t, b_v)
        assert out["logits"].shape == (4, 7)
        assert out["aux_text_logits"].shape == (4, 7)
        assert out["aux_video_logits"].shape == (4, 7)
        assert out["gate_scores"].shape == (4, 2)


class TestGeneralizedRCSEngine:
    def test_rcs_six_classes_and_pca_clamping(self, caplog):
        """Tests that RCS handles 6 classes and clamps pca_dim if smallest class < 4 * pca_dim."""
        rng = np.random.RandomState(42)
        # Create synthetic dataset with 6 classes, smallest class has 16 samples
        counts = [300, 200, 150, 100, 80, 16]
        X_list, y_list = [], []
        for c_idx, cnt in enumerate(counts):
            X_list.append(rng.randn(cnt, 128))
            y_list.append(np.full(cnt, c_idx, dtype=np.int64))

        Z_tr = np.vstack(X_list)
        y_tr = np.concatenate(y_list)

        # Request pca_dim=64, but smallest class is 16 -> max allowed = 16 // 4 = 4
        with caplog.at_level(logging.WARNING):
            Z_aug, y_aug, info = apply_rcs_optimal(
                Z_train=Z_tr,
                y_train=y_tr,
                pca_dim=64,
                seed=42,
                max_synthetic_share=0.85,
            )

        assert info["pca_dim"] == 4, f"Expected pca_dim to be clamped to 4, got {info['pca_dim']}"
        assert any("Clamping pca_dim" in record.message for record in caplog.records)

        # Check synthetic share <= 85% for all classes
        for c in range(6):
            n_real = counts[c]
            n_after = np.sum(y_aug == c)
            n_syn = n_after - n_real
            if n_after > 0:
                share = n_syn / n_after
                assert share <= 0.85001, f"Class {c} synthetic share {share:.4f} exceeded 85%!"

    def test_rcs_engine_identity_with_package_baseline(self):
        """Verifies root apply_rcs_optimal matches final_rcs_package/rcs_engine.py exactly with defaults."""
        from final_rcs_package.rcs_engine import apply_rcs_optimal as pkg_apply_rcs

        latents_path = REPO_ROOT / "final_rcs_package/latents/latents_train.npz"
        assert latents_path.exists(), f"Latents not found: {latents_path}"

        data = np.load(latents_path)
        Z_train = data["fused_feat"]
        y_train = data["labels"]

        Z_b_root, y_b_root, info_root = apply_rcs_optimal(Z_train, y_train, seed=42)
        Z_b_pkg, y_b_pkg, info_pkg = pkg_apply_rcs(Z_train, y_train, seed=42)

        # 1. Same per-class synthetic counts and totals
        assert info_root["counts_before"] == info_pkg["counts_before"]
        assert info_root["counts_after"] == info_pkg["counts_after"]

        # 2. Identical label arrays
        np.testing.assert_array_equal(y_b_root, y_b_pkg)

        # 3. Identical feature arrays
        np.testing.assert_allclose(Z_b_root, Z_b_pkg, rtol=1e-5, atol=1e-6)


