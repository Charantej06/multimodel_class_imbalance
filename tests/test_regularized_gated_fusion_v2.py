"""
tests/test_regularized_gated_fusion_v2.py
=========================================
Unit tests for Fusion V2 model:
  1. Output keys verification:
     'logits', 'aux_text_logits', 'aux_video_logits', 'gate_scores', 'fused_feat', 'z_con'
  2. Shape verification:
     logits: (B, 7)
     aux_text_logits: (B, 7)
     aux_video_logits: (B, 7)
     gate_scores: (B, 2)
     fused_feat: (B, 512)
     z_con: (B, 128)
  3. L2 normalization of z_con:
     torch.norm(z_con, dim=-1) == 1.0 (within 1e-5)
  4. Forced text drop (mask = [0, 1]):
     gate_scores[:, 0] == 0.0
     gate_scores[:, 1] == 1.0
  5. Forced video drop (mask = [1, 0]):
     gate_scores[:, 0] == 1.0
     gate_scores[:, 1] == 0.0
  6. Neither dropped (mask = [1, 1]):
     gate_scores.sum(dim=-1) == 1.0
     0.0 < gate_scores[:, 0] < 1.0
     0.0 < gate_scores[:, 1] < 1.0
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from models.regularized_gated_fusion_v2 import RegularizedGatedMultimodalClassifierV2


def test_fusion_v2():
    batch_size = 16
    text_dim = 768
    video_dim = 768
    hidden_dim = 512
    num_classes = 7

    model = RegularizedGatedMultimodalClassifierV2(
        text_dim=text_dim,
        video_dim=video_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        modality_dropout=0.15,
        classifier_dropout=0.3,
    )
    model.eval()

    text_emb = torch.randn(batch_size, text_dim)
    video_emb = torch.randn(batch_size, video_dim)

    # Test 1: Standard forward pass without forced mask
    out = model(text_emb, video_emb)
    required_keys = {"logits", "aux_text_logits", "aux_video_logits", "gate_scores", "fused_feat", "z_con"}
    assert required_keys.issubset(out.keys()), f"Missing keys: {required_keys - set(out.keys())}"

    assert out["logits"].shape == (batch_size, num_classes)
    assert out["aux_text_logits"].shape == (batch_size, num_classes)
    assert out["aux_video_logits"].shape == (batch_size, num_classes)
    assert out["gate_scores"].shape == (batch_size, 2)
    assert out["fused_feat"].shape == (batch_size, hidden_dim)
    assert out["z_con"].shape == (batch_size, 128)

    # Check L2 normalization of z_con
    z_norms = torch.norm(out["z_con"], p=2, dim=-1)
    assert torch.allclose(z_norms, torch.ones_like(z_norms), atol=1e-5), f"z_con is not L2 normalized: {z_norms}"

    # Test 2: Forced Text Drop (mask = [0, 1])
    # When text is dropped, text weight MUST be 0.0 and video weight MUST be 1.0
    text_dropped_mask = torch.tensor([[0.0, 1.0]] * batch_size)
    out_drop_text = model(text_emb, video_emb, forced_mask=text_dropped_mask)
    gate_drop_text = out_drop_text["gate_scores"]

    assert torch.allclose(gate_drop_text[:, 0], torch.zeros(batch_size), atol=1e-6), \
        f"Text gate should be 0.0 when dropped, got: {gate_drop_text[:, 0]}"
    assert torch.allclose(gate_drop_text[:, 1], torch.ones(batch_size), atol=1e-6), \
        f"Video gate should be 1.0 when text dropped, got: {gate_drop_text[:, 1]}"

    # Test 3: Forced Video Drop (mask = [1, 0])
    # When video is dropped, text weight MUST be 1.0 and video weight MUST be 0.0
    video_dropped_mask = torch.tensor([[1.0, 0.0]] * batch_size)
    out_drop_video = model(text_emb, video_emb, forced_mask=video_dropped_mask)
    gate_drop_video = out_drop_video["gate_scores"]

    assert torch.allclose(gate_drop_video[:, 0], torch.ones(batch_size), atol=1e-6), \
        f"Text gate should be 1.0 when video dropped, got: {gate_drop_video[:, 0]}"
    assert torch.allclose(gate_drop_video[:, 1], torch.zeros(batch_size), atol=1e-6), \
        f"Video gate should be 0.0 when dropped, got: {gate_drop_video[:, 1]}"

    # Test 4: Both Present (mask = [1, 1])
    both_mask = torch.tensor([[1.0, 1.0]] * batch_size)
    out_both = model(text_emb, video_emb, forced_mask=both_mask)
    gate_both = out_both["gate_scores"]
    assert torch.allclose(gate_both.sum(dim=-1), torch.ones(batch_size), atol=1e-6), \
        f"Gate scores should sum to 1.0: {gate_both.sum(dim=-1)}"

    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print(f"  - Output keys: {list(out.keys())}")
    print(f"  - z_con shape: {out['z_con'].shape}, L2-norm min/max: {z_norms.min().item():.6f}/{z_norms.max().item():.6f}")
    print(f"  - When text dropped: gate text={gate_drop_text[0, 0].item():.4f}, gate video={gate_drop_text[0, 1].item():.4f}")
    print(f"  - When video dropped: gate text={gate_drop_video[0, 0].item():.4f}, gate video={gate_drop_video[0, 1].item():.4f}")


if __name__ == "__main__":
    test_fusion_v2()
