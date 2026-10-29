import pytest
import torch
from models.regularized_gated_fusion import (
    RegularizedGatedMultimodalClassifier,
    compute_multimodal_loss,
)

def test_model_forward_shapes():
    B = 16
    text_dim = 768
    video_dim = 768
    hidden_dim = 512
    num_classes = 7

    model = RegularizedGatedMultimodalClassifier(
        text_dim=text_dim,
        video_dim=video_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        modality_dropout=0.2,
    )

    t_emb = torch.randn(B, text_dim)
    v_emb = torch.randn(B, video_dim)

    # Train mode
    model.train()
    out = model(t_emb, v_emb)
    assert out["logits"].shape == (B, num_classes)
    assert out["aux_text_logits"].shape == (B, num_classes)
    assert out["aux_video_logits"].shape == (B, num_classes)
    assert out["gate_scores"].shape == (B, 2)
    assert torch.allclose(out["gate_scores"].sum(dim=-1), torch.ones(B), atol=1e-5)
    assert out["fused_feat"].shape == (B, hidden_dim)

    # Eval mode
    model.eval()
    out_eval = model(t_emb, v_emb)
    assert out_eval["logits"].shape == (B, num_classes)
    assert out_eval["gate_scores"].shape == (B, 2)


def test_loss_computation():
    B = 8
    num_classes = 7
    model = RegularizedGatedMultimodalClassifier(num_classes=num_classes)
    t_emb = torch.randn(B, 768)
    v_emb = torch.randn(B, 768)
    targets = torch.randint(0, num_classes, (B,))

    criterion = torch.nn.CrossEntropyLoss()
    outputs = model(t_emb, v_emb)

    total_loss, metrics = compute_multimodal_loss(
        outputs, targets, criterion, alpha_text=0.3, beta_video=0.3
    )

    assert total_loss.ndim == 0
    assert total_loss.item() > 0.0
    assert "loss_total" in metrics
    assert "loss_fused" in metrics
    assert "loss_text" in metrics
    assert "loss_video" in metrics

    # Test backpropagation
    total_loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
