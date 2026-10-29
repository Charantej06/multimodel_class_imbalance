"""
models/regularized_gated_fusion.py
==================================
Regularized Gated Multimodal Fusion with:
  1. LayerNorm + Linear + tanh projection per modality
  2. Modality Dropout during training (anti-collapse mechanism)
  3. Shared Gate Scorer (Softmax dynamic gating weights)
  4. Auxiliary Unimodal Classifier Heads (for multi-task supervision)
  5. InfoNCE Contrastive Loss (text-video cross-modal alignment)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularizedGatedMultimodalClassifier(nn.Module):
    """
    Multimodal classification model with regularized gated fusion.

    Parameters
    ----------
    text_dim : int
        Dimension of incoming text embedding (default: 768 for ModernBERT).
    video_dim : int
        Dimension of incoming video embedding (default: 768 for VideoMAE).
    hidden_dim : int
        Projected common representation dimension (default: 512).
    num_classes : int
        Number of target emotion classes (default: 7 for MELD).
    modality_dropout : float
        Probability of dropping one of the modalities during training (default: 0.15).
    classifier_dropout : float
        Dropout probability in classification heads (default: 0.3).
    """

    def __init__(
        self,
        text_dim: int = 768,
        video_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 7,
        modality_dropout: float = 0.15,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.modality_dropout = modality_dropout

        # 1. LayerNorm + Linear + tanh projection per modality
        self.proj_text = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.Tanh(),
        )

        self.proj_video = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, hidden_dim),
            nn.Tanh(),
        )

        # 2. Contrastive projection heads (L2-normalized for InfoNCE)
        contrastive_dim = hidden_dim // 2
        self.contra_proj_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, contrastive_dim),
        )
        self.contra_proj_video = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, contrastive_dim),
        )

        # 3. Shared Gate Scorer: takes concatenated [h_t; h_v] -> 2 dynamic weights (sum=1)
        self.gate_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

        # 3. Main fused classifier head
        self.fused_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # 4. Auxiliary unimodal classifier heads (direct gradient flow for tail classes)
        self.aux_text_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.aux_video_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _apply_modality_dropout(
        self, h_text: torch.Tensor, h_video: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply modality dropout during training.
        Randomly zeroes out either text or video with probability `modality_dropout`.
        Ensures both modalities are never dropped simultaneously.
        """
        if not self.training or self.modality_dropout <= 0.0:
            return h_text, h_video, torch.ones(h_text.size(0), 2, device=h_text.device)

        batch_size = h_text.size(0)
        # Sample masks: 0 = drop text, 1 = drop video, 2 = keep both
        rand_vals = torch.rand(batch_size, device=h_text.device)
        p = self.modality_dropout

        mask_text = torch.ones(batch_size, 1, device=h_text.device)
        mask_video = torch.ones(batch_size, 1, device=h_text.device)

        # Drop text only when rand < p
        drop_text = rand_vals < p
        mask_text[drop_text] = 0.0

        # Drop video only when p <= rand < 2*p
        drop_video = (rand_vals >= p) & (rand_vals < 2 * p)
        mask_video[drop_video] = 0.0

        h_text_dropped = h_text * mask_text
        h_video_dropped = h_video * mask_video
        modality_mask = torch.cat([mask_text, mask_video], dim=-1)

        return h_text_dropped, h_video_dropped, modality_mask

    def forward(
        self, text_emb: torch.Tensor, video_emb: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        text_emb : Tensor of shape (B, 768)
        video_emb : Tensor of shape (B, 768)

        Returns
        -------
        dict with:
            "logits": Main fused class logits (B, num_classes)
            "aux_text_logits": Auxiliary text-only logits (B, num_classes)
            "aux_video_logits": Auxiliary video-only logits (B, num_classes)
            "gate_scores": Gate importance weights (B, 2) [text_weight, video_weight]
            "fused_feat": Fused representation before classifier (B, hidden_dim)
            "z_text": L2-normalized contrastive projection of text (B, contrastive_dim)
            "z_video": L2-normalized contrastive projection of video (B, contrastive_dim)
        """
        # Step 1: Modality Projection
        h_text = self.proj_text(text_emb)    # (B, hidden_dim)
        h_video = self.proj_video(video_emb) # (B, hidden_dim)

        # Step 2: Contrastive Projections (L2-normalized for InfoNCE)
        z_text = F.normalize(self.contra_proj_text(h_text), dim=-1)   # (B, D/2)
        z_video = F.normalize(self.contra_proj_video(h_video), dim=-1) # (B, D/2)

        # Step 3: Unimodal Auxiliary Predictions (computed prior to dropout)
        aux_text_logits = self.aux_text_classifier(h_text)
        aux_video_logits = self.aux_video_classifier(h_video)

        # Step 4: Modality Dropout
        h_text_d, h_video_d, _ = self._apply_modality_dropout(h_text, h_video)

        # Step 5: Shared Gate Scorer
        gate_input = torch.cat([h_text_d, h_video_d], dim=-1) # (B, hidden_dim * 2)
        gate_scores = self.gate_scorer(gate_input)            # (B, 2)
        s_text = gate_scores[:, 0:1]                           # (B, 1)
        s_video = gate_scores[:, 1:2]                          # (B, 1)

        # Step 6: Feature Blending & Final Classification
        h_fused = (s_text * h_text_d) + (s_video * h_video_d) # (B, hidden_dim)
        fused_logits = self.fused_classifier(h_fused)         # (B, num_classes)

        return {
            "logits": fused_logits,
            "aux_text_logits": aux_text_logits,
            "aux_video_logits": aux_video_logits,
            "gate_scores": gate_scores,
            "fused_feat": h_fused,
            "z_text": z_text,
            "z_video": z_video,
        }


def info_nce_loss(
    z_text: torch.Tensor,
    z_video: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE / NT-Xent Contrastive Loss for cross-modal alignment.

    Each sample i has a positive pair (z_text[i], z_video[i]).
    All other j != i in the batch are treated as negatives.

    Loss is computed symmetrically:
        L = 0.5 * (L_text_to_video + L_video_to_text)

    Parameters
    ----------
    z_text  : L2-normalized text projections,  shape (B, D)
    z_video : L2-normalized video projections, shape (B, D)
    temperature : InfoNCE temperature scalar (default: 0.07)

    Returns
    -------
    Scalar contrastive loss.
    """
    B = z_text.size(0)
    # Similarity matrix: (B, B) where entry [i,j] = cosine_sim(text_i, video_j)
    logits = torch.matmul(z_text, z_video.t()) / temperature  # (B, B)
    # Diagonal entries are positive pairs
    labels = torch.arange(B, device=z_text.device)
    # Symmetric: text->video and video->text
    loss_t2v = F.cross_entropy(logits, labels)
    loss_v2t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_t2v + loss_v2t)


def compute_multimodal_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    criterion: nn.Module,
    alpha_text: float = 0.3,
    beta_video: float = 0.3,
    gamma_contrastive: float = 0.5,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes joint loss:
        L_total = L_CE(fused) + alpha * L_CE(text) + beta * L_CE(video)
                + gamma * L_InfoNCE(z_text, z_video)
    """
    loss_fused = criterion(outputs["logits"], targets)
    loss_text = criterion(outputs["aux_text_logits"], targets)
    loss_video = criterion(outputs["aux_video_logits"], targets)
    loss_contra = info_nce_loss(
        outputs["z_text"], outputs["z_video"], temperature=temperature
    )

    total_loss = (
        loss_fused
        + (alpha_text * loss_text)
        + (beta_video * loss_video)
        + (gamma_contrastive * loss_contra)
    )

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_fused": float(loss_fused.item()),
        "loss_text": float(loss_text.item()),
        "loss_video": float(loss_video.item()),
        "loss_contrastive": float(loss_contra.item()),
    }
    return total_loss, metrics
