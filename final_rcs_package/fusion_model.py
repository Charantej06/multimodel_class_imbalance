"""
models/regularized_gated_fusion_v2.py
=====================================
Fusion V2 Architecture:
  1. Gate masking: Gate scorer outputs LOGITS (nn.Softmax removed).
     Modality mask (B, 2) sets logits of dropped modalities to -inf before softmax,
     guaranteeing exact renormalization over available modalities.
  2. h_fused = sum_m weight_m * h_m (dropped modality contributes exactly 0).
  3. Outputs: 'logits', 'aux_text_logits', 'aux_video_logits', 'gate_scores', 'fused_feat', 'z_con'.
  4. Class-aware contrastive projection head on fused_feat:
     Linear(hidden, hidden) -> ReLU -> Linear(hidden, 128), L2-normalized -> 'z_con'.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularizedGatedMultimodalClassifierV2(nn.Module):
    """
    Multimodal classification model with Fusion V2 regularized gated fusion.

    Parameters
    ----------
    text_dim : int
        Dimension of incoming text embedding (default: 768).
    video_dim : int
        Dimension of incoming video embedding (default: 768).
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

        # 2. Gate Scorer: takes concatenated [h_t; h_v] -> LOGITS (B, 2)
        # Note: Softmax removed so dropped modalities can be masked with -inf
        self.gate_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, 2),
        )

        # 3. Main fused classifier head
        self.fused_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # 4. Auxiliary unimodal classifier heads
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

        # 5. Class-aware contrastive projection head on fused_feat
        # Linear(hidden, hidden) -> ReLU -> Linear(hidden, 128), L2-normalized
        self.con_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )

    def _apply_modality_dropout(
        self,
        h_text: torch.Tensor,
        h_video: torch.Tensor,
        forced_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply modality dropout during training or when forced_mask is provided.
        Ensures both modalities are never dropped simultaneously.
        Returns:
            h_text_dropped, h_video_dropped, modality_mask (B, 2)
            where modality_mask has 1.0 for kept modality, 0.0 for dropped modality.
        """
        batch_size = h_text.size(0)
        device = h_text.device

        if forced_mask is not None:
            mask_text = forced_mask[:, 0:1]
            mask_video = forced_mask[:, 1:2]
            return h_text * mask_text, h_video * mask_video, forced_mask

        if not self.training or self.modality_dropout <= 0.0:
            return h_text, h_video, torch.ones(batch_size, 2, device=device)

        rand_vals = torch.rand(batch_size, device=device)
        p = self.modality_dropout

        mask_text = torch.ones(batch_size, 1, device=device)
        mask_video = torch.ones(batch_size, 1, device=device)

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
        self,
        text_emb: torch.Tensor,
        video_emb: torch.Tensor,
        forced_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        text_emb : (B, 768)
        video_emb : (B, 768)
        forced_mask : Optional (B, 2) tensor for unit testing or controlled masking.

        Returns
        -------
        dict with:
            'logits': (B, num_classes)
            'aux_text_logits': (B, num_classes)
            'aux_video_logits': (B, num_classes)
            'gate_scores': (B, 2) renormalized gate weights
            'fused_feat': (B, hidden_dim)
            'z_con': (B, 128) L2-normalized contrastive projection
        """
        # Step 1: Modality Projection
        h_text = self.proj_text(text_emb)    # (B, hidden_dim)
        h_video = self.proj_video(video_emb) # (B, hidden_dim)

        # Step 2: Auxiliary Unimodal Predictions (prior to dropout)
        aux_text_logits = self.aux_text_classifier(h_text)
        aux_video_logits = self.aux_video_classifier(h_video)

        # Step 3: Modality Dropout
        h_text_d, h_video_d, modality_mask = self._apply_modality_dropout(
            h_text, h_video, forced_mask=forced_mask
        )

        # Step 4: Shared Gate Scorer (outputs raw logits)
        gate_input = torch.cat([h_text_d, h_video_d], dim=-1) # (B, hidden_dim * 2)
        gate_logits = self.gate_scorer(gate_input)            # (B, 2)

        # Step 5: Gate Masking & Renormalization
        # Set logits of dropped modalities to -inf before softmax
        masked_gate_logits = gate_logits.masked_fill(modality_mask == 0.0, float("-inf"))
        gate_scores = F.softmax(masked_gate_logits, dim=-1)    # (B, 2)
        s_text = gate_scores[:, 0:1]                           # (B, 1)
        s_video = gate_scores[:, 1:2]                          # (B, 1)

        # Step 6: Dynamic Feature Blending
        # Dropped modality contributes exactly 0
        h_fused = (s_text * h_text_d) + (s_video * h_video_d) # (B, hidden_dim)
        fused_logits = self.fused_classifier(h_fused)         # (B, num_classes)

        # Step 7: Class-aware contrastive projection (L2 normalized)
        z_con = F.normalize(self.con_proj(h_fused), p=2, dim=-1) # (B, 128)

        return {
            "logits": fused_logits,
            "aux_text_logits": aux_text_logits,
            "aux_video_logits": aux_video_logits,
            "gate_scores": gate_scores,
            "fused_feat": h_fused,
            "z_con": z_con,
        }
