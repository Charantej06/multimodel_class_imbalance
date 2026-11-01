"""
final_rcs_package/fusion_model_food_101.py
===========================================
Fusion V2 Architecture configured for Food-101 (101 classes):
  1. Gated fusion over ModernBERT (text titles) and DINOv2 (images) (both 768-d).
  2. Modality dropout (p=0.15) ensuring robustness against missing modalities.
  3. Class-aware contrastive projection head on fused_feat (512 -> 128, L2-normalized).
  4. Classification heads configured for 101 fine-grained culinary categories.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularizedGatedMultimodalClassifierFood101(nn.Module):
    """
    Multimodal classification model with Fusion V2 regularized gated fusion
    specifically tailored for Food-101 (101 classes).
    """

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 101,
        modality_dropout: float = 0.15,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.modality_dropout = modality_dropout

        # 1. LayerNorm + Linear + tanh projection per modality
        self.proj_text = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.Tanh(),
        )

        self.proj_image = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.Tanh(),
        )

        # 2. Gate Scorer: takes concatenated [h_t; h_img] -> LOGITS (B, 2)
        self.gate_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, 2),
        )

        # 3. Main fused classifier head (101 classes)
        self.fused_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # 4. Auxiliary unimodal classifier heads (101 classes)
        self.aux_text_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.aux_image_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # 5. Class-aware contrastive projection head on fused_feat
        self.con_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )

    def _apply_modality_dropout(
        self,
        h_text: torch.Tensor,
        h_image: torch.Tensor,
        forced_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = h_text.size(0)
        device = h_text.device

        if forced_mask is not None:
            mask_text = forced_mask[:, 0:1]
            mask_image = forced_mask[:, 1:2]
            return h_text * mask_text, h_image * mask_image, forced_mask

        if not self.training or self.modality_dropout <= 0.0:
            return h_text, h_image, torch.ones(batch_size, 2, device=device)

        rand_vals = torch.rand(batch_size, device=device)
        p = self.modality_dropout

        mask_text = torch.ones(batch_size, 1, device=device)
        mask_image = torch.ones(batch_size, 1, device=device)

        # Drop text only when rand < p
        drop_text = rand_vals < p
        mask_text[drop_text] = 0.0

        # Drop image only when p <= rand < 2*p
        drop_image = (rand_vals >= p) & (rand_vals < 2 * p)
        mask_image[drop_image] = 0.0

        h_text_dropped = h_text * mask_text
        h_image_dropped = h_image * mask_image
        modality_mask = torch.cat([mask_text, mask_image], dim=-1)

        return h_text_dropped, h_image_dropped, modality_mask

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        forced_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h_text = self.proj_text(text_emb)
        h_image = self.proj_image(image_emb)

        h_t_drop, h_img_drop, mod_mask = self._apply_modality_dropout(h_text, h_image, forced_mask)

        # Gate logits (B, 2)
        h_cat = torch.cat([h_text, h_image], dim=-1)
        gate_logits = self.gate_scorer(h_cat)

        # Mask dropped modality with -inf before softmax
        inf_mask = (1.0 - mod_mask) * -1e9
        gate_logits_masked = gate_logits + inf_mask
        gate_scores = F.softmax(gate_logits_masked, dim=-1)

        w_text = gate_scores[:, 0:1]
        w_image = gate_scores[:, 1:2]

        # Gated combination
        fused_feat = w_text * h_t_drop + w_image * h_img_drop

        # Classifiers (101 classes)
        logits = self.fused_classifier(fused_feat)
        aux_text_logits = self.aux_text_classifier(h_text)
        aux_image_logits = self.aux_image_classifier(h_image)

        # Contrastive projection
        z_con = F.normalize(self.con_proj(fused_feat), p=2, dim=-1)

        return {
            "logits": logits,
            "aux_text_logits": aux_text_logits,
            "aux_image_logits": aux_image_logits,
            "gate_scores": gate_scores,
            "fused_feat": fused_feat,
            "z_con": z_con,
        }

    def extract_fused_features(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Extracts the 512-d fused multimodal representation."""
        out = self.forward(text_emb, image_emb)
        return out["fused_feat"]

