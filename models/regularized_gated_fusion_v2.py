
"""
models/regularized_gated_fusion_v2.py
=====================================
Modality- and class-count-agnostic Regularized Gated Multimodal Fusion (Fusion V2):
  1. Arbitrary modalities list (default: ["text", "video"] for MELD backward compatibility)
     with per-modality input dimensions.
  2. Modality projection: LayerNorm + Linear + tanh per modality.
  3. Modality dropout: Anti-collapse mechanism with guaranteed non-empty active set
     (all modalities are NEVER dropped simultaneously for any sample).
  4. Shared Gate Scorer: Takes concatenated representations -> raw logits, masked with -inf
     for dropped modalities before softmax renormalization.
  5. Dynamic Feature Blending: Fused representation h_fused = sum_m (gate_m * h_m_dropped).
  6. Auxiliary Unimodal Classifiers: Separate classifier per modality for direct supervision.
  7. Class-Aware Contrastive Projection Head: Linear(hidden, hidden) -> ReLU -> Linear(hidden, 128)
     with L2 normalization for Decoupled Supervised Contrastive Loss.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularizedGatedMultimodalClassifierV2(nn.Module):
    """
    Generalized multimodal classification model with Fusion V2 regularized gated fusion.

    Parameters
    ----------
    text_dim : int
        Dimension of incoming text embedding (default: 768 for backward compatibility).
    video_dim : int
        Dimension of incoming video embedding (default: 768 for backward compatibility).
    hidden_dim : int
        Projected common representation dimension (default: 512).
    num_classes : int
        Number of target emotion classes (default: 7 for MELD).
    modality_dropout : float
        Probability of dropping one of the modalities during training (default: 0.15).
    classifier_dropout : float
        Dropout probability in classification heads (default: 0.3).
    modalities : Optional[List[str]]
        List of modality names (default: ["text", "video"]).
    modality_dims : Optional[Dict[str, int]]
        Mapping of modality name to embedding dimension.
    """

    def __init__(
        self,
        text_dim: int = 768,
        video_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 7,
        modality_dropout: float = 0.15,
        classifier_dropout: float = 0.3,
        modalities: Optional[List[str]] = None,
        modality_dims: Optional[Dict[str, int]] = None,
        use_gate: bool = True,
    ):
        super().__init__()
        if modalities is None:
            self.modalities = ["text", "video"]
        else:
            self.modalities = list(modalities)

        self.num_modalities = len(self.modalities)
        if self.num_modalities < 1:
            raise ValueError("Must provide at least 1 modality.")

        self.use_gate = use_gate

        # Resolve modality dimensions
        self.modality_dims: Dict[str, int] = {}
        for m in self.modalities:
            if modality_dims and m in modality_dims:
                self.modality_dims[m] = modality_dims[m]
            elif m == "text":
                self.modality_dims[m] = text_dim
            elif m == "video":
                self.modality_dims[m] = video_dim
            elif m == "audio":
                self.modality_dims[m] = 768
            else:
                self.modality_dims[m] = 768

        self.text_dim = self.modality_dims.get("text", text_dim)
        self.video_dim = self.modality_dims.get("video", video_dim)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.modality_dropout = modality_dropout
        self.classifier_dropout = classifier_dropout

        # 1. LayerNorm + Linear + tanh projection per modality
        self.projections = nn.ModuleDict({
            m: nn.Sequential(
                nn.LayerNorm(self.modality_dims[m]),
                nn.Linear(self.modality_dims[m], hidden_dim),
                nn.Tanh(),
            )
            for m in self.modalities
        })

        # 2. Gate Scorer: takes concatenated [h_1; ...; h_M] -> LOGITS (B, M)
        self.gate_scorer = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, self.num_modalities),
        )

        # 3. Main fused classifier head
        self.fused_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # 4. Auxiliary unimodal classifier heads per modality
        self.aux_classifiers = nn.ModuleDict({
            m: nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
            for m in self.modalities
        })

        # 5. Class-aware contrastive projection head on fused_feat
        self.con_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )

    def _apply_modality_dropout(
        self,
        h_dict: Dict[str, torch.Tensor],
        forced_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply modality dropout during training or when forced_mask is provided.
        Guarantees that all modalities are NEVER dropped simultaneously for any sample.
        """
        first_h = next(iter(h_dict.values()))
        batch_size = first_h.size(0)
        device = first_h.device
        M = self.num_modalities

        if forced_mask is not None:
            mask = forced_mask.to(device)
            dropped = {
                m: h_dict[m] * mask[:, idx : idx + 1]
                for idx, m in enumerate(self.modalities)
            }
            return dropped, mask

        if not self.training or self.modality_dropout <= 0.0 or M <= 1:
            mask = torch.ones(batch_size, M, device=device)
            return h_dict, mask

        # For M == 2, preserve exact original behavior for reproducibility:
        if M == 2:
            rand_vals = torch.rand(batch_size, device=device)
            p = self.modality_dropout
            mask = torch.ones(batch_size, 2, device=device)
            # Drop mod 0 when rand < p
            drop_0 = rand_vals < p
            mask[drop_0, 0] = 0.0
            # Drop mod 1 when p <= rand < 2*p
            drop_1 = (rand_vals >= p) & (rand_vals < 2 * p)
            mask[drop_1, 1] = 0.0
        else:
            rand_vals = torch.rand(batch_size, M, device=device)
            mask = (rand_vals >= self.modality_dropout).float()
            # Guarantee that at least one modality is active per sample
            all_dropped = (mask.sum(dim=-1) == 0)
            if all_dropped.any():
                kept_idx = torch.randint(0, M, (int(all_dropped.sum().item()),), device=device)
                mask[all_dropped, kept_idx] = 1.0

        dropped = {
            m: h_dict[m] * mask[:, idx : idx + 1]
            for idx, m in enumerate(self.modalities)
        }
        return dropped, mask

    def forward(
        self,
        *args,
        forced_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass. Supports positional embeddings matching self.modalities order,
        keyword arguments (e.g. text_emb, video_emb, audio_emb), or a dictionary of tensors.
        """
        input_dict: Dict[str, torch.Tensor] = {}
        if len(args) == 1 and isinstance(args[0], dict):
            input_dict = dict(args[0])
        elif len(args) > 0:
            for m, t in zip(self.modalities, args):
                input_dict[m] = t
        for m in self.modalities:
            if m in kwargs:
                input_dict[m] = kwargs[m]
            elif f"{m}_emb" in kwargs:
                input_dict[m] = kwargs[f"{m}_emb"]

        # Validate that all required modalities are present
        for m in self.modalities:
            if m not in input_dict or input_dict[m] is None:
                raise ValueError(f"Missing required embedding for modality '{m}'")

        # Step 1: Modality Projection
        h_dict = {m: self.projections[m](input_dict[m]) for m in self.modalities}

        # Step 2: Auxiliary Unimodal Predictions (prior to dropout)
        aux_dict = {m: self.aux_classifiers[m](h_dict[m]) for m in self.modalities}

        # Step 3: Modality Dropout
        h_dropped, modality_mask = self._apply_modality_dropout(h_dict, forced_mask=forced_mask)

        if self.use_gate:
            # Step 4: Shared Gate Scorer
            gate_input = torch.cat([h_dropped[m] for m in self.modalities], dim=-1)
            gate_logits = self.gate_scorer(gate_input)

            # Step 5: Gate Masking & Renormalization
            masked_gate_logits = gate_logits.masked_fill(modality_mask == 0.0, float("-inf"))
            gate_scores = F.softmax(masked_gate_logits, dim=-1)

            # Step 6: Dynamic Feature Blending
            h_fused = torch.zeros_like(next(iter(h_dict.values())))
            for idx, m in enumerate(self.modalities):
                s_m = gate_scores[:, idx : idx + 1]
                h_fused = h_fused + (s_m * h_dropped[m])
        else:
            # No-gate ablation: uniform average blending of active modalities
            mask_sum = modality_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            gate_scores = modality_mask / mask_sum
            h_sum = torch.zeros_like(next(iter(h_dict.values())))
            for m in self.modalities:
                h_sum = h_sum + h_dropped[m]
            h_fused = h_sum / mask_sum

        fused_logits = self.fused_classifier(h_fused)

        # Step 7: Class-aware contrastive projection (L2 normalized)
        z_con = F.normalize(self.con_proj(h_fused), p=2, dim=-1)

        result: Dict[str, Any] = {
            "logits": fused_logits,
            "gate_scores": gate_scores,
            "fused_feat": h_fused,
            "z_con": z_con,
            "aux_logits": aux_dict,
        }
        # Backward compatibility aliases for per-modality aux logits
        for m in self.modalities:
            result[f"aux_{m}_logits"] = aux_dict[m]

        return result

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Translates legacy MELD checkpoint keys for seamless backward compatibility."""
        for k in list(state_dict.keys()):
            new_k = k
            if k.startswith(prefix + "proj_text."):
                new_k = prefix + "projections.text." + k[len(prefix + "proj_text.") :]
            elif k.startswith(prefix + "proj_video."):
                new_k = prefix + "projections.video." + k[len(prefix + "proj_video.") :]
            elif k.startswith(prefix + "aux_text_classifier."):
                new_k = prefix + "aux_classifiers.text." + k[len(prefix + "aux_text_classifier.") :]
            elif k.startswith(prefix + "aux_video_classifier."):
                new_k = prefix + "aux_classifiers.video." + k[len(prefix + "aux_video_classifier.") :]
            if new_k != k:
                state_dict[new_k] = state_dict.pop(k)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith("proj_") and hasattr(self, "projections"):
                mod = name[5:]
                if mod in self.projections:
                    return self.projections[mod]
            if name.startswith("aux_") and name.endswith("_classifier") and hasattr(self, "aux_classifiers"):
                mod = name[4:-11]
                if mod in self.aux_classifiers:
                    return self.aux_classifiers[mod]
            raise
