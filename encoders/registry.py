"""
encoders/registry.py
====================
Pretrained-encoder registry for multimodal feature extraction.

Supported modalities and their verified HuggingFace checkpoints
---------------------------------------------------------------
  text  : answerdotai/ModernBERT-base
          Standard HF transformers ≥ 4.48.0
          AutoTokenizer + AutoModel  (encoder-only, BERT-style)
          CLS-token embedding → shape (B, 768)

  image : facebook/dinov2-base
          Standard HF transformers
          AutoImageProcessor + AutoModel
          CLS-token embedding → shape (B, 768)

  audio : hance-ai/audiomae
          Custom remote code (trust_remote_code=True) – verified against
          the hance-ai/audiomae HuggingFace repository.
          AutoModel accepts a raw file path string per sample.
          Raw output → (768, 8, 64); mean-pooled → shape (B, 768)

  video : MCG-NJU/videomae-base
          Standard HF transformers
          VideoMAEImageProcessor + VideoMAEModel
          Expects 16-frame clips → CLS-token → shape (B, 768)

Interface contract
------------------
Every encoder returned by ``load_encoder`` is a dict::

    {
        "model":     nn.Module,           # frozen pretrained weights
        "processor": callable | None,     # HF processor / tokenizer
        "embed_dim": int,                 # output embedding dimension
        "modality":  str,                 # "text" | "image" | "audio" | "video"
        "model_id":  str,                 # HF model identifier
    }

Models are frozen (``requires_grad = False``) immediately after loading.
They are NOT moved to a device here; the caller manages device placement.

Verified interfaces
-------------------
- ModernBERT : standard AutoModel/AutoTokenizer (transformers ≥ 4.48.0)
- DINOv2     : standard AutoModel/AutoImageProcessor
- AudioMAE   : custom remote code confirmed via hance-ai/audiomae repo;
               forward pass takes a file-path string, returns (768, 8, 64)
- VideoMAE   : VideoMAEImageProcessor + VideoMAEModel (16-frame input)

NOTE: AudioMAE's forward() accepts a file path directly (no separate
processor needed). The output tensor has shape (768, 8, 64) and must
be mean-pooled to obtain a fixed-length embedding.

WARNING: Do NOT call model.train() on these encoders during Stage 1.
Unfreeze selectively only if you intend to fine-tune.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default checkpoint IDs (can be overridden via config)
# ---------------------------------------------------------------------------

SUPPORTED_MODALITIES: tuple = ("text", "image", "audio", "video")

DEFAULT_CHECKPOINTS: Dict[str, str] = {
    "text":  "answerdotai/ModernBERT-base",
    "image": "facebook/dinov2-base",
    "audio": "hance-ai/audiomae",
    "video": "MCG-NJU/videomae-base",
}

# Expected CLS / pooled output dimension for each checkpoint
EMBED_DIMS: Dict[str, int] = {
    "text":  768,
    "image": 768,
    "audio": 768,  # after mean-pooling the (768, 8, 64) spatial output
    "video": 768,
}


# ---------------------------------------------------------------------------
# Internal loader functions (one per modality)
# ---------------------------------------------------------------------------

def _load_text_encoder(model_id: str) -> Dict[str, Any]:
    """
    Load ModernBERT (or any BERT-style) encoder via AutoModel / AutoTokenizer.

    Requires: transformers >= 4.48.0
    Embedding: last_hidden_state[:, 0, :]  (CLS token)
    """
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading text encoder: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    _freeze(model)
    model.eval()
    logger.info("Text encoder loaded. Parameters: %s", _param_count(model))
    return {
        "model":     model,
        "processor": tokenizer,
        "embed_dim": EMBED_DIMS["text"],
        "modality":  "text",
        "model_id":  model_id,
    }


def _load_image_encoder(model_id: str) -> Dict[str, Any]:
    """
    Load DINOv2 (or any ViT) encoder via AutoModel / AutoImageProcessor.

    Embedding: last_hidden_state[:, 0, :]  (CLS token)
    """
    from transformers import AutoImageProcessor, AutoModel

    logger.info("Loading image encoder: %s", model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    _freeze(model)
    model.eval()
    logger.info("Image encoder loaded. Parameters: %s", _param_count(model))
    return {
        "model":     model,
        "processor": processor,
        "embed_dim": EMBED_DIMS["image"],
        "modality":  "image",
        "model_id":  model_id,
    }


def _load_audio_encoder(model_id: str) -> Dict[str, Any]:
    """
    Load AudioMAE via trust_remote_code=True.

    VERIFIED against hance-ai/audiomae (October 2024):
    - Uses custom AudioMAEConfig and AudioMAEEncoder defined in the repo's model.py
    - forward(audio_path: str) -> Tensor of shape (768, 8, 64)
    - Preprocessing (mel-spectrogram, normalisation) is internal to the model
    - Max audio length: 10 seconds (longer inputs are truncated internally)
    - Requires: torchaudio, timm, einops, ffmpeg (>=4.4, <7 recommended)

    No separate processor is needed.
    Embedding: output.mean(dim=(-2, -1))  shape (B, 768)
    """
    from transformers import AutoModel

    logger.info("Loading audio encoder: %s", model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    _freeze(model)
    model.eval()
    logger.info("Audio encoder loaded. Parameters: %s", _param_count(model))
    return {
        "model":     model,
        "processor": None,   # preprocessing is internal to AudioMAE
        "embed_dim": EMBED_DIMS["audio"],
        "modality":  "audio",
        "model_id":  model_id,
        # Runtime note for feature_extraction.py:
        "_audio_pooling": "mean_spatial",  # pool (768,8,64) -> (768,)
    }


def _load_video_encoder(model_id: str) -> Dict[str, Any]:
    """
    Load VideoMAE via VideoMAEImageProcessor + VideoMAEModel.

    VERIFIED against MCG-NJU/videomae-base:
    - VideoMAEImageProcessor: resize 224x224, ImageNet normalisation
    - VideoMAEModel: expects pixel_values of shape (B, 16, 3, 224, 224)
    - Embedding: last_hidden_state (no CLS token; (B, 1568, 768) patch tokens, pooled across sequence)
    - Standard frame count: 16 (callers must sample/pad to 16 frames)
    """
    from transformers import VideoMAEImageProcessor, VideoMAEModel

    logger.info("Loading video encoder: %s", model_id)
    processor = VideoMAEImageProcessor.from_pretrained(model_id)
    model = VideoMAEModel.from_pretrained(model_id)
    _freeze(model)
    model.eval()
    logger.info("Video encoder loaded. Parameters: %s", _param_count(model))
    return {
        "model":     model,
        "processor": processor,
        "embed_dim": EMBED_DIMS["video"],
        "modality":  "video",
        "model_id":  model_id,
        "_num_frames": 16,  # expected frame count; caller must enforce this
    }


# Dispatch table
_LOADERS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "text":  _load_text_encoder,
    "image": _load_image_encoder,
    "audio": _load_audio_encoder,
    "video": _load_video_encoder,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_encoder(
    modality: str,
    model_id: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a pretrained encoder for *modality* and move it to *device*.

    Parameters
    ----------
    modality : str
        One of ``"text"``, ``"image"``, ``"audio"``, ``"video"``.
    model_id : str, optional
        HuggingFace Hub model identifier.  Defaults to the curated
        checkpoint for each modality (see ``DEFAULT_CHECKPOINTS``).
    device : torch.device, optional
        Target device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    dict with keys: model, processor, embed_dim, modality, model_id
    """
    if modality not in _LOADERS:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Choose from: {sorted(_LOADERS.keys())}"
        )

    if model_id is None:
        model_id = DEFAULT_CHECKPOINTS[modality]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading encoder | modality=%s | device=%s", modality, device)
    encoder = _LOADERS[modality](model_id)
    encoder["model"] = encoder["model"].to(device)
    encoder["device"] = device
    return encoder


def load_encoders(
    modalities: list[str],
    checkpoints: Optional[Dict[str, str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load multiple encoders at once.

    Parameters
    ----------
    modalities  : list of modality strings to load
    checkpoints : optional per-modality override, e.g. {"text": "bert-base-uncased"}
    device      : shared target device for all encoders

    Returns
    -------
    dict[modality_str -> encoder_dict]
    """
    checkpoints = checkpoints or {}
    encoders: Dict[str, Dict[str, Any]] = {}
    for mod in modalities:
        mid = checkpoints.get(mod, DEFAULT_CHECKPOINTS.get(mod))
        encoders[mod] = load_encoder(mod, model_id=mid, device=device)
    return encoders


def freeze_encoder(encoder: Dict[str, Any]) -> None:
    """Freeze all parameters of an encoder's model (idempotent)."""
    _freeze(encoder["model"])


def unfreeze_encoder(encoder: Dict[str, Any]) -> None:
    """Unfreeze all parameters of an encoder's model (use with caution)."""
    for p in encoder["model"].parameters():
        p.requires_grad = True
    logger.warning(
        "Encoder '%s' unfrozen. Make sure this is intentional.", encoder["modality"]
    )


def encoder_info(encoder: Dict[str, Any]) -> str:
    """Return a human-readable summary string for an encoder."""
    mod = encoder.get("model")
    frozen = all(not p.requires_grad for p in mod.parameters()) if mod else "?"
    return (
        f"[{encoder['modality'].upper()}] "
        f"{encoder['model_id']} | "
        f"embed_dim={encoder['embed_dim']} | "
        f"frozen={frozen} | "
        f"device={encoder.get('device', '?')} | "
        f"params={_param_count(mod)}"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _freeze(model: nn.Module) -> None:
    """Disable gradient computation for all parameters."""
    for p in model.parameters():
        p.requires_grad = False


def _param_count(model: Optional[nn.Module]) -> str:
    if model is None:
        return "?"
    total = sum(p.numel() for p in model.parameters())
    return f"{total / 1e6:.1f}M"


# ---------------------------------------------------------------------------
# Quick self-test (run: python -m encoders.registry --modality text)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Test-load a single encoder.")
    parser.add_argument(
        "--modality",
        choices=list(_LOADERS.keys()),
        required=True,
        help="Modality to load",
    )
    parser.add_argument(
        "--model_id", default=None,
        help="Override HuggingFace checkpoint (optional)",
    )
    args = parser.parse_args()

    enc = load_encoder(args.modality, model_id=args.model_id)
    print(encoder_info(enc))
    print("Processor:", type(enc["processor"]).__name__)
    print("All parameters frozen:", all(not p.requires_grad for p in enc["model"].parameters()))
