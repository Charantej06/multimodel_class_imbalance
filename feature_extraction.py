"""
feature_extraction.py
======================
Modality-specific preprocessing → pretrained encoder → cached embeddings.

Pipeline per sample
-------------------
  dataset_adapter  →  raw file path  →  preprocessor  →  encoder  →  embedding
                                                                         ↓
                                                                       cache

Caching strategy
----------------
Each modality gets its own cache directory::

    <cache_root>/<modality>/<split>/<sample_id>.pt

A cache is considered valid when:
  1. The .pt file exists.
  2. The stored metadata (model_id, embed_dim) matches the current config.

Cache metadata is written to ``<cache_root>/<modality>/cache_meta.json``.

Audio notes (verified interface)
---------------------------------
hance-ai/audiomae accepts a raw file-path string in its forward() call.
Preprocessing (mel-spectrogram, normalisation) is done internally.
The output tensor has shape ``(768, 8, 64)``; we mean-pool the last two
spatial dimensions to obtain a ``(768,)`` embedding.

Video notes (verified interface)
---------------------------------
MCG-NJU/videomae-base expects exactly 16 frames of size 224×224.
We sample/pad to 16 frames from the full video using decord (preferred)
or torchvision.io as a fallback.

Text notes (verified interface)
---------------------------------
answerdotai/ModernBERT-base: AutoTokenizer + AutoModel.
CLS-token (index 0) of last_hidden_state → (768,).
Supports up to 8192 tokens.

Image notes (verified interface)
----------------------------------
facebook/dinov2-base: AutoImageProcessor + AutoModel.
CLS-token (index 0) of last_hidden_state → (768,).

Supported config keys
---------------------
config = {
    # --- Required ---
    "cache_root":   "embeddings/",  # root dir for cached .pt files
    "modalities":   ["text", "image", "audio", "video"],

    # --- Optional overrides ---
    "checkpoints":  {"audio": "hance-ai/audiomae"},   # per-modality HF id
    "device":       "cuda",          # "cuda" | "cpu" | None (auto)
    "batch_size":   32,
    "num_workers":  0,

    # Text
    "text_max_length": 512,          # tokens; ModernBERT supports up to 8192

    # Video
    "video_num_frames":   16,        # frames sampled per clip
    "video_frame_stride": 1,         # sample every N-th frame (0 = uniform)
}
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_adapter import (
    DatasetAdapter,
    MultimodalDataset,
    multimodal_collate_fn,
)
from encoders.registry import load_encoders, encoder_info

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_MODALITIES = ("text", "image", "audio", "video")
_CACHE_META_FILE = "cache_meta.json"
# Pooling recorded per modality. Both video and text are tracked.
_POOLING = {"video": "mean_tokens", "text": "mean_tokens"}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """
    Simple per-modality embedding cache backed by individual ``.pt`` files.

    Directory layout::

        <root>/
          <modality>/
            cache_meta.json        # {"model_id": ..., "embed_dim": ...}
            train/
              <sample_id>.pt
            val/
              <sample_id>.pt
            test/
              <sample_id>.pt
    """

    def __init__(self, root: str, modality: str, model_id: str, embed_dim: int,
                 pooling: Optional[str] = None):
        self.pooling = pooling
        self.root = Path(root) / modality
        self.modality = modality
        self.model_id = model_id
        self.embed_dim = embed_dim
        self._meta_path = self.root / _CACHE_META_FILE
        self._valid: Optional[bool] = None

    # ------------------------------------------------------------------
    def is_valid(self) -> bool:
        """Return True if the on-disk cache matches the current config."""
        if self._valid is not None:
            return self._valid
        if not self._meta_path.exists():
            return False
        with open(self._meta_path) as fh:
            meta = json.load(fh)
        self._valid = (
            meta.get("model_id") == self.model_id
            and meta.get("embed_dim") == self.embed_dim
            and (self.pooling is None or meta.get("pooling") == self.pooling)
        )
        if not self._valid:
            logger.warning(
                "Cache for '%s' is STALE (stored model_id='%s', current='%s'). "
                "Re-extraction required.",
                self.modality, meta.get("model_id"), self.model_id,
            )
        return self._valid

    def invalidate(self):
        """Force cache to be treated as invalid (e.g. after checkpoint change)."""
        self._valid = False

    def has_files(self) -> bool:
        return self.root.exists() and any(self.root.rglob("*.pt"))

    def _ensure_dirs(self, split: str):
        (self.root / split).mkdir(parents=True, exist_ok=True)
        if not self._meta_path.exists():
            with open(self._meta_path, "w") as fh:
                meta = {"model_id": self.model_id, "embed_dim": self.embed_dim}
                if self.pooling is not None:
                    meta["pooling"] = self.pooling
                json.dump(meta, fh)
            self._valid = True

    def path(self, sample_id: str, split: str) -> Path:
        return self.root / split / f"{_safe_id(sample_id)}.pt"

    def exists(self, sample_id: str, split: str) -> bool:
        return self.path(sample_id, split).exists()

    def save(self, sample_id: str, split: str, embedding: torch.Tensor):
        self._ensure_dirs(split)
        torch.save(embedding.cpu(), self.path(sample_id, split))

    def load(self, sample_id: str, split: str) -> torch.Tensor:
        return torch.load(self.path(sample_id, split), map_location="cpu",
                          weights_only=True)

    def load_split(
        self,
        dataset: MultimodalDataset,
        split: str,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Load all cached embeddings for *split* that exist on disk.

        Returns
        -------
        ids        : List[str]
        embeddings : Tensor  (N, embed_dim)
        labels     : Tensor  (N,)
        """
        ids, embs, labels = [], [], []
        for i in range(len(dataset)):
            item = dataset[i]
            sid = item["id"]
            if self.exists(sid, split):
                ids.append(sid)
                embs.append(self.load(sid, split))
                labels.append(item["label"])
        if embs:
            return ids, torch.stack(embs), torch.tensor(labels, dtype=torch.long)
        return ids, torch.empty(0, self.embed_dim), torch.empty(0, dtype=torch.long)


def _safe_id(sample_id: str) -> str:
    """Make sample_id filesystem-safe."""
    return sample_id.replace("/", "_").replace("\\", "_").replace(":", "_")


# ---------------------------------------------------------------------------
# Modality-specific preprocessors
# ---------------------------------------------------------------------------

def _preprocess_text(
    batch: Dict[str, Any],
    processor,   # AutoTokenizer
    max_length: int,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Tokenise a batch of text strings or file paths (reads file if path exists)."""
    texts = []
    for t in batch.get("text", []):
        if t is None:
            return None
        p = Path(t)
        texts.append(p.read_text(encoding="utf-8") if p.exists() else t)

    if not texts:
        return None

    return processor(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def _is_video_path(path: Union[str, Path, None]) -> bool:
    if path is None:
        return False
    return Path(str(path)).suffix.lower() in VIDEO_EXTENSIONS


def _get_ffmpeg_exe() -> Optional[str]:
    """Return path to ffmpeg binary (system ffmpeg or imageio-ffmpeg bundled binary)."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _write_silent_wav(output_wav_path: Path, duration_sec: float = 1.0, sample_rate: int = 16000):
    """Write a silent WAV file (16kHz mono int16) as a graceful fallback."""
    import wave
    import struct
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(output_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))


def _extract_audio_from_video(
    video_path: Union[str, Path],
    output_wav_path: Union[str, Path],
    sample_rate: int = 16000,
) -> str:
    """
    Extract the audio track from a video file into a 16kHz mono WAV file.
    Reuses existing cached WAV file if present and non-empty.
    """
    import subprocess
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    if output_wav_path.exists() and output_wav_path.stat().st_size > 0:
        return str(output_wav_path)

    ffmpeg_exe = _get_ffmpeg_exe()
    if not ffmpeg_exe:
        raise RuntimeError(
            "ffmpeg is required to extract audio from video files. "
            "Install ffmpeg on your system or run 'pip install imageio-ffmpeg'."
        )

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        str(output_wav_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 or not output_wav_path.exists() or output_wav_path.stat().st_size == 0:
        logger.warning(
            "ffmpeg failed to extract audio from '%s'. Creating silent WAV fallback.",
            video_path,
        )
        _write_silent_wav(output_wav_path, duration_sec=1.0, sample_rate=sample_rate)

    return str(output_wav_path)


def _sample_representative_frame(video_path: Union[str, Path]) -> Optional[Any]:
    """
    Sample a single representative frame (middle frame) from a video file in memory.
    Tries decord first, then OpenCV (cv2), then torchvision.
    Returns a PIL.Image in RGB format or None on failure.
    """
    from PIL import Image

    video_path_str = str(video_path)
    if not Path(video_path_str).exists():
        logger.error("Video file does not exist: %s", video_path_str)
        return None

    # 1. decord
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path_str)
        total = len(vr)
        if total > 0:
            mid_idx = total // 2
            frame_tensor = vr[mid_idx]  # (H, W, C)
            return Image.fromarray(frame_tensor.numpy().astype("uint8"))
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("decord frame sampling failed for '%s': %s", video_path_str, exc)

    # 2. cv2 (OpenCV)
    try:
        import cv2
        cap = cv2.VideoCapture(video_path_str)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid_idx = max(0, total // 2) if total > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("cv2 frame sampling failed for '%s': %s", video_path_str, exc)

    # 3. torchvision fallback
    try:
        import torchvision.io as tvio
        if hasattr(tvio, "read_video"):
            frames, _, _ = tvio.read_video(video_path_str, pts_unit="sec")
            if frames.shape[0] > 0:
                mid_idx = frames.shape[0] // 2
                return Image.fromarray(frames[mid_idx].numpy().astype("uint8"))
    except Exception as exc:
        logger.debug("torchvision frame sampling failed for '%s': %s", video_path_str, exc)

    logger.error("Failed to sample representative frame from '%s'", video_path_str)
    return None


def _preprocess_image(
    batch: Dict[str, Any],
    processor,  # AutoImageProcessor
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load and preprocess a batch of images from file paths.
    If image paths are video files (or if image is absent and video is present),
    samples a representative frame from each video in memory (for DINOv2).
    """
    from PIL import Image

    paths = batch.get("image")
    if not paths or all(p is None for p in paths):
        paths = batch.get("video", [])
    if not paths or any(p is None for p in paths):
        return None

    images = []
    for p in paths:
        try:
            if _is_video_path(p):
                img = _sample_representative_frame(p)
                if img is None:
                    return None
            else:
                img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as exc:
            logger.error("Cannot load image or sample video frame from '%s': %s", p, exc)
            return None

    inputs = processor(images=images, return_tensors="pt")
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {k: v.to(device) for k, v in inputs.items()}


def _preprocess_audio(
    batch: Dict[str, Any],
    device: torch.device,
    audio_cache_dir: Optional[Union[str, Path]] = None,
) -> Optional[List[str]]:
    """
    Validate or prepare audio files for AudioMAE.
    If given video paths (or if audio is absent and video is present),
    extracts each video's audio track into a cached WAV file and returns the WAV path.
    """
    paths = batch.get("audio")
    if not paths or all(p is None for p in paths):
        paths = batch.get("video", [])
    if not paths or any(p is None for p in paths):
        return None

    sample_ids = batch.get("id", [Path(p).stem for p in paths])
    cache_dir = Path(audio_cache_dir or "embeddings/extracted_audio")

    valid = []
    for sid, p in zip(sample_ids, paths):
        if not Path(p).exists():
            logger.error("Media file not found: %s", p)
            return None

        if _is_video_path(p):
            wav_path = cache_dir / f"{_safe_id(str(sid))}.wav"
            try:
                extracted = _extract_audio_from_video(p, wav_path)
                valid.append(extracted)
            except Exception as exc:
                logger.error("Failed to extract audio track from '%s': %s", p, exc)
                return None
        else:
            valid.append(str(p))

    return valid


def _sample_video_frames(
    video_path: str,
    num_frames: int,
    frame_stride: int,
) -> Optional[List[Any]]:
    """
    Sample *num_frames* frames from a video file.

    Tries decord first (fast), then OpenCV (cv2), then torchvision.io.
    Returns a list of PIL Images or None on failure.
    """
    import numpy as np
    from PIL import Image

    # --- decord (preferred) ---
    try:
        import decord  # pip install decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path)
        total = len(vr)
        if total > 0:
            if frame_stride > 0:
                indices = list(range(0, total, frame_stride))[:num_frames]
            else:
                indices = [int(i * total / num_frames) for i in range(num_frames)]
            # Pad if necessary
            while len(indices) < num_frames:
                indices.append(indices[-1])
            frames_tensor = vr.get_batch(indices)  # (T, H, W, C)
            return [
                Image.fromarray(frames_tensor[i].numpy().astype("uint8"))
                for i in range(num_frames)
            ]
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("decord failed for '%s': %s. Trying cv2.", video_path, exc)

    # --- cv2 fallback ---
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                if frame_stride > 0:
                    indices = list(range(0, total, frame_stride))[:num_frames]
                else:
                    indices = [int(i * total / num_frames) for i in range(num_frames)]
                while len(indices) < num_frames:
                    indices.append(indices[-1])
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    elif frames:
                        frames.append(frames[-1])
                cap.release()
                if len(frames) == num_frames:
                    return frames
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("cv2 video frames sampling failed for '%s': %s", video_path, exc)

    # --- torchvision fallback ---
    try:
        import torchvision.io as tvio
        if hasattr(tvio, "read_video"):
            frames, _, _ = tvio.read_video(video_path, pts_unit="sec")  # (T, H, W, C)
            total = frames.shape[0]
            if total > 0:
                if frame_stride > 0:
                    indices = list(range(0, total, frame_stride))[:num_frames]
                else:
                    indices = [int(i * total / num_frames) for i in range(num_frames)]
                while len(indices) < num_frames:
                    indices.append(indices[-1])
                return [
                    Image.fromarray(frames[i].numpy().astype("uint8"))
                    for i in range(num_frames)
                ]
    except Exception as exc:
        logger.error("torchvision failed for '%s': %s", video_path, exc)

    logger.error("All video readers failed for '%s'", video_path)
    return None


_READER_LOGGED = False


def _log_video_reader_once():
    global _READER_LOGGED
    if _READER_LOGGED:
        return
    _READER_LOGGED = True
    try:
        import decord  # noqa: F401
        logger.info("Video reader: decord available (preferred).")
    except Exception as exc:
        logger.warning("decord NOT usable (%s). Using OpenCV/torchvision (slower).", exc)


def _preprocess_video(
    batch: Dict[str, Any],
    processor,   # VideoMAEImageProcessor
    num_frames: int,
    frame_stride: int,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load, sample and preprocess a batch of video file paths."""
    paths = batch.get("video", [])
    if not paths or any(p is None for p in paths):
        return None

    _log_video_reader_once()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(paths), 8)) as ex:
        all_clips = list(ex.map(
            lambda p: _sample_video_frames(p, num_frames, frame_stride), paths))
    if any(c is None for c in all_clips):
        return None

    # processor expects List[List[PIL.Image]] for batched input
    inputs = processor(all_clips, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Per-modality extraction (single forward pass, returns embeddings)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_text(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Attention-masked mean pooling for ModernBERT / BERT-family models."""
    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state  # (B, L, 768)
    mask = inputs["attention_mask"].unsqueeze(-1).expand_as(last_hidden).float()
    sum_embeddings = torch.sum(last_hidden * mask, dim=1)
    sum_mask = torch.clamp(inputs["attention_mask"].sum(dim=1, keepdim=True), min=1)
    return sum_embeddings / sum_mask  # (B, 768)


@torch.no_grad()
def _extract_image(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """CLS-token from DINOv2 / ViT-family models."""
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]   # (B, 768)


@torch.no_grad()
def _extract_audio(model, paths: List[str]) -> torch.Tensor:
    """
    AudioMAE: forward() accepts a file path string.
    Output shape: (768, 8, 64) per sample.
    Mean-pool spatial dims → (768,).
    Batched manually (one call per sample, then stack).
    """
    embeddings = []
    for path in paths:
        out = model(path)                          # (768, 8, 64)
        emb = out.mean(dim=(-2, -1))               # (768,)
        embeddings.append(emb)
    return torch.stack(embeddings)                 # (B, 768)


@torch.no_grad()
def _extract_video(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """VideoMAE has NO CLS token: output is (B, 1568, 768) patch tokens.
    Clip embedding = mean over all tokens (same as HF use_mean_pooling=True)."""
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)   # (B, 768)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Orchestrates modality-specific preprocessing, encoder forward passes,
    and embedding caching.

    Usage::

        extractor = FeatureExtractor(config)
        extractor.extract_all(adapter)
        # → embeddings on disk under config["cache_root"]

        # Load cached embeddings for downstream use:
        ids, embs, labels = extractor.load_embeddings("text", "train", dataset)
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self._device = _resolve_device(config.get("device"))
        if self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.80, device=0)
        self._encoders: Dict[str, Dict[str, Any]] = {}
        self._caches: Dict[str, EmbeddingCache] = {}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def load_encoders(self):
        """Load (and freeze) only the encoders required by config['modalities']."""
        mods = self.cfg.get("modalities", list(SUPPORTED_MODALITIES))
        checkpoints = self.cfg.get("checkpoints", {})
        logger.info("Loading encoders for modalities: %s", mods)
        self._encoders = load_encoders(mods, checkpoints=checkpoints, device=self._device)
        for mod, enc in self._encoders.items():
            logger.info(encoder_info(enc))

    def _get_cache(self, modality: str) -> EmbeddingCache:
        if modality not in self._caches:
            enc = self._encoders[modality]
            self._caches[modality] = EmbeddingCache(
                root=self.cfg.get("cache_root", "embeddings"),
                modality=modality,
                model_id=enc["model_id"],
                embed_dim=enc["embed_dim"],
                pooling=_POOLING.get(modality),
            )
        return self._caches[modality]

    # ------------------------------------------------------------------
    # Extraction entry points
    # ------------------------------------------------------------------

    def extract_all(self, adapter: DatasetAdapter):
        """
        Extract embeddings for all modalities × all splits.
        Skips samples whose embeddings are already cached.
        """
        if not self._encoders:
            self.load_encoders()

        for modality in self._encoders:
            for split in ("train", "val", "test"):
                dataset = adapter.get_dataset(split)
                self.extract_split(modality, split, dataset)

    def extract_split(
        self,
        modality: str,
        split: str,
        dataset: MultimodalDataset,
    ):
        """Extract and cache embeddings for one modality × one split."""
        if modality not in self._encoders:
            raise ValueError(
                f"Encoder for '{modality}' not loaded. "
                "Call load_encoders() first or include it in config['modalities']."
            )

        cache = self._get_cache(modality)
        # Check whether the whole cache is stale
        force_recompute = not cache.is_valid()
        if force_recompute and cache.has_files():
            raise RuntimeError(
                f"Stale '{modality}' cache at {cache.root} (model/pooling mismatch). "
                f"Rename or move that folder first (e.g. {modality}_old), then rerun."
            )

        # Determine which samples still need extraction
        pending_indices = [
            i for i in range(len(dataset))
            if force_recompute or not cache.exists(dataset[i]["id"], split)
        ]

        if not pending_indices:
            logger.info(
                "[%s/%s] All %d embeddings already cached. Skipping.",
                modality, split, len(dataset),
            )
            return

        logger.info(
            "[%s/%s] Extracting %d / %d embeddings.",
            modality, split, len(pending_indices), len(dataset),
        )

        batch_size = self.cfg.get(
            f"{modality}_batch_size",
            self.cfg.get("batch_size", 8 if modality == "video" else 32),
        )
        pending_ds = _SubsetDataset(dataset, pending_indices)
        loader = DataLoader(
            pending_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.cfg.get("num_workers", 0),
            collate_fn=multimodal_collate_fn,
        )

        enc = self._encoders[modality]
        model = enc["model"]
        processor = enc["processor"]
        total_batches = len(loader)

        for b_idx, batch in enumerate(loader):
            batch_ids: List[str] = batch["id"]
            embeddings = self._forward(modality, model, processor, batch)

            if embeddings is None:
                logger.warning(
                    "[%s/%s] Batch %d with IDs %s returned None; skipping.",
                    modality, split, b_idx + 1, batch_ids,
                )
                continue

            for idx, (sid, emb) in enumerate(zip(batch_ids, embeddings)):
                cache.save(sid, split, emb)

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == total_batches:
                logger.info(
                    "[%s/%s] Progress: %d / %d batches completed (%d samples).",
                    modality, split, b_idx + 1, total_batches, min((b_idx + 1) * batch_size, len(pending_indices)),
                )

        logger.info("[%s/%s] Extraction complete.", modality, split)

    # ------------------------------------------------------------------
    # Loading cached embeddings
    # ------------------------------------------------------------------

    def load_embeddings(
        self,
        modality: str,
        split: str,
        dataset: MultimodalDataset,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Load all cached embeddings for *modality* / *split*.

        Returns
        -------
        ids        : List[str]
        embeddings : Tensor  (N, embed_dim)  on *device*
        labels     : Tensor  (N,)            on *device*
        """
        cache = self._get_cache(modality) if modality in self._caches else EmbeddingCache(
            root=self.cfg.get("cache_root", "embeddings"),
            modality=modality,
            model_id=self.cfg.get("checkpoints", {}).get(
                modality,
                _default_model_id(modality),
            ),
            embed_dim=768,
        )
        ids, embs, labels = cache.load_split(dataset, split)
        if device:
            embs = embs.to(device)
            labels = labels.to(device)
        logger.info(
            "Loaded %d cached embeddings [%s/%s].", len(ids), modality, split,
        )
        return ids, embs, labels

    # ------------------------------------------------------------------
    # Private: dispatch forward pass by modality
    # ------------------------------------------------------------------

    def _forward(
        self,
        modality: str,
        model: nn.Module,
        processor,
        batch: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        try:
            if modality == "text":
                max_len = self.cfg.get("text_max_length", 512)
                inputs = _preprocess_text(batch, processor, max_len, self._device)
                if inputs is None:
                    return None
                return _extract_text(model, inputs)

            elif modality == "image":
                inputs = _preprocess_image(batch, processor, self._device)
                if inputs is None:
                    return None
                return _extract_image(model, inputs)

            elif modality == "audio":
                # AudioMAE runs sample-by-sample; processor is None
                cache_dir = Path(self.cfg.get("cache_root", "embeddings")) / "extracted_audio"
                paths = _preprocess_audio(batch, self._device, audio_cache_dir=cache_dir)
                if paths is None:
                    return None
                # Move model output to device consistently
                embs = _extract_audio(model, paths)
                return embs.to(self._device)

            elif modality == "video":
                num_frames  = self.cfg.get("video_num_frames", 16)
                frame_stride = self.cfg.get("video_frame_stride", 0)
                inputs = _preprocess_video(
                    batch, processor, num_frames, frame_stride, self._device
                )
                if inputs is None:
                    return None
                return _extract_video(model, inputs)

        except Exception as exc:
            logger.error("Error during %s forward pass: %s", modality, exc, exc_info=True)
            return None

        return None


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _resolve_device(device_cfg: Optional[str]) -> torch.device:
    if device_cfg is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device_cfg)
    logger.info("Using device: %s", dev)
    return dev


def _default_model_id(modality: str) -> str:
    """Return the default HF checkpoint for a modality."""
    from encoders.registry import DEFAULT_CHECKPOINTS
    return DEFAULT_CHECKPOINTS.get(modality, "unknown")


class _SubsetDataset:
    """Wraps a MultimodalDataset to expose only a list of indices."""

    def __init__(self, dataset: MultimodalDataset, indices: List[int]):
        self._dataset = dataset
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._dataset[self._indices[idx]]


# ---------------------------------------------------------------------------
# Example configuration (importable reference)
# ---------------------------------------------------------------------------

EXAMPLE_CONFIG: Dict[str, Any] = {
    # DatasetAdapter config
    "metadata_path": "data/metadata.csv",
    "id_col":        "id",
    "label_col":     "label",
    "split_col":     "split",
    "modality_cols": {
        "text":  "text_path",
        "image": "image_path",
        "audio": "audio_path",
        "video": "video_path",
    },

    # FeatureExtractor config
    "cache_root":  "embeddings",
    "modalities":  ["text", "image", "audio", "video"],
    "checkpoints": {
        "text":  "answerdotai/ModernBERT-base",
        "image": "facebook/dinov2-base",
        "audio": "hance-ai/audiomae",
        "video": "MCG-NJU/videomae-base",
    },
    "device":      None,   # auto-detect
    "batch_size":  16,
    "num_workers": 0,

    # Modality-specific
    "text_max_length":    512,
    "video_num_frames":   16,
    "video_frame_stride": 0,  # 0 = uniform sampling
}


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Extract and cache multimodal embeddings.",
    )
    parser.add_argument("--meta",       required=True, help="Path to metadata CSV/JSON")
    parser.add_argument("--cache_root", default="embeddings")
    parser.add_argument(
        "--modalities", nargs="+",
        default=["text", "image", "audio", "video"],
        choices=list(SUPPORTED_MODALITIES),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device",     default=None, help="cuda | cpu | None")
    args = parser.parse_args()

    cfg = dict(EXAMPLE_CONFIG)
    cfg.pop("modality_cols", None)
    cfg["metadata_path"] = args.meta
    cfg["cache_root"]    = args.cache_root
    cfg["modalities"]    = args.modalities
    cfg["batch_size"]    = args.batch_size
    cfg["device"]        = args.device

    adapter = DatasetAdapter(cfg).load()
    print(adapter.summary())

    extractor = FeatureExtractor(cfg)
    extractor.load_encoders()
    extractor.extract_all(adapter)

    print("\nEmbeddings cached under:", args.cache_root)
