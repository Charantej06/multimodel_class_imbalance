"""
food101_lt/verify.py
====================
Verification suite for the Long-Tailed Food-101 (101 classes) dataset
designed for the RCS paper pipeline (Autoencoder + SupCon -> GMM / Calibrated Gaussian Sampling -> ResNet32).

Verifications performed:
  1. Reload images.pt and labels.pt.
  2. Reload val and test splits (folders and/or manifest).
  3. Assert tensor shapes:
       - images is 4D (N, C, H, W) or (N, H, W, C) with N = sum(counts) = 23,430, C = 3.
       - labels is 1D (N,) matching image count.
  4. Assert value range:
       - images values lie strictly in [-1.0, 1.0].
  5. Assert bincount == planned counts:
       - bincount exactly matches profile.make_counts(700, 40, 101).
  6. Assert non-increasing class order:
       - counts[i] >= counts[i+1] for all i in [0, C-2].
       - label 0 is largest (700), label 100 is smallest (40).
  7. Assert val and test exactly balanced:
       - val has exactly 50 samples per class across all 101 classes (5,050 total).
       - test has exactly 250 samples per class across all 101 classes (25,250 total).
  8. Assert no file overlap between splits:
       - pairwise disjoint: train & val == empty, train & test == empty, val & test == empty.
  9. Print comprehensive metrics:
       - Imbalance Ratio (IR = n_max / n_min).
       - #GMM classes vs #minority classes at eta=7 and eta=4.
       - Predicted GMM component counts.
       - Train, Val, and Test split sizes.
 10. Save report visualizations to data/food101_lt/report/:
       - Bar chart of per-class train counts on log-y scale (class_counts_logy.png).
       - 10x10 sample grid of largest class (sample_grid_largest.png).
       - 10x10 sample grid of smallest class (sample_grid_smallest.png).
 11. Exit non-zero (sys.exit(1)) if any assertion fails.
"""

from __future__ import annotations
import os
import sys
import csv
import math
import zlib
import struct
import pickle
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

# Ensure local imports work whether run from repo root or package dir
sys.path.insert(0, str(Path(__file__).parent.resolve()))
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

try:
    from food101_profile import make_counts, check_profile
except ImportError:
    try:
        from food_101_imbalanced_making.food101_profile import make_counts, check_profile
    except ImportError:
        def make_counts(n_max: int = 700, n_min: int = 40, C: int = 101) -> List[int]:
            return [round(n_max * (n_min / n_max) ** (i / (C - 1))) for i in range(C)]

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


class MockImageData:
    """Mock container for 4D image data when PyTorch is not installed."""
    def __init__(self, n: int, c: int, h: int, w: int):
        self.shape = (n, c, h, w)

    def min(self) -> float:
        return -1.0

    def max(self) -> float:
        return 1.0

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int) -> Any:
        return [[[((idx + ch + y + x) % 256) / 127.5 - 1.0 for x in range(self.shape[3])] for y in range(self.shape[2])] for ch in range(self.shape[1])]


# ==============================================================================
# Mock Tensor Container (Used for pure-Python fallback when torch is unavailable)
# ==============================================================================
class MockTensor:
    """Lightweight 4D/1D tensor wrapper providing tensor API when torch is missing."""
    def __init__(self, data: Any, shape: Tuple[int, ...]):
        self.data = data
        self.shape = shape

    def min(self) -> float:
        if isinstance(self.data, (list, tuple)):
            if len(self.data) == 0:
                return 0.0
            if isinstance(self.data[0], (list, tuple)):
                return min(min(row) for row in self.data)
            return float(min(self.data))
        return float(self.data)

    def max(self) -> float:
        if isinstance(self.data, (list, tuple)):
            if len(self.data) == 0:
                return 0.0
            if isinstance(self.data[0], (list, tuple)):
                return max(max(row) for row in self.data)
            return float(max(self.data))
        return float(self.data)

    def tolist(self) -> List[Any]:
        if isinstance(self.data, list):
            return self.data
        return list(self.data)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: Any) -> Any:
        return self.data[idx]


# ==============================================================================
# Pure Python PNG Generation (Standard Library zlib/struct, zero external deps)
# ==============================================================================
def write_rgb_png(filepath: Path, width: int, height: int, rgb_bytes: bytearray) -> None:
    """Writes a 24-bit Truecolor PNG file directly using standard library zlib/struct."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    raw_data = bytearray()
    row_stride = width * 3
    for y in range(height):
        raw_data.append(0)  # Filter type 0 (None)
        raw_data.extend(rgb_bytes[y * row_stride : (y + 1) * row_stride])

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    png_data = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_data.extend(make_chunk(b"IHDR", ihdr_data))
    png_data.extend(make_chunk(b"IDAT", zlib.compress(bytes(raw_data), 9)))
    png_data.extend(make_chunk(b"IEND", b""))

    with open(filepath, "wb") as f:
        f.write(png_data)


def render_fallback_logy_chart(counts: List[int], filepath: Path) -> None:
    """Renders a pixel-perfect logarithmic bar chart in pure Python."""
    w, h = 900, 520
    # Background: clean dark-slate theme
    bg_r, bg_g, bg_b = 248, 250, 252  # Light neutral
    buf = bytearray([bg_r, bg_g, bg_b] * (w * h))

    def set_pixel(x: int, y: int, r: int, g: int, b: int):
        if 0 <= x < w and 0 <= y < h:
            idx = (y * w + x) * 3
            buf[idx] = r
            buf[idx + 1] = g
            buf[idx + 2] = b

    def draw_rect(x1: int, y1: int, x2: int, y2: int, r: int, g: int, b: int):
        for y in range(max(0, y1), min(h, y2 + 1)):
            for x in range(max(0, x1), min(w, x2 + 1)):
                set_pixel(x, y, r, g, b)

    # Plot margins
    plot_x1, plot_y1 = 80, 60
    plot_x2, plot_y2 = 850, 450
    plot_w = plot_x2 - plot_x1
    plot_h = plot_y2 - plot_y1

    # White plotting area
    draw_rect(plot_x1, plot_y1, plot_x2, plot_y2, 255, 255, 255)

    # Y-axis is log-scale: from min_val=20 to max_val=1000
    log_min = math.log10(20)
    log_max = math.log10(1000)

    def val_to_y(val: float) -> int:
        frac = (math.log10(max(val, 20)) - log_min) / (log_max - log_min)
        return int(plot_y2 - frac * plot_h)

    # Gridlines and reference lines
    grid_vals = [20, 50, 100, 200, 500, 700, 1000]
    for gv in grid_vals:
        gy = val_to_y(gv)
        draw_rect(plot_x1, gy, plot_x2, gy, 226, 232, 240)

    # Threshold lines: zeta=100 (eta=7) in red, zeta=175 (eta=4) in orange
    y_eta7 = val_to_y(100)
    y_eta4 = val_to_y(175)
    for x in range(plot_x1, plot_x2, 4):
        draw_rect(x, y_eta7, min(x + 2, plot_x2), y_eta7, 220, 38, 38)
        draw_rect(x, y_eta4, min(x + 2, plot_x2), y_eta4, 217, 119, 6)

    # Draw bars for 101 classes
    num_classes = len(counts)
    bar_width = max(1, (plot_w // num_classes) - 1)
    for i, c in enumerate(counts):
        bx = plot_x1 + int(i * (plot_w / num_classes))
        by = val_to_y(c)
        # Bar color: cool steel blue gradient
        draw_rect(bx, by, bx + bar_width, plot_y2, 37, 99, 235)

    # Border around plot
    for x in range(plot_x1, plot_x2 + 1):
        set_pixel(x, plot_y1, 71, 85, 105)
        set_pixel(x, plot_y2, 71, 85, 105)
    for y in range(plot_y1, plot_y2 + 1):
        set_pixel(plot_x1, y, 71, 85, 105)
        set_pixel(plot_x2, y, 71, 85, 105)

    write_rgb_png(filepath, w, h, buf)


def render_fallback_grid(sample_images: List[Any], filepath: Path, title: str) -> None:
    """Renders a 10x10 image grid (100 tiles) to a PNG file using pure Python."""
    grid_rows, grid_cols = 10, 10
    total_tiles = grid_rows * grid_cols
    assert len(sample_images) == total_tiles, f"Expected {total_tiles} tiles, got {len(sample_images)}"

    tile_size = 48
    border = 2
    cell_size = tile_size + border
    w = grid_cols * cell_size + border
    h = grid_rows * cell_size + border

    # Background: dark navy grid lines
    buf = bytearray([30, 41, 59] * (w * h))

    for idx, img in enumerate(sample_images):
        row = idx // grid_cols
        col = idx % grid_cols
        x_start = border + col * cell_size
        y_start = border + row * cell_size

        for ty in range(tile_size):
            for tx in range(tile_size):
                # Extract RGB from sample image or generate gradient pattern
                if hasattr(img, "__getitem__"):
                    try:
                        # (3, H, W) or (H, W, 3)
                        if hasattr(img, "shape") and len(img.shape) == 3 and img.shape[0] == 3:
                            h_orig, w_orig = img.shape[1], img.shape[2]
                            sy = int(ty * (h_orig / tile_size))
                            sx = int(tx * (w_orig / tile_size))
                            r = int(min(max((float(img[0][sy][sx]) + 1.0) * 127.5, 0.0), 255.0))
                            g = int(min(max((float(img[1][sy][sx]) + 1.0) * 127.5, 0.0), 255.0))
                            b = int(min(max((float(img[2][sy][sx]) + 1.0) * 127.5, 0.0), 255.0))
                        else:
                            r = (idx * 7 + tx * 3) % 256
                            g = (idx * 13 + ty * 3) % 256
                            b = (idx * 19 + (tx + ty)) % 256
                    except Exception:
                        r = (idx * 5) % 256
                        g = (idx * 11) % 256
                        b = (idx * 17) % 256
                else:
                    r = (idx * 5) % 256
                    g = (idx * 11) % 256
                    b = (idx * 17) % 256

                px = x_start + tx
                py = y_start + ty
                p_idx = (py * w + px) * 3
                buf[p_idx] = r
                buf[p_idx + 1] = g
                buf[p_idx + 2] = b

    write_rgb_png(filepath, w, h, buf)


# ==============================================================================
# Verification Logic
# ==============================================================================
class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == "MockImageData":
            return MockImageData
        if name == "MockTensor":
            return MockTensor
        return super().find_class(module, name)


def load_tensors(
    images_path: Path,
    labels_path: Path,
    allow_mock_fallback: bool = False,
) -> Tuple[Any, Any]:
    """
    Reloads images.pt and labels.pt.
    Supports PyTorch tensors and pure-Python pickled MockTensor fallback.
    """
    assert images_path.exists(), (
        f"Assert failed: images.pt not found at '{images_path}'. "
        f"Provide existing file or pass --mock to generate verification tensors."
    )
    assert labels_path.exists(), (
        f"Assert failed: labels.pt not found at '{labels_path}'. "
        f"Provide existing file or pass --mock to generate verification tensors."
    )

    if TORCH_AVAILABLE:
        try:
            images = torch.load(images_path, weights_only=False)
            labels = torch.load(labels_path, weights_only=False)
            return images, labels
        except Exception as e:
            if not allow_mock_fallback:
                raise AssertionError(f"Assert failed: torch.load failed for tensors: {e}")

    # Fallback to pickle loading with safe class resolver
    with open(images_path, "rb") as f:
        images = _SafeUnpickler(f).load()
    with open(labels_path, "rb") as f:
        labels = _SafeUnpickler(f).load()

    return images, labels


def generate_mock_tensors(
    images_path: Path,
    labels_path: Path,
    counts: List[int],
    height: int = 32,
    width: int = 32,
) -> None:
    """Creates synthetic compliant tensors for verification testing."""
    images_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = sum(counts)
    rng = random.Random(42)

    # Build labels list
    labels_list: List[int] = []
    for cls_idx, cnt in enumerate(counts):
        labels_list.extend([cls_idx] * cnt)

    assert len(labels_list) == total_samples, f"Labels count mismatch: {len(labels_list)} vs {total_samples}"

    if TORCH_AVAILABLE:
        # Generate random values in [-1.0, 1.0]
        images_tensor = torch.empty((total_samples, 3, height, width), dtype=torch.float32).uniform_(-1.0, 1.0)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        torch.save(images_tensor, images_path)
        torch.save(labels_tensor, labels_path)
    else:
        # Pure Python MockTensor container
        mock_images = MockImageData(total_samples, 3, height, width)
        mock_labels = MockTensor(labels_list, (total_samples,))

        with open(images_path, "wb") as f:
            pickle.dump(mock_images, f)
        with open(labels_path, "wb") as f:
            pickle.dump(mock_labels, f)


def verify_food101_lt(
    data_dir: Path,
    report_dir: Path,
    images_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
    val_dir: Optional[Path] = None,
    test_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    eta_list: List[int] = [7, 4],
    n_max: int = 700,
    n_min: int = 40,
    num_classes: int = 101,
    mock: bool = False,
) -> Dict[str, Any]:
    """
    Executes full assertion suite for Long-Tailed Food-101.
    Exits non-zero (sys.exit(1)) if any assertion fails.
    """
    data_dir = data_dir.resolve()
    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    if images_path is None:
        images_path = data_dir / "images.pt"
    if labels_path is None:
        labels_path = data_dir / "labels.pt"
    if manifest_path is None:
        manifest_path = data_dir / "manifest.csv"
    if val_dir is None:
        val_dir = data_dir / "val"
    if test_dir is None:
        test_dir = data_dir / "test"

    planned_counts = make_counts(n_max=n_max, n_min=n_min, C=num_classes)
    total_planned_train = sum(planned_counts)

    print("=" * 80)
    print("               FOOD-101 LONG-TAILED (RCS) VERIFICATION SUITE")
    print("=" * 80)

    # 1. Handle mock mode or tensor loading
    if mock and (not images_path.exists() or not labels_path.exists()):
        print("[INFO] --mock flag detected: Generating compliant mock tensors...")
        generate_mock_tensors(images_path, labels_path, planned_counts)
        print(f"[INFO] Mock tensors saved to '{images_path}' and '{labels_path}'")

    print(f"[STEP 1] Reloading tensors from:\n         - Images: {images_path}\n         - Labels: {labels_path}")
    images, labels = load_tensors(images_path, labels_path, allow_mock_fallback=mock)

    # 2. Assert Shapes
    print("[STEP 2] Asserting tensor shapes...")
    assert hasattr(images, "shape"), "Assert failed: images object missing 'shape' attribute."
    assert hasattr(labels, "shape"), "Assert failed: labels object missing 'shape' attribute."

    # images must be 4D
    assert len(images.shape) == 4, (
        f"Assert failed (shapes): images tensor must be 4D (N, C, H, W) or (N, H, W, C), "
        f"got shape {images.shape} (ndim={len(images.shape)})."
    )
    # labels must be 1D
    assert len(labels.shape) == 1, (
        f"Assert failed (shapes): labels tensor must be 1D (N,), got shape {labels.shape}."
    )
    # samples count match
    assert images.shape[0] == total_planned_train, (
        f"Assert failed (shapes): images batch dimension {images.shape[0]} != planned total {total_planned_train}."
    )
    assert labels.shape[0] == total_planned_train, (
        f"Assert failed (shapes): labels length {labels.shape[0]} != planned total {total_planned_train}."
    )
    assert images.shape[0] == labels.shape[0], (
        f"Assert failed (shapes): images.shape[0] ({images.shape[0]}) != labels.shape[0] ({labels.shape[0]})."
    )

    # Color channels check (3 channels)
    has_3_channels = (images.shape[1] == 3) or (images.shape[3] == 3)
    assert has_3_channels, (
        f"Assert failed (shapes): expected 3 color channels (RGB) in dim 1 or dim 3, got shape {images.shape}."
    )
    print(f"         [PASS] images.shape = {tuple(images.shape)}, labels.shape = {tuple(labels.shape)}")

    # 3. Assert Value Range [-1, 1]
    print("[STEP 3] Asserting value range [-1.0, 1.0]...")
    min_val = float(images.min())
    max_val = float(images.max())
    assert min_val >= -1.0 - 1e-4, (
        f"Assert failed (value range): images.min() = {min_val:.6f} < -1.0. "
        f"Tensor must be strictly normalized to [-1, 1]."
    )
    assert max_val <= 1.0 + 1e-4, (
        f"Assert failed (value range): images.max() = {max_val:.6f} > 1.0. "
        f"Tensor must be strictly normalized to [-1, 1]."
    )
    print(f"         [PASS] min = {min_val:.6f} (>= -1.0), max = {max_val:.6f} (<= 1.0)")

    # 4. Assert Bincount == Planned Counts
    print("[STEP 4] Asserting bincount == planned counts...")
    if hasattr(labels, "tolist"):
        labels_list = labels.tolist()
    else:
        labels_list = list(labels)

    label_counts = Counter(labels_list)
    bincount = [label_counts.get(i, 0) for i in range(num_classes)]

    assert len(bincount) == num_classes, (
        f"Assert failed (bincount): number of classes in bincount ({len(bincount)}) != {num_classes}."
    )

    for i in range(num_classes):
        assert bincount[i] == planned_counts[i], (
            f"Assert failed (bincount): class {i} observed count {bincount[i]} != "
            f"planned count {planned_counts[i]}."
        )
    print(f"         [PASS] All {num_classes} classes match planned profile exactly.")

    # 5. Assert Non-Increasing Class Order
    print("[STEP 5] Asserting non-increasing class order...")
    for i in range(num_classes - 1):
        assert bincount[i] >= bincount[i + 1], (
            f"Assert failed (monotonicity): counts are not non-increasing at index {i}: "
            f"class {i} ({bincount[i]}) < class {i+1} ({bincount[i+1]})."
        )
    assert bincount[0] == n_max, (
        f"Assert failed (monotonicity): largest class count {bincount[0]} != n_max ({n_max})."
    )
    assert bincount[-1] == n_min, (
        f"Assert failed (monotonicity): smallest class count {bincount[-1]} != n_min ({n_min})."
    )
    print(f"         [PASS] Monotonically non-increasing: Rank 0 ({bincount[0]}) -> Rank {num_classes-1} ({bincount[-1]}).")

    # 6. Assert Val and Test are Exactly Balanced & No File Overlap
    print("[STEP 6] Asserting balanced val/test folders and zero split file overlap...")
    train_files: set[str] = set()
    val_files: set[str] = set()
    test_files: set[str] = set()

    val_counts_per_class: Dict[int, int] = Counter()
    test_counts_per_class: Dict[int, int] = Counter()

    # (A) Check physical folders if they exist
    val_folder_verified = False
    test_folder_verified = False

    if val_dir.exists() and val_dir.is_dir():
        val_subdirs = [d for d in val_dir.iterdir() if d.is_dir()]
        if len(val_subdirs) == num_classes:
            for idx, d in enumerate(sorted(val_subdirs)):
                files = [f for f in d.iterdir() if f.is_file()]
                val_counts_per_class[idx] = len(files)
                for f in files:
                    val_files.add(str(f.resolve()))
            val_folder_verified = True

    if test_dir.exists() and test_dir.is_dir():
        test_subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if len(test_subdirs) == num_classes:
            for idx, d in enumerate(sorted(test_subdirs)):
                files = [f for f in d.iterdir() if f.is_file()]
                test_counts_per_class[idx] = len(files)
                for f in files:
                    test_files.add(str(f.resolve()))
            test_folder_verified = True

    # (B) If physical folders were not directly present or partially verified, check manifest.csv
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row["split"]
                lbl = int(row["label"])
                path = row["image_path"]
                if split == "train":
                    train_files.add(path)
                elif split == "val":
                    val_files.add(path)
                    val_counts_per_class[lbl] += 1
                elif split == "test":
                    test_files.add(path)
                    test_counts_per_class[lbl] += 1

    # Assert val balance: exactly 50 per class (5,050 total)
    assert len(val_counts_per_class) == num_classes, (
        f"Assert failed (val balance): expected {num_classes} classes in val, got {len(val_counts_per_class)}."
    )
    for c in range(num_classes):
        cnt = val_counts_per_class.get(c, 0)
        assert cnt == 50, (
            f"Assert failed (val balance): class {c} has {cnt} val samples, expected exactly 50."
        )
    total_val = sum(val_counts_per_class.values())
    assert total_val == 50 * num_classes, (
        f"Assert failed (val balance): total val samples {total_val} != {50 * num_classes}."
    )

    # Assert test balance: exactly 250 per class (25,250 total)
    assert len(test_counts_per_class) == num_classes, (
        f"Assert failed (test balance): expected {num_classes} classes in test, got {len(test_counts_per_class)}."
    )
    for c in range(num_classes):
        cnt = test_counts_per_class.get(c, 0)
        assert cnt == 250, (
            f"Assert failed (test balance): class {c} has {cnt} test samples, expected exactly 250."
        )
    total_test = sum(test_counts_per_class.values())
    assert total_test == 250 * num_classes, (
        f"Assert failed (test balance): total test samples {total_test} != {250 * num_classes}."
    )

    # Assert no file overlap between splits
    overlap_train_val = train_files & val_files
    assert len(overlap_train_val) == 0, (
        f"Assert failed (disjoint splits): train and val have {len(overlap_train_val)} overlapping files: "
        f"{list(overlap_train_val)[:5]}"
    )
    overlap_train_test = train_files & test_files
    assert len(overlap_train_test) == 0, (
        f"Assert failed (disjoint splits): train and test have {len(overlap_train_test)} overlapping files: "
        f"{list(overlap_train_test)[:5]}"
    )
    overlap_val_test = val_files & test_files
    assert len(overlap_val_test) == 0, (
        f"Assert failed (disjoint splits): val and test have {len(overlap_val_test)} overlapping files: "
        f"{list(overlap_val_test)[:5]}"
    )

    print(f"         [PASS] Val: exactly 50 samples / class across {num_classes} classes ({total_val:,} total).")
    print(f"         [PASS] Test: exactly 250 samples / class across {num_classes} classes ({total_test:,} total).")
    print(f"         [PASS] Split file overlap: Train & Val = 0, Train & Test = 0, Val & Test = 0.")

    # 7. Print IR, #GMM vs #Minority, Component Counts, Split Sizes
    print("-" * 80)
    print("                     STATISTICAL SUMMARY & METRICS")
    print("-" * 80)
    imbalance_ratio = bincount[0] / bincount[-1]
    print(f"  - Imbalance Ratio (IR)      : {imbalance_ratio:.2f} ({bincount[0]} / {bincount[-1]})")
    print(f"  - Train Split Size          : {total_planned_train:,} samples (imbalanced)")
    print(f"  - Val Split Size            : {total_val:,} samples (balanced, 50/class)")
    print(f"  - Test Split Size           : {total_test:,} samples (balanced, 250/class)")
    print(f"  - Total Dataset Size        : {total_planned_train + total_val + total_test:,} samples")

    eta_metrics: Dict[int, Dict[str, Any]] = {}
    for eta in eta_list:
        zeta = bincount[0] // eta
        gmm_classes = [c for c in bincount if c > zeta]
        minority_classes = [c for c in bincount if c <= zeta]
        gmm_components = sum(math.ceil(c / 100) for c in gmm_classes)
        total_components = sum(math.ceil(c / 100) if c > 100 else 1 for c in bincount)
        eta_metrics[eta] = {
            "zeta": zeta,
            "num_gmm": len(gmm_classes),
            "num_minority": len(minority_classes),
            "gmm_components": gmm_components,
            "total_components": total_components,
        }
        print(f"\n  [Threshold eta = {eta}]:")
        print(f"    * Majority Threshold (zeta) : {zeta} (n_max // {eta})")
        print(f"    * # GMM Classes (> zeta)    : {len(gmm_classes)}")
        print(f"    * # Minority Classes (<= zeta): {len(minority_classes)}")
        print(f"    * GMM Components (Majority) : {gmm_components}")
        print(f"    * Total Predicted Components: {total_components} (with minority singletons)")

    # 8. Save Visualizations to report/
    print("-" * 80)
    print(f"[STEP 7] Generating and saving report plots to: {report_dir}")

    # (A) Bar chart of per-class train counts (log-y)
    bar_chart_path = report_dir / "class_counts_logy.png"
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(num_classes), bincount, color="#2563eb", edgecolor="#1e40af", alpha=0.85, width=0.8)
        ax.set_yscale("log")
        ax.axhline(100, color="#dc2626", linestyle="--", linewidth=1.5, label=r"$\eta=7$ GMM Threshold ($\zeta=100$)")
        ax.axhline(175, color="#d97706", linestyle=":", linewidth=1.5, label=r"$\eta=4$ GMM Threshold ($\zeta=175$)")
        ax.set_title("Food-101 Long-Tail: Per-Class Train Sample Counts (Log-Y Scale)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Class Rank (0 = Largest, 100 = Smallest)", fontsize=11)
        ax.set_ylabel("Train Sample Count (log scale)", fontsize=11)
        ax.set_xlim(-1, num_classes)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(bar_chart_path, dpi=150)
        plt.close(fig)
    else:
        render_fallback_logy_chart(bincount, bar_chart_path)
    print(f"         [SAVED] Bar chart (log-y)   : {bar_chart_path}")

    # Extract sample images for Largest Class (Label 0) and Smallest Class (Label num_classes - 1)
    largest_indices = [idx for idx, lbl in enumerate(labels_list) if lbl == 0]
    smallest_indices = [idx for idx, lbl in enumerate(labels_list) if lbl == (num_classes - 1)]

    assert len(largest_indices) == n_max, (
        f"Assert failed: expected {n_max} samples for largest class, got {len(largest_indices)}"
    )
    assert len(smallest_indices) == n_min, (
        f"Assert failed: expected {n_min} samples for smallest class, got {len(smallest_indices)}"
    )

    # 10x10 = 100 tiles for largest class
    largest_samples = [images[largest_indices[i]] for i in range(100)]

    # 10x10 = 100 tiles for smallest class (tile/repeat 40 samples across 100 cells)
    smallest_samples = [images[smallest_indices[i % n_min]] for i in range(100)]

    largest_grid_path = report_dir / "sample_grid_largest.png"
    smallest_grid_path = report_dir / "sample_grid_smallest.png"

    if MATPLOTLIB_AVAILABLE:
        # Render largest class 10x10 grid
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        for idx, ax in enumerate(axes.flat):
            img = largest_samples[idx]
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()
            if hasattr(img, "shape") and len(img.shape) == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img_norm = ((img + 1.0) * 127.5).clip(0, 255).astype("uint8")
            ax.imshow(img_norm)
            ax.axis("off")
        fig.suptitle(f"Class 0 (Largest Class, n={n_max}) - 10x10 Sample Grid", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(largest_grid_path, dpi=150)
        plt.close(fig)

        # Render smallest class 10x10 grid (40 unique samples tiled)
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        for idx, ax in enumerate(axes.flat):
            img = smallest_samples[idx]
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()
            if hasattr(img, "shape") and len(img.shape) == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img_norm = ((img + 1.0) * 127.5).clip(0, 255).astype("uint8")
            ax.imshow(img_norm)
            ax.axis("off")
        fig.suptitle(f"Class {num_classes-1} (Smallest Class, n={n_min}) - 10x10 Sample Grid (Tiled)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(smallest_grid_path, dpi=150)
        plt.close(fig)
    else:
        render_fallback_grid(largest_samples, largest_grid_path, title=f"Class 0 (n={n_max})")
        render_fallback_grid(smallest_samples, smallest_grid_path, title=f"Class {num_classes-1} (n={n_min})")

    print(f"         [SAVED] Largest class grid  : {largest_grid_path}")
    print(f"         [SAVED] Smallest class grid : {smallest_grid_path}")

    print("=" * 80)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY [OK]")
    print("=" * 80)

    return {
        "imbalance_ratio": imbalance_ratio,
        "total_train": total_planned_train,
        "total_val": total_val,
        "total_test": total_test,
        "eta_metrics": eta_metrics,
        "bar_chart_path": str(bar_chart_path),
        "largest_grid_path": str(largest_grid_path),
        "smallest_grid_path": str(smallest_grid_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Long-Tailed Food-101 dataset split, tensor shapes, value ranges, and balance."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/food101_lt"),
        help="Directory containing manifest.csv, class_map.json, config.json, images.pt, labels.pt.",
    )
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=Path("data/food101_lt/report"),
        help="Output directory for generated plots and grids.",
    )
    parser.add_argument(
        "--images_pt",
        type=Path,
        default=None,
        help="Path to images.pt tensor (defaults to <data_dir>/images.pt).",
    )
    parser.add_argument(
        "--labels_pt",
        type=Path,
        default=None,
        help="Path to labels.pt tensor (defaults to <data_dir>/labels.pt).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.csv (defaults to <data_dir>/manifest.csv).",
    )
    parser.add_argument(
        "--val_dir",
        type=Path,
        default=None,
        help="Path to val directory (defaults to <data_dir>/val).",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=None,
        help="Path to test directory (defaults to <data_dir>/test).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate compliant synthetic mock tensors if images.pt / labels.pt do not exist.",
    )
    parser.add_argument(
        "--n_max",
        type=int,
        default=700,
        help="Maximum class count (default: 700).",
    )
    parser.add_argument(
        "--n_min",
        type=int,
        default=40,
        help="Minimum class count (default: 40).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=101,
        help="Number of classes (default: 101).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        verify_food101_lt(
            data_dir=args.data_dir,
            report_dir=args.report_dir,
            images_path=args.images_pt,
            labels_path=args.labels_pt,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            manifest_path=args.manifest,
            eta_list=[7, 4],
            n_max=args.n_max,
            n_min=args.n_min,
            num_classes=args.num_classes,
            mock=args.mock,
        )
    except AssertionError as err:
        print(f"\n[FATAL ASSERTION ERROR]: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR]: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
