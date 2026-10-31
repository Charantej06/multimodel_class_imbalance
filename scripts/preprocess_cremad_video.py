#!/usr/bin/env python3
"""
scripts/preprocess_cremad_video.py
==================================
Preprocesses CREMA-D video files for VideoMAE / frozen video backbone feature extraction (Step 5).

Pipeline for each manifest clip id:
1. Resolves input video path: {media_root}/VideoFlash/{id}.flv
   (falls back to ffmpeg conversion if OpenCV cannot decode).
2. Uniformly samples N frames (default 16) across the video duration.
3. Detects face in sampled frames using MTCNN (or fallback detector).
4. Computes median face bounding box across detected frames.
5. Applies 20% margin to the median box (with square aspect preservation).
6. Crops and resizes all 16 frames to (224, 224) RGB uint8.
7. If face detection fails across all sampled frames, falls back to a center crop
   and logs the clip id to {output_dir}/failed_faces.csv.
8. Saves the preprocessed tensor as (16, 224, 224, 3) uint8 npy to {output_dir}/{id}.npy.
9. Resumable (skips existing valid .npy files).
10. Supports multiprocessing for high throughput.
11. Generates 8 random sample 4x4 visual grids as PNG for verification.
"""

from __future__ import annotations
import argparse
import csv
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocess_cremad_video")

# Global detector instance for worker process initialization
_WORKER_DETECTOR = None


def init_worker_detector(device_str: str = "cpu"):
    """Initializes the face detector within each worker process."""
    global _WORKER_DETECTOR
    try:
        from facenet_pytorch import MTCNN
        _WORKER_DETECTOR = MTCNN(
            keep_all=False,
            select_largest=True,
            post_process=False,
            device=device_str,
        )
    except Exception as e:
        logger.warning("Could not initialize MTCNN in worker: %s. Will fallback to Haar/Center crop.", e)
        _WORKER_DETECTOR = None


def get_detector(device_str: str = "cpu"):
    """Returns or lazily creates a detector instance."""
    global _WORKER_DETECTOR
    if _WORKER_DETECTOR is None:
        init_worker_detector(device_str)
    return _WORKER_DETECTOR


def decode_video_frames(video_path: Path) -> List[np.ndarray]:
    """
    Decodes all frames from a video file using OpenCV.
    If OpenCV fails, attempts conversion via ffmpeg.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        cap.release()

    if len(frames) > 0:
        return frames

    # Fallback: try converting FLV to MP4 using ffmpeg
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            f"OpenCV failed to decode {video_path} and 'ffmpeg' binary was not found on PATH. "
            "Please ensure ffmpeg is installed or OpenCV has FLV codec support."
        )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_mp4 = Path(tmp_file.name)

    try:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-pix_fmt", "bgr24",
            "-loglevel", "error",
            str(tmp_mp4),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        cap = cv2.VideoCapture(str(tmp_mp4))
        frames = []
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
            cap.release()
    finally:
        if tmp_mp4.exists():
            try:
                tmp_mp4.unlink()
            except OSError:
                pass

    if len(frames) == 0:
        raise RuntimeError(f"Failed to decode any frames from {video_path} using both OpenCV and ffmpeg.")

    return frames


def detect_face_box(
    frame_rgb: np.ndarray,
    detector: Any,
    min_confidence: float = 0.80,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Runs face detection on an RGB frame.
    Returns (x1, y1, x2, y2) if detected with sufficient confidence, else None.
    """
    if detector is not None:
        try:
            boxes, probs = detector.detect(frame_rgb)
            if boxes is not None and len(boxes) > 0:
                prob = probs[0] if probs is not None else 1.0
                if prob is not None and prob >= min_confidence:
                    b = boxes[0]
                    return float(b[0]), float(b[1]), float(b[2]), float(b[3])
        except Exception as e:
            logger.debug("Detector error: %s", e)

    # Secondary fallback: OpenCV Haar Cascade if available
    try:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(faces) > 0:
            # Largest face
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest
            return float(x), float(y), float(x + w), float(y + h)
    except Exception:
        pass

    return None


def calculate_crop_window(
    median_box: Optional[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    margin: float = 0.20,
    square: bool = True,
) -> Tuple[int, int, int, int]:
    """
    Computes [y1, y2, x1, x2] cropping window from the median face box with margin.
    If median_box is None, returns center crop window.
    """
    if median_box is None:
        # Fallback center crop
        crop_size = min(img_w, img_h)
        cx, cy = img_w // 2, img_h // 2
        x1 = max(0, cx - crop_size // 2)
        x2 = min(img_w, x1 + crop_size)
        y1 = max(0, cy - crop_size // 2)
        y2 = min(img_h, y1 + crop_size)
        return y1, y2, x1, x2

    x1_b, y1_b, x2_b, y2_b = median_box
    bw = max(1.0, x2_b - x1_b)
    bh = max(1.0, y2_b - y1_b)
    cx = (x1_b + x2_b) / 2.0
    cy = (y1_b + y2_b) / 2.0

    if square:
        # Expand larger dimension by margin on each side: side = max(bw, bh) * (1 + 2 * margin)
        side = max(bw, bh) * (1.0 + 2.0 * margin)
        half = side / 2.0
        x1 = int(round(cx - half))
        x2 = int(round(cx + half))
        y1 = int(round(cy - half))
        y2 = int(round(cy + half))

        # Shift inside bounds if shifted off edge
        if x1 < 0:
            x2 = min(img_w, x2 - x1)
            x1 = 0
        if x2 > img_w:
            x1 = max(0, x1 - (x2 - img_w))
            x2 = img_w
        if y1 < 0:
            y2 = min(img_h, y2 - y1)
            y1 = 0
        if y2 > img_h:
            y1 = max(0, y1 - (y2 - img_h))
            y2 = img_h

        # Final sanity clamp
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)
    else:
        dx = bw * margin
        dy = bh * margin
        x1 = max(0, int(round(x1_b - dx)))
        x2 = min(img_w, int(round(x2_b + dx)))
        y1 = max(0, int(round(y1_b - dy)))
        y2 = min(img_h, int(round(y2_b + dy)))

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        # Degradation protection: center crop
        return calculate_crop_window(None, img_w, img_h, margin=margin, square=square)

    return y1, y2, x1, x2


def process_single_clip(
    clip_id: str,
    media_root: Path,
    output_dir: Path,
    num_frames: int = 16,
    image_size: int = 224,
    margin: float = 0.20,
    square: bool = True,
    skip_existing: bool = True,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """
    Processes one video clip:
    Reads FLV -> uniformly samples 16 frames -> detects face -> crops with margin -> resizes -> saves NPY.
    """
    out_npy = output_dir / f"{clip_id}.npy"
    if skip_existing and out_npy.is_file():
        try:
            arr = np.load(out_npy)
            if arr.shape == (num_frames, image_size, image_size, 3) and arr.dtype == np.uint8:
                return {
                    "clip_id": clip_id,
                    "status": "cached",
                    "face_detected": True,
                    "detected_count": num_frames,
                    "error": None,
                }
        except Exception:
            pass  # corrupt cache, re-process

    video_path = media_root / "VideoFlash" / f"{clip_id}.flv"
    if not video_path.is_file():
        return {
            "clip_id": clip_id,
            "status": "error",
            "face_detected": False,
            "detected_count": 0,
            "error": f"Missing video file: {video_path}",
        }

    try:
        raw_frames = decode_video_frames(video_path)
    except Exception as e:
        return {
            "clip_id": clip_id,
            "status": "error",
            "face_detected": False,
            "detected_count": 0,
            "error": f"Decode error: {e}",
        }

    total_frames = len(raw_frames)
    if total_frames == 0:
        return {
            "clip_id": clip_id,
            "status": "error",
            "face_detected": False,
            "detected_count": 0,
            "error": "0 frames decoded",
        }

    # Uniformly sample num_frames indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_bgr = [raw_frames[i] for i in indices]

    # Convert to RGB
    sampled_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled_bgr]

    detector = get_detector(device_str)
    img_h, img_w, _ = sampled_rgb[0].shape

    detected_boxes: List[Tuple[float, float, float, float]] = []
    for f_rgb in sampled_rgb:
        box = detect_face_box(f_rgb, detector=detector)
        if box is not None:
            detected_boxes.append(box)

    face_detected = len(detected_boxes) > 0
    median_box = None
    if face_detected:
        med_x1 = float(np.median([b[0] for b in detected_boxes]))
        med_y1 = float(np.median([b[1] for b in detected_boxes]))
        med_x2 = float(np.median([b[2] for b in detected_boxes]))
        med_y2 = float(np.median([b[3] for b in detected_boxes]))
        median_box = (med_x1, med_y1, med_x2, med_y2)

    y1, y2, x1, x2 = calculate_crop_window(
        median_box, img_w=img_w, img_h=img_h, margin=margin, square=square
    )

    cropped_frames: List[np.ndarray] = []
    for f_rgb in sampled_rgb:
        crop = f_rgb[y1:y2, x1:x2]
        # High quality interpolation
        interp = cv2.INTER_AREA if (crop.shape[0] > image_size or crop.shape[1] > image_size) else cv2.INTER_LINEAR
        resized = cv2.resize(crop, (image_size, image_size), interpolation=interp)
        cropped_frames.append(resized)

    # Stack to (16, 224, 224, 3) uint8
    video_tensor = np.stack(cropped_frames, axis=0).astype(np.uint8)

    # Save to temp then atomically rename
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = output_dir / f".tmp_{clip_id}_{os.getpid()}_{random.randint(1000, 9999)}.npy"
    np.save(tmp_out, video_tensor)
    tmp_out.replace(out_npy)

    return {
        "clip_id": clip_id,
        "status": "success",
        "face_detected": face_detected,
        "detected_count": len(detected_boxes),
        "error": None,
        "crop_window": (y1, y2, x1, x2),
    }


def save_sample_grid(
    npy_path: Path,
    out_png: Path,
    title: str = "",
):
    """
    Creates and saves a 4x4 visual montage grid from the 16 preprocessed frames.
    """
    arr = np.load(npy_path)  # (16, 224, 224, 3) RGB uint8
    assert arr.shape == (16, 224, 224, 3), f"Invalid shape: {arr.shape}"

    rows = []
    for r in range(4):
        cols = []
        for c in range(4):
            idx = r * 4 + c
            frame_rgb = arr[idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            # Add frame index stamp
            cv2.putText(
                frame_bgr,
                f"F{idx:02d}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cols.append(frame_bgr)
        rows.append(np.hstack(cols))
    grid = np.vstack(rows)  # (896, 896, 3)

    # Add header banner with metadata
    banner_height = 40
    banner = np.zeros((banner_height, grid.shape[1], 3), dtype=np.uint8)
    if title:
        cv2.putText(
            banner,
            title,
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    final_img = np.vstack([banner, grid])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), final_img)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CREMA-D video frames with face detection and crop.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/cremad_manifest.csv",
        help="Path to cremad_manifest.csv (default: data/cremad_manifest.csv)",
    )
    parser.add_argument(
        "--media_root",
        type=str,
        default="CREMAD",
        help="Root directory containing VideoFlash and AudioWAV folders (default: CREMAD)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cache/cremad_video",
        help="Directory to save preprocessed .npy files (default: cache/cremad_video)",
    )
    parser.add_argument(
        "--failed_csv",
        type=str,
        default=None,
        help="Path to CSV logging clips where face detection failed (default: {output_dir}/failed_faces.csv)",
    )
    parser.add_argument(
        "--sample_grids_dir",
        type=str,
        default=None,
        help="Directory to save visual sample grid PNGs (default: {output_dir}/sample_grids)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to uniformly sample per clip (default: 16)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Target cropped frame resolution (default: 224)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.20,
        help="Margin to add around detected face bounding box (default: 0.20 for 20%%)",
    )
    parser.add_argument(
        "--square_crop",
        action="store_true",
        default=True,
        help="Preserve 1:1 aspect ratio when cropping face to prevent distortion (default: True)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4, 0 for sequential)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to first N clips from manifest (for testing / smoke checks)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip already processed .npy files (default: True)",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Force reprocessing even if .npy files exist",
    )
    parser.add_argument(
        "--num_sample_grids",
        type=int,
        default=8,
        help="Number of sample grids to generate as PNG (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample grid selection (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for face detector: cpu or cuda (default: cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    media_root = Path(args.media_root)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    # Resolve manifest path if relative
    if not manifest_path.is_file():
        alt_paths = [
            Path("data/cremad_manifest.csv"),
            Path("CREMAD/cremad_manifest.csv"),
            Path("../data/cremad_manifest.csv"),
        ]
        for ap in alt_paths:
            if ap.is_file():
                manifest_path = ap
                break

    if not manifest_path.is_file():
        logger.error("FATAL: Manifest not found at %s", manifest_path)
        sys.exit(1)

    # Resolve media root
    if not (media_root / "VideoFlash").is_dir():
        alt_roots = [
            Path("CREMAD"),
            Path("multimodel_class_imbalance/CREMAD"),
            Path("../CREMAD"),
        ]
        for ar in alt_roots:
            if (ar / "VideoFlash").is_dir():
                media_root = ar
                break

    if not (media_root / "VideoFlash").is_dir():
        logger.error("FATAL: VideoFlash directory not found under %s", media_root)
        sys.exit(1)

    failed_csv = Path(args.failed_csv) if args.failed_csv else output_dir / "failed_faces.csv"
    sample_grids_dir = Path(args.sample_grids_dir) if args.sample_grids_dir else output_dir / "sample_grids"

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_grids_dir.mkdir(parents=True, exist_ok=True)

    df_manifest = pd.read_csv(manifest_path)
    if "id" not in df_manifest.columns:
        logger.error("FATAL: 'id' column missing in manifest %s", manifest_path)
        sys.exit(1)

    clip_ids = df_manifest["id"].tolist()
    if args.limit is not None and args.limit > 0:
        logger.info("Applying --limit %d (from %d total clips in manifest)", args.limit, len(clip_ids))
        clip_ids = clip_ids[: args.limit]
        df_manifest = df_manifest.iloc[: args.limit].copy()

    total_clips = len(clip_ids)
    logger.info("=" * 80)
    logger.info("CREMA-D VIDEO PREPROCESSING (Step 5)")
    logger.info("=" * 80)
    logger.info("Total clips to process: %d", total_clips)
    logger.info("Media root:             %s", media_root)
    logger.info("Output directory:       %s", output_dir)
    logger.info("Target frames:          %d frames @ %dx%d (uint8)", args.num_frames, args.image_size, args.image_size)
    logger.info("Face margin:            %.1f%% (square_crop=%s)", args.margin * 100, args.square_crop)
    logger.info("Workers:                %d (device: %s)", args.workers, args.device)
    logger.info("Skip existing:          %s", args.skip_existing)
    logger.info("-" * 80)

    start_time = time.time()
    results: List[Dict[str, Any]] = []

    if args.workers <= 1:
        # Sequential execution
        init_worker_detector(args.device)
        for idx, cid in enumerate(clip_ids, 1):
            res = process_single_clip(
                clip_id=cid,
                media_root=media_root,
                output_dir=output_dir,
                num_frames=args.num_frames,
                image_size=args.image_size,
                margin=args.margin,
                square=args.square_crop,
                skip_existing=args.skip_existing,
                device_str=args.device,
            )
            results.append(res)
            if idx % 10 == 0 or idx == total_clips:
                elapsed = time.time() - start_time
                rate = idx / max(1e-5, elapsed)
                logger.info(
                    "[%d/%d] %.1f%% done | Elapsed: %.1fs | %.1f clips/sec | Detected: %s",
                    idx, total_clips, (idx / total_clips) * 100, elapsed, rate, res["face_detected"],
                )
    else:
        # Multiprocessing execution
        logger.info("Launching ProcessPoolExecutor with %d workers...", args.workers)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker_detector,
            initargs=(args.device,),
        ) as executor:
            future_to_id = {
                executor.submit(
                    process_single_clip,
                    clip_id=cid,
                    media_root=media_root,
                    output_dir=output_dir,
                    num_frames=args.num_frames,
                    image_size=args.image_size,
                    margin=args.margin,
                    square=args.square_crop,
                    skip_existing=args.skip_existing,
                    device_str=args.device,
                ): cid
                for cid in clip_ids
            }

            done_count = 0
            for future in as_completed(future_to_id):
                res = future.result()
                results.append(res)
                done_count += 1
                if done_count % 10 == 0 or done_count == total_clips:
                    elapsed = time.time() - start_time
                    rate = done_count / max(1e-5, elapsed)
                    logger.info(
                        "[%d/%d] %.1f%% done | Elapsed: %.1fs | %.1f clips/sec | Detected: %s",
                        done_count, total_clips, (done_count / total_clips) * 100, elapsed, rate, res["face_detected"],
                    )

    elapsed_total = time.time() - start_time

    # Aggregate results
    cached_count = sum(1 for r in results if r["status"] == "cached")
    success_count = sum(1 for r in results if r["status"] in ("success", "cached"))
    error_count = sum(1 for r in results if r["status"] == "error")
    failed_face_count = sum(1 for r in results if not r["face_detected"] and r["status"] != "error")
    detected_face_count = sum(1 for r in results if r["face_detected"])

    # Log failed face detections to failed_faces.csv
    failed_clips = [r for r in results if not r["face_detected"] and r["status"] != "error"]
    failed_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "status", "fallback", "detected_count", "error"])
        for r in failed_clips:
            writer.writerow([r["clip_id"], r["status"], "center_crop", r["detected_count"], r["error"] or ""])

    # Log errors if any
    error_clips = [r for r in results if r["status"] == "error"]
    if len(error_clips) > 0:
        err_csv = output_dir / "processing_errors.csv"
        with open(err_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "error"])
            for r in error_clips:
                writer.writerow([r["clip_id"], r["error"]])
        logger.error("WARNING: %d clips failed with processing errors! Logged to %s", len(error_clips), err_csv)

    failure_rate = (failed_face_count / max(1, success_count)) * 100.0

    print("\n" + "=" * 80)
    print("CREMA-D VIDEO PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total Clips Processed:    {total_clips:,}")
    print(f"Successful (.npy saved):  {success_count:,}")
    print(f"  - Cached (skipped):     {cached_count:,}")
    print(f"  - Newly processed:      {success_count - cached_count:,}")
    print(f"Face Detection Success:   {detected_face_count:,} ({detected_face_count / max(1, success_count) * 100:.2f}%)")
    print(f"Fallback to Center Crop:  {failed_face_count:,} ({failure_rate:.2f}%)")
    print(f"Processing Errors:        {error_count:,}")
    print(f"Failed Detections CSV:    {failed_csv}")
    print(f"Total Elapsed Time:       {elapsed_total:.2f}s ({elapsed_total / max(1, total_clips):.3f}s / clip)")
    print("=" * 80)

    # 11. Generate random visual check grids
    if args.num_sample_grids > 0:
        rng = random.Random(args.seed)
        valid_res = [r for r in results if r["status"] in ("success", "cached")]
        num_to_sample = min(args.num_sample_grids, len(valid_res))
        sample_picks = rng.sample(valid_res, num_to_sample)

        logger.info("Generating %d sample visual grids into %s...", num_to_sample, sample_grids_dir)
        grid_paths = []
        for g_idx, r in enumerate(sample_picks, 1):
            cid = r["clip_id"]
            npy_path = output_dir / f"{cid}.npy"
            png_path = sample_grids_dir / f"grid_{g_idx:02d}_{cid}.png"

            # Lookup emotion and actor if available in manifest
            row = df_manifest[df_manifest["id"] == cid]
            emo = row["label"].values[0] if len(row) > 0 and "label" in row else "unknown"
            act = row["actor"].values[0] if len(row) > 0 and "actor" in row else "unknown"
            det_label = "Face Detected (20% margin)" if r["face_detected"] else "FALLBACK: Center Crop"

            title = f"Clip: {cid} | Actor: {act} | Emotion: {emo} | {det_label}"
            save_sample_grid(npy_path, png_path, title=title)
            grid_paths.append(png_path)
            logger.info("Saved visual grid %d: %s", g_idx, png_path)

        print(f"\nSaved {len(grid_paths)} visual verification grids in: {sample_grids_dir}")

    # Check fatal exit condition
    if error_count > 0 and error_count == total_clips:
        logger.error("FATAL: All clips failed to process!")
        sys.exit(1)

    logger.info("Step 5 preprocessing run complete!")
    return 0


if __name__ == "__main__":
    main()

