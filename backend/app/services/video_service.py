"""
CivicPulse AI — Video Processing Service
==========================================
Extracts keyframes from uploaded videos, runs the vision pipeline on each,
picks the worst-case (highest severity) frame, and optionally extracts the
audio track for Whisper transcription.

Supports: .mp4, .avi, .mov, .webm, .mkv

How it works:
  1. Open video with OpenCV
  2. Extract keyframes at fixed intervals (default: every 2 seconds)
  3. Run run_vision_pipeline() on each frame
  4. Return the frame with the highest defect confidence + severity
  5. (Optional) Extract audio track via ffmpeg → Whisper transcription
"""

import os
import uuid
import subprocess
import shutil
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image

from backend.app.config import settings

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v", ".flv"}


def is_video_file(filename: str) -> bool:
    """Checks if a filename has a video extension."""
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in VIDEO_EXTENSIONS


# =============================================================================
# Keyframe Extraction
# =============================================================================

def extract_keyframes(
    video_path: str,
    output_dir: str,
    max_frames: int = 8,
    interval_seconds: float = 2.0,
) -> List[str]:
    """
    Extracts up to `max_frames` keyframes from a video at fixed time intervals.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frame images.
        max_frames: Maximum number of frames to extract.
        interval_seconds: Time gap between extracted frames.

    Returns:
        List of saved frame image file paths.
    """
    if not HAS_CV2:
        # Fallback: try to extract a single frame using PIL (limited format support)
        return _extract_single_frame_pillow(video_path, output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VideoService] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Calculate frame interval
    frame_gap = max(1, int(fps * interval_seconds))

    # If video is very short (< 3 seconds), just grab 3 evenly spaced frames
    if duration_sec < 3.0:
        frame_gap = max(1, total_frames // 3)

    frame_paths = []
    frame_idx = 0
    saved = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_gap == 0:
            frame_name = f"vframe_{uuid.uuid4().hex[:8]}_{saved:03d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_paths.append(frame_path)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"[VideoService] Extracted {len(frame_paths)} keyframes from {os.path.basename(video_path)} "
          f"(duration: {duration_sec:.1f}s, fps: {fps:.1f})")
    return frame_paths


def _extract_single_frame_pillow(video_path: str, output_dir: str) -> List[str]:
    """Fallback: use ffmpeg (if available) to extract a single thumbnail frame."""
    frame_name = f"vframe_{uuid.uuid4().hex[:8]}_000.jpg"
    frame_path = os.path.join(output_dir, frame_name)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            subprocess.run(
                [ffmpeg, "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path, "-y"],
                capture_output=True, timeout=30, check=True
            )
            if os.path.exists(frame_path):
                print("[VideoService] Extracted 1 frame via ffmpeg fallback")
                return [frame_path]
        except Exception as e:
            print(f"[VideoService] ffmpeg frame extraction failed: {e}")
    return []


# =============================================================================
# Video Metadata
# =============================================================================

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Returns basic video metadata (duration, resolution, fps, codec)."""
    meta = {
        "file_size_bytes": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
        "duration_seconds": 0.0,
        "fps": 0.0,
        "resolution": "unknown",
        "total_frames": 0,
    }

    if not HAS_CV2:
        return meta

    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta["fps"] = round(fps, 1)
            meta["total_frames"] = total
            meta["resolution"] = f"{w}x{h}"
            meta["duration_seconds"] = round(total / fps, 1) if fps > 0 else 0.0
            cap.release()
    except Exception as e:
        print(f"[VideoService] Metadata extraction error: {e}")

    return meta


# =============================================================================
# Audio Extraction from Video
# =============================================================================

def extract_audio_from_video(video_path: str, output_dir: str) -> Optional[str]:
    """
    Extracts audio track from video using ffmpeg.
    Returns path to extracted .mp3 file, or None if ffmpeg unavailable / no audio.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[VideoService] ffmpeg not found — skipping audio extraction from video")
        return None

    audio_name = f"vaudio_{uuid.uuid4().hex[:8]}.mp3"
    audio_path = os.path.join(output_dir, audio_name)

    try:
        result = subprocess.run(
            [ffmpeg, "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"],
            capture_output=True, timeout=120
        )
        if result.returncode == 0 and os.path.exists(audio_path):
            size = os.path.getsize(audio_path)
            if size > 1024:  # At least 1KB of audio
                print(f"[VideoService] Audio extracted: {audio_name} ({size} bytes)")
                return audio_path
            else:
                os.remove(audio_path)
                print("[VideoService] Extracted audio too small — video likely has no audio track")
                return None
        else:
            print(f"[VideoService] ffmpeg audio extraction returned code {result.returncode}")
            return None
    except subprocess.TimeoutExpired:
        print("[VideoService] Audio extraction timed out (120s)")
        return None
    except Exception as e:
        print(f"[VideoService] Audio extraction error: {e}")
        return None


# =============================================================================
# Full Video Analysis Pipeline
# =============================================================================

def analyze_video(
    video_path: str,
    upload_dir: str,
    max_frames: int = 8,
) -> Tuple[List[Dict[str, Any]], float, float, str, str, Dict[str, Any]]:
    """
    Full video analysis pipeline:
      1. Extract keyframes at 2-second intervals
      2. Run 3-layer vision pipeline on EACH frame
      3. Pick the worst-case frame (highest severity)
      4. Save the best annotated frame as the ticket's annotated image

    Returns:
        (detections, c_conf, r_area, best_frame_path, annotated_frame_path, video_meta)

    The return signature is designed to plug directly into reports.py,
    providing the same (detections, c_conf, r_area) as run_vision_pipeline().
    """
    from backend.app.services.vision_service import run_vision_pipeline

    video_meta = get_video_metadata(video_path)
    frame_paths = extract_keyframes(video_path, upload_dir, max_frames=max_frames)

    if not frame_paths:
        print("[VideoService] No frames extracted — returning baseline scores")
        return [], 0.35, 0.08, "", "", video_meta

    best_detections: List[Dict[str, Any]] = []
    best_c_conf = 0.0
    best_r_area = 0.0
    best_frame_path = frame_paths[0]
    best_annotated_path = ""
    best_score = 0.0  # Combined score for ranking frames

    frame_results = []

    for i, frame_path in enumerate(frame_paths):
        annotated_name = f"annotated_vf_{os.path.splitext(os.path.basename(frame_path))[0]}.jpg"
        annotated_path = os.path.join(upload_dir, annotated_name)

        detections, c_conf, r_area, _ = run_vision_pipeline(frame_path, annotated_path)

        # Score this frame: higher confidence + larger area ratio = worse defect
        frame_score = c_conf * 0.6 + r_area * 0.4
        frame_results.append({
            "frame_index": i,
            "frame_path": frame_path,
            "annotated_path": annotated_path,
            "detections": len(detections),
            "confidence": c_conf,
            "area_ratio": r_area,
            "score": round(frame_score, 3),
        })

        if frame_score > best_score:
            best_score = frame_score
            best_detections = detections
            best_c_conf = c_conf
            best_r_area = r_area
            best_frame_path = frame_path
            best_annotated_path = annotated_path

        print(f"[VideoService] Frame {i+1}/{len(frame_paths)}: "
              f"conf={c_conf}, area={r_area}, detections={len(detections)}, score={frame_score:.3f}")

    video_meta["frames_analyzed"] = len(frame_paths)
    video_meta["frame_results"] = frame_results
    video_meta["best_frame_index"] = next(
        (r["frame_index"] for r in frame_results
         if r["frame_path"] == best_frame_path), 0
    )

    # Cleanup non-best frames to save disk space (keep best frame + annotated)
    for fp in frame_paths:
        if fp != best_frame_path and os.path.exists(fp):
            try:
                os.remove(fp)
            except OSError:
                pass

    print(f"[VideoService] Best frame: #{video_meta['best_frame_index']} "
          f"(conf={best_c_conf}, area={best_r_area}, detections={len(best_detections)})")

    return (
        best_detections,
        round(best_c_conf, 2),
        round(best_r_area, 2),
        best_frame_path,
        best_annotated_path,
        video_meta,
    )
