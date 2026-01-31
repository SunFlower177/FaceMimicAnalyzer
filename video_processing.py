import os
import math
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp

# ---------------------------------------------
# This module extracts frames from a video OR
# treats an image as a single-frame "video".
# It returns a list of saved frame image paths.
# ---------------------------------------------

# Optional face filter for videos
mp_face_detection = mp.solutions.face_detection

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_image_path(p: str) -> bool:
    """Return True if the path points to an image file we support."""
    return Path(p).suffix.lower() in IMG_EXTS


def _apply_rotation(img, rotate_mode: Optional[str] = None):
    """Rotate image if rotate_mode provided: 'cw' | 'ccw' | '180'."""
    if img is None or not rotate_mode:
        return img
    if rotate_mode == "cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate_mode == "ccw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate_mode == "180":
        return cv2.rotate(img, cv2.ROTATE_180)
    return img


def _resize_shorter(img, shorter: Optional[int] = None):
    """If shorter side > `shorter`, downscale while keeping aspect ratio."""
    if img is None or not shorter:
        return img
    h, w = img.shape[:2]
    if min(h, w) <= shorter:
        return img
    if h < w:
        new_h = shorter
        new_w = int(w * (shorter / h))
    else:
        new_w = shorter
        new_h = int(h * (shorter / w))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_valid_frames(
    media_path: str,
    video_path: Optional[str],
    output_dir: str,
    target_fps: float = 10.0,
    min_confidence: float = 0.5,
    *,
    face_required: bool = True,
    rotate_mode: Optional[str] = None,
    resize_shorter_val: Optional[int] = None,
) -> List[str]:
    """
    Extract frames from a video or treat an image as a single frame.

    Parameters
    ----------
    media_path : str
        Path to the uploaded file (image or video). Images will be handled directly.
    video_path : Optional[str]
        Path to open with cv2.VideoCapture when input is a video. For images, pass the
        same path as `media_path` (it will be ignored).
    output_dir : str
        Directory to save extracted frame images.
    target_fps : float
        Approximate sampling fps for videos.
    min_confidence : float
        Minimum face detection confidence when `face_required=True`.
    face_required : bool
        If True (default), keep video frames only when a face is detected.
    rotate_mode : Optional[str]
        Optional rotation for all frames: 'cw' | 'ccw' | '180'.
    resize_shorter_val : Optional[int]
        If set, downscale frames so the shorter side equals this value.

    Returns
    -------
    List[str]
        A list of saved frame file paths (e.g., ["frame_00001.jpg", ...]).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ----- Image path: output a single frame -----
    if is_image_path(media_path):
        img = cv2.imread(media_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {media_path}")
        img = _apply_rotation(img, rotate_mode)
        img = _resize_shorter(img, resize_shorter_val)
        out_path = os.path.join(output_dir, "frame_00001.jpg")
        cv2.imwrite(out_path, img)
        return [out_path]

    # ----- Video path: sample frames around target_fps -----
    video_path = video_path or media_path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = target_fps
    step = max(int(round(fps / max(target_fps, 1))), 1)

    saved: List[str] = []
    idx = 0

    detector = None
    if face_required:
        detector = mp_face_detection.FaceDetection(min_detection_confidence=min_confidence)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step != 0:
                idx += 1
                continue
            frame = _apply_rotation(frame, rotate_mode)
            frame = _resize_shorter(frame, resize_shorter_val)

            if detector is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = detector.process(rgb)
                if not res.detections:
                    idx += 1
                    continue

            out_path = os.path.join(output_dir, f"frame_{len(saved)+1:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
            idx += 1
    finally:
        cap.release()
        if detector is not None:
            detector.close()

    # Fallback: ensure at least one frame exists
    if not saved:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"No frames extracted from: {video_path}")
        frame = _apply_rotation(frame, rotate_mode)
        frame = _resize_shorter(frame, resize_shorter_val)
        out_path = os.path.join(output_dir, "frame_00001.jpg")
        cv2.imwrite(out_path, frame)
        saved = [out_path]

    return saved
