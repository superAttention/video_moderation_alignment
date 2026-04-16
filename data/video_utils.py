"""
Video frame extraction for Qwen3-VL inputs.

Qwen3-VL expects a list of PIL images (sampled frames) alongside the text prompt.
These are passed as model_input to Tinker alongside the tokenized text.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


VIDEO_ROOT = "data/videos"


def resolve_path(video_path: str) -> str:
    """Resolve dataset-relative path (e.g. 'video/...') to local path."""
    return str(Path(VIDEO_ROOT) / video_path)


def extract_frames(video_path: str, num_frames: int = 8) -> list[Image.Image]:
    """
    Sample num_frames evenly from a video file.
    Returns a list of PIL Images ready to pass to the Qwen3-VL processor.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, total - 1, num_frames, dtype=int))
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from: {video_path}")
    return frames


def load_frame_from_path(frame_path: str) -> Image.Image:
    """Load a single pre-extracted frame from disk."""
    return Image.open(frame_path).convert("RGB")
