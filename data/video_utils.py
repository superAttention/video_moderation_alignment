"""
Video frame extraction for Qwen3-VL inputs.

Qwen3-VL expects a list of PIL images (sampled frames) alongside the text prompt.
These are passed as model_input to Tinker alongside the tokenized text.
"""
from pathlib import Path
from PIL import Image


VIDEO_ROOT = "data/videos"  # set by scripts/download_data.py


def resolve_path(video_path: str) -> str:
    """Resolve dataset-relative path (e.g. 'video/...') to local path."""
    return str(Path(VIDEO_ROOT) / Path(video_path).relative_to("video"))


def extract_frames(video_path: str, num_frames: int = 8) -> list[Image.Image]:
    """
    Sample num_frames evenly from a video file.
    Returns a list of PIL Images ready to pass to the Qwen3-VL processor.
    """
    raise NotImplementedError


def load_frame_from_path(frame_path: str) -> Image.Image:
    """Load a single pre-extracted frame from disk."""
    return Image.open(frame_path).convert("RGB")
