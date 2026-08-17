from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoSegment:
    ordinal: int
    start_seconds: float
    end_seconds: float
    frame_seconds: float
    frame_path: Path
    clip_path: Path


def video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_video_segments(path: Path, output_dir: Path, segment_seconds: int = 30) -> list[VideoSegment]:
    duration = video_duration(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments: list[VideoSegment] = []
    start = 0.0
    ordinal = 0
    while start < duration:
        end = min(start + segment_seconds, duration)
        midpoint = (start + end) / 2
        frame_path = output_dir / f"segment-{ordinal:05d}.jpg"
        clip_path = output_dir / f"segment-{ordinal:05d}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(midpoint), "-i", str(path), "-frames:v", "1", "-q:v", "3", str(frame_path)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(start), "-i", str(path), "-t", str(end - start),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-c:a", "aac", "-b:a", "64k",
                "-movflags", "+faststart", str(clip_path),
            ],
            check=True,
            capture_output=True,
        )
        segments.append(VideoSegment(ordinal, start, end, midpoint, frame_path, clip_path))
        ordinal += 1
        start = end
    return segments


def timecode(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600:02d}:{(whole % 3600) // 60:02d}:{whole % 60:02d}"
