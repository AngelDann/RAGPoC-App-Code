"""Adapt evidence media to what a local multimodal model can actually consume.

The retrieval layer hands back raw evidence: full images, rendered PDF pages,
30-second MP4 clips and whole audio files. Only the first two are directly
usable by the ONNX vision encoder, which expects ``pixel_values`` produced from
still images. Everything else has to be reduced before it reaches the model:

* **video** -> a handful of JPEG frames, since a frame sequence is what the
  vision tower understands and an MP4 blob is not;
* **audio** -> a transcript, because the shipped chat model is vision-only and
  a text transcript is both smaller and more useful to the agent than raw
  waveform.

The reduction is deliberately lossy and explicit about it: the adapted media
carries a note describing what was dropped, so the answer can be qualified
rather than silently wrong.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ragpoc.local_llm import MediaInput

FFMPEG_TIMEOUT_S = 60


class Transcriber(Protocol):
    def transcribe(self, media: MediaInput) -> str: ...


@dataclass
class AdaptedMedia:
    """Media reduced to what the local model accepts, plus provenance notes."""

    images: list[MediaInput] = field(default_factory=list)
    text: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.images and not self.text


def image_media_types() -> frozenset[str]:
    return frozenset({"image/png", "image/jpeg", "image/webp"})


def is_image(media: MediaInput) -> bool:
    return media.media_type in image_media_types()


def is_video(media: MediaInput) -> bool:
    return media.media_type.startswith("video/")


def is_audio(media: MediaInput) -> bool:
    return media.media_type.startswith("audio/")


def video_to_frames(media: MediaInput, *, max_frames: int = 3, width: int = 640) -> list[MediaInput]:
    """Sample a video into JPEG frames with ffmpeg.

    Frames are taken evenly across the clip so a short retrieval segment still
    yields its start, middle and end rather than three identical posters.
    """
    if shutil.which("ffmpeg") is None:
        raise MediaAdaptError("ffmpeg is required to turn video evidence into frames")
    if not media.data or max_frames < 1:
        return []
    frames: list[MediaInput] = []
    with tempfile.TemporaryDirectory(prefix="ragpoc-frames-") as tmp:
        tmpdir = Path(tmp)
        source = tmpdir / "clip.bin"
        source.write_bytes(media.data)
        for index in range(max_frames):
            fraction = (index + 0.5) / max_frames
            target = tmpdir / f"frame{index:02d}.jpg"
            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{fraction:.4f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                f"scale={width}:-2",
                "-q:v",
                "4",
                str(target),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    timeout=FFMPEG_TIMEOUT_S,
                    creationflags=_no_window(),
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise MediaAdaptError(f"ffmpeg could not extract frame {index}: {exc}") from exc
            if target.exists():
                data = target.read_bytes()
                if data:
                    frames.append(MediaInput(data=data, media_type="image/jpeg"))
    return frames


def _no_window() -> int:
    # ffmpeg is a console app; without this the desktop build flashes windows.
    return subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


def composite_images(images: Sequence[MediaInput], *, max_width: int = 1024) -> MediaInput | None:
    """Tile several images into one JPEG.

    The ONNX vision path accepts a single image per processor call
    (``Images.open_bytes`` replaces rather than accumulates), so when a turn
    carries more than one piece of visual evidence they are laid out side by
    side instead of dropping all but the first.
    """
    valid = [img for img in images if img.data]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    from io import BytesIO

    from PIL import Image

    tiles: list[Any] = []
    for item in valid:
        try:
            tile = Image.open(BytesIO(item.data)).convert("RGB")
        except Exception:
            continue
        tiles.append(tile)
    if not tiles:
        return None
    if len(tiles) == 1:
        return MediaInput(data=_encode_jpeg(tiles[0]), media_type="image/jpeg")

    cell_h = min(512, max(t.height for t in tiles))
    scaled = []
    for tile in tiles:
        ratio = cell_h / tile.height
        scaled.append(tile.resize((max(1, int(tile.width * ratio)), cell_h)))
    total_w = sum(t.width for t in scaled)
    sheet = Image.new("RGB", (total_w, cell_h), (255, 255, 255))
    x = 0
    for tile in scaled:
        sheet.paste(tile, (x, 0))
        x += tile.width
    if sheet.width > max_width:
        ratio = max_width / sheet.width
        sheet = sheet.resize((max_width, max(1, int(sheet.height * ratio))))
    return MediaInput(data=_encode_jpeg(sheet), media_type="image/jpeg")


def _encode_jpeg(image: Any) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def adapt_media(
    media: Sequence[MediaInput],
    *,
    max_frames: int = 3,
    max_images: int = 1,
    transcriber: Transcriber | None = None,
) -> AdaptedMedia:
    """Reduce evidence media to model-consumable images plus explanatory text.

    ``max_images`` is 1 because the ONNX vision path takes a single image per
    call; anything beyond that is tiled into one sheet rather than discarded.
    """
    result = AdaptedMedia()
    visuals: list[MediaInput] = []
    for item in media:
        if is_image(item):
            visuals.append(item)
        elif is_video(item):
            frames = video_to_frames(item, max_frames=max_frames)
            if frames:
                visuals.extend(frames)
                result.notes.append(
                    f"Video reducido a {len(frames)} fotogramas; no se reprodujo el audio ni la duracion completa."
                )
            else:
                result.notes.append("Video recuperado pero no se pudieron extraer fotogramas.")
        elif is_audio(item):
            transcript = _transcribe(item, transcriber)
            if transcript:
                result.text = (result.text + "\n" + transcript).strip()
                result.notes.append("Audio transcrito a texto; el sonido original no se reprodujo.")
            else:
                result.notes.append("Audio recuperado pero no se pudo transcribir.")
        else:
            result.notes.append(f"Tipo de medio no soportado por el modelo local: {item.media_type}.")

    if visuals:
        single = composite_images(visuals)
        if single is not None:
            result.images = [single]
            if len(visuals) > 1:
                result.notes.append(
                    f"{len(visuals)} imagenes compuestas en una sola lamina para el modelo local."
                )
    return result


def _transcribe(media: MediaInput, transcriber: Transcriber | None) -> str:
    if transcriber is None:
        return ""
    try:
        return transcriber.transcribe(media) or ""
    except Exception:
        return ""


class MediaAdaptError(RuntimeError):
    pass


__all__ = [
    "AdaptedMedia",
    "MediaAdaptError",
    "Transcriber",
    "adapt_media",
    "composite_images",
    "is_audio",
    "is_image",
    "is_video",
    "video_to_frames",
]
