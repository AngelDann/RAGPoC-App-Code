from io import BytesIO
from pathlib import Path

from PIL import Image

from ragpoc.local_llm import MediaInput
from ragpoc.local_media import (
    AdaptedMedia,
    MediaAdaptError,
    adapt_media,
    composite_images,
    is_audio,
    is_image,
    is_video,
    video_to_frames,
)


def png_bytes(color=(200, 40, 40), size=(80, 80)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_size(data: bytes) -> tuple[int, int]:
    return Image.open(BytesIO(data)).size


# --- classification ---


def test_image_types_are_recognised():
    assert is_image(MediaInput(data=b"", media_type="image/png"))
    assert is_image(MediaInput(data=b"", media_type="image/jpeg"))
    assert is_image(MediaInput(data=b"", media_type="image/webp"))
    assert not is_image(MediaInput(data=b"", media_type="video/mp4"))


def test_video_and_audio_are_recognised():
    assert is_video(MediaInput(data=b"", media_type="video/mp4"))
    assert is_audio(MediaInput(data=b"", media_type="audio/wav"))
    assert not is_audio(MediaInput(data=b"", media_type="image/png"))


# --- composite_images ---


def test_composite_single_image_passes_through_unchanged():
    blob = png_bytes()
    result = composite_images([MediaInput(data=blob, media_type="image/png")])
    assert result is not None
    assert result.data == blob


def test_composite_multiple_images_returns_one_jpeg():
    a = MediaInput(data=png_bytes((255, 0, 0)), media_type="image/png")
    b = MediaInput(data=png_bytes((0, 0, 255)), media_type="image/png")
    result = composite_images([a, b])
    assert result is not None
    assert result.media_type == "image/jpeg"
    width, height = jpeg_size(result.data)
    assert width > 80  # side by side, wider than one tile
    assert height == 80


def test_composite_respects_max_width():
    tiles = [MediaInput(data=png_bytes(size=(400, 100)), media_type="image/png") for _ in range(4)]
    result = composite_images(tiles, max_width=500)
    assert result is not None
    width, height = jpeg_size(result.data)
    assert width <= 500
    assert height > 0


def test_composite_skips_corrupt_images():
    good = MediaInput(data=png_bytes(), media_type="image/png")
    bad = MediaInput(data=b"not an image", media_type="image/png")
    result = composite_images([bad, good])
    assert result is not None


def test_composite_empty_returns_none():
    assert composite_images([]) is None


def test_composite_ignores_empty_payloads():
    assert composite_images([MediaInput(data=b"", media_type="image/png")]) is None


# --- video_to_frames ---


def test_video_to_frames_rejects_empty_payload():
    assert video_to_frames(MediaInput(data=b"", media_type="video/mp4")) == []


def test_video_to_frames_returns_nothing_for_rubbish():
    import pytest

    with pytest.raises(MediaAdaptError):
        video_to_frames(MediaInput(data=b"not a video", media_type="video/mp4"), max_frames=1)


def _tiny_video(tmp_path: Path) -> bytes:
    import subprocess

    target = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=5:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        check=True,
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return target.read_bytes()


def test_video_to_frames_extracts_jpegs(tmp_path: Path):
    import shutil

    if shutil.which("ffmpeg") is None:
        import pytest

        pytest.skip("ffmpeg not available")
    blob = _tiny_video(tmp_path)
    frames = video_to_frames(MediaInput(data=blob, media_type="video/mp4"), max_frames=2)
    assert len(frames) == 2
    assert all(f.media_type == "image/jpeg" for f in frames)
    assert all(len(f.data) > 0 for f in frames)


# --- adapt_media ---


def test_adapt_media_passes_images_through():
    result = adapt_media([MediaInput(data=png_bytes(), media_type="image/png")])
    assert isinstance(result, AdaptedMedia)
    assert len(result.images) == 1
    assert result.text == ""


def test_adapt_media_tiles_several_images_into_one():
    a = MediaInput(data=png_bytes((255, 0, 0)), media_type="image/png")
    b = MediaInput(data=png_bytes((0, 255, 0)), media_type="image/png")
    result = adapt_media([a, b])
    assert len(result.images) == 1
    assert any("lamina" in note or "compuestas" in note for note in result.notes)


def test_adapt_media_uses_transcriber_for_audio():
    class FakeTranscriber:
        def transcribe(self, media):
            return "hola mundo"

    audio = MediaInput(data=b"\x00\x01", media_type="audio/wav")
    result = adapt_media([audio], transcriber=FakeTranscriber())
    assert "hola mundo" in result.text
    assert any("transcrito" in note for note in result.notes)
    assert result.images == []


def test_adapt_media_notes_when_audio_cannot_be_transcribed():
    audio = MediaInput(data=b"\x00\x01", media_type="audio/wav")
    result = adapt_media([audio], transcriber=None)
    assert any("no se pudo transcribir" in note for note in result.notes)


def test_adapt_media_notes_unsupported_type():
    result = adapt_media([MediaInput(data=b"x", media_type="application/pdf")])
    assert any("no soportado" in note for note in result.notes)


def test_adapt_media_swallows_transcriber_errors():
    class Exploding:
        def transcribe(self, media):
            raise RuntimeError("boom")

    audio = MediaInput(data=b"\x00", media_type="audio/wav")
    result = adapt_media([audio], transcriber=Exploding())
    assert result.text == ""
    assert result.notes
