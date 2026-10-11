"""End-to-end multimodal: agent receives image + video + audio evidence."""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402
from pydantic_ai import Agent, BinaryContent  # noqa: E402

from knowledge.local_agent_model import build_local_agent_model  # noqa: E402
from ragpoc.budget import MemoryBudget  # noqa: E402
from ragpoc.local_llm import LocalLlmEngine  # noqa: E402
from ragpoc.metrics import PeakMemoryTracker, process_rss_bytes  # noqa: E402
from ragpoc.whisper_transcribe import WhisperTranscriber  # noqa: E402

MB = 1024 * 1024
OUT = ROOT / "scripts" / "multimodal_e2e.txt"
lines: list[str] = []


def w(s: object = "") -> None:
    lines.append(str(s))


async def main() -> None:
    tmp = ROOT / "scripts" / "_tmp_media"
    tmp.mkdir(exist_ok=True)

    red = tmp / "red.png"
    Image.new("RGB", (320, 320), (210, 35, 35)).save(red)
    red_bytes = red.read_bytes()

    # Tiny video: 2 s colour bars so ffmpeg has something to sample.
    video = tmp / "clip.mp4"
    import subprocess

    if not video.exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=size=320x240:rate=5:duration=2",
                "-pix_fmt", "yuv420p", str(video),
            ],
            check=True,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    video_bytes = video.read_bytes()

    wav = ROOT / "data" / "uploads" / "3137f9ed3021cd8b-test_audio.wav"
    audio_bytes = wav.read_bytes() if wav.exists() else b""

    engine = LocalLlmEngine(
        ROOT / "data" / "models" / "qwen3-vl-2b",
        dtype="q4f16",
        device="cpu",
        context_length=1536,
        max_new_tokens=140,
        budget=MemoryBudget(),
    )
    transcriber = WhisperTranscriber(ROOT / "data" / "models" / "whisper-tiny", language="es")

    with PeakMemoryTracker(interval=0.05) as tracker:
        engine.load()
        w(f"rss after chat load : {process_rss_bytes() / MB:.1f} MB")
        adapter = build_local_agent_model(engine, name="local-mm", transcriber=transcriber)
        agent = Agent(adapter.model)

        w("")
        w("--- 1) imagen: debe ver el color ---")
        r = await agent.run(
            [
                "Describe en una frase lo que ves en esta imagen.",
                BinaryContent(data=red_bytes, media_type="image/png"),
            ]
        )
        w("respuesta: " + str(r.output)[:300])
        w("")

        w("--- 2) video: se reduce a fotogramas ---")
        r = await agent.run(
            [
                "Que se ve en este clip de video? Una frase.",
                BinaryContent(data=video_bytes, media_type="video/mp4"),
            ]
        )
        w("respuesta: " + str(r.output)[:300])
        w("")

        w("--- 3) audio: se transcribe con whisper ---")
        if audio_bytes:
            r = await agent.run(
                [
                    "Que se escucha en este audio? Una frase.",
                    BinaryContent(data=audio_bytes, media_type="audio/wav"),
                ]
            )
            w("respuesta: " + str(r.output)[:300])
        else:
            w("(sin audio de prueba)")

    w("")
    w(f"peak rss  : {tracker.peak_mb:.1f} MB")
    engine.unload()
    transcriber.unload()

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


asyncio.run(main())
