"""Integration test with the real EmbeddingGemma 2 ONNX model (q4, local cache).

Run manually: python tests/integration_local_embedding.py
Requires the model in C:\\Users\\angl_\\AppData\\Local\\Temp\\opencode\\embeddinggemma2-onnx
"""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragpoc.config import Settings
from ragpoc.embeddings_local import LocalEmbeddingProvider


def main():
    tmp = Path(r"C:\Users\angl_\AppData\Local\Temp\opencode")
    settings = Settings(
        _env_file=None,
        data_dir=tmp / "data",
        allowed_upload_dir=tmp / "data" / "uploads",
        embedding_provider="local",
        local_embedding_dtype="q4",
        local_embedding_device="auto",
        local_embedding_dim=768,
        local_embedding_vision_tokens=70,
        local_embedding_cache_dir=tmp / "embeddinggemma2-onnx-parent",
    )
    # point to the already-downloaded model
    settings = settings.model_copy(
        update={"local_embedding_cache_dir": tmp}
    )

    provider = LocalEmbeddingProvider(settings)

    print("=== EmbeddingGemma 2 ONNX integration test ===")

    # 1) Text
    t0 = time.perf_counter()
    vecs = asyncio.run(provider.embed_texts(["The quick brown fox jumps over the lazy dog."]))
    t1 = time.perf_counter()
    v = vecs[0]
    print(f"TEXT: dim={len(v)} norm={np.linalg.norm(v):.4f} time={t1-t0:.3f}s")
    assert len(v) == 768
    assert abs(np.linalg.norm(v) - 1.0) < 0.01

    # 2) Query (with task prefix)
    t0 = time.perf_counter()
    qv = asyncio.run(provider.embed_query("a fox jumping"))
    t1 = time.perf_counter()
    print(f"QUERY: dim={len(qv)} norm={np.linalg.norm(qv):.4f} time={t1-t0:.3f}s")
    assert len(qv) == 768

    # 3) Cross-modal: text-text similarity should be high
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    v2 = asyncio.run(provider.embed_texts(["A fox is jumping over a sleeping dog."]))[0]
    v3 = asyncio.run(provider.embed_texts(["Quantum chromodynamics governs quark interactions."]))[0]
    s_fox = cos(qv, v2)
    s_quark = cos(qv, v3)
    print(f"text-text fox-related:  {s_fox:.4f}")
    print(f"text-text unrelated:    {s_quark:.4f}")
    assert s_fox > s_quark, f"Fox-related ({s_fox:.4f}) should score higher than unrelated ({s_quark:.4f})"

    # 4) Image
    from PIL import Image
    img_path = tmp / "test_integration.png"
    Image.new("RGB", (640, 480), color=(255, 128, 0)).save(img_path)
    t0 = time.perf_counter()
    iv = asyncio.run(provider.embed_image(img_path))
    t1 = time.perf_counter()
    print(f"IMAGE: dim={len(iv)} norm={np.linalg.norm(iv):.4f} time={t1-t0:.3f}s")
    assert len(iv) == 768

    # 5) Cross-modal: text query vs image
    q_img = asyncio.run(provider.embed_query("a warm orange rectangle"))
    s_img = cos(q_img, iv)
    print(f"text-image similarity:  {s_img:.4f}")

    # 6) Audio (generate 1s of a sine wave)
    import struct
    import wave

    wav_path = tmp / "test_integration.wav"
    rate = 16000
    samples = [int(12000 * np.sin(2 * np.pi * 440 * i / rate)) for i in range(rate)]
    with wave.open(str(wav_path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    t0 = time.perf_counter()
    av = asyncio.run(provider.embed_audio(wav_path))
    t1 = time.perf_counter()
    print(f"AUDIO: dim={len(av)} norm={np.linalg.norm(av):.4f} time={t1-t0:.3f}s")
    assert len(av) == 768

    # 7) Cross-modal: audio vs text
    q_audio = asyncio.run(provider.embed_query("a tone or sound"))
    s_audio = cos(q_audio, av)
    print(f"text-audio similarity:  {s_audio:.4f}")

    print()
    print("=== ALL INTEGRATION TESTS PASSED ===")


if __name__ == "__main__":
    main()
