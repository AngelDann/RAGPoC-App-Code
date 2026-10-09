import asyncio
from pathlib import Path

import numpy as np
import pytest

from ragpoc.config import Settings
from ragpoc.embeddings import EmbeddingProvider, get_provider


def make_settings(tmp_path: Path, **overrides) -> Settings:
    defaults = dict(_env_file=None, data_dir=tmp_path / "data", allowed_upload_dir=tmp_path / "data" / "uploads")
    defaults.update(overrides)
    return Settings(**defaults)


# --- factory tests ---

def test_get_provider_defaults_to_openrouter(tmp_path):
    settings = make_settings(tmp_path)
    provider = get_provider(settings)
    assert type(provider).__name__ == "OpenRouterEmbeddingProvider"


def test_get_provider_local_returns_local_provider(tmp_path):
    settings = make_settings(tmp_path, embedding_provider="local")
    provider = get_provider(settings)
    assert type(provider).__name__ == "LocalEmbeddingProvider"


def test_get_provider_local_is_embedding_provider(tmp_path):
    settings = make_settings(tmp_path, embedding_provider="local")
    provider = get_provider(settings)
    assert isinstance(provider, EmbeddingProvider)


def test_get_provider_unknown_provider_raises(tmp_path):
    settings = make_settings(tmp_path, embedding_provider="bogus")
    with pytest.raises((ValueError, NotImplementedError)):
        get_provider(settings)


# --- LocalEmbeddingProvider unit tests (mocked engine) ---

class _FakeLocalEngine:
    """Deterministic 768-dim engine that mirrors the real ONNX output contract."""

    dimension = 768

    def embed_text_ids(self, input_ids, attention_mask):
        rng = np.random.default_rng(abs(hash(input_ids.tobytes())) % (2**32))
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def embed_multimodal(self, input_ids, attention_mask, image_features=None, video_features=None, audio_features=None):
        seed = 0
        if image_features is not None and len(image_features):
            seed += int(image_features.shape[0])
        if video_features is not None and len(video_features):
            seed += int(video_features.shape[0]) * 7
        if audio_features is not None and len(audio_features):
            seed += int(audio_features.shape[0]) * 13
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def embed_pixel_values(self, pixel_values, pixel_position_ids):
        rng = np.random.default_rng(pixel_values.shape[1])
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def embed_audio_features(self, input_features, input_features_mask):
        rng = np.random.default_rng(input_features.shape[1])
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def empty_media(self):
        return np.zeros((0, 512), dtype=np.float32)


def _make_local_provider(tmp_path, **overrides):
    from ragpoc.embeddings_local import LocalEmbeddingProvider

    settings = make_settings(tmp_path, embedding_provider="local", **overrides)
    provider = LocalEmbeddingProvider(settings)
    provider._engine = _FakeLocalEngine()
    provider._tokenizer = _FakeTokenizer()
    provider._image_processor = _FakeImageProcessor()
    provider._audio_extractor = _FakeAudioExtractor()
    provider._loaded = True
    return provider


class _FakeTokenizer:
    def __call__(self, text, return_tensors=None):
        ids = np.array([[2] + [ord(c) % 1000 for c in text[:50]] + [1]], dtype=np.int64)
        mask = np.ones_like(ids)
        return {"input_ids": ids, "attention_mask": mask}


class _FakeImageProcessor:
    max_soft_tokens = 280

    def __call__(self, images=None, return_tensors=None):
        n = 2520
        return {
            "pixel_values": np.zeros((1, n, 768), dtype=np.float32),
            "image_position_ids": np.zeros((1, n, 2), dtype=np.int64),
            "num_soft_tokens_per_image": np.array([266]),
        }


class _FakeAudioExtractor:
    def __call__(self, audio, sampling_rate=None, return_tensors=None):
        frames = max(1, len(audio[0]) // 160) if audio else 10
        return {
            "input_features": np.zeros((1, frames, 128), dtype=np.float32),
            "input_features_mask": np.ones((1, frames), dtype=bool),
        }


def test_local_embed_texts_returns_normalized_vectors(tmp_path):
    provider = _make_local_provider(tmp_path)
    vectors = asyncio.run(provider.embed_texts(["hello", "world"]))
    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == 768
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4


def test_local_embed_image_returns_normalized_vector(tmp_path):
    from PIL import Image

    provider = _make_local_provider(tmp_path)
    img_path = tmp_path / "test.png"
    Image.new("RGB", (100, 80), color=(255, 0, 0)).save(img_path)
    vec = asyncio.run(provider.embed_image(img_path))
    assert len(vec) == 768
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4


def test_local_embed_audio_returns_normalized_vector(tmp_path):
    from unittest.mock import patch

    provider = _make_local_provider(tmp_path)
    wav_path = tmp_path / "test.wav"
    _write_wav(wav_path, 0.5)
    with patch.object(
        type(provider), "_ffmpeg_resample_audio", return_value=np.zeros(8000, dtype=np.float32)
    ):
        vec = asyncio.run(provider.embed_audio(wav_path))
    assert len(vec) == 768
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4


def test_local_embed_video_returns_normalized_vector(tmp_path):
    from unittest.mock import patch

    from PIL import Image

    provider = _make_local_provider(tmp_path)
    vid_path = tmp_path / "test.mp4"
    _write_dummy_mp4(vid_path)
    fake_frames = [Image.new("RGB", (100, 80), color=(0, 128, 255)) for _ in range(4)]
    with patch.object(type(provider), "_ffmpeg_extract_frames", return_value=fake_frames):
        vec = asyncio.run(provider.embed_video(vid_path))
    assert len(vec) == 768
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4


def test_local_provider_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.embedding_provider == "openrouter"
    assert settings.local_embedding_dtype == "q8"
    assert settings.local_embedding_device == "auto"
    assert settings.local_embedding_dim == 768
    assert settings.local_embedding_vision_tokens == 140


def test_local_provider_dimension_property(tmp_path):
    provider = _make_local_provider(tmp_path, local_embedding_dim=512)
    assert provider.dimension == 512


# --- helpers ---

def _write_wav(path: Path, seconds: float):
    import struct
    import wave

    rate = 16000
    n = int(rate * seconds)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n}h", *([0] * n)))


def _write_dummy_mp4(path: Path):
    """Write a minimal file the provider can detect as .mp4 (content is stubbed in tests)."""
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
