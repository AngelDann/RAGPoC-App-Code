from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import numpy as np

from ragpoc.config import Settings

_DTYPE_SUFFIX = {
    "fp32": "",
    "fp16": "_fp16",
    "q4": "_q4",
    "q4f16": "_q4f16",
    "q8": "_quantized",
}

_MRL_DIMS = {128, 256, 512, 768}


class LocalEmbeddingError(RuntimeError):
    pass


class _OnnxEngine:
    """Thin wrapper around three ONNX Runtime sessions (text / vision / audio)."""

    def __init__(self, model_dir: Path, dtype: str, device: str):
        self.model_dir = model_dir
        self.dtype = dtype
        self.device = device
        self.dimension = 768
        self._text = None
        self._vision = None
        self._audio = None

    # -- session management --

    def _providers(self, prefer_dml: bool) -> list[str]:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if prefer_dml and self.device in ("auto", "dml") and "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _opts():
        import onnxruntime as ort

        o = ort.SessionOptions()
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return o

    def _load(self, name: str, prefer_dml: bool):
        import onnxruntime as ort

        suffix = _DTYPE_SUFFIX.get(self.dtype, "")
        path = self.model_dir / "onnx" / f"{name}{suffix}.onnx"
        if not path.exists():
            raise LocalEmbeddingError(f"ONNX model file not found: {path}")
        return ort.InferenceSession(
            str(path),
            providers=self._providers(prefer_dml),
            sess_options=self._opts(),
        )

    @property
    def text(self):
        if self._text is None:
            self._text = self._load("model", prefer_dml=True)
        return self._text

    @property
    def vision(self):
        if self._vision is None:
            # Vision encoder has a DML Reshape incompatibility; force CPU.
            self._vision = self._load("vision_encoder", prefer_dml=False)
        return self._vision

    @property
    def audio(self):
        if self._audio is None:
            self._audio = self._load("audio_encoder", prefer_dml=False)
        return self._audio

    # -- inference --

    def embed_text_ids(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        empty = np.zeros((0, 512), dtype=np.float32)
        out = self.text.run(
            ["sentence_embedding"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_features": empty,
                "video_features": empty,
                "audio_features": empty,
            },
        )
        return out[0][0]

    def embed_multimodal(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        image_features: np.ndarray | None = None,
        video_features: np.ndarray | None = None,
        audio_features: np.ndarray | None = None,
    ) -> np.ndarray:
        empty = np.zeros((0, 512), dtype=np.float32)
        out = self.text.run(
            ["sentence_embedding"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_features": image_features if image_features is not None else empty,
                "video_features": video_features if video_features is not None else empty,
                "audio_features": audio_features if audio_features is not None else empty,
            },
        )
        return out[0][0]

    def embed_pixel_values(
        self, pixel_values: np.ndarray, pixel_position_ids: np.ndarray
    ) -> np.ndarray:
        out = self.vision.run(
            ["image_features"],
            {
                "pixel_values": pixel_values.astype(np.float32),
                "pixel_position_ids": pixel_position_ids,
            },
        )
        return out[0]

    def embed_audio_features(
        self, input_features: np.ndarray, input_features_mask: np.ndarray
    ) -> np.ndarray:
        out = self.audio.run(
            ["audio_features"],
            {
                "input_features": input_features.astype(np.float32),
                "input_features_mask": input_features_mask.astype(bool),
            },
        )
        return out[0]


class LocalEmbeddingProvider:
    """Multimodal embedding provider backed by EmbeddingGemma 2 via ONNX Runtime.

    Implements the ``EmbeddingProvider`` protocol (texts, image, video, audio)
    entirely on-device.  Text runs on DirectML (GPU) when available; the vision
    and audio encoders fall back to CPU due to a DirectML Reshape limitation
    in the vision encoder graph.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine: _OnnxEngine | None = None
        self._tokenizer = None
        self._image_processor = None
        self._audio_extractor = None
        self._loaded = False

    # -- lazy setup --

    @property
    def dimension(self) -> int:
        return self.settings.local_embedding_dim or 768

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        model_dir = self.settings.local_model_dir
        if not (model_dir / "tokenizer.json").exists():
            self._download_model(model_dir)
        self._tokenizer = self._load_tokenizer(model_dir)
        self._image_processor = self._load_image_processor(model_dir)
        self._audio_extractor = self._load_audio_extractor(model_dir)
        self._engine = _OnnxEngine(
            model_dir, self.settings.local_embedding_dtype, self.settings.local_embedding_device
        )
        self._loaded = True

    @staticmethod
    def _download_model(target: Path) -> None:
        from huggingface_hub import snapshot_download

        target.mkdir(parents=True, exist_ok=True)
        suffix = _DTYPE_SUFFIX.get("q8", "_quantized")
        patterns = [
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "config_sentence_transformers.json",
        ]
        for name in ("model", "vision_encoder", "audio_encoder"):
            patterns.append(f"onnx/{name}{suffix}.onnx")
            patterns.append(f"onnx/{name}{suffix}.onnx_data")
        snapshot_download(
            "onnx-community/embeddinggemma-2-ONNX",
            repo_type="model",
            local_dir=str(target),
            allow_patterns=patterns,
        )

    @staticmethod
    def _load_tokenizer(model_dir: Path):
        from transformers import AutoTokenizer

        # `extra_special_tokens={}` overrides the repo's legacy list-form config key: recent
        # transformers (>=4.57) assumes a dict there and crashes on the shipped list. The
        # special tokens themselves live in tokenizer.json's added_tokens and still resolve.
        return AutoTokenizer.from_pretrained(str(model_dir), extra_special_tokens={})

    @staticmethod
    def _load_image_processor(model_dir: Path):
        from transformers.models.gemma4.image_processing_pil_gemma4_fast import (
            Gemma4ImageProcessorPil,
        )

        proc = Gemma4ImageProcessorPil.from_pretrained(str(model_dir))
        return proc

    @staticmethod
    def _load_audio_extractor(model_dir: Path):
        from transformers import Gemma4AudioFeatureExtractor

        return Gemma4AudioFeatureExtractor.from_pretrained(str(model_dir))

    # -- media helpers --

    @staticmethod
    def _ffmpeg_resample_audio(src: Path) -> np.ndarray:
        """Decode *src* to mono 16 kHz float32 PCM via ffmpeg."""
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error",
            "-i", str(src),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", "16000",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0:
            raise LocalEmbeddingError(
                f"ffmpeg audio decode failed: {proc.stderr.decode('utf-8', errors='replace')[:300]}"
            )
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    @staticmethod
    def _ffmpeg_extract_frames(src: Path, fps: int = 1, max_frames: int = 32) -> list:
        """Extract video frames as PIL Images at *fps*, subsampled to *max_frames*."""
        from PIL import Image

        cmd = [
            "ffmpeg", "-nostdin", "-v", "error",
            "-i", str(src),
            "-vf", f"fps={fps}",
            "-f", "image2pipe", "-vcodec", "ppm",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise LocalEmbeddingError(
                f"ffmpeg frame extraction failed: {proc.stderr.decode('utf-8', errors='replace')[:300]}"
            )
        raw = proc.stdout
        frames = []
        pos = 0
        while pos < len(raw) and len(frames) < max_frames:
            if raw[pos : pos + 2] != b"P6":
                break
            pos += 2
            # skip whitespace + width + whitespace + height + whitespace + maxval + single whitespace
            while pos < len(raw) and raw[pos : pos + 1].isspace():
                pos += 1
            w_start = pos
            while pos < len(raw) and raw[pos : pos + 1].isdigit():
                pos += 1
            width = int(raw[w_start:pos])
            while pos < len(raw) and raw[pos : pos + 1].isspace():
                pos += 1
            h_start = pos
            while pos < len(raw) and raw[pos : pos + 1].isdigit():
                pos += 1
            height = int(raw[h_start:pos])
            while pos < len(raw) and raw[pos : pos + 1].isspace():
                pos += 1
            mv_start = pos
            while pos < len(raw) and raw[pos : pos + 1].isdigit():
                pos += 1
            _maxval = int(raw[mv_start:pos])
            pos += 1  # single whitespace after maxval
            n_bytes = width * height * 3
            pixel_data = raw[pos : pos + n_bytes]
            pos += n_bytes
            if len(pixel_data) < n_bytes:
                break
            frames.append(Image.frombytes("RGB", (width, height), pixel_data))
        if not frames:
            raise LocalEmbeddingError("ffmpeg produced no video frames")
        # uniform subsample if we got more than max_frames
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        return frames

    def _encode_images(self, images: list) -> np.ndarray:
        """Run a list of PIL images through the vision encoder; returns [N, 512]."""
        self._image_processor.max_soft_tokens = self.settings.local_embedding_vision_tokens
        feats = []
        for img in images:
            out = self._image_processor(images=img, return_tensors="np")
            feat = self._engine.embed_pixel_values(
                out["pixel_values"], out["image_position_ids"]
            )
            feats.append(feat)
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, 512), dtype=np.float32)

    def _normalize(self, vec: np.ndarray) -> list[float]:
        dim = self.dimension
        vec = vec[:dim]
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    # -- EmbeddingProvider protocol --

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_texts_sync, texts)

    async def embed_query(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._embed_query_sync, query)

    async def embed_image(self, path: Path) -> list[float]:
        return await asyncio.to_thread(self._embed_image_sync, path)

    async def embed_video(self, path: Path) -> list[float]:
        return await asyncio.to_thread(self._embed_video_sync, path)

    async def embed_audio(self, path: Path) -> list[float]:
        return await asyncio.to_thread(self._embed_audio_sync, path)

    # -- synchronous implementations --

    def _embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        results = []
        for text in texts:
            enc = self._tokenizer(text, return_tensors="np")
            vec = self._engine.embed_text_ids(enc["input_ids"], enc["attention_mask"])
            results.append(self._normalize(vec))
        return results

    def _embed_query_sync(self, query: str) -> list[float]:
        self._ensure_loaded()
        prefixed = f"task: search result | query: {query}"
        enc = self._tokenizer(prefixed, return_tensors="np")
        vec = self._engine.embed_text_ids(enc["input_ids"], enc["attention_mask"])
        return self._normalize(vec)

    def _embed_image_sync(self, path: Path) -> list[float]:
        self._ensure_loaded()
        from PIL import Image

        img = Image.open(path).convert("RGB")
        feats = self._encode_images([img])
        vec = self._engine.embed_multimodal(
            *self._media_token_input("<|image|>"), image_features=feats
        )
        return self._normalize(vec)

    def _embed_video_sync(self, path: Path) -> list[float]:
        self._ensure_loaded()
        frames = self._ffmpeg_extract_frames(
            path,
            fps=1,
            max_frames=min(32, self.settings.video_segment_seconds),
        )
        feats = self._encode_images(frames)
        vec = self._engine.embed_multimodal(
            *self._media_token_input("<|video|>"), video_features=feats
        )
        return self._normalize(vec)

    def _embed_audio_sync(self, path: Path) -> list[float]:
        self._ensure_loaded()
        pcm = self._ffmpeg_resample_audio(path)
        if pcm.size == 0:
            raise LocalEmbeddingError(f"Audio file decoded to zero samples: {path}")
        out = self._audio_extractor([pcm], sampling_rate=16000, return_tensors="np")
        feats = self._engine.embed_audio_features(
            out["input_features"], out["input_features_mask"]
        )
        vec = self._engine.embed_multimodal(
            *self._media_token_input("<|audio|>"), audio_features=feats
        )
        return self._normalize(vec)

    def _media_token_input(self, placeholder: str):
        """Tokenize a bare placeholder so the text model positions the media features."""
        enc = self._tokenizer(placeholder, return_tensors="np")
        return enc["input_ids"], enc["attention_mask"]
