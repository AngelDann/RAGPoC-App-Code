from __future__ import annotations

import sys
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if getattr(sys, "frozen", False):
    # PyInstaller build: anchor to the .exe's own folder, not the bootloader's
    # extraction tempdir (_MEIPASS) or an arbitrary launch cwd, so data/.env
    # travel with the executable no matter how/where it's started.
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _anchor(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        env_prefix="RAGPOC_",
        extra="ignore",
        # API keys carry a validation_alias so they can be supplied as env vars
        # without the RAGPOC_ prefix. Without populate_by_name the field name is
        # rejected as an input key, so programmatic construction (tests, presets,
        # the settings overlay) silently dropped the credential.
        populate_by_name=True,
    )

    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    google_api_key: str | None = Field(default=None, validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"))
    llm_provider: str = "openrouter"  # "openrouter" | "google"
    ui_username: str = "operator"
    ui_password: str | None = None
    desktop_mode: bool = False
    embedding_provider: str = "openrouter"  # "openrouter" | "google" | "local"
    embedding_model: str = "google/gemini-embedding-2-preview"
    local_embedding_model: str = "onnx-community/embeddinggemma-2-ONNX"
    local_embedding_dtype: str = "q8"  # "q4" | "q8" | "fp16" | "fp32"
    local_embedding_device: str = "auto"  # "auto" | "dml" | "cpu"
    local_embedding_dim: int | None = 768  # MRL truncation: 768, 512, 256, 128
    local_embedding_vision_tokens: int = Field(default=140, ge=70, le=1120)
    local_embedding_cache_dir: Path = Field(default_factory=lambda: BASE_DIR / "data" / "models")

    # --- local chat LLM (ONNX Runtime GenAI) ---
    # "openrouter" | "google" | "local" is selected via llm_provider; these tune
    # the local path only.
    local_chat_model: str = "qwen3-vl-2b"
    local_chat_dtype: str = "q4f16"  # informational: the GenAI trees are int4
    local_chat_device: str = "auto"  # "auto" | "dml" | "cpu"
    # Hard cap on the prompt window. The exported genai_config advertises 262144
    # with past_present_share_buffer, which would pre-allocate a KV cache far
    # past the memory budget; this value is what actually gets applied.
    local_chat_context_length: int = Field(default=2048, ge=512, le=32768)
    local_chat_max_new_tokens: int = Field(default=512, ge=32, le=8192)
    local_chat_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    local_chat_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    local_chat_top_k: int = Field(default=40, ge=0, le=200)
    agent_max_tool_rounds: int = Field(default=3, ge=0, le=12)
    memory_budget_mb: int = Field(default=3072, ge=512, le=32768)
    local_chat_cache_dir: Path = Field(default_factory=lambda: BASE_DIR / "data" / "models")

    chat_model: str = "google/gemini-3.7-flash"
    image_model: str = "google/gemini-3.1-flash-image"
    tts_model: str = "google/gemini-3.1-flash-tts-preview"
    tts_voice_a: str = "Kore"
    tts_voice_b: str = "Puck"
    web_search_engine: str = "duckduckgo"
    searxng_url: str = ""
    web_search_deep_read: bool = True
    embedding_dimension: int | None = None
    data_dir: Path = Field(default_factory=lambda: BASE_DIR / "data")
    max_pdf_bytes: int = 524_288_000
    max_video_seconds: int = 7_200
    render_concurrency: int = Field(default=2, ge=1, le=8)
    embedding_concurrency: int = Field(default=4, ge=1, le=16)
    embedding_batch_size: int = Field(default=32, ge=1, le=128)
    pipeline_queue_size: int = Field(default=16, ge=1, le=128)
    video_segment_seconds: int = Field(default=30, ge=1, le=300)
    pdf_render_dpi: int = Field(default=180, ge=72, le=300)
    max_top_k: int = Field(default=50, ge=1, le=100)
    allowed_upload_dir: Path = Field(default_factory=lambda: BASE_DIR / "data" / "uploads")

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def empty_dimension_means_auto(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator(
        "data_dir",
        "allowed_upload_dir",
        "local_embedding_cache_dir",
        "local_chat_cache_dir",
        mode="after",
    )
    @classmethod
    def anchor_relative_paths(cls, value: Path) -> Path:
        # A relative override (e.g. RAGPOC_DATA_DIR=mydata in .env) should still
        # resolve next to the executable/project root, not the launch cwd.
        return _anchor(value)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "ragpoc.db"

    @property
    def local_model_dir(self) -> Path:
        return self.local_embedding_cache_dir / "embeddinggemma2-onnx"

    @property
    def local_chat_model_dir(self) -> Path:
        return self.local_chat_cache_dir / self.local_chat_model

    @property
    def renders_dir(self) -> Path:
        return self.data_dir / "renders"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "derived" / "video_frames"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "derived" / "artifacts"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.allowed_upload_dir, self.renders_dir, self.frames_dir, self.artifacts_dir, self.local_embedding_cache_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
