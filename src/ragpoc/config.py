from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field, field_validator
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
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", env_prefix="RAGPOC_", extra="ignore"
    )

    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    ui_username: str = "operator"
    ui_password: str | None = None
    desktop_mode: bool = False
    embedding_model: str = "google/gemini-embedding-2-preview"
    chat_model: str = "google/gemini-3.7-flash"
    image_model: str = "google/gemini-3.1-flash-image"
    tts_model: str = "google/gemini-3.1-flash-tts-preview"
    tts_voice_a: str = "Kore"
    tts_voice_b: str = "Puck"
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

    @field_validator("data_dir", "allowed_upload_dir", mode="after")
    @classmethod
    def anchor_relative_paths(cls, value: Path) -> Path:
        # A relative override (e.g. RAGPOC_DATA_DIR=mydata in .env) should still
        # resolve next to the executable/project root, not the launch cwd.
        return _anchor(value)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "ragpoc.db"

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
        for path in (self.data_dir, self.allowed_upload_dir, self.renders_dir, self.frames_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
