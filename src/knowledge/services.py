from __future__ import annotations

import sqlite3

from asgiref.sync import sync_to_async

from knowledge.settings_store import get_effective_settings as get_settings
from ragpoc.config import Settings
from ragpoc.db import initialize_database
from ragpoc.embeddings import EmbeddingProvider, get_provider
from ragpoc.ingestion import Ingestor
from ragpoc.retrieval import Retriever


async def _embedding_usage_hook(payload: dict) -> None:
    from knowledge.usage import record_usage

    await sync_to_async(record_usage)(
        category="embedding",
        action=f"embed_{payload.get('kind', 'text')}",
        model=payload.get("model", ""),
        input_tokens=payload.get("input_tokens", 0),
        total_tokens=payload.get("total_tokens", 0),
        duration_ms=payload.get("duration_ms"),
        status=payload.get("status", "success"),
        error_message=payload.get("error_message", ""),
        request_count=payload.get("count", 1),
        metadata={"kind": payload.get("kind")},
    )


class RAGCoreService:
    def __init__(self, settings: Settings | None = None, provider: EmbeddingProvider | None = None):
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.provider = provider or get_provider(self.settings, usage_hook=_embedding_usage_hook)
        self._connection: sqlite3.Connection | None = None
        self._ingestor: Ingestor | None = None
        self._retriever: Retriever | None = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = initialize_database(self.settings.database_path, self.settings.embedding_dimension)
        return self._connection

    @property
    def ingestor(self) -> Ingestor:
        if self._ingestor is None:
            self._ingestor = Ingestor(self.connection, self.settings, self.provider)
        return self._ingestor

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.connection, self.provider, self.settings.max_top_k)
        return self._retriever


_global_service: RAGCoreService | None = None


def get_rag_service(settings: Settings | None = None, provider: EmbeddingProvider | None = None) -> RAGCoreService:
    global _global_service
    if settings is not None or provider is not None:
        return RAGCoreService(settings, provider)
    if _global_service is None:
        _global_service = RAGCoreService()
    return _global_service


def reset_rag_service() -> None:
    """Drop the cached service so the next call rebuilds it (e.g. after the API key changes)."""
    global _global_service
    _global_service = None
