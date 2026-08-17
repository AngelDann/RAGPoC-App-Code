from __future__ import annotations

from fastmcp import FastMCP

from ragpoc.config import get_settings
from ragpoc.db import initialize_database
from ragpoc.embeddings import get_provider
from ragpoc.retrieval import Retriever

mcp = FastMCP("RAGPoC")


def _retriever() -> Retriever:
    settings = get_settings()
    connection = initialize_database(settings.database_path, settings.embedding_dimension)
    return Retriever(connection, get_provider(settings), settings.max_top_k)


@mcp.tool()
async def rag_search(query: str, top_k: int = 8, media_type: str | None = None) -> dict:
    """Search the local RAG store and return traceable source evidence."""
    results = await _retriever().search(query, top_k, media_type)
    return {"results": results}


@mcp.tool()
def rag_get_document(document_id: str) -> dict:
    """Get metadata for one indexed document."""
    result = _retriever().get_document(document_id)
    return result or {"error": "document_not_found", "document_id": document_id}


@mcp.tool()
def rag_get_chunk(chunk_id: str) -> dict:
    """Get one retrieved chunk and its stored provenance."""
    result = _retriever().get_chunk(chunk_id)
    return result or {"error": "chunk_not_found", "chunk_id": chunk_id}


@mcp.tool()
def rag_list_documents(limit: int = 100, offset: int = 0) -> dict:
    """List local indexed document metadata; this tool is read-only."""
    return {"documents": _retriever().list_documents(limit, offset)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
