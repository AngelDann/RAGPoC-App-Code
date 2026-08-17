import asyncio
from pathlib import Path

from PIL import Image

from ragpoc.config import Settings
from ragpoc.db import initialize_database
from ragpoc.embeddings import FakeEmbeddingProvider
from ragpoc.ingestion import Ingestor
from ragpoc.retrieval import Retriever


def test_video_result_exposes_timecode_evidence(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"not-a-real-video")
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (12, 12), "green").save(frame)

    class Segment:
        ordinal = 0
        start_seconds = 0.0
        end_seconds = 30.0
        frame_path = frame
        clip_path = frame

    monkeypatch.setattr("ragpoc.ingestion.extract_video_segments", lambda *_: [Segment()])
    settings = Settings(_env_file=None, data_dir=data, allowed_upload_dir=data / "uploads")
    connection = initialize_database(settings.database_path)
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("doc", str(source), source.name, "video", "test", 0, "pending", "now", None, None),
    )
    connection.commit()
    provider = FakeEmbeddingProvider()
    ingestor = Ingestor(connection, settings, provider)
    asyncio.run(ingestor._video("doc", source))
    connection.execute("UPDATE documents SET status = 'indexed' WHERE id = 'doc'")
    connection.commit()

    result = asyncio.run(Retriever(connection, provider).search("green", media_type=None))
    assert result[0]["metadata"]["start_timecode"] == "00:00:00"
    assert result[0]["metadata"]["end_timecode"] == "00:00:30"
    assert result[0]["start_seconds"] == 0.0
    assert result[0]["end_seconds"] == 30.0
