from pathlib import Path

from ragpoc.chunking import ApproximateTokenCounter, chunk_text
from ragpoc.config import BASE_DIR, Settings
from ragpoc.db import configured_dimension, initialize_database, persist_dimension
from ragpoc.updater import _write_updater_script


def test_settings_default_to_local_data_directory():
    settings = Settings(_env_file=None)
    # Anchored to BASE_DIR (the .exe's folder when frozen, project root in dev)
    # rather than the launch cwd, so data/.env travel with the app.
    assert settings.database_path == BASE_DIR / "data" / "ragpoc.db"
    assert settings.allowed_upload_dir == BASE_DIR / "data" / "uploads"


def test_database_persists_embedding_dimension(tmp_path):
    connection = initialize_database(tmp_path / "ragpoc.db")
    persist_dimension(connection, 3)
    assert configured_dimension(connection) == 3
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    assert {"documents", "chunks", "chunks_fts", "chunk_vectors"} <= tables


def test_text_is_chunked_near_target_with_overlap():
    counter = ApproximateTokenCounter()
    text = "\n\n".join(f"Sentence {i}. " * 12 for i in range(12))
    chunks = chunk_text(text, counter, target_tokens=80, overlap_tokens=15)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 110 for chunk in chunks)
    assert chunks[0].content.split("\n\n")[-1] in chunks[1].content


def test_empty_text_has_no_chunks():
    assert chunk_text(" \n ", ApproximateTokenCounter()) == []


def test_updater_script_retries_the_exe_swap_before_giving_up():
    # A bare, unretried `move` lost the race against Windows still holding the just-exited
    # process's own .exe locked, so it silently kept running the old exe forever while a
    # "_new.exe" piled up unused next to it -- the script must retry instead of trying once.
    current_exe = Path("C:/apps/RAGPoC/RAGPoC.exe")
    new_exe = Path("C:/apps/RAGPoC/RAGPoC_new.exe")
    script_path = _write_updater_script(pid=4242, current_exe=current_exe, new_exe=new_exe)
    try:
        content = script_path.read_text(encoding="utf-8")
        assert content.count(f'move /y "{new_exe}" "{current_exe}"') == 1
        assert ":move_retry" in content
        assert "goto move_retry" in content
        assert f'start "" "{current_exe}"' in content
    finally:
        script_path.unlink(missing_ok=True)
