import sys
from pathlib import Path

from ragpoc.chunking import ApproximateTokenCounter, chunk_text
from ragpoc.config import BASE_DIR, Settings
from ragpoc.db import configured_dimension, initialize_database, persist_dimension
from ragpoc.updater import _write_updater_script, cleanup_stale_update_files


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


def test_updater_script_renames_old_exe_out_of_the_way_before_placing_the_new_one():
    # Overwriting current_exe directly kept losing a race against Windows/antivirus still
    # holding the just-exited process's own .exe locked, silently: the app just kept running
    # the untouched old exe while a "_new.exe" piled up unused next to it forever. Renaming
    # current_exe out of the way first (Windows allows renaming an in-use exe, unlike
    # overwriting it) then moving new_exe into the now-free path avoids that race entirely.
    current_exe = Path("C:/apps/RAGPoC/RAGPoC.exe")
    new_exe = Path("C:/apps/RAGPoC/RAGPoC_new.exe")
    old_backup = Path("C:/apps/RAGPoC/RAGPoC.exe.old")
    script_path = _write_updater_script(pid=4242, current_exe=current_exe, new_exe=new_exe)
    try:
        content = script_path.read_text(encoding="utf-8")
        assert f'move /y "{current_exe}" "{old_backup}"' in content
        assert f'move /y "{new_exe}" "{current_exe}"' in content
        assert ":rename_retry" in content
        assert "goto rename_retry" in content
        # The second move (placing new_exe into position) races the same AV-lock hazard as the
        # first -- new_exe just landed from the internet unsigned, which is exactly what
        # Windows Defender scans on-write -- so it needs its own retry loop too, not just the
        # first rename.
        assert ":install_retry" in content
        assert "goto install_retry" in content
        # A failed new-exe swap must roll back to the renamed-away original, not leave the
        # app pointing at nothing.
        assert f'move /y "{old_backup}" "{current_exe}"' in content
        assert f'start "" "{current_exe}"' in content
    finally:
        script_path.unlink(missing_ok=True)


def test_cleanup_stale_update_files_removes_leftover_old_exe(tmp_path, monkeypatch):
    fake_exe = tmp_path / "RAGPoC.exe"
    fake_exe.write_text("fake")
    stale_old = tmp_path / "RAGPoC.exe.old"
    stale_old.write_text("stale leftover from a previous update")
    # Leftover from a failed install-retry (see _write_updater_script): the rename-away of
    # current_exe succeeded but the AV-locked move of new_exe into place exhausted its
    # retries, so this file was never cleaned up by the updater script itself.
    stale_new = tmp_path / "RAGPoC_new.exe"
    stale_new.write_text("stale leftover download from a failed update")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    cleanup_stale_update_files()

    assert not stale_old.exists()
    assert not stale_new.exists()
    assert fake_exe.exists()  # only the leftovers are swept, never the real exe
