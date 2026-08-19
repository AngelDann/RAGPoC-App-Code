import subprocess
import sys
from pathlib import Path

from ragpoc import updater
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
        # A brief pause before relaunch, so a transient lock/scan on the file that was JUST
        # written (UPX-packed builds draw more AV/EDR attention -- see ragpoc.spec) has time to
        # clear before the bootloader's self-extraction tries to load the embedded Python DLL
        # from it. Must sit between the install and the relaunch, not before or after.
        install_pos = content.index(f'move /y "{new_exe}" "{current_exe}"')
        start_pos = content.index(f'start "" "{current_exe}"')
        delay_pos = content.index("ping -n 4 127.0.0.1")
        assert install_pos < delay_pos < start_pos
    finally:
        script_path.unlink(missing_ok=True)


def test_updater_script_avoids_commands_that_need_a_console():
    # This script runs under a windowless cmd.exe, where two constructs are fatal or useless:
    # a `|` pipeline (cmd re-launches itself per stage and dies if it can't -- the old
    # `tasklist | find` wait loop killed the script on line one, so no swap ever happened and a
    # stray _new.exe was left behind every time), and `timeout` (needs a console stdin, exits
    # instantly otherwise, making every retry pause a busy spin).
    script_path = _write_updater_script(
        pid=4242,
        current_exe=Path("C:/apps/RAGPoC/RAGPoC.exe"),
        new_exe=Path("C:/apps/RAGPoC/RAGPoC_new.exe"),
    )
    try:
        content = script_path.read_text(encoding="utf-8")
        assert "|" not in content
        assert "timeout" not in content
        assert "ping -n 2 127.0.0.1" in content
        # An early death must still leave a trace, which the old script couldn't do: every one of
        # its log lines sat downstream of the pipeline that was killing it.
        assert "swap script started" in content.split(":wait")[0]
    finally:
        script_path.unlink(missing_ok=True)


def test_apply_update_launches_swap_script_with_a_console(monkeypatch, tmp_path):
    # DETACHED_PROCESS leaves cmd.exe with no console at all, which is what broke the swap
    # script's pipelines; CREATE_NO_WINDOW keeps the updater invisible while still giving it one.
    # The std handles must be pinned to DEVNULL too -- a windowed build's own handles are invalid
    # and children inherit them by default.
    fake_exe = tmp_path / "RAGPoC.exe"
    fake_exe.write_text("fake")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield b"new exe payload"

    monkeypatch.setattr(updater.httpx, "stream", lambda *a, **kw: _FakeResponse())
    monkeypatch.setattr(updater.threading, "Timer", lambda *a, **kw: type("T", (), {"start": lambda self: None})())

    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return None

    monkeypatch.setattr(updater.subprocess, "Popen", fake_popen)

    updater.apply_update("https://github.com/AngelDann/RAGPoC-App-Code/releases/download/v9/RAGPoC.exe")

    flags = captured["kwargs"]["creationflags"]
    assert flags & subprocess.CREATE_NO_WINDOW
    assert not flags & subprocess.DETACHED_PROCESS
    assert captured["kwargs"]["stdin"] == subprocess.DEVNULL
    assert captured["kwargs"]["stdout"] == subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] == subprocess.DEVNULL
    Path(captured["args"][2]).unlink(missing_ok=True)


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
