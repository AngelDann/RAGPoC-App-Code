from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse

import httpx
from packaging.version import Version

from ragpoc import __version__
from ragpoc.http import SSL_CONTEXT
from ragpoc.http import get as http_get

GITHUB_REPO = "AngelDann/RAGPoC-App-Code"
_RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_ALLOWED_DOWNLOAD_HOSTS = {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}


class UpdateError(RuntimeError):
    pass


def check_for_update() -> dict:
    """Queries GitHub Releases for the latest published version. Works whether or not the app
    is frozen (only apply_update() requires a frozen build), so it degrades gracefully when run
    from source -- update_available will just never fire since __version__ tracks HEAD there."""
    response = http_get(_RELEASES_API, timeout=10, headers={"Accept": "application/vnd.github+json"})
    response.raise_for_status()
    data = response.json()
    latest_tag = (data.get("tag_name") or "").lstrip("v")
    asset = next((a for a in data.get("assets", []) if a["name"].lower().endswith(".exe")), None)

    available = bool(latest_tag) and Version(latest_tag) > Version(__version__)
    return {
        "current_version": __version__,
        "latest_version": latest_tag or __version__,
        "update_available": available,
        "download_url": asset["browser_download_url"] if asset else None,
        "release_notes": data.get("body") or "",
        "release_url": data.get("html_url"),
    }


def apply_update(download_url: str) -> None:
    """Downloads the new .exe next to the running one, then hands off to a detached batch
    script that waits for this process to exit, replaces the executable, and relaunches it.
    Windows won't let a running process overwrite its own .exe file, hence the external helper.
    Only touches the executable itself -- data/ and .env live alongside it, untouched."""
    if not getattr(sys, "frozen", False):
        raise UpdateError("El auto-actualizador solo funciona en la build compilada (.exe).")
    if urlparse(download_url).hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise UpdateError("download_url debe apuntar a un asset de GitHub Releases.")

    current_exe = Path(sys.executable).resolve()
    new_exe = current_exe.with_name(current_exe.stem + "_new.exe")

    with httpx.stream("GET", download_url, timeout=180, follow_redirects=True, verify=SSL_CONTEXT) as response:
        response.raise_for_status()
        with new_exe.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    script_path = _write_updater_script(os.getpid(), current_exe, new_exe)
    subprocess.Popen(
        ["cmd", "/c", str(script_path)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        close_fds=True,
    )
    # Give the current HTTP response time to flush back to the browser/webview before this
    # process (and the uvicorn thread serving it) disappears out from under the connection.
    threading.Timer(1.5, lambda: os._exit(0)).start()


def _write_updater_script(pid: int, current_exe: Path, new_exe: Path) -> Path:
    # A single unretried `move` of new_exe straight over current_exe kept losing a race against
    # Windows/antivirus still holding the just-exited process's own .exe file locked: it failed
    # silently (nothing checked its exit code), the app just kept launching the untouched old
    # exe while a "_new.exe" sat next to it forever, and the next update attempt then treated
    # THAT stale copy as current_exe -- one silent failure compounding into a pile of duplicate
    # binaries. A retry loop around that same overwrite helped but wasn't enough: it can still
    # lose if the lock outlasts the retry window (large exe under antivirus scan, etc.).
    #
    # The robust fix (the pattern mature Windows auto-updaters use, e.g. Squirrel.Windows) is to
    # never overwrite an in-use file at all: Windows allows *renaming* a running process's own
    # exe even while it's still mapped/executing (unlike overwriting it), so current_exe is
    # renamed out of the way first -- which almost always succeeds immediately -- and only then
    # is new_exe moved into the now-free canonical path. That second move can still lose a lock
    # race though: new_exe just landed from the internet and is unsigned (no Authenticode signing
    # in release.yml), which is exactly what Windows Defender/SmartScreen scans on-write, so it
    # gets its own retry loop too. The renamed-away old exe is deleted best-effort; if that one
    # step still fails because of a lingering handle, main() sweeps up any leftover *.exe.old
    # (or, if the second move exhausts its retries, *_new.exe) on the next app start (see
    # cleanup_stale_update_files), once that lock is long gone.
    log_path = current_exe.with_name("ragpoc.log")
    old_backup = current_exe.with_suffix(current_exe.suffix + ".old")
    script_path = Path(tempfile.gettempdir()) / "ragpoc_update.bat"
    script_path.write_text(
        "@echo off\r\n"
        ":wait\r\n"
        f'tasklist /fi "PID eq {pid}" | find "{pid}" >nul\r\n'
        "if not errorlevel 1 (\r\n"
        "  timeout /t 1 /nobreak >nul\r\n"
        "  goto wait\r\n"
        ")\r\n"
        f'if exist "{old_backup}" del /f /q "{old_backup}" >nul 2>&1\r\n'
        "set attempts=0\r\n"
        ":rename_retry\r\n"
        f'move /y "{current_exe}" "{old_backup}" >nul 2>&1\r\n'
        "if errorlevel 1 (\r\n"
        "  set /a attempts+=1\r\n"
        "  if %attempts% LSS 15 (\r\n"
        "    timeout /t 1 /nobreak >nul\r\n"
        "    goto rename_retry\r\n"
        "  )\r\n"
        f'  echo [%date% %time%] Update failed: could not rename "{current_exe}" out of the way after 15 attempts >> "{log_path}"\r\n'
        "  goto relaunch\r\n"
        ")\r\n"
        "set attempts=0\r\n"
        ":install_retry\r\n"
        f'move /y "{new_exe}" "{current_exe}" >nul 2>&1\r\n'
        "if errorlevel 1 (\r\n"
        "  set /a attempts+=1\r\n"
        "  if %attempts% LSS 15 (\r\n"
        "    timeout /t 1 /nobreak >nul\r\n"
        "    goto install_retry\r\n"
        "  )\r\n"
        f'  echo [%date% %time%] Update failed: could not move new exe into place after 15 attempts, restoring previous version >> "{log_path}"\r\n'
        f'  move /y "{old_backup}" "{current_exe}" >nul 2>&1\r\n'
        "  goto relaunch\r\n"
        ")\r\n"
        f'del /f /q "{old_backup}" >nul 2>&1\r\n'
        ":relaunch\r\n"
        f'start "" "{current_exe}"\r\n'
        f'del "%~f0"\r\n',
        encoding="utf-8",
    )
    return script_path


def cleanup_stale_update_files() -> None:
    """Best-effort sweep for *.exe.old and *_new.exe leftovers from an update whose final
    cleanup step lost a lock race (see _write_updater_script) -- called on every startup of the
    frozen build, by which point whatever process held the file locked has long since exited.
    Never raises: this is opportunistic housekeeping, not something that should ever block
    startup. Safe to run unconditionally: it only fires after the app is already up and running
    again, i.e. never while an update is actually in flight."""
    if not getattr(sys, "frozen", False):
        return
    try:
        current_exe = Path(sys.executable).resolve()
        for pattern in ("*.exe.old", "*_new.exe"):
            for stale in current_exe.parent.glob(pattern):
                try:
                    stale.unlink()
                except OSError:
                    pass
    except OSError:
        pass
