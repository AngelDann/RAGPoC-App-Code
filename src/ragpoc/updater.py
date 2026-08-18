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
    # Windows can keep an exited process's own .exe file locked (memory-mapped image unmap,
    # antivirus real-time scan of the freshly-downloaded new_exe, etc.) for a moment after
    # tasklist already stops listing its PID -- a single unretried `move` can lose that race,
    # and since nothing here checked its exit code, the failure was silent: the app just kept
    # launching the untouched old exe while a "_new.exe" sat next to it forever. Next update
    # attempt then treats THAT stale copy as current_exe and downloads a "_new_new.exe" on
    # top of it, compounding one silent failure into a pile of duplicate binaries. Retrying
    # the move for up to ~15s covers that race; a log line on the rare case it still fails
    # makes the failure discoverable instead of silent.
    log_path = current_exe.with_name("ragpoc.log")
    script_path = Path(tempfile.gettempdir()) / "ragpoc_update.bat"
    script_path.write_text(
        "@echo off\r\n"
        ":wait\r\n"
        f'tasklist /fi "PID eq {pid}" | find "{pid}" >nul\r\n'
        "if not errorlevel 1 (\r\n"
        "  timeout /t 1 /nobreak >nul\r\n"
        "  goto wait\r\n"
        ")\r\n"
        "set attempts=0\r\n"
        ":move_retry\r\n"
        f'move /y "{new_exe}" "{current_exe}" >nul 2>&1\r\n'
        "if errorlevel 1 (\r\n"
        "  set /a attempts+=1\r\n"
        "  if %attempts% LSS 15 (\r\n"
        "    timeout /t 1 /nobreak >nul\r\n"
        "    goto move_retry\r\n"
        "  )\r\n"
        f'  echo [%date% %time%] Update swap failed after 15 attempts: could not move "{new_exe}" over "{current_exe}" >> "{log_path}"\r\n'
        ")\r\n"
        f'start "" "{current_exe}"\r\n'
        f'del "%~f0"\r\n',
        encoding="utf-8",
    )
    return script_path
