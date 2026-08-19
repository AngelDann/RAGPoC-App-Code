from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
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
    # A zip of the onedir build's contents (RAGPoC.exe + _internal\), not a standalone .exe --
    # see ragpoc.spec and release.yml. A build from before this switch only ever published a
    # bare .exe and won't find a matching asset here; its update button will report "no download
    # available" rather than corrupt anything, and needs installing by hand once to bootstrap
    # onto the onedir layout, exactly like any other update-mechanism-changing release.
    asset = next((a for a in data.get("assets", []) if a["name"].lower().endswith(".zip")), None)

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
    """Downloads the new build's zip (RAGPoC.exe + _internal\\, see ragpoc.spec) to a staging
    directory, extracts it, then hands off to a detached batch script that waits for this process
    to exit, replaces the install in place, and relaunches it. Windows won't let a running process
    overwrite its own .exe file, hence the external helper. Only touches the exe and _internal\\ --
    data/ and .env live alongside them, untouched."""
    if not getattr(sys, "frozen", False):
        raise UpdateError("El auto-actualizador solo funciona en la build compilada (.exe).")
    if urlparse(download_url).hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise UpdateError("download_url debe apuntar a un asset de GitHub Releases.")

    current_exe = Path(sys.executable).resolve()
    current_dir = current_exe.parent
    tmp_dir = Path(tempfile.gettempdir())
    zip_path = tmp_dir / "ragpoc_update.zip"
    # Cleared and re-extracted on every attempt, so any leftover from an interrupted previous
    # update (crash, power loss) self-heals instead of needing a separate stale-state sweep.
    staging_dir = tmp_dir / "ragpoc_update_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", download_url, timeout=180, follow_redirects=True, verify=SSL_CONTEXT) as response:
        response.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(staging_dir)

    script_path = _write_updater_script(
        pid=os.getpid(),
        current_exe=current_exe,
        staging_exe=staging_dir / current_exe.name,
        current_internal=current_dir / "_internal",
        staging_internal=staging_dir / "_internal",
        staging_dir=staging_dir,
        zip_path=zip_path,
    )
    # CREATE_NO_WINDOW, not DETACHED_PROCESS: detached gives cmd.exe no console at all, and cmd
    # runs every `|` pipeline stage by re-launching itself, which fails without one -- cmd died
    # on the swap script's very first pipeline and nothing after it ever ran (silently, since the
    # failure log line was itself past that point). CREATE_NO_WINDOW still shows no window but
    # gives the child its own console, and an explicitly-detached parent isn't needed for it to
    # outlive us. The DEVNULL handles matter too: a windowed build's own std handles are invalid,
    # and children inherit them by default.
    subprocess.Popen(
        ["cmd", "/c", str(script_path)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    # Give the current HTTP response time to flush back to the browser/webview before this
    # process (and the uvicorn thread serving it) disappears out from under the connection.
    threading.Timer(1.5, lambda: os._exit(0)).start()


def _write_updater_script(
    pid: int,
    current_exe: Path,
    staging_exe: Path,
    current_internal: Path,
    staging_internal: Path,
    staging_dir: Path,
    zip_path: Path,
) -> Path:
    # Two parts to every swap: the bulk of the build (_internal\, dozens of DLLs/data files) and
    # the exe itself. They need different techniques.
    #
    # _internal\ is synced with `robocopy /MIR`: by the time this runs, the old process (whose
    # pid we waited out below) has already exited, so nothing has these DLLs open the way a
    # running process holds its own .exe -- a plain in-place overwrite is fine, no rename-first
    # dance needed. robocopy's /R and /W give it its own built-in retry against a transient
    # lock/scan (e.g. antivirus briefly touching a freshly-written DLL), so no hand-rolled retry
    # loop is needed here either. Its exit codes do NOT follow normal cmd conventions though --
    # 0-7 all mean success (0 = nothing changed, 1 = files copied, etc.), only >=8 is a real
    # failure, hence `if %errorlevel% GEQ 8` rather than `if errorlevel 1`.
    #
    # The exe still needs the rename-first pattern (the pattern mature Windows auto-updaters use,
    # e.g. Squirrel.Windows): Windows allows *renaming* a running process's own exe even while
    # still mapped/executing (unlike overwriting it) -- but that only matters while it's still
    # running, and by this point in the script it no longer is. The rename-first swap is kept
    # anyway since it's already proven reliable and gives a rollback path (old_backup) for free.
    # A single unretried move here previously lost races against Windows/antivirus locks
    # silently (nothing checked its exit code); both moves below have their own retry loop for
    # the same reason robocopy gets one -- new_exe just landed from the internet and is unsigned
    # (no Authenticode signing in release.yml), which is exactly what Defender/SmartScreen scans
    # on-write. The renamed-away old exe is deleted best-effort; if that step fails because of a
    # lingering handle, main() sweeps up any leftover *.exe.old on the next app start (see
    # cleanup_stale_update_files), once that lock is long gone.
    #
    # Two commands are deliberately avoided throughout the script because they need a console,
    # which a background-launched updater can easily end up without (see apply_update):
    #   - `a | b` pipelines: cmd runs each stage by re-launching itself, and dies outright if it
    #     can't. The old `tasklist | find` wait loop killed the script on its first line, which is
    #     why every swap silently no-op'd and left a stray copy behind -- redirect to a temp file
    #     and `find` in that file instead.
    #   - `timeout`: exits immediately with "input redirection is not supported" when stdin isn't
    #     a console, turning every retry pause into a busy spin -- `ping -n 2 127.0.0.1` sleeps
    #     ~1s with no such requirement.
    # Both hand-rolled retry loops re-expand %attempts% correctly despite being inside
    # parenthesised blocks: `goto` makes cmd re-read and re-parse from the label each pass, so
    # the value is never stale.
    log_path = current_exe.with_name("ragpoc.log")
    old_backup = current_exe.with_suffix(current_exe.suffix + ".old")
    pid_probe = Path(tempfile.gettempdir()) / "ragpoc_update_pid.txt"
    script_path = Path(tempfile.gettempdir()) / "ragpoc_update.bat"
    script_path.write_text(
        "@echo off\r\n"
        # Logged before anything can go wrong, so a swap that dies early leaves a trace instead of
        # looking like the updater was never invoked at all.
        f'echo [%date% %time%] Update: swap script started, waiting for pid {pid} to exit >> "{log_path}"\r\n'
        "set attempts=0\r\n"
        ":wait\r\n"
        f'tasklist /nh /fo csv /fi "PID eq {pid}" > "{pid_probe}" 2>nul\r\n'
        f'find "{pid}" "{pid_probe}" >nul 2>&1\r\n'
        "if not errorlevel 1 (\r\n"
        "  set /a attempts+=1\r\n"
        "  if %attempts% LSS 60 (\r\n"
        "    ping -n 2 127.0.0.1 >nul 2>&1\r\n"
        "    goto wait\r\n"
        "  )\r\n"
        f'  echo [%date% %time%] Update failed: pid {pid} still running after 60s, aborting swap >> "{log_path}"\r\n'
        "  goto cleanup\r\n"
        ")\r\n"
        f'del /f /q "{pid_probe}" >nul 2>&1\r\n'
        f'if exist "{old_backup}" del /f /q "{old_backup}" >nul 2>&1\r\n'
        f'robocopy "{staging_internal}" "{current_internal}" /MIR /R:15 /W:1 /NFL /NDL /NJH /NJS /NP >nul 2>&1\r\n'
        "if %errorlevel% GEQ 8 (\r\n"
        f'  echo [%date% %time%] Update failed: robocopy could not sync _internal (exit %errorlevel%) >> "{log_path}"\r\n'
        "  goto relaunch\r\n"
        ")\r\n"
        "set attempts=0\r\n"
        ":rename_retry\r\n"
        f'move /y "{current_exe}" "{old_backup}" >nul 2>&1\r\n'
        "if errorlevel 1 (\r\n"
        "  set /a attempts+=1\r\n"
        "  if %attempts% LSS 15 (\r\n"
        "    ping -n 2 127.0.0.1 >nul 2>&1\r\n"
        "    goto rename_retry\r\n"
        "  )\r\n"
        f'  echo [%date% %time%] Update failed: could not rename "{current_exe}" out of the way after 15 attempts >> "{log_path}"\r\n'
        "  goto relaunch\r\n"
        ")\r\n"
        "set attempts=0\r\n"
        ":install_retry\r\n"
        f'move /y "{staging_exe}" "{current_exe}" >nul 2>&1\r\n'
        "if errorlevel 1 (\r\n"
        "  set /a attempts+=1\r\n"
        "  if %attempts% LSS 15 (\r\n"
        "    ping -n 2 127.0.0.1 >nul 2>&1\r\n"
        "    goto install_retry\r\n"
        "  )\r\n"
        f'  echo [%date% %time%] Update failed: could not move new exe into place after 15 attempts, restoring previous version >> "{log_path}"\r\n'
        f'  move /y "{old_backup}" "{current_exe}" >nul 2>&1\r\n'
        "  goto relaunch\r\n"
        ")\r\n"
        f'del /f /q "{old_backup}" >nul 2>&1\r\n'
        f'echo [%date% %time%] Update: new build installed successfully >> "{log_path}"\r\n'
        ":relaunch\r\n"
        # A brief pause before the first launch of the exe that was JUST written to disk. UPX
        # packs the build (see ragpoc.spec), which some AV/EDR products scan more aggressively
        # than an uncompressed exe precisely because packers are a common malware-evasion
        # technique -- a scan (or any other transient handle on the freshly-written files)
        # overlapping the very first launch is exactly the kind of race this pause, and the
        # onedir switch itself (no more self-extraction step at all), both guard against.
        "ping -n 4 127.0.0.1 >nul 2>&1\r\n"
        f'start "" "{current_exe}"\r\n'
        ":cleanup\r\n"
        f'rd /s /q "{staging_dir}" >nul 2>&1\r\n'
        f'del /f /q "{zip_path}" >nul 2>&1\r\n'
        f'del /f /q "{pid_probe}" >nul 2>&1\r\n'
        f'del "%~f0"\r\n',
        encoding="utf-8",
    )
    return script_path


def cleanup_stale_update_files() -> None:
    """Best-effort sweep for *.exe.old leftovers from an update whose final cleanup step lost a
    lock race (see _write_updater_script) -- called on every startup of the frozen build, by
    which point whatever process held the file locked has long since exited. Never raises: this
    is opportunistic housekeeping, not something that should ever block startup. Safe to run
    unconditionally: it only fires after the app is already up and running again, i.e. never
    while an update is actually in flight."""
    if not getattr(sys, "frozen", False):
        return
    try:
        current_exe = Path(sys.executable).resolve()
        for stale in current_exe.parent.glob("*.exe.old"):
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass
